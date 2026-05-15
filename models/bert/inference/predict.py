import re
import math
import torch
from typing import List, Tuple
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from backend.core import _CASE_FOLDING, UNK_TOKEN
from packages.hcb_infilling.hcb_infilling.decode import (
    decode_modified_BestToWorst_vectorized,
    decode_modified_LeftToRight_vectorized,
    decode_standard_LeftToRight_vectorized,
    decode_standard_BestToWorst_vectorized,
)

from backend.core.preprocess import (
    normalize_greek,
    process_editorial_marks,
    remove_punctuation,
)
from models.bert.finetuning import GAP_TOKEN, get_model_config


def p_gaptoks_prior(k: int, k_min: int, k_max: int, n_chars: int) -> float:
    """
    Step 5 baseline: Prior P(gaptoks = k | n_chars).
    Usa una distribuzione uniforme tra k_min e k_max.
    Futuri sviluppi: Implementare la FCNN (Multi-layer perceptron) stile Logion, in futuro questa funzione può
    inviare n_chars in one-hot ad un modello PyTorch / scikit-learn.
    """
    return 1.0 / (k_max - k_min + 1)


def fill_mask(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    checkpoint: str = None,  
    n_chars: int = None,
    K: int = 20,
    beam_size: int = 20,
    method: str = "modified_best_to_worst",
    return_raw: bool = False,
    normalize_probs: bool = False,
) -> List[Tuple[str | List[int], float]]:

    device = next(model.parameters()).device
    model.eval()

    if checkpoint is None: 
        config = {
            "remove_punct": False,
            "strip_diacritics": True,
            "case_folding": "fold",
        }
    else:
        try:
            config = get_model_config(checkpoint)
        except ValueError as e:
            print(f"{e}")

    if tokenizer.unk_token:
        text = text.replace(UNK_TOKEN, tokenizer.unk_token)

    # trasformazioni model-specific del testo in input
    text = process_editorial_marks(text, preserve_lacunae=True)
    text = normalize_greek(
        text,
        case_folding=config.get("case_folding", "upper"),
        strip_diacritics_flag=config.get("strip_diacritics", True),
    )
    if config.get("remove_punct"):
        text = remove_punctuation(text, preserve_lacunae=True)


    if n_chars is None:
        match = re.search(r"\[(\.+)\]", text)
        if match:
            n_chars = len(match.group(1))
        else:
            raise ValueError("n_chars non fornito e non trovato nel testo")

    text = re.sub(r"\[\.+\]", GAP_TOKEN, text, count=1)

    k_min = 1
    k_max_theoretical = math.ceil(n_chars / 2) + 1
    k_max = min(k_max_theoretical, 3)

    all_candidates: List[Tuple[str, float]] = []

    # Scelta del metodo di generazione
    if method == "modified_best_to_worst":
        decode_fn = decode_modified_BestToWorst_vectorized
    elif method == "modified_left_to_right":
        decode_fn = decode_modified_LeftToRight_vectorized
    elif method == "standard_left_to_right":
        decode_fn = decode_standard_LeftToRight_vectorized
    elif method == "standard_best_to_worst":
        decode_fn = decode_standard_BestToWorst_vectorized
    else:
        raise ValueError(f"Metodo {method} non supportato.")

    for k in range(k_min, k_max + 1):

        mask_str = " ".join([tokenizer.mask_token] * k)
        masked_text = text.replace(GAP_TOKEN, mask_str)

        # Preveniamo la perdita della lacuna se il testo è troppo lungo (> 512 token)
        # Tokenizziamo senza troncamento automatico per trovare la posizione della maschera
        full_enc = tokenizer(masked_text, add_special_tokens=False, truncation=False)
        full_input_ids = full_enc["input_ids"]

        mask_id = tokenizer.mask_token_id
        mask_indices = [i for i, tid in enumerate(full_input_ids) if tid == mask_id]

        # Limite massimo del modello (solitamente 512)
        max_len = int(tokenizer.model_max_length)
        if max_len > 10000:  # Alcuni modelli non hanno il limite impostato nel tokenizer
            max_len = 512

        # Riserviamo 2 token per [CLS] e [SEP]
        max_tokens_body = max_len - 2

        if len(full_input_ids) > max_tokens_body:
            if mask_indices:
                # Centriamo la finestra intorno alla lacuna
                center = (mask_indices[0] + mask_indices[-1]) // 2
                start = max(0, center - (max_tokens_body // 2))
                end = start + max_tokens_body

                # Se sforiamo a destra, spostiamo a sinistra
                if end > len(full_input_ids):
                    end = len(full_input_ids)
                    start = max(0, end - max_tokens_body)

                input_ids_chunk = full_input_ids[start:end]
            else:
                # Se non c'è maschera (non dovrebbe succedere), troncamento standard
                input_ids_chunk = full_input_ids[:max_tokens_body]
        else:
            input_ids_chunk = full_input_ids

        # Aggiungiamo i token speciali e convertiamo in tensore
        cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 101
        sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 102
        input_ids = [cls_id] + input_ids_chunk + [sep_id]
        input_ids = torch.tensor([input_ids]).to(device)
        attention_mask = torch.ones_like(input_ids)
        mask_id = tokenizer.mask_token_id

        if (input_ids == mask_id).sum().item() != k:
            continue

        with torch.no_grad():
            out = decode_fn(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_size=beam_size,
                mask_id=mask_id,
            )

        batch_output = out[0]

        for cand in batch_output:
            log_p_hcb = cand[0]
            token_ids = cand[1:]

            prior_prob = p_gaptoks_prior(k, k_min, k_max_theoretical, n_chars)
            log_prior = math.log(prior_prob + 1e-12)
            final_score = log_p_hcb + log_prior

            
            if return_raw:
                all_candidates.append((token_ids, final_score))
                continue

            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # sanitizzazione per confronto stringhe: rimuoviamo artefatti di tokenizzazione come 
            # "##" o "Ġ" prodotti da tokenizzatori WordPiece o Byte-Pair Encoding, e normalizziamo spazi bianchi e caratteri invisibili
            decoded = decoded.replace("##", "").replace("Ġ", "").replace("Ċ", "")
            decoded = re.sub(r'[\s\u200B-\u200D\uFEFF]', '', decoded)
            
            if not decoded.strip():
                continue

            all_candidates.append((decoded, final_score))

    all_candidates.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_candidates = []
    for item, score in all_candidates:
        key = (
            tuple(item)
            if return_raw
            else normalize_greek(
                item,
                case_folding="fold",
                strip_diacritics_flag=config.get("strip_diacritics"),
            )
        )
        if key not in seen:
            seen.add(key)
            unique_candidates.append((item if return_raw else key, score))
        if len(unique_candidates) == K:
            break

    if normalize_probs and unique_candidates:
        max_score = max(s for _, s in unique_candidates)
        unnorm = [math.exp(s - max_score) for _, s in unique_candidates]
        total = sum(unnorm)
        unique_candidates = [
            (item, p / total) for (item, _), p in zip(unique_candidates, unnorm)
        ]

    return unique_candidates
