import re
import math
import torch
from typing import List, Tuple
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from backend.core import UNK_TOKEN
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


def estimate_mask_range(
    n_chars: int, tokenizer: PreTrainedTokenizer, is_partial_word: bool = False
) -> Tuple[int, int, int]:
    """
    Stima il range di token mascherati (k_min, k_max, k_max_theoretical)
    in base al numero di caratteri e al tokenizzatore.
    Ottimizzato in base all'analisi del vocabolario (avg tokens/word: 1.2 - 1.8).
    """
    tokenizer_class = type(tokenizer).__name__

    if "Roberta" in tokenizer_class or "BPE" in tokenizer_class:
        min_chars_per_token = 2.5
        max_chars_per_token = 4.5
    elif "Bert" in tokenizer_class or "WordPiece" in tokenizer_class:
        min_chars_per_token = 3.0
        max_chars_per_token = 5.0
    else:
        min_chars_per_token = 2.5
        max_chars_per_token = 4.0

    if is_partial_word:
        # Per le lacune parziali, il calcolo deve comunque supportare
        # i BPE token necessari a coprire n_chars.
        k_min = 1
        k_max = min(4, max(2, math.ceil(n_chars / min_chars_per_token)))
        k_max_theoretical = k_max + 1
        return k_min, k_max, k_max_theoretical

    k_min = max(1, math.floor(n_chars / max_chars_per_token))
    k_max_theoretical = math.ceil(n_chars / min_chars_per_token) + 1

    # Cap basato sulla nuova densità dei token
    k_max = min(k_max_theoretical, max(2, math.ceil(n_chars / min_chars_per_token)))

    if k_min > k_max:
        k_min = max(1, k_max - 1)

    return k_min, k_max, k_max_theoretical


def get_contextual_embeddings(
    text_with_gap: str,
    candidates: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device = None,
    batch_size: int = 32,
) -> List[torch.Tensor]:
    """
    Estrae l'embedding contestuale per ciascun candidato (e per la gold label) sostituendo
    la lacuna nel testo e applicando un mean-pooling sugli stati nascosti dei suoi token.
    Utilizza il mapping degli offset (tramite Fast Tokenizer) per essere robusto rispetto a
    qualsiasi tokenizzatore e al merging dei sub-word token (es. BPE).
    Restituisce una lista di tensori (uno per ciascun candidato).
    """
    if not candidates:
        return []

    if device is None:
        device = next(model.parameters()).device

    embeddings = []
    model.eval()

    # Troviamo l'indice di inizio e fine della lacuna nella stringa testuale originale
    gap_match = re.search(r"\[\.+\]", text_with_gap)
    if not gap_match:
        return [torch.zeros(model.config.hidden_size).to(device) for _ in candidates]

    gap_start_char = gap_match.start()
    gap_original_end = gap_match.end()

    completed_texts = []
    for cand in candidates:
        completed_text = text_with_gap[:gap_start_char] + cand + text_with_gap[gap_original_end:]
        completed_texts.append(completed_text)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    # Processiamo a chunk in caso ci siano troppi candidati per prevenire OOM
    for batch_start in range(0, len(completed_texts), batch_size):
        batch_texts = completed_texts[batch_start : batch_start + batch_size]
        batch_candidates = candidates[batch_start : batch_start + batch_size]

        if tokenizer.is_fast:
            batch_inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            offsets = batch_inputs.pop("offset_mapping")
        else:
            batch_inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            offsets = None

        batch_inputs = batch_inputs.to(device)

        with torch.no_grad():
            outputs = model(**batch_inputs, output_hidden_states=True)
            # Supporto per last_hidden_state come fallback se hidden_states non disponibile
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                batch_last_hidden = outputs.hidden_states[-1]
            else:
                batch_last_hidden = outputs.last_hidden_state

        for i, cand in enumerate(batch_candidates):
            cand_indices = []
            
            if offsets is not None:
                cand_len = len(cand)
                gap_end_char = gap_start_char + cand_len
                
                # Analizziamo gli offset per trovare i token sovrapposti all'inserimento
                for token_idx, (start_offset, end_offset) in enumerate(offsets[i]):
                    if start_offset == 0 and end_offset == 0:
                        continue
                        
                    overlap_start = max(start_offset.item(), gap_start_char)
                    overlap_end = min(end_offset.item(), gap_end_char)
                    
                    if overlap_start < overlap_end:
                        cand_indices.append(token_idx)
            else:
                # Fallback approssimativo per slow tokenizer
                prefix_text = text_with_gap[:gap_start_char]
                suffix_text = text_with_gap[gap_original_end:]
                
                prefix_inputs = tokenizer(prefix_text, add_special_tokens=False)
                suffix_inputs = tokenizer(suffix_text, add_special_tokens=False)
                inputs_no_special = tokenizer(batch_texts[i], add_special_tokens=False)
                
                true_total_len = len(batch_inputs["input_ids"][i][batch_inputs["attention_mask"][i] == 1])
                num_special = true_total_len - len(inputs_no_special["input_ids"])
                
                start_idx = len(prefix_inputs["input_ids"]) + (num_special // 2)
                num_cand_tokens = true_total_len - len(prefix_inputs["input_ids"]) - len(suffix_inputs["input_ids"]) - num_special
                
                if num_cand_tokens > 0:
                    cand_indices = list(range(start_idx, start_idx + num_cand_tokens))

            if not cand_indices:
                embeddings.append(torch.zeros(model.config.hidden_size).to(device))
                continue

            cand_indices_tensor = torch.tensor(cand_indices).to(device)
            cand_emb = batch_last_hidden[i][cand_indices_tensor]
            embeddings.append(torch.mean(cand_emb, dim=0))

    return embeddings


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

    # Rilevamento parola parziale: controlliamo se c'è un carattere alfabetico attaccato alla lacuna
    # (es. "φ[..]ερώτερον" oppure "[.....]ας") prima di sostituire la lacuna.
    is_partial = bool(re.search(r"[^\W\d_]\[\.+\]|\[\.+\][^\W\d_]", text))

    text = re.sub(r"\[\.+\]", GAP_TOKEN, text, count=1)

    k_min, k_max, k_max_theoretical = estimate_mask_range(
        n_chars, tokenizer, is_partial_word=is_partial
    )

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

        # Limite massimo del modello (512)
        max_len = int(tokenizer.model_max_length)
        if (
            max_len > 10000
        ):  # Alcuni modelli non hanno il limite impostato nel tokenizer
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
            # Nota: 'Ġ' viene prodotto da GreBerta per codificare lo spazio, 'Ċ' per l'interpunzione
            decoded = (
                decoded.replace("##", "").replace("Ġ", "").replace("Ċ", "").strip()
            )

            if not decoded:
                continue

            # Euristica di filtraggio e ricostruzione per parole parziali
            if is_partial:
                # Una parola parziale non deve contenere spazi o punteggiatura spuria nel pezzo generato
                if " " in decoded or any(p in decoded for p in ".,;:!?'\"()[]{}"):
                    continue
                # Pur applicando il controllo per evitare spazi spuri, restituiamo solo il frammento mancante
                # affinché combaci con la Gold Label e con la sostituzione in get_contextual_embeddings
                candidate_str = decoded
            else:
                candidate_str = decoded

            all_candidates.append((candidate_str, final_score))

    all_candidates.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    unique_candidates = []
    for item, score in all_candidates:
        key = (
            tuple(item)
            if return_raw
            else normalize_greek(
                item,
                case_folding=config.get("case_folding", "fold"),
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
