import re
import math
import torch
import numpy as np
from typing import List, Tuple, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer

from packages.hcb_infilling.hcb_infilling.decode import (
    decode_modified_BestToWorst_vectorized,
    decode_modified_LeftToRight_vectorized,
    decode_standard_LeftToRight_vectorized,
    decode_standard_BestToWorst_vectorized
)

from backend.core.preprocess import normalize_greek
from models.bert.inference import GAP_TOKEN

def p_gaptoks_prior(k: int, k_min: int, k_max: int, n_chars: int) -> float:
    """
    Step 5 baseline: Prior P(gaptoks = k | n_chars).
    Usa una distribuzione uniforme tra k_min e k_max.
    Futuri sviluppi: Implementare la FCNN (Multi-layer perceptron) stile Logion, in futuro questa funzione può 
    inviare n_chars in one-hot ad un modello PyTorch / scikit-learn.
    """
    return 1.0 / (k_max - k_min + 1) # Uniform baseline

def fill_mask(
    text: str, 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    n_chars: int = None,
    K: int = 10, 
    beam_size: int = 10, 
    method: str = "modified_best_to_worst",
    case_folding: bool = True,
    return_raw: bool = False,
    normalize_probs: bool = False
) -> List[Tuple[str | List[int], float]]:
    """
    Esegue il processo di text infilling tramite HCB (Hammersley-Clifford-Besag) su 
    `k` maschere variabili.
    Il processo è articolato in 5 step: 
    1. Sostituzione lacuna con `<GAP_TOKEN>`.
        Questo placeholder viene aggiunto al vocabolario del tokenizer
        del modello usato per la generazione e sostituito con la lacuna presente nel testo.
        La sostituzione nel testo avviene tramite regex. 
    2. Calcolo range di `k` (numero di [MASK] consecutivi)
        Il range di `k` è calcolato in base al numero di caratteri presenti nella lacuna (n_chars).
        Limitiamo k_max empiricamente per via della degradazione di BERT sui mask consecutivi (k <= 3)

    3. Costruzione sequenza mascherata
        La lacuna viene sostituita con `k` [MASK] consecutivi.
        Dopodichè avviene la tokenizzazione del modello. 

    4. HCB Beam Search
        Viene eseguito il beam search per trovare i migliori suggerimenti.
        `decode_modified_BestToWorst_vectorized` usa `hcb_update` internamente
        `hcb_update` fa: f_HCB(x_i) = log p(x_i | x_!=i) - log p(MASK | x_!=i)

    5. Aggregazione e Scoring (log p(HCB) + log p(k | n_chars))
        Viene calcolato lo score finale combinando il log p(HCB) e il log p(k | n_chars).
    
    Args:
        text (str): Testo di input con lacuna definita da '[...]', es. "ΑΛΛΑ [.....] ΕΧΕΙ"
        n_chars (int): Numero atteso di caratteri nella lacuna
        model (PreTrainedModel): Modello HuggingFace (es. GreBERTa)
        tokenizer (PreTrainedTokenizer): Tokenizer del modello
        K (int): Numero finale di suggerimenti da ritornare (top-K)
        beam_size (int): Beam size per il beam search
        
    Returns:
        List[Tuple[str, float]]: Lista di tuple (suggerimento_in_testo, score)
    """
    device = next(model.parameters()).device
    model.eval()

    # STEP 1
    if GAP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [GAP_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
        
    if n_chars is None:
        match = re.search(r'\[(\.+)\]', text)
        if match:
            n_chars = len(match.group(1))
        else:
            raise ValueError("n_chars non fornito e non trovato nel testo")

    text = re.sub(r'\[\.+\]', GAP_TOKEN, text, count=1)
    
    # STEP 2

    k_min = 1
    k_max_theoretical = math.ceil(n_chars / 2) + 1
    k_max = min(k_max_theoretical, 3) # k <= 3 (o fino a 5 a discrezione)
    
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
        
        # STEP 3
        mask_str = " ".join([tokenizer.mask_token] * k)
        masked_text = text.replace(GAP_TOKEN, mask_str)
        
        # Tokenizzazione finale
        inputs = tokenizer(masked_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        mask_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        
        # Verifica se i mask ci sono (sanity check)
        if (input_ids == mask_id).sum().item() != k:
            continue # Salta se tokenizer ha fatto merge strano (raro)

        # STEP 4
        with torch.no_grad():
            out = decode_fn(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_size=beam_size,
                mask_id=mask_id
            )
        
        batch_output = out[0] # out ha batch_dim=0
        
        # Iterazione candidati generati per questo k
        for cand in batch_output:
            log_p_hcb = cand[0]
            token_ids = cand[1:]
            
            # STEP 5
            prior_prob = p_gaptoks_prior(k, k_min, k_max_theoretical, n_chars)
            log_prior = math.log(prior_prob + 1e-12) # con l'addizione si evita log(0)
            
            # Score combinato
            final_score = log_p_hcb + log_prior
            
            if return_raw:
                all_candidates.append((token_ids, final_score))
            else:
                suggestion = tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
                all_candidates.append((suggestion, final_score))

    # Ordiniamo e riportiamo Top-K
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Rimuoviamo duplicati mantenendo il migliore
    seen = set()
    unique_candidates = []
    for item, score in all_candidates:
        hashable_item = tuple(item) if return_raw else normalize_greek(item, case_folding=case_folding)
        
        if hashable_item not in seen:
            seen.add(hashable_item)
            unique_candidates.append((item if return_raw else hashable_item, score))
            if len(unique_candidates) == K:
                break
                
    if normalize_probs and len(unique_candidates) > 0:
        max_score = max(score for _, score in unique_candidates)
        unnorm_probs = [math.exp(score - max_score) for _, score in unique_candidates]
        sum_probs = sum(unnorm_probs)
        norm_probs = [p / sum_probs for p in unnorm_probs]
        unique_candidates = [(item, norm_prob) for (item, _), norm_prob in zip(unique_candidates, norm_probs)]
                
    return unique_candidates
