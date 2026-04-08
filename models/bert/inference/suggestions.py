import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from collections.abc import Sequence

# Aggiungiamo i percorsi al sys.path se necessario, ma assumiamo PYTHONPATH corretto
from packages.hcb_infilling.hcb_infilling.decode import (
    decode_modified_BestToWorst_vectorized,
    decode_modified_LeftToRight_vectorized,
    decode_standard_LeftToRight_vectorized,
    decode_standard_BestToWorst_vectorized
)
from models.bert.dataset.dev_set import DevCase
from models.bert.inference.utils import convert_lacuna_to_masks

def generate_hcb_suggestions(
    case: DevCase,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    num_suggestions: int = 10,
    mask_token: str = "[MASK]",
    method: str = "modified_best_to_worst",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list[tuple[float, list[int], str]]:
    """
    Genera i suggerimenti per un dato DevCase utilizzando i metodi di hcb_infilling.
    
    Args:
        case (DevCase): Il caso del dev_set contenente la frase (x) e la gold label (y).
        tokenizer (PreTrainedTokenizer): Tokenizer HF.
        model (PreTrainedModel): Modello HF (BERT/RoBERTa like).
        num_suggestions (int): Numero K di suggerimenti.
        mask_token (str): Il token maschera da utilizzare.
        method (str): Il metodo di decoding hcb.
        device (str): Device su cui eseguire l'inferenza (es. "cuda", "cpu").
        
    Returns:
        Una lista di suggerimenti, ognuno rappresentato da una tupla:
        (log_prob, token_ids, text_suggestion)
    """
    model.to(device)
    model.eval()

    # Prepara il testo: sostituisce la lacuna [..] con la sequenza di MASK corretta per HCB
    # convert_lacuna_to_masks trova la lacuna usando SUPPLEMENTS_REGEX e restituisce il testo
    # sostituendola con un singolo mask_token temporalmente (o multipli a seconda di come HCB lo vuole).
    # Tuttavia, hcb_infilling per una lacuna di lunghezza `gap_length` necessita `gap_length` tokens mascherati.
    # Quindi creiamo una stringa con N mask_tokens separati da spazio, o uniti se character level.
    
    # Costruiamo la sequenza di MASK appropriata per gap_length
    # Supponiamo un tokenizzatore a livello carattere o dove gap_length = numero di MASK
    mask_sequence = " ".join([mask_token] * case.gap_length)
    
    # Il pattern della lacuna in x è letteralmente "[..]" dove ci sono gap_length punti.
    lacuna_str = f"[{'.' * case.gap_length}]"
    
    # Sostituzione brute-force in x, visto che DevCase.x è formato per avere quel placeholder.
    if lacuna_str in case.x:
        x_masked = case.x.replace(lacuna_str, mask_sequence, 1)
    else:
        # Fallback provando convert_lacuna_to_masks
        res = convert_lacuna_to_masks(case.x, mask_sequence)
        if res is not None:
            x_masked = res[0]
        else:
            raise ValueError(f"Impossibile trovare la lacuna {lacuna_str} in {case.x}")

    # Tokenizza
    inputs = tokenizer(x_masked, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    mask_token_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids(mask_token)

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

    # Genera i suggerimenti chiamando la funzione hcb
    # L'output ha forma: per ogni batch (qui 1), una lista di `num_suggestions` liste: 
    # [prob, token1, token2, ...]
    out = decode_fn(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        beam_size=num_suggestions,
        mask_id=mask_token_id
    )

    batch_output = out[0]
    
    suggestions = []
    for cand in batch_output:
        prob = cand[0]
        token_ids = cand[1:]
        text_suggestion = tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "")
        suggestions.append((prob, token_ids, text_suggestion))
        
    # Ordiniamo per log_prob descrescente per sicurezza
    suggestions.sort(key=lambda s: s[0], reverse=True)

    return suggestions
