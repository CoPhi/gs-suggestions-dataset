import re
import torch
from predictions.utils import (
    get_BERT_model,
    get_tokenizer,
    convert_lacuna_to_masks,
)
from utils.preprocess import normalize_greek


def to_greek_lower(text: str) -> str:
    """
    Converte una stringa (teoricamente in greco antico) in minuscolo e sostituisce i caratteri greci 'ϲ' e 'Ϲ' con 'σ'.

    Args:
        text (str): La stringa di input da convertire.

    Returns:
        str: La stringa convertita in minuscolo con i caratteri 'ϲ' e 'Ϲ' sostituiti da 'σ'.
    """

    return text.lower().replace("ϲ", "σ").replace("Ϲ", "σ")

def string_to_regex(pattern: str) -> str:
    regex = ''.join(['.' if char == '.' else re.escape(char) for char in pattern])
    return regex

def fill_mask(model, tokenizer, context, k, alpha=500):
    """
    Riempie le maschere presenti nel testo fornito utilizzando un modello BERT.
    Le maschere nel testo in input sono individuate da `convert_lacuna_to_masks`, che attua una conversione
    delle lacune (indicate da "[...]" o simili) nel mask token del tokenizer.
    Args:
        model: Il modello BERT pre-addestrato utilizzato per la predizione delle maschere.
        tokenizer: Il tokenizer associato al modello, utilizzato per tokenizzare il testo e gestire il token di maschera.
        context (str): Il testo di input contenente lacune da riempire, che verranno convertite in token di maschera.
        k (int): Il numero massimo di predizioni da restituire per ogni maschera individuata.
        alpha (int, opzionale): Fattore di espansione per la generazione dei suggerimenti (default=3).
    Returns:
        list: Una lista di tuple, ciascuna contenente:
            - Il testo con la maschera riempita (in minuscolo e normalizzato),
            - Il token suggerito per la maschera (in minuscolo e normalizzato),
            - La probabilità associata alla predizione.
        Se il contesto è vuoto o non contiene maschere, restituisce una lista vuota.
        Se non vengono trovate posizioni di maschera, restituisce None.
    Note:
        - La funzione filtra i suggerimenti in base alla lunghezza del token rispetto alla lacuna originale.
        - Utilizza funzioni di normalizzazione specifiche per il greco.
        - Gestisce i token che iniziano con "##" (subword tokens) per garantire la correttezza del suggerimento.
    """

    print(f"Filling masks in context: {context} {k} {alpha}")

    if not context:
        return []

    mask_token = tokenizer.mask_token
    result = convert_lacuna_to_masks(text=context.upper(), mask_token=mask_token)

    print(f"Converted lacuna to masks: {result}")

    if result is None:
        return []
    
    context, lacuna_length, maskWithChars = result # faccio unpack della tupla

    print(f"Context after conversion: {context}, lacuna_length: {lacuna_length}, maskWithChars: {maskWithChars}")

    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]
    if mask_positions.nelement() == 0:
        return None

    with torch.no_grad():
        logits = model(**inputs).logits

    results = []
    for mask_pos in mask_positions:
        mask_logits = logits[0, mask_pos]
        top_k = torch.topk(mask_logits, k * alpha)

        for token_id, prob in zip(
            top_k.indices.tolist(), torch.softmax(top_k.values, dim=0)
        ):
            token_raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
            token_str = tokenizer.convert_tokens_to_string([token_raw]).strip()

            print(f"Token string after conversion: {token_str}")

            # fallback se inizia con ##
            if token_str == "" or token_str.startswith("##"):
                token_str = token_raw.replace("##", "").strip()

            # filtro per lunghezza
            if len(token_str) != lacuna_length:
                continue
            
            regex_pattern = string_to_regex(maskWithChars)
            print(f"Regex pattern: {regex_pattern} {bool(re.match(regex_pattern, token_str))}")

            if not bool(re.match(regex_pattern, token_str)):
                continue

            filled_text = normalize_greek(context.replace(mask_token, token_str, 1))

            results.append(
                (
                    to_greek_lower(filled_text),
                    to_greek_lower(token_str),
                    float(prob.item()),
                )
            )
            
            if len(results) >= k:
                break

    return results


# Script di prova
if __name__ == "__main__":
    # μὲ]ν εὐπαρακολού̣θητ̣α̣ π̣[ᾶ]ϲιν
    test = "μὲν εὐπαρακολού̣θητ̣α̣ π̣[...]ϲιν"
    model_checkpoint = "CNR-ILC/gs-GreBerta"

    model = get_BERT_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)

    print(fill_mask(model, tokenizer, test, 10))
