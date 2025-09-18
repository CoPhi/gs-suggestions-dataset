import torch
from finetuning import (
    get_BERT_model,
    get_tokenizer,
    convert_lacuna_to_masks,
)
from utils.preprocess import normalize_greek


def to_greek_lower(text: str) -> str:

    return text.lower().replace("ϲ", "σ").replace("Ϲ", "σ")


def fill_mask(model, tokenizer, context, k):
    """
    Riempie i token [MASK] presenti nel testo di input utilizzando un modello di language modeling.
    Args:
        model: Modello BERT, preso da HuggingFace.
        tokenizer: Il tokenizer associato al modello.
        context (str): Il testo di input contenente uno o più lacune da riempire (indicate con la notazione tra parentesi quadre `[...]`), che verranno convertite in token [MASK].
        k (int): Il numero di predizioni top-k da restituire per ciascun token [MASK].
    Returns:
        list of tuple or None: Una lista di tuple per ciascun token [MASK] trovato, dove ogni tupla contiene:
            - Il testo completo con il token predetto inserito.
            - Il token predetto come stringa.
            - La probabilità associata alla predizione.
        Restituisce None se il testo di input è vuoto o non contiene alcun token [MASK].
    """
    mask_token = tokenizer.mask_token
    context = convert_lacuna_to_masks(text=context.upper(), mask_token=mask_token)
    if not context:
        return None

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
        top_k = torch.topk(mask_logits, k)

        for token_id, prob in zip(
            top_k.indices.tolist(), torch.softmax(top_k.values, dim=0)
        ):
            token_raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]

            token_str = tokenizer.convert_tokens_to_string([token_raw]).strip()

            if token_str == "" or token_str.startswith("##"):
                token_str = token_raw.replace("##", "").strip()

            filled_text = normalize_greek(context.replace(mask_token, token_str, 1))

            results.append(
                (
                    to_greek_lower(filled_text),
                    to_greek_lower(token_str),
                    float(prob.item()),
                )
            )

    return results


# Script di prova
if __name__ == "__main__":
    # μὲ]ν εὐπαρακολού̣θητ̣α̣ π̣[ᾶ]ϲιν
    test = "μὲν εὐπαρακολού̣θητ̣α̣ π̣[.]ϲιν"
    model_checkpoint = "CNR-ILC/gs-GreBerta"

    model = get_BERT_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)

    print(fill_mask(model, tokenizer, test, 10))
