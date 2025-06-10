import torch
from finetuning import (
    get_BERT_model,
    get_tokenizer,
    convert_lacuna_to_masks,
)


def fill_mask(model, tokenizer, context, k, mask_token="[MASK]"):

    context = convert_lacuna_to_masks(
        text=context, mask_token=tokenizer.decode(tokenizer.mask_token_id)
    )
    if not context:
        return None

    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]

    if mask_token_index.nelement() == 0:  # Se non ci sono [MASK]
        print("Nessun token [MASK] trovato negli input_ids!")
        return None

    with torch.no_grad():
        logits = model(**inputs).logits

    top_k_tokens = []
    for mask_pos in mask_token_index:
        mask_logits = logits[0, mask_pos]
        top_k_ids = torch.topk(mask_logits, k).indices
        top_k_probs = torch.softmax(torch.topk(mask_logits, k).values, dim=0)
        top_k_tokens.extend(
            [
                (
                    context.replace(
                        mask_token, token.replace("##", "")
                    ),  # Testo completato
                    token.replace("##", ""),  # Token predetto (senza ##)
                    prob.item(),  # Probabilità
                )
                for token, prob in zip(
                    tokenizer.convert_ids_to_tokens(top_k_ids), top_k_probs
                )
            ]
        )

    return top_k_tokens


# Script di prova
if __name__ == "__main__":
    test = "α̣λλα μην εν τωι κατ̣[.]ς̣κευαζειν"
    model_checkpoint = "CNR-ILC/gs-aristoBERTo"

    model = get_BERT_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)

    print(fill_mask(model, tokenizer, test, 10))
