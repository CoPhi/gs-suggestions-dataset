import torch
from finetuning import (
    get_BERT_model,
    get_tokenizer,
    convert_lacuna_to_masks,
)


def fill_mask(model, tokenizer, context, k, mask_token="[MASK]") -> list[tuple[str, float, str]] | None:
    context = convert_lacuna_to_masks(text=context, mask_token=mask_token)
    if not context:
        return

    inputs = tokenizer(context, return_tensors="pt")
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[
        1
    ]  # indice del token mascherato

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Top-k predizioni
    mask_logits = logits[0, mask_token_index.item()]
    top_k_logits, top_k_ids = torch.topk(mask_logits, k)

    # Calcolo softmax sui top-k per avere probabilità relative
    top_k_probs = torch.nn.functional.softmax(
        top_k_logits, dim=0
    )  # Normalizzazione dei logit mediante softmax
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids)

    return [
        (
            token.replace("##", ""),
            score,
            context.replace(mask_token, token.replace("##", "")),
        )
        for token, score in zip(top_k_tokens, top_k_probs.tolist())
    ]


# Script di prova
if __name__ == "__main__":
    test = "α̣λλα μην εν τωι κατ̣[.]ς̣κευαζειν"
    model_checkpoint = "CNR-ILC/gs-aristoBERTo"

    model = get_BERT_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)

    print(fill_mask(model, tokenizer, test, 10))
