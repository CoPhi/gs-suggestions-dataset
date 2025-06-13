import torch
from finetuning import (
    get_BERT_model,
    get_tokenizer,
    convert_lacuna_to_masks,
)

def to_greek_lower(text: str) -> str:
    
    return (text
            .lower()
            .replace("ϲ", "σ")  
            .replace("Ϲ", "σ"))  

def fill_mask(model, tokenizer, context, k):
    mask_token = tokenizer.mask_token
    context = convert_lacuna_to_masks(text=context.upper(), mask_token=mask_token)
    if not context:
        return None

    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]

    if mask_positions.nelement() == 0:  # Se non ci sono [MASK]
        return None

    with torch.no_grad():
        logits = model(**inputs).logits

    results = []
    for mask_pos in mask_positions:
        mask_logits = logits[0, mask_pos]
        top_k = torch.topk(mask_logits, k)

        for token_id, prob in zip(top_k.indices, torch.softmax(top_k.values, dim=0)):
            # Crea una copia degli input_ids con il token predetto
            new_input_ids = inputs["input_ids"].clone()
            new_input_ids[0, mask_pos] = token_id
            
            filled_text = tokenizer.decode(new_input_ids[0], skip_special_tokens=True) # Decodifica l'intera sequenza
            token_str = tokenizer.decode([token_id]) # Decodifica il token singolo
            results.append((to_greek_lower(filled_text), to_greek_lower(token_str), prob.item()))

    return results

# Script di prova
if __name__ == "__main__":
    test = "μὲν εὐπαρακολού̣θητ̣α̣ π̣[.]ϲιν"
    model_checkpoint = "CNR-ILC/gs-aristoBERTo"

    model = get_BERT_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)
    
    print(fill_mask(model, tokenizer, test, 10))
