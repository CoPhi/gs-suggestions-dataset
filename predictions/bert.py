import re
import torch
import torch.nn.functional as F
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
    Riempie le maschere presenti nel testo fornito utilizzando un modello BERT con ri-ordinamento semantico.
    
    Le maschere nel testo in input sono individuate da `convert_lacuna_to_masks` e convertite con il token mascherato usato dal modello BERT utilizzato (`model`).
    Utilizza la similarità del coseno tra gli embedding del contesto e i candidati per migliorare il ranking.
    In particolare, calcola un vettore centroide del contesto (escludendo i token speciali) e confronta ogni candidato con questo centroide per ottenere una misura di similarità semantica con il contesto.
    

    Args:
        model: Il modello BERT pre-addestrato utilizzato per la predizione delle maschere.
        tokenizer: Il tokenizer associato al modello.
        context (str): Il testo di input contenente lacune.
        k (int): Il numero massimo di predizioni da restituire.
        alpha (int, opzionale): Fattore di espansione per la generazione dei candidati (default=500).

    Returns:
        list: Una lista di tuple (testo_riempito, token_suggerito, score_combinato).
    """

    if not context:
        return []

    mask_token = tokenizer.mask_token
    result = convert_lacuna_to_masks(text=context.upper(), mask_token=mask_token)

    if result is None:
        return []
    
    context, lacuna_length, maskWithChars = result
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]
    
    if mask_positions.nelement() == 0:
        return None

    # 1. Inference con output_hidden_states per il reranking semantico
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        # Prendiamo l'ultimo layer degli hidden states per rappresentare il contesto
        last_hidden_state = outputs.hidden_states[-1] 

    # 2. Calcolo del Centroide del Contesto (escludendo token speciali [CLS], [SEP], [MASK])
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, mask_token_id]
    context_mask = ~torch.isin(inputs["input_ids"], torch.tensor(special_tokens).to(model.device))
    
    if context_mask.any():
        context_embeddings = last_hidden_state[0][context_mask[0]]
        context_centroid = torch.mean(context_embeddings, dim=0, keepdim=True)
    else:
        context_centroid = None

    # Otteniamo la matrice degli embedding statici (word embeddings) per il confronto
    word_embeddings = model.get_input_embeddings().weight

    results = []
    mask_pos = mask_positions[0] 
    
    mask_logits = logits[0, mask_pos]
    
    top_k_raw = torch.topk(mask_logits, k * alpha) # Usiamo `alpha` alto per pescare molti candidati da riordinare semanticamente
    probs = torch.softmax(top_k_raw.values, dim=0)

    regex_pattern = string_to_regex(maskWithChars) #Si escapano i caratteri speciali

    for token_id, prob in zip(top_k_raw.indices.tolist(), probs):
        token_raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        token_str = tokenizer.convert_tokens_to_string([token_raw]).strip()

        if token_str == "" or token_str.startswith("##"):
            token_str = token_raw.replace("##", "").strip()

        # Filtro lunghezza rigido
        if len(token_str) != lacuna_length:
            continue
        
        # Filtro Regex (Head/Tail)
        if not bool(re.match(regex_pattern, token_str)):
            continue

        # 3. Calcolo della Cosine Similarity semantica
        current_token_embedding = word_embeddings[token_id].unsqueeze(0)
        if context_centroid is not None:
            semantic_sim = F.cosine_similarity(context_centroid, current_token_embedding).item()
        else:
            semantic_sim = 0.0

        # 4. Scoring Ponderato (40% probabilità MLM, 60% similarità semantica)
        # La similarità viene clippata a 0 per evitare contributi negativi
        final_score = (prob.item() * 0.4) + (max(0, semantic_sim) * 0.6)

        filled_text = normalize_greek(context.replace(mask_token, token_str, 1))
        
        results.append({
            "text": to_greek_lower(filled_text),
            "token": to_greek_lower(token_str),
            "score": float(final_score)
        })

    # 5. Reranking finale basato sullo score combinato e restituzione dei top k
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]

    return [(r["text"], r["token"], r["score"]) for r in results]


# Script di prova
if __name__ == "__main__":
    # μὲ]ν εὐπαρακολού̣θητ̣α̣ π̣[ᾶ]ϲιν
    test = "μὲν εὐπαρακολού̣θητ̣α̣ π̣[...]ϲιν"
    model_checkpoint = "CNR-ILC/gs-GreBerta"

    model = get_BERT_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)

    print(fill_mask(model, tokenizer, test, 10))
