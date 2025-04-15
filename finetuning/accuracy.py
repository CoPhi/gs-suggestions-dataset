"""
Script per la valutazione dell'accuracy dei modelli BERT su MAAT, dividendo il dataset in:
- training set: frasi pulite in greco antico senza la presenza di token sconosciuti, usate per fare finetuning.
- dev set: dataset usato per monitorare la loss durante il processo di finetuning, e per regolare gli iperparametri dei modelli BERT.
- test set: dataset di test, frasi pulite in greco antico con la presenza di token mascherati, usate per valutare l'accuracy del modello.
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer, default_data_collator
from config.settings import K_PRED
import torch
import numpy as np
from finetuning.utils import get_model, get_tokenizer, convert_lacuna_to_masks
from utils.preprocess import clean_text_from_gaps
import collections


def hcb_beam_search(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    masked_text: str,
    k: int = K_PRED,
    beam_size: int = K_PRED,
) -> list[str]:
    inputs = tokenizer(masked_text, return_tensors="pt")
    input_ids = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask
    mask_indices = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()

    beam = [(input_ids.clone(), 0.0)]  # (input_ids, score)

    for mask_idx in mask_indices:
        new_beam = []

        for seq, score in beam:
            with torch.no_grad():
                outputs = model(input_ids=seq, attention_mask=attention_mask)
                logits = outputs.logits[0, mask_idx, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Calcola log p(x_i | x_<i, [M]_{i:k}, x_>k)
                topk_log_probs, topk_ids = torch.topk(log_probs, k=k)

                # Calcola log p([MASK]_i | x_<i, [M]_{i:k}, x_>k)
                mask_log_prob = log_probs[tokenizer.mask_token_id].item()

                for token_id, log_prob in zip(topk_ids, topk_log_probs):
                    new_seq = seq.clone()
                    new_seq[0, mask_idx] = token_id.item()

                    hcb_score = score + (log_prob.item() - mask_log_prob)
                    new_beam.append((new_seq, hcb_score))

        # Mantieni solo i migliori beam_size candidati
        beam = sorted(new_beam, key=lambda x: x[1] ,reverse=True)[:beam_size]

    # Estrai le top-k sequenze finali
    result = []
    for seq, score in beam[:k]:
        decoded_tokens = [seq[0, idx].item() for idx in mask_indices]
        decoded_sequence = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
        normalized_score = score #da implementare
        result.append((decoded_sequence, normalized_score))

    return result


def one_word_masking_data_collator(features, tokenizer, wwm_probability=1.0):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Mappa le parole agli indici dei token
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)

        # Scegli UNA parola random da mascherare
        if len(mapping) == 0:
            continue  # niente da mascherare

        chosen_word = np.random.choice(list(mapping.keys()))
        for idx in mapping[chosen_word]:
            new_labels[idx] = labels[idx]
            input_ids[idx] = tokenizer.mask_token_id

        feature["labels"] = new_labels

    return default_data_collator(features)


def main():
    pass


if __name__ == "__main__":
    test = "ἀντίγραφον ἀπʼ ἀντιγράφου Αἰγυ[...]ίας" #πτ
    model_checkpoint = "CNR-ILC/gs-aristoBERTo"
    dataset_checkpoint = "ILC-CNR/gs-maat"
    model = get_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)
    
    print(
        hcb_beam_search(
            model=model,
            tokenizer=tokenizer,
            masked_text=convert_lacuna_to_masks(test, tokenizer),
        )
    )
    """print (convert_lacuna_to_masks(test, tokenizer))"""
