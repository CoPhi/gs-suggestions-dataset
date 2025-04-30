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
from finetuning import (
    get_model,
    get_tokenizer,
    get_dataset,
    convert_lacuna_to_masks,
)
import collections

from hcb.hcb_infilling.decode import decode_modified_BestToWorst_vectorized
from hcb.hcb_infilling.metrics import score_batch
from hcb.hcb_infilling.utils import mask_tokens_batch


def hcb_beam_search(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    masked_text: tuple,  # (masked_text, attached_left, attached_right)
    k: int = K_PRED,
    beam_size: int = K_PRED,
) -> list[tuple[str, float, str]]:
    inputs = tokenizer(masked_text[0], return_tensors="pt")
    input_ids = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask
    mask_indices = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()

    if not mask_indices:
        return []

    beam = [(input_ids.clone(), 0.0, tokenizer.decode(input_ids[0]))]

    for mask_idx in mask_indices:
        new_beam = []

        for seq, score, text in beam:
            with torch.no_grad():
                logits = model(input_ids=seq, attention_mask=attention_mask).logits[
                    0, mask_idx, :
                ]  # logits sul primo batch (0) sul token mascherato indicato da `mask_idx` sul vocabolario `:`
                log_probs = torch.nn.functional.log_softmax(
                    logits, dim=-1
                )  # normalizzazione dei logits con una softmax logaritmica

                topk_log_probs, topk_ids = torch.topk(
                    log_probs, k=beam_size
                )  # IDs e probabilità logartimiche dei `beam_size` token più probabili
                mask_log_prob = log_probs[tokenizer.mask_token_id].item()

                for token_id, log_prob in zip(topk_ids, topk_log_probs):
                    new_seq = seq.clone()
                    new_seq[0, mask_idx] = token_id.item()
                    hcb_score = score + (log_prob.item() - mask_log_prob)
                    new_beam.append(
                        (
                            new_seq,
                            hcb_score,
                            tokenizer.decode(new_seq[0], skip_special_tokens=True),
                        )
                    )

        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]

    return _extract_top_k_results(beam, mask_indices, tokenizer, masked_text, k)


def _extract_top_k_results(beam, mask_indices, tokenizer, masked_text, k) -> list[tuple[str, float, str]]:
    result = []

    for seq, score, _ in beam[:k]:
        decoded_mask_tokens = [seq[0, idx].item() for idx in mask_indices]
        decoded_mask_sequence = tokenizer.decode(
            decoded_mask_tokens, skip_special_tokens=True
        )

        left_context = (
            tokenizer.decode(seq[0, : mask_indices[0]], skip_special_tokens=True)
            if mask_indices
            else ""
        )
        right_context = (
            tokenizer.decode(seq[0, mask_indices[-1] + 1 :], skip_special_tokens=True)
            if mask_indices
            else ""
        )

        text = _reconstruct_text(
            left_context, decoded_mask_sequence, right_context, masked_text
        )
        result.append((decoded_mask_sequence, np.exp(-score), text))

    return result


def _reconstruct_text(left_context, decoded_mask_sequence, right_context, masked_text):
    left_context = left_context.strip() if masked_text[1] else left_context
    right_context = right_context.strip() if masked_text[2] else right_context
    reconstructed_text = f"{left_context}{decoded_mask_sequence}{right_context}"
    return reconstructed_text.replace(
        "#", ""
    )  # Si eliminano riferimento a subword tokens


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


def hcb_topK_accuracy(
    model,
    tokenizer,
    dataset,
    beam_size,
    k,
    num_masks,
    num_examples=16,
    num_experiments=32,
    report_period=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    rng = np.random.default_rng(
        seed=42
    )  # si rende il calcolo riproducibile (deterministico)
    num_total = 0
    num_correct = np.zeros(beam_size)

    for batch_num in range(num_experiments):
        if batch_num % report_period == 0:
            print("Starting batch", batch_num + 1)
            print()
        example_nums = rng.choice(
            np.arange(len(dataset["input_ids"])), size=num_examples, replace=False
        ).tolist()  # Convert to a Python list
        input_ids = torch.tensor([dataset["input_ids"][i] for i in example_nums]).to(device)
        attention_mask = torch.tensor([dataset["attention_mask"][i] for i in example_nums]).to(device)
        total = input_ids.shape[1]
        if batch_num % report_period == 0:
            print("Example text:")
            print(" ".join(tokenizer.convert_ids_to_tokens(input_ids[0][:10])))
        mask_start_ind = rng.choice(np.arange(1, total - num_masks))
        if batch_num % report_period == 0:
            print(f"Index: {mask_start_ind} / {total-num_masks}")
        masked_positions = list(range(mask_start_ind, mask_start_ind + num_masks))
        masked_inputs_batch, true_ids_batch = mask_tokens_batch(
            input_ids, masked_positions, tokenizer.mask_token_id, tokenizer.pad_token_id
        )
        suggestions_batch = decode_modified_BestToWorst_vectorized(
            model,
            masked_inputs_batch,
            attention_mask,
            beam_size,
            tokenizer.mask_token_id,
        )
        count_batch, num_correct_batch = score_batch(
            suggestions_batch, true_ids_batch, tokenizer
        )
        num_correct += num_correct_batch
        num_total = +count_batch

        if batch_num % report_period == 0:
            print()
            print(f"Modified Best-to-Worst Correct: {num_correct}/{num_total}")
            print()

    return {"topK-accuracy": sum(num_correct[:k]) / num_total if num_total != 0 else 0}


def get_test_set_chunked(
    dataset,
    chunk_size=64,
    desired_columns=["input_ids", "attention_mask", "labels", "word_ids"],
):

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )  # Applica la tokenizzazione mantenendo il tipo di dato
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    lm_datasets = lm_datasets.select_columns(desired_columns)
    return lm_datasets


if __name__ == "__main__":
    test = "ἀντίγραφον ἀπʼ ἀντιγράφου Αἰγυ[....]ίας"  # πτ
    model_checkpoint = "CNR-ILC/gs-aristoBERTo"
    dataset_checkpoint = "ILC-CNR/gs-maat"
    model = get_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)
    test_set = get_test_set_chunked(get_dataset(dataset_checkpoint)["test"])
    print (hcb_topK_accuracy(model=model, tokenizer=tokenizer, dataset=test_set, beam_size=K_PRED*2, k=K_PRED, num_masks=3))

    # print ("Frase con sequenza mascherata: ", convert_lacuna_to_masks(test))
    """print(
        hcb_beam_search(
            model=model,
            tokenizer=tokenizer,
            masked_text=convert_lacuna_to_masks(test),
        )
    )"""
    """print (convert_lacuna_to_masks(test, tokenizer))"""
