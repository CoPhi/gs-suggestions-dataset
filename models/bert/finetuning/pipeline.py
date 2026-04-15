"""
Pipeline di finetuning MLM per modelli BERT su testi in greco antico.

Il flusso di preprocessing è:
1. Caricamento del dataset grezzo (model-agnostic) da HuggingFace Hub
2. Normalizzazione model-specific via `prepare_dataset_for_model`:
   - normalize_grc (normalizzazione Unicode)
   - strip_diacritics (rimozione spiriti/accenti) se previsto dalla config
   - remove_punctuation se previsto dalla config
   - case_folding ("upper"/"lower"/None) secondo la config del modello
   - Filtraggio qualità (soglia UNK token)
   - Tokenizzazione sub-word con il tokenizer del modello target
3. Chunking in blocchi di lunghezza fissa per MLM
4. Training con DataCollatorForLanguageModeling

La configurazione model-specific è centralizzata in
`models.bert.finetuning.BERT_MODEL_CONFIG`.
"""

import torch
from itertools import chain
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
)

from models.bert.dataset import MAAT_CORPUS_CHECKPOINT, MAAT_EVAL_CHECKPOINT
from models.bert.dataset.load import prepare_dataset_for_model
from models.bert.dataset.dev_set import DevCase
from models.bert.finetuning import get_model_config
from models.bert.finetuning.callback import HCBEvaluationCallback


def prepare_data(checkpoint: str, chunk_size: int = 128):
    """
    Carica il corpus MAAT e l'eval set da HuggingFace Hub, applica la
    normalizzazione model-specific e raggruppa in chunk per il training MLM.

    Args:
        checkpoint: Checkpoint fine-tuned target (es. "CNR-ILC/gs-GreBerta").
        chunk_size: Lunghezza dei blocchi di input_ids per MLM.

    Returns:
        (lm_datasets, dev_cases): DatasetDict pronti per il Trainer e lista DevCase.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # --- Corpus MLM ---
    print(f"Loading raw corpus from '{MAAT_CORPUS_CHECKPOINT}'...")
    corpus_dataset = load_dataset(MAAT_CORPUS_CHECKPOINT)

    print(f"Applying model-specific normalization for [{checkpoint}]...")
    normalized_datasets = {}
    for split_name in corpus_dataset:
        normalized_datasets[split_name] = prepare_dataset_for_model(
            corpus_dataset[split_name], checkpoint
        )

    # Chunking: raggruppa i blocchi tokenizzati in sequenze di lunghezza fissa
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # Padding per arrivare a multiplo di chunk_size
        if total_length % chunk_size != 0:
            padding_length = chunk_size - (total_length % chunk_size)
            for key in concatenated:
                pad_value = 0 if key == "attention_mask" else tokenizer.pad_token_id
                concatenated[key] += [pad_value] * padding_length

        total_length = (len(concatenated["input_ids"]) // chunk_size) * chunk_size
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = DatasetDict(
        {
            split_name: ds.map(
                group_texts, batched=True, desc=f"Chunking [{split_name}]"
            ).select_columns(["input_ids", "attention_mask", "labels"])
            for split_name, ds in normalized_datasets.items()
        }
    )

    # --- Eval set per HCB callback ---
    print(f"Loading eval set from '{MAAT_EVAL_CHECKPOINT}'...")
    eval_dataset = load_dataset(MAAT_EVAL_CHECKPOINT)

    dev_cases = []
    for row in eval_dataset["dev"].to_list():
        if 1 <= row["gap_length"] <= 5:
            dev_cases.append(
                DevCase(
                    x=row["x"],
                    y=row["y"],
                    gap_length=row["gap_length"],
                    corpus_id=row["corpus_id"],
                    file_id=row["file_id"],
                )
            )

    return lm_datasets, dev_cases


def pipeline_finetuning(
    checkpoint: str,
    base_model: str,
    batch_size: int = 16,
    chunk_size: int = 128,
    epochs: int = 10,
    lr: float = 5e-5,
    push_to_hub: bool = False,
):
    """
    Esegue la pipeline completa di finetuning MLM.

    Args:
        checkpoint: Checkpoint fine-tuned target (es. "CNR-ILC/gs-GreBerta").
                    Determina tokenizer e normalizzazione model-specific.
        base_model: Checkpoint base da cui caricare i pesi del modello
                    (es. "bowphs/GreBerta").
        batch_size: Batch size per il training.
        chunk_size: Lunghezza dei blocchi di input_ids per MLM.
        epochs: Numero di epoche di training.
        lr: Learning rate.
        push_to_hub: Se True, carica il modello su HuggingFace Hub al termine.
    """
    print(f"Checkpoint target: {checkpoint}")
    print(f"Base model (pesi): {base_model}")
    print(f"Config: {get_model_config(checkpoint)}")

    model = AutoModelForMaskedLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = 512

    print("Preparazione Dataset...")
    lm_datasets, hcb_dev_cases = prepare_data(checkpoint, chunk_size=chunk_size)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    # Directory di output basate sul nome del checkpoint
    ckpt_short = checkpoint.split("/")[-1]
    output_dir = f"./models/bert/finetuning/gs/{ckpt_short}"
    logs_dir = f"./models/bert/finetuning/gs/{ckpt_short}-logs"

    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logs_dir,
        report_to="wandb",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        seed=42,
        fp16=True,
        logging_strategy="epoch",
        eval_accumulation_steps=4,
        torch_empty_cache_steps=5000,
        dataloader_drop_last=True,
        save_total_limit=2,
        hub_model_id=checkpoint if push_to_hub else None,
        push_to_hub=push_to_hub,
    )

    hcb_callback = HCBEvaluationCallback(
        dev_cases_pool=hcb_dev_cases,
        tokenizer=tokenizer,
        max_eval_cases=50,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["dev"],
        data_collator=data_collator,
        callbacks=[hcb_callback],
    )

    print(f"Avvio Finetuning MLM [{checkpoint}] (con check HCB epochs callback)")
    trainer.train()

    # Salvataggio metriche
    trainer.save_metrics("train", trainer.state.log_history[-1])
    metrics = trainer.evaluate()
    metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    if push_to_hub:
        print(f"Push del modello su HuggingFace Hub [{checkpoint}]...")
        trainer.push_to_hub()

    print("Finetuning completato.")
    return trainer
