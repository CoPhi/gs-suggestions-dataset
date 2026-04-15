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

import math

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
from models.bert.inference.predict import fill_mask

import wandb


def prepare_data(checkpoint: str, chunk_size: int = 128):
    """
    Carica il corpus MAAT e l'eval set da HuggingFace Hub, applica la
    normalizzazione model-specific e raggruppa in chunk per il training MLM.

    Args:
        checkpoint: Checkpoint fine-tuned target (es. "CNR-ILC/gs-GreBerta").
        chunk_size: Lunghezza dei blocchi di input_ids per MLM.

    Returns:
        (lm_datasets, dev_cases, test_cases): DatasetDict pronti per il Trainer,
        lista DevCase per il dev set e lista DevCase per il test set.
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

    print(f"Loading eval set from '{MAAT_EVAL_CHECKPOINT}'...")
    eval_dataset = load_dataset(MAAT_EVAL_CHECKPOINT)

    def _load_eval_split(split_name: str) -> list[DevCase]:
        cases = []
        for row in eval_dataset[split_name].to_list():
            if 1 <= row["gap_length"] <= 5:
                cases.append(
                    DevCase(
                        x=row["x"],
                        y=row["y"],
                        gap_length=row["gap_length"],
                        corpus_id=row["corpus_id"],
                        file_id=row["file_id"],
                    )
                )
        return cases

    dev_cases = _load_eval_split("dev")
    test_cases = _load_eval_split("test")

    return lm_datasets, dev_cases, test_cases


def _evaluate_hcb_on_split(
    split_name: str,
    cases: list[DevCase],
    model,
    tokenizer,
    max_cases: int | None = None,
) -> dict[str, float]:
    """
    Esegue la valutazione HCB (TopK + BERTscore) su un insieme di DevCase.

    Args:
        split_name: Nome dello split ("dev" o "test"), usato per il logging.
        cases: Lista di DevCase su cui valutare.
        model: Modello HuggingFace in eval mode.
        tokenizer: Tokenizer del modello.
        max_cases: Numero massimo di casi da valutare (None = tutti).

    Returns:
        Dizionario con tutte le metriche (topK + bertscore).
    """
    from models.bert.evaluation.metrics import (
        evaluate_topK_text,
        evaluate_bertscore_text,
    )

    pool = cases[:max_cases] if max_cases else cases
    model.eval()

    predictions_text: list[list[tuple[str, float]]] = []
    gold_labels: list[str] = []

    for case in pool:
        try:
            suggestions = fill_mask(
                text=case.x,
                n_chars=case.gap_length,
                model=model,
                tokenizer=tokenizer,
                K=10,
                beam_size=10,
                method="modified_best_to_worst",
                return_raw=False,
                case_folding=False,
            )
            predictions_text.append(suggestions)
            gold_labels.append(case.y)
        except Exception:
            pass

    if not predictions_text:
        return {}

    top_k = evaluate_topK_text(predictions_text, gold_labels)
    bert_s = evaluate_bertscore_text(predictions_text, gold_labels)
    all_metrics = {**top_k, **bert_s}

    print(
        f"[HCB {split_name}] "
        f"Top1: {top_k.get('top1', 0):.2f}% | "
        f"Top5: {top_k.get('top5', 0):.2f}% | "
        f"BERTscore F1: {bert_s.get('bertscore_f1', 0):.2f}%"
    )
    return all_metrics


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
    lm_datasets, hcb_dev_cases, hcb_test_cases = prepare_data(
        checkpoint, chunk_size=chunk_size
    )

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

    # Salvataggio metriche di training
    trainer.save_metrics("train", trainer.state.log_history[-1])
    metrics = trainer.evaluate()
    metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    # Valutazione finale HCB sul test set 
    print("\n" + "=" * 60)
    print("Valutazione HCB finale sul TEST set...")
    print("=" * 60)
    test_metrics = _evaluate_hcb_on_split(
        split_name="test",
        cases=hcb_test_cases,
        model=model,
        tokenizer=tokenizer,
    )

    if test_metrics:
        test_logs = {f"test_hcb_{k}": v for k, v in test_metrics.items()}
        trainer.save_metrics("test_hcb", test_logs)

        if wandb.run is not None:
            wandb.log(test_logs)

    if push_to_hub:
        print(f"Push del modello su HuggingFace Hub [{checkpoint}]...")
        trainer.push_to_hub()

    print("Finetuning completato.")
    return trainer
