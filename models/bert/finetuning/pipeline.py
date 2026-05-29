"""
Pipeline di finetuning MLM per modelli BERT su testi in greco antico.

Il flusso di preprocessing è:
1. Caricamento del dataset grezzo (model-agnostic) da HuggingFace Hub
2. Normalizzazione model-specific via `prepare_dataset_for_model`:
   - `normalize_grc` (normalizzazione Unicode)
   - `strip_diacritics` (rimozione spiriti/accenti) se previsto dalla config
   - `remove_punctuation` se previsto dalla config
   - `case_folding` ("upper"/"lower"/None) secondo la config del modello
   - Filtraggio qualità (soglia UNK token)
   - Tokenizzazione sub-word con il tokenizer del modello target
3. Chunking in blocchi di lunghezza fissa per MLM
4. Training con DataCollatorForSpanMLM

La configurazione model-specific è centralizzata in
`models.bert.finetuning.BERT_MODEL_CONFIG`.
"""

from collections import defaultdict
import math
import random
from typing import Any

import torch
from itertools import chain
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from torch.optim import AdamW

from models.bert.dataset import CORPUS_CHECKPOINT, EVAL_CHECKPOINT
from models.bert.dataset.load import prepare_dataset_for_model
from models.bert.dataset.dev_set import DevCase
from models.bert.finetuning import get_model_config, GAP_TOKEN, WANDB_PROJECT
from models.bert.finetuning.callback import HCBEvaluationCallback
from models.bert.finetuning.collator import DataCollatorForSpanMLM
from models.bert.evaluation.metrics import (
    reset_scorer_cache,
    evaluate_contextual_similarity,
)
from models.bert.inference.predict import fill_mask, get_contextual_embeddings

import wandb


def _init_wandb(
    checkpoint: str,
    base_model: str,
    lr: float,
    batch_size: int,
    chunk_size: int,
    epochs: int,
) -> str:
    """
    Inizializza una run W&B sul progetto principale 'gs-suggestions'.
    Restituisce il run_name generato per coerenza con TrainingArguments.
    """
    ckpt_short = checkpoint.split("/")[-1]
    run_name = f"{ckpt_short}_lr{lr}_bs{batch_size}_ep{epochs}"

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "checkpoint": checkpoint,
            "base_model": base_model,
            "learning_rate": lr,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "epochs": epochs,
            "mlm_probability": 0.15,
            "max_span_length": 3,
            "gap_token": GAP_TOKEN,
        },
        tags=[ckpt_short, "finetuning", "mlm"],
        resume="allow",
    )
    return run_name


def prepare_data(
    checkpoint: str,
    tokenizer: PreTrainedTokenizer,
    chunk_size: int = 128,
) -> tuple[DatasetDict, list[DevCase], list[DevCase]]:
    """
    Carica il training set e l'eval set da HuggingFace Hub, applica la
    normalizzazione model-specific e raggruppa in chunk per il training MLM.

    Args:
        checkpoint: Checkpoint fine-tuned target (es. "CNR-ILC/gs-GreBerta").
        tokenizer: Tokenizer già istanziato, coerente con il checkpoint.
        chunk_size: Lunghezza dei blocchi di input_ids per MLM.

    Returns:
        (lm_datasets, dev_cases, test_cases): DatasetDict pronti per il Trainer,
        lista DevCase per il dev set e lista DevCase per il test set.
    """
    print(f"Loading raw corpus from '{CORPUS_CHECKPOINT}'...")
    corpus_dataset = load_dataset(CORPUS_CHECKPOINT)

    print(f"Applying model-specific normalization for [{checkpoint}]...")
    normalized_datasets = {}
    for split_name in corpus_dataset:
        normalized_datasets[split_name] = prepare_dataset_for_model(
            corpus_dataset[split_name],
            checkpoint,
        )

    def group_texts(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Consideriamo solo le colonne utili per MLM presenti nel batch (es. input_ids, attention_mask)
        keys_to_group = [
            k
            for k in ["input_ids", "attention_mask", "token_type_ids"]
            if k in examples
        ]
        concatenated = {k: list(chain(*examples[k])) for k in keys_to_group}
        total_length = len(concatenated["input_ids"])

        total_length = (total_length // chunk_size) * chunk_size

        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated.items()
        }

        # The mask will be applied dynamically by the DataCollatorForSpanMLM
        result["labels"] = result["input_ids"].copy()

        return result

    lm_datasets = DatasetDict(
        {
            split_name: ds.map(
                group_texts,
                batched=True,
                desc=f"Chunking [{split_name}]",
            ).select_columns(["input_ids", "attention_mask", "labels"])
            for split_name, ds in normalized_datasets.items()
        }
    )

    print(f"Loading eval set from '{EVAL_CHECKPOINT}'...")
    eval_dataset = load_dataset(EVAL_CHECKPOINT)

    def _load_eval_split(split_name: str) -> list[DevCase]:
        cases = []
        for row in eval_dataset[split_name].to_list():
            
            # Ci interessano le lacune di lunghezza compresa tra 1 e 6 (inclusi)
            if 1 <= row["gap_length"] <= 6:
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


def stratified_sample_by_gap(
    cases: list[DevCase], n: int, seed: int = 42
) -> list[DevCase]:
    """
    Esegue un campionamento stratificato per gap_length, mantenendo la distribuzione
    originale dei gap_length nel pool di DevCase.
    Restituisce una lista di n casi campionati stratificati per la lunghezza della lacuna.
    """
    buckets = defaultdict(list)
    for case in cases:
        buckets[case.gap_length].append(case)

    total = len(cases)
    sampled = []
    rng = random.Random(seed)

    for gap_len, bucket in sorted(buckets.items()):
        quota = max(1, round(n * len(bucket) / total))
        sampled.extend(rng.sample(bucket, min(quota, len(bucket))))

    rng.shuffle(sampled)
    return sampled[:n]


def _build_optimizer(
    model: PreTrainedModel,
    lr: float,
    weight_decay: float = 0.01,
    num_layers_to_freeze: int = 6,  # i primi 6 layer su 12
    freeze_embeddings: bool = True,
) -> AdamW:
    """
    Costruisce un ottimizzatore AdamW con weight decay selettivo e supporto al freezing dei layer:
    - Congela gli embeddings e i primi N layer dell'encoder se specificato.
    - Applica il weight decay solo ai parametri attivi (escludendo bias e LayerNorm
      seguendo le best practice del paper originale BERT/AdamW).

    Args:
        model: Il modello BERT da ottimizzare.
        lr: Learning rate per l'ottimizzatore.
        weight_decay: Valore di L2 regularization per i parametri attivi ammissibili.
        num_layers_to_freeze: Numero di layer iniziali dell'encoder da congelare (requires_grad = False).
        freeze_embeddings: Se True, congela i parametri dello strato di embedding.

    Returns:
        Istanza di AdamW con i gruppi di parametri attivi configurati.
    """
    # Si disabilita il calcolo dei gradienti per il freezing dei layer
    for name, param in model.named_parameters():
        # Si congelano gli embeddings
        if freeze_embeddings and "embeddings" in name:
            param.requires_grad = False
            continue

        # Si congelano i primi N layer dell'encoder
        is_layer_to_freeze = False
        for i in range(num_layers_to_freeze):
            if f"encoder.layer.{i}." in name:
                is_layer_to_freeze = True
                break

        if is_layer_to_freeze:
            param.requires_grad = False

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            # Parametri attivi con weight decay
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            # Parametri attivi senza weight decay (bias, LayerNorm attivi)
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)


def evaluate_metrics_on_test_set(
    split_name: str,
    cases: list[DevCase],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    checkpoint: str,
    max_cases: int | None = None,
) -> dict[str, float]:
    """
    Esegue la valutazione completa (TopK, BERTscore, Cosine Similarity Contestuale) su un insieme di test.
    """
    from models.bert.evaluation.metrics import (
        evaluate_topK_text,
        evaluate_bertscore_topk_text,
        evaluate_cosine_similarity_topk,
    )

    pool = cases[:max_cases] if max_cases else cases
    model.eval()

    predictions_text: list[list[tuple[str, float]]] = []
    gold_labels: list[str] = []
    contexts: list[str] = []
    all_similarities: list[list[float]] = []

    for case in pool:
        try:
            suggestions = fill_mask(
                text=case.x,
                checkpoint=checkpoint,
                n_chars=case.gap_length,
                model=model,
                tokenizer=tokenizer,
                K=20,
                beam_size=20,
                method="modified_best_to_worst",
                return_raw=False,
            )

            # --- INTEGRAZIONE COSINE SIMILARITY ---
            cand_texts = [s[0] for s in suggestions]

            if cand_texts:
                cand_embs = get_contextual_embeddings(
                    text_with_gap=case.x,
                    candidates=cand_texts,
                    model=model,
                    tokenizer=tokenizer,
                )

                gold_text = " ".join(case.y) if isinstance(case.y, list) else case.y
                gold_emb = get_contextual_embeddings(
                    text_with_gap=case.x,
                    candidates=[gold_text],
                    model=model,
                    tokenizer=tokenizer,
                )[0]

                similarities = evaluate_contextual_similarity(cand_embs, gold_emb)
                all_similarities.append(similarities)
            else:
                all_similarities.append([])

            predictions_text.append(suggestions)
            gold_labels.append(case.y)
            contexts.append(case.x)
        except Exception as e:
            print(f"[Eval Error] fill_mask/embeddings ha generato un'eccezione: {e}")
            print(f"[Eval Error] Case: {case}")
            continue

    if not predictions_text:
        return {}

    # 1. Calcolo Exact Match (Top-K testuale)
    topk_metrics = evaluate_topK_text(predictions_text, gold_labels)

    # 2. Calcolo BERTscore@K
    bert_s = evaluate_bertscore_topk_text(
        predictions_text,
        gold_labels,
        contexts=contexts,
        k_values=[1, 5, 10, 20],
        checkpoint=checkpoint,
    )

    # 3. Calcolo Cosine Similarity @K
    cos_sim_metrics = evaluate_cosine_similarity_topk(
        similarities_list=all_similarities, k_values=[1, 5, 10, 20]
    )

    # Uniamo le metriche
    all_metrics = {**topk_metrics, **bert_s, **cos_sim_metrics}

    print(
        f"[{split_name.upper()} SET]\n"
        f"  Exact Match:   Top-1 EM: {all_metrics.get('top1', 0):.2f}% | Top-5 EM: {all_metrics.get('top5', 0):.2f}% | Top-10 EM: {all_metrics.get('top10', 0):.2f}% | Top-20 EM: {all_metrics.get('top20', 0):.2f}%\n"
        f"  BERTScore:     F1@1: {all_metrics.get('bertscore_f1_top1', 0):.2f}% | F1@5: {all_metrics.get('bertscore_f1_top5', 0):.2f}% | F1@10: {all_metrics.get('bertscore_f1_top10', 0):.2f}% | F1@20: {all_metrics.get('bertscore_f1_top20', 0):.2f}%\n"
        f"  CosSim (Max):  @1: {all_metrics.get('cos_sim_top1_max', 0):.2f}% | @5: {all_metrics.get('cos_sim_top5_max', 0):.2f}% | @10: {all_metrics.get('cos_sim_top10_max', 0):.2f}% | @20: {all_metrics.get('cos_sim_top20_max', 0):.2f}%\n"
        f"  CosSim (Mean): @1: {all_metrics.get('cos_sim_top1_mean', 0):.2f}% | @5: {all_metrics.get('cos_sim_top5_mean', 0):.2f}% | @10: {all_metrics.get('cos_sim_top10_mean', 0):.2f}% | @20: {all_metrics.get('cos_sim_top20_mean', 0):.2f}%"
    )
    return all_metrics


def pipeline_finetuning(
    checkpoint: str,
    base_model: str,
    batch_size: int = 128,
    chunk_size: int = 128,
    epochs: int = 4,
    lr: float = 2e-5,
    logging_steps: int = 50,
    push_to_hub: bool = False,
) -> Trainer:
    """
    Esegue la pipeline completa di finetuning MLM.
    """

    # Svuota la cache degli scorer BERTScore ad ogni run per evitare che istanze
    # costruite con parametri errati (es. rescale_with_baseline=True su modelli
    # senza baseline) vengano riutilizzate in ambienti long-running (Jupyter, server).
    reset_scorer_cache()

    print(f"Checkpoint target: {checkpoint}")
    print(f"Base model (pesi): {base_model}")
    print(f"Config: {get_model_config(checkpoint)}")

    model = AutoModelForMaskedLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = 512

    if GAP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [GAP_TOKEN]})
        print(f"[setup] GAP token '{GAP_TOKEN}' aggiunto al vocabolario.")

    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    # Dataset
    print("Preparazione Dataset...")
    lm_datasets, hcb_dev_cases, hcb_test_cases = prepare_data(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
    )

    # W&B init
    ckpt_short = checkpoint.split("/")[-1]
    run_name = _init_wandb(
        checkpoint=checkpoint,
        base_model=base_model,
        lr=lr,
        batch_size=batch_size,
        chunk_size=chunk_size,
        epochs=epochs,
    )

    # Training setup
    output_dir = f"./models/bert/finetuning/gs/{ckpt_short}"
    logs_dir = f"./models/bert/finetuning/gs/{ckpt_short}-logs"

    data_collator = DataCollatorForSpanMLM(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_span_length=3,
    )

    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        hub_strategy="end",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        seed=42,
        bf16=True,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_accumulation_steps=4,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        warmup_steps=0.1,
        save_total_limit=1,
        hub_model_id=checkpoint if push_to_hub else None,
        push_to_hub=push_to_hub,
        load_best_model_at_end=True,
    )

    # Calcolo del numero totale di training steps per lo scheduler
    num_training_steps = (len(lm_datasets["train"]) // batch_size) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    # Ottimizzatore custom con weight decay selettivo
    # (no decay su bias e LayerNorm, come da best practice BERT/AdamW)
    optimizer = _build_optimizer(model, lr=lr, weight_decay=0.01)

    # Scheduler lineare con warmup coerente con warmup_ratio=0.06
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    dev_pool = stratified_sample_by_gap(hcb_dev_cases, n=300, seed=42)

    hcb_callback = HCBEvaluationCallback(
        dev_cases_pool=dev_pool,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        max_eval_cases=300,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["dev"],
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        callbacks=[hcb_callback, EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Training
    print(f"Avvio Finetuning MLM [{checkpoint}] (con check HCB epochs callback)")
    trainer.train()

    trainer.save_metrics("train", trainer.state.log_history[-1])
    metrics = trainer.evaluate()
    metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    # Valutazione HCB finale sul test set
    print("Valutazione HCB finale sul TEST set...")
    test_metrics = evaluate_metrics_on_test_set(
        split_name="test",
        cases=hcb_test_cases,
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
    )

    if test_metrics:
        test_logs = {f"test/hcb_{k}": v for k, v in test_metrics.items()}
        trainer.save_metrics("test_hcb", test_logs)
        if wandb.run is not None:
            wandb.log(test_logs)

    if push_to_hub:
        print(f"Push del modello su HuggingFace Hub [{checkpoint}]...")
        trainer.push_to_hub()

    wandb.finish()
    print("Finetuning completato.")
    return trainer
