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

import logging
import math
import random
import re
import torch
import wandb

from torch.optim import AdamW
from collections import defaultdict
from typing import Any
from itertools import chain
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from backend.core.preprocess import normalize_greek
from models.bert.dataset.load import prepare_dataset_for_model
from models.bert.dataset.dev_set import DevCase
from models.bert.finetuning import get_model_config, GAP_TOKEN, WANDB_PROJECT
from models.bert.finetuning.callback import CustomEvaluationCallback
from transformers import DataCollatorForLanguageModeling
from models.bert.evaluation.metrics import (
    reset_scorer_cache,
    evaluate_contextual_similarity,
)
from models.bert.inference.predict import fill_mask, get_contextual_embeddings

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def _init_wandb(
    checkpoint: str,
    base_model: str,
    lr: float,
    batch_size: int,
    chunk_size: int,
    epochs: int,
    num_layers_to_freeze: int,
    weight_decay: float,
    warmup_ratio: float,
    mlm_probability: float,
    max_span_length: int,
    lr_scheduler_type: str,
) -> str:
    """
    Inizializza una run W&B sul progetto principale 'gs-suggestions'.
    Se c'è già una run attiva (es. avviata da uno sweep), si limita ad aggiornarne il nome e la configurazione extra.
    Restituisce il run_name generato per coerenza con TrainingArguments.
    """
    ckpt_short = checkpoint.split("/")[-1]
    run_name = f"{ckpt_short}_lr{lr}_bs{batch_size}_ep{epochs}"

    config_dict = {
        "checkpoint": checkpoint,
        "base_model": base_model,
        "learning_rate": lr,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "epochs": epochs,
        "num_layers_to_freeze": num_layers_to_freeze,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "mlm_probability": mlm_probability,
        "max_span_length": max_span_length,
        "lr_scheduler_type": lr_scheduler_type,
        "gap_token": GAP_TOKEN,
    }

    if wandb.run is None:
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=config_dict,
            tags=[ckpt_short, "finetuning", "mlm"],
            resume="allow",
        )
    else:
        wandb.run.name = run_name
        wandb.config.update(config_dict, allow_val_change=True)

    return run_name


def generate_synthetic_cases(
    dataset: Dataset, n: int, min_gap: int = 1, max_gap: int = 6, seed: int = 42
) -> list[DevCase]:
    """
    Genera casi di test sintetici mascherando sottostringhe casuali (di lunghezza min_gap - max_gap)
    all'interno delle parole delle frasi del dataset.
    """
    rng = random.Random(seed)
    cases = []

    texts = dataset["text"]
    indices = list(range(len(texts)))
    rng.shuffle(indices)

    for idx in indices:
        if len(cases) >= n:
            break

        text = texts[idx]

        # Troviamo parole con caratteri alfabetici
        words_matches = list(re.finditer(r"[^\W\d_]+", text))
        valid_matches = [m for m in words_matches if len(m.group()) >= min_gap]

        if not valid_matches:
            continue

        match = rng.choice(valid_matches)
        word = match.group()

        max_possible_gap = min(max_gap, len(word))
        gap_length = rng.randint(min_gap, max_possible_gap)
        start_in_word = rng.randint(0, len(word) - gap_length)

        placeholder = f"[{'.' * gap_length}]"
        masked_word = (
            word[:start_in_word] + placeholder + word[start_in_word + gap_length :]
        )

        start_idx = match.start()
        end_idx = match.end()
        x_text = text[:start_idx] + masked_word + text[end_idx:]

        missing_fragment = word[start_in_word : start_in_word + gap_length]

        cases.append(
            DevCase(
                x=x_text,
                y=[missing_fragment],
                gap_length=gap_length,
                corpus_id="synthetic",
                file_id="synthetic",
            )
        )

    return cases


def _load_eval_split(eval_dataset: DatasetDict, split_name: str) -> list[DevCase]:
    """Estrae i DevCase (con lacune di lunghezza 1-6) da un dato split del dataset di valutazione."""
    cases = []
    for row in eval_dataset[split_name].to_list():
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


def group_texts(
    examples: dict[str, list[Any]], chunk_size: int = 128
) -> dict[str, list[Any]]:
    """Raggruppa le frasi in blocchi contigui di lunghezza `chunk_size`."""
    keys_to_group = [
        k for k in ["input_ids", "attention_mask", "token_type_ids"] if k in examples
    ]
    concatenated = {k: list(chain(*examples[k])) for k in keys_to_group}
    total_length = len(concatenated["input_ids"])

    total_length = (total_length // chunk_size) * chunk_size

    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated.items()
    }

    # La maschera MLM verrà applicata dinamicamente dal DataCollatorForSpanMLM
    result["labels"] = result["input_ids"].copy()
    return result


def prepare_data(
    checkpoint: str,
    dataset_name: str = "CNR-ILC/gs-dataset-tlg-uncased",
    eval_dataset_name: str | None = None,
) -> tuple[DatasetDict, list[DevCase], list[DevCase]]:
    """
    Carica il training set e l'eval set da HuggingFace Hub, applica la
    normalizzazione model-specific e raggruppa in chunk per il training MLM.

    Args:
        checkpoint: Checkpoint fine-tuned target (es. "CNR-ILC/gs-GreBerta").
        tokenizer: Tokenizer già istanziato, coerente con il checkpoint.
        chunk_size: Lunghezza dei blocchi di input_ids per MLM.
        dataset_name: Nome del dataset da caricare. Se ha solo lo split 'train', verrà splittato.
        eval_dataset_name: Nome del dataset di validazione con i test cases originali. Se None, i test cases verranno generati in maniera sintetica.

    Returns:
        (lm_datasets, dev_cases, test_cases): DatasetDict pronti per il Trainer,
        lista DevCase per il dev set e lista DevCase per il test set.
    """

    print(f"Loading raw corpus from '{dataset_name}'...")
    main_dataset = load_dataset(dataset_name)

    if "dev" in main_dataset and "test" in main_dataset:
        corpus_dataset = DatasetDict(
            {
                "train": main_dataset["train"],
                "dev": main_dataset["dev"],
                "test": main_dataset["test"],
            }
        )
    elif "dev" not in main_dataset and "test" not in main_dataset:
        print(
            f"[{dataset_name}] contains only train split. Dynamically splitting into train/dev/test..."
        )
        # 90% train, 5% dev, 5% test
        split_1 = main_dataset["train"].train_test_split(test_size=0.1, seed=42)
        split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
        corpus_dataset = DatasetDict(
            {
                "train": split_1["train"],
                "dev": split_2["train"],
                "test": split_2["test"],
            }
        )
    else:
        corpus_dataset = main_dataset

    print("Generazione dei test cases...")
    if eval_dataset_name:
        print(f"Loading eval set from '{eval_dataset_name}'...")
        eval_dataset = load_dataset(eval_dataset_name)
        dev_cases = _load_eval_split(eval_dataset, "dev")
        test_cases = _load_eval_split(eval_dataset, "test")
    else:
        print(
            "eval_dataset_name non fornito, genero casi sintetici da 'dev' e 'test' set..."
        )

        dev_cases = generate_synthetic_cases(corpus_dataset["dev"], n=300, max_gap=6)
        test_cases = generate_synthetic_cases(corpus_dataset["test"], n=1000, max_gap=6)

    print(f"Applying model-specific normalization for [{checkpoint}]...")
    normalized_datasets = {}

    for split_name in ["train", "dev"]:
        if split_name in corpus_dataset:
            normalized_datasets[split_name] = prepare_dataset_for_model(
                corpus_dataset[split_name],
                checkpoint,
            )

    lm_datasets = DatasetDict(normalized_datasets)

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
                beam_size=50,
                method="modified_best_to_worst",
                return_raw=False,
            )

            # --- INTEGRAZIONE COSINE SIMILARITY ---
            cand_texts = [s[0] for s in suggestions]

            if cand_texts:
                gold_text = " ".join(case.y) if isinstance(case.y, list) else case.y

                # eseguiamo una singola estrazione batch per candidati e gold text
                all_texts_to_embed = cand_texts + [gold_text]

                embs = get_contextual_embeddings(
                    text_with_gap=case.x,
                    candidates=all_texts_to_embed,
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint=checkpoint,
                )

                gold_emb = embs[-1]
                cand_embs = embs[:-1]

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

    config = get_model_config(checkpoint)
    is_strip = config.get("strip_diacritics", True)
    case_fold = config.get("case_folding", "fold")

    normalized_gold_labels = []
    for case_y in gold_labels:
        if isinstance(case_y, list):
            norm_y = [
                normalize_greek(
                    y, case_folding=case_fold, strip_diacritics_flag=is_strip
                )
                for y in case_y
            ]
        else:
            norm_y = normalize_greek(
                case_y, case_folding=case_fold, strip_diacritics_flag=is_strip
            )
        normalized_gold_labels.append(norm_y)

    # 1. Calcolo Exact Match (Top-K testuale)
    topk_metrics = evaluate_topK_text(predictions_text, normalized_gold_labels)

    # 2. Calcolo BERTscore@K
    bert_s = evaluate_bertscore_topk_text(
        predictions_text,
        normalized_gold_labels,
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

    import wandb

    if wandb.run is not None:
        columns = ["Context", "Gold Label", "Top 20 Predictions"]
        data = []
        # Logghiamo i primi 100 casi per non appesantire troppo W&B
        for i, case in enumerate(pool[:100]):
            if i < len(predictions_text):
                gold = ", ".join(case.y) if isinstance(case.y, list) else case.y
                # Formattiamo i primi 20 suggerimenti senza assegnare il punteggio
                top_preds = " | ".join([s[0] for s in predictions_text[i][:20]])
                data.append([case.x, gold, top_preds])

        table = wandb.Table(columns=columns, data=data)
        wandb.log({f"{split_name}/predictions_table": table})

    return all_metrics


def pipeline_finetuning(
    checkpoint: str,
    base_model: str,
    batch_size: int,
    chunk_size: int,
    epochs: int,
    lr: float,
    num_layers_to_freeze: int,
    weight_decay: float,
    warmup_ratio: float,
    mlm_probability: float,
    max_span_length: int,
    lr_scheduler_type: str,
    logging_steps: int = 50,
    push_to_hub: bool = False,
    dataset_name: str = "CNR-ILC/gs-dataset-tlg-uncased",
    eval_dataset_name: str | None = None,
    evaluate_on_test: bool = True,
    max_eval_cases: int = 500,
) -> Trainer:
    """
    Esegue la pipeline completa di finetuning MLM.
    Gli iperparametri devono essere passati esplicitamente.
    """
    config = get_model_config(checkpoint)

    # Svuota la cache degli scorer BERTScore ad ogni run per evitare che istanze
    # costruite con parametri errati (es. rescale_with_baseline=True su modelli
    # senza baseline) vengano riutilizzate in ambienti long-running (Jupyter, server).
    reset_scorer_cache()

    print(f"Checkpoint target: {checkpoint}")
    print(f"Base model (pesi): {base_model}")
    print(f"Config & Hyperparams: {config}")

    model = AutoModelForMaskedLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.model_max_length = 512

    # Decommenta se utilizzi il DataCollatorForSyntheticGap

    # if GAP_TOKEN not in tokenizer.get_vocab():
    #     tokenizer.add_special_tokens({"additional_special_tokens": [GAP_TOKEN]})
    #     print(f"[setup] GAP token '{GAP_TOKEN}' aggiunto al vocabolario.")

    # model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    # Dataset
    print("Preparazione Dataset...")
    lm_datasets, eval_dev_cases, eval_test_cases = prepare_data(
        checkpoint=checkpoint,
        dataset_name=dataset_name,
        eval_dataset_name=eval_dataset_name,
    )

    lm_datasets = lm_datasets.map(
        group_texts,
        batched=True,
        fn_kwargs={"chunk_size": chunk_size},
        desc=f"Grouping texts in chunks of {chunk_size}",
        remove_columns=lm_datasets["train"].column_names,
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
        num_layers_to_freeze=num_layers_to_freeze,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        mlm_probability=mlm_probability,
        max_span_length=max_span_length,
        lr_scheduler_type=lr_scheduler_type,
    )

    pre_ft_metrics = None
    if evaluate_on_test:
        print("Valutazione baseline sul TEST set (Pre-FT)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        pre_ft_metrics = evaluate_metrics_on_test_set(
            split_name="test_pre_ft",
            cases=eval_test_cases,
            model=model,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
        )

        if pre_ft_metrics and wandb.run is not None:
            pre_ft_logs = {f"test_pre_ft/{k}": v for k, v in pre_ft_metrics.items()}
            wandb.log(pre_ft_logs)

    # Training setup
    output_dir = f"./models/bert/finetuning/gs/{ckpt_short}"

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
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
        remove_unused_columns=True,
    )

    # Calcolo del numero totale di training steps per lo scheduler
    num_training_steps = (len(lm_datasets["train"]) // batch_size) * epochs
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    # Ottimizzatore custom con weight decay selettivo e freezing configurabile
    optimizer = _build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        num_layers_to_freeze=num_layers_to_freeze,
    )

    scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    dev_pool = stratified_sample_by_gap(eval_dev_cases, n=300, seed=42)

    eval_callback = CustomEvaluationCallback(
        dev_cases_pool=dev_pool,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        max_eval_cases=max_eval_cases,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["dev"],
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        callbacks=[eval_callback, EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Training
    print(f"Avvio Finetuning di [{checkpoint}]")
    trainer.train()

    trainer.save_metrics("train", trainer.state.log_history[-1])
    metrics = trainer.evaluate()
    metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    post_ft_metrics = None
    if evaluate_on_test:
        print("Valutazione finale sul TEST set (Post-FT)...")
        post_ft_metrics = evaluate_metrics_on_test_set(
            split_name="test_post_ft",
            cases=eval_test_cases,
            model=model,
            tokenizer=tokenizer,
            checkpoint=checkpoint,
        )

        if post_ft_metrics:
            post_ft_logs = {f"test_post_ft/{k}": v for k, v in post_ft_metrics.items()}
            trainer.save_metrics("test_post_ft", post_ft_logs)
            if wandb.run is not None:
                wandb.log(post_ft_logs)

    # Generazione tabella comparativa
    if pre_ft_metrics and post_ft_metrics:
        print("\n" + "-" * 70)
        print("                  CONFRONTO METRICHE: PRE-FT VS POST-FT (TEST SET)")
        print("-" * 70)
        print(f"{'Metrica':<25} | {'Pre-FT':<10} | {'Post-FT':<10} | {'Delta':<10}")
        print("-" * 70)

        comparison_keys = [
            ("top1", "Exact Match @1"),
            ("top5", "Exact Match @5"),
            ("top10", "Exact Match @10"),
            ("top20", "Exact Match @20"),
            ("bertscore_f1_top1", "BERTScore F1 @1"),
            ("bertscore_f1_top5", "BERTScore F1 @5"),
            ("bertscore_f1_top10", "BERTScore F1 @10"),
            ("bertscore_f1_top20", "BERTScore F1 @20"),
            ("cos_sim_top1_max", "CosSim Max @1"),
            ("cos_sim_top5_max", "CosSim Max @5"),
            ("cos_sim_top10_max", "CosSim Max @10"),
            ("cos_sim_top20_max", "CosSim Max @20"),
        ]

        table_data = []
        for key, name in comparison_keys:
            val_pre = pre_ft_metrics.get(key, 0.0)
            val_post = post_ft_metrics.get(key, 0.0)
            delta = val_post - val_pre
            delta_str = f"{delta:+.2f}%" if delta != 0 else "0.00%"
            print(f"{name:<25} | {val_pre:>8.2f}% | {val_post:>8.2f}% | {delta_str:>8}")
            table_data.append([name, val_pre, val_post, delta])

        print("=" * 80 + "\n")

        if wandb.run is not None:
            wb_table = wandb.Table(
                columns=["Metrica", "Pre-FT (%)", "Post-FT (%)", "Delta (%)"]
            )
            for row in table_data:
                wb_table.add_data(*row)
            wandb.log({"confronto_pre_post_ft": wb_table})

    if push_to_hub:
        print(f"Push del modello su HuggingFace Hub [{checkpoint}]...")
        trainer.push_to_hub()

    wandb.finish()
    print("Finetuning completato.")
    return trainer
