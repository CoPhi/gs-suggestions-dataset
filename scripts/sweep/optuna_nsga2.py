"""
Ottimizzazione Multi-Obiettivo con algoritmo genetico NSGA-II (Deb et al., 2002).

Utilizza optuna.samplers.NSGAIISampler, che implementa:
1. Fast Non-dominated Sorting (ordinamento rapido dei fronti di non dominanza)
2. Crowding Distance Assignment (stima della densità per preservare la diversità)
3. Selezione Elitaria (garantisce che le migliori soluzioni Pareto non vengano perse)

NON utilizza il vecchio NSGA-1 (privo di elitismo e con complessità cubica O(MN^3)).
"""

import os
import sys
import gc
import argparse
import numpy as np
import torch

# Assicura che la root del progetto sia nel sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import optuna
    from optuna.samplers import NSGAIISampler
except ImportError:
    optuna = None

from dotenv import load_dotenv
from models.bert.finetuning import ModelRegistry, HF_TOKEN, WANDB_API_KEY, wandb_login
from models.bert.finetuning.pipeline import pipeline_finetuning
from huggingface_hub import login

load_dotenv()


ITHACA_CHECKPOINT_ALIASES = {
    "CNR-ILC/gs-ithaca-tlg",
    "gs-ithaca-tlg",
    "ithaca",
    "deepmind/ithaca",
}


def is_ithaca_checkpoint(checkpoint: str) -> bool:
    """Restituisce True se il checkpoint indicato si riferisce al modello Ithaca."""
    return "ithaca" in checkpoint.lower()


def extract_eval_metrics_from_trainer(trainer) -> dict[str, float]:
    """
    Estrae l'ultimo dizionario di metriche registrato dal CustomEvaluationCallback
    all'interno di trainer.state.log_history.
    """
    if not hasattr(trainer, "state") or not trainer.state.log_history:
        return {}

    for entry in reversed(trainer.state.log_history):
        if "eval_top1" in entry:
            return entry

    return {}


def _format_objectives_tuple(
    top1_em: float,
    cluster_inc: float,
    cos_sim_top1: float,
    char_acc: float,
    mrr_pct: float,
    objectives_mode: str,
) -> tuple[float, ...]:
    """Restituisce la tupla di valori obiettivo in base alla modalità scelta."""
    if objectives_mode == "3d":
        return (top1_em, cluster_inc, cos_sim_top1)
    if objectives_mode == "em_cer":
        return (top1_em, char_acc)
    if objectives_mode == "em_cer_mrr":
        return (top1_em, char_acc, mrr_pct)
    return (top1_em, cluster_inc)


def _zero_objectives_tuple(objectives_mode: str) -> tuple[float, ...]:
    """Restituisce la tupla di fallback (0.0, ...) in caso di errore/OOM."""
    if objectives_mode in ("3d", "em_cer_mrr"):
        return (0.0, 0.0, 0.0)
    return (0.0, 0.0)


def create_objective(
    checkpoint: str,
    base_model: str,
    model_default_config: dict,
    dataset_name: str,
    eval_dataset_name: str | None,
    max_eval_cases: int,
    objectives_mode: str = "2d",
    study_name: str = "optuna_nsga2",
):
    """
    Crea la funzione obiettivo multi-obiettivo da passare ad Optuna per modelli BERT.
    """

    def objective(trial: "optuna.Trial") -> tuple[float, ...]:
        current_study_name = (
            study_name
            or (trial.study.study_name if hasattr(trial, "study") else "optuna_nsga2")
        )
        # Pulizia preventiva della memoria GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 1. Definizione dello spazio di ricerca degli iperparametri
        lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
        epochs = trial.suggest_int("epochs", 2, 4)
        chunk_size = trial.suggest_categorical("chunk_size", [128, 256])
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        num_layers_to_freeze = trial.suggest_categorical(
            "num_layers_to_freeze", [0, 4, 6, 8, 10]
        )
        weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.1])
        warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.05, 0.10, 0.15])
        mlm_probability = trial.suggest_categorical(
            "mlm_probability", [0.10, 0.15, 0.20]
        )
        max_span_length = trial.suggest_categorical("max_span_length", [2, 3])
        lr_scheduler_type = trial.suggest_categorical(
            "lr_scheduler_type", ["cosine", "linear"]
        )

        print("\n" + "=" * 80)
        ckpt_short = checkpoint.split("/")[-1]
        trial_run_name = f"trial_{trial.number:03d}_{ckpt_short}"
        trial_tags = [ckpt_short, "optuna", "nsga2", f"obj_{objectives_mode}"]

        # 2. Esecuzione del finetuning
        try:
            trainer = pipeline_finetuning(
                checkpoint=checkpoint,
                base_model=base_model,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                chunk_size=chunk_size,
                num_layers_to_freeze=num_layers_to_freeze,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                mlm_probability=mlm_probability,
                max_span_length=max_span_length,
                lr_scheduler_type=lr_scheduler_type,
                push_to_hub=False,
                evaluate_on_test=False,
                dataset_name=dataset_name,
                eval_dataset_name=eval_dataset_name,
                max_eval_cases=max_eval_cases,
                logging_steps=50,
                run_name=trial_run_name,
                wandb_group=current_study_name,
                wandb_tags=trial_tags,
                wandb_job_type="optuna_trial",
                extra_config={
                    "optuna_study": current_study_name,
                    "optuna_trial": trial.number,
                    "objectives_mode": objectives_mode,
                },
            )

            metrics = extract_eval_metrics_from_trainer(trainer)

        except Exception as e:
            print(f"[Optuna Trial {trial.number} ERROR] Fallimento durante il training: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return _zero_objectives_tuple(objectives_mode)

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 3. Estrazione delle metriche multi-obiettivo
        top1_em = float(metrics.get("eval_top1", 0.0))
        cluster_inc = float(metrics.get("eval_cluster_inclusion_rate", 0.0))
        cos_sim_top1 = float(metrics.get("eval_cos_sim_top1_max", 0.0))
        char_acc = float(metrics.get("eval_char_acc", top1_em))
        mrr_pct = float(metrics.get("eval_mrr_pct", top1_em))

        # Registrazione delle metriche ausiliarie nei user_attrs del trial
        trial.set_user_attr("top1", top1_em)
        trial.set_user_attr("top5", float(metrics.get("eval_top5", 0.0)))
        trial.set_user_attr("cluster_inclusion_rate", cluster_inc)
        trial.set_user_attr("cos_sim_top1_max", cos_sim_top1)
        trial.set_user_attr("mean_inclusion_margin", float(metrics.get("eval_mean_inclusion_margin", -1.0)))
        trial.set_user_attr("mean_gold_centroid_cosine_sim", float(metrics.get("eval_mean_gold_centroid_cosine_sim", 0.0)))

        print("\n" + "-" * 60)
        print(f" RISULTATI TRIAL {trial.number}:")
        print(f"   Top-1 Exact Match:      {top1_em:.2f}%")
        print(f"   Cluster Inclusion Rate: {cluster_inc:.2f}%")
        if objectives_mode == "3d":
            print(f"   CosSim Top-1 Max:       {cos_sim_top1:.2f}%")
        print("-" * 60 + "\n")

        return _format_objectives_tuple(
            top1_em=top1_em,
            cluster_inc=cluster_inc,
            cos_sim_top1=cos_sim_top1,
            char_acc=char_acc,
            mrr_pct=mrr_pct,
            objectives_mode=objectives_mode,
        )

    return objective


def create_ithaca_objective(
    checkpoint_path: str = "checkpoints/ithaca/checkpoint_v1.pkl",
    train_path: str = "data/ithaca/tlg_train.jsonl",
    val_path: str = "data/ithaca/tlg_val.jsonl",
    dataset_name: str = "CNR-ILC/gs-dataset-tlg-uncased",
    max_eval_cases: int = 300,
    beam_size: int = 20,
    objectives_mode: str = "2d",
    study_name: str = "nsga_gs_ithaca_tlg_2d",
    log_to_wandb: bool = True,
):
    """
    Crea la funzione obiettivo multi-obiettivo per Ithaca (JAX/Flax).
    1. Pre-carica i dataset JSONL e calcola UNA SOLA VOLTA le metriche della baseline Pre-FT
       sul set di validazione di riferimento (`pre_ft_cached_metrics`).
    2. Ad ogni trial campiona gli iperparametri di training JAX/Optax e di decodifica HCB.
    3. Esegue il fine-tuning e pubblica su Weights & Biases le tabelle comparative
       `confronto_pre_post_ft_{policy}` e `predictions_table`.
    """
    from models.ithaca.finetuning.checkpoint import (
        ensure_checkpoint_exists,
        load_ithaca_checkpoint,
    )
    from models.ithaca.finetuning.train import (
        load_jsonl_dataset,
        load_eval_cases_jsonl,
        pipeline_ithaca_finetuning,
    )
    from models.ithaca.evaluation.compare import evaluate_checkpoint_on_cases
    from models.ithaca.inference.predict import clear_ithaca_cache

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(
            f"[Ithaca Setup] Dataset JSONL non trovato in '{train_path}'. "
            f"Generazione automatica da '{dataset_name}'..."
        )
        from models.ithaca.dataset.prepare_tlg import prepare_tlg_dataset

        prepare_tlg_dataset(
            dataset_name=dataset_name,
            output_dir=os.path.dirname(train_path) or "data/ithaca",
        )

    ensure_checkpoint_exists(checkpoint_path)
    ckpt_data = load_ithaca_checkpoint(checkpoint_path)
    base_params = ckpt_data["params"]
    alphabet = ckpt_data["alphabet"]
    base_config = ckpt_data["config"]
    max_len = int(base_config.get("max_len", 768))

    print(f"[Ithaca NSGA-II] Pre-caricamento dataset di training e validazione in memoria...")
    train_data_cached = load_jsonl_dataset(train_path, alphabet, max_len=max_len)
    val_data_cached = load_jsonl_dataset(val_path, alphabet, max_len=max_len)
    eval_cases_cached = load_eval_cases_jsonl(val_path, max_cases=max_eval_cases)

    print(
        f"[Ithaca NSGA-II] Calcolo iniziale baseline Pre-FT su {len(eval_cases_cached)} casi "
        f"(verrà riutilizzato in tutti i trial per generare le tabelle W&B Pre vs Post-FT)..."
    )
    pre_ft_cached_metrics = evaluate_checkpoint_on_cases(
        checkpoint_path=checkpoint_path,
        cases=eval_cases_cached,
        strategy="hcb_best_to_worst",
        beam_size=beam_size,
        temperature=1.0,
        params_override=base_params,
        alphabet_override=alphabet,
        config_override=base_config,
        compute_semantic_metrics=True,
    )

    def objective(trial: "optuna.Trial") -> tuple[float, ...]:
        current_study_name = (
            study_name
            or (trial.study.study_name if hasattr(trial, "study") else "nsga_gs_ithaca_tlg")
        )
        clear_ithaca_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 1. Spazio di ricerca degli iperparametri (Training JAX/Optax + Decodifica HCB)
        lr = trial.suggest_float("lr", 5e-6, 1e-4, log=True)
        epochs = trial.suggest_int("epochs", 2, 5)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        num_layers_to_freeze = trial.suggest_categorical(
            "num_layers_to_freeze", [0, 1, 2, 3, 4]
        )
        weight_decay = trial.suggest_categorical(
            "weight_decay", [1e-4, 1e-3, 1e-2, 1e-1]
        )
        warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.05, 0.10, 0.15])
        lr_scheduler_type = trial.suggest_categorical(
            "lr_scheduler_type", ["cosine", "linear"]
        )
        dropout_rate = trial.suggest_categorical(
            "dropout_rate", [0.05, 0.10, 0.15, 0.20]
        )
        strategy = trial.suggest_categorical(
            "strategy", ["hcb_best_to_worst", "hcb_left_to_right"]
        )
        temperature = trial.suggest_categorical(
            "temperature", [0.70, 0.85, 1.00, 1.15]
        )

        print("\n" + "=" * 85)
        print(
            f"[Ithaca Trial #{trial.number}] lr={lr:.2e}, ep={epochs}, bs={batch_size}, "
            f"freeze={num_layers_to_freeze}/6, wd={weight_decay}, drop={dropout_rate}, "
            f"sched={lr_scheduler_type}, strat={strategy}, temp={temperature}"
        )
        print("=" * 85)

        trial_run_name = f"trial_{trial.number:03d}_gs-ithaca-tlg"
        trial_tags = [
            "gs-ithaca-tlg",
            "ithaca",
            "optuna",
            "nsga2",
            f"obj_{objectives_mode}",
        ]

        try:
            result = pipeline_ithaca_finetuning(
                checkpoint_path=checkpoint_path,
                train_path=train_path,
                val_path=val_path,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                num_layers_to_freeze=num_layers_to_freeze,
                dropout_rate=dropout_rate,
                freeze_attribution_heads=True,
                strategy=strategy,
                beam_size=beam_size,
                temperature=temperature,
                max_eval_cases=max_eval_cases,
                evaluate_hcb=True,
                evaluate_on_test=False,
                pre_ft_cached_metrics=pre_ft_cached_metrics,
                eval_cases_override=eval_cases_cached,
                train_data_override=train_data_cached,
                val_data_override=val_data_cached,
                save_checkpoint=False,
                log_to_wandb=log_to_wandb,
                run_name=trial_run_name,
                wandb_group=current_study_name,
                wandb_tags=trial_tags,
                wandb_job_type="optuna_trial",
                extra_config={
                    "optuna_study": current_study_name,
                    "optuna_trial": trial.number,
                    "objectives_mode": objectives_mode,
                },
                seed=42 + trial.number,
            )
            metrics = result.get("eval_metrics", {})

        except Exception as e:
            print(f"[Optuna Ithaca Trial {trial.number} ERROR] Fallimento durante il training: {e}")
            clear_ithaca_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return _zero_objectives_tuple(objectives_mode)

        finally:
            clear_ithaca_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        top1_em = float(metrics.get("top1", 0.0))
        top5_em = float(metrics.get("top5", 0.0))
        top20_em = float(metrics.get("top20", 0.0))
        mean_cer = float(metrics.get("mean_cer", 1.0))
        char_acc = float(metrics.get("char_acc", max(0.0, (1.0 - mean_cer) * 100.0)))
        mrr_val = float(metrics.get("mrr", 0.0))
        mrr_pct = float(metrics.get("mrr_pct", mrr_val * 100.0))
        cluster_inc = float(metrics.get("cluster_inclusion_rate", 0.0))
        cos_sim_top1 = float(metrics.get("cos_sim_top1_max", 0.0))

        # Salvataggio completo delle metriche negli user_attrs del trial Optuna
        trial.set_user_attr("top1", top1_em)
        trial.set_user_attr("top5", top5_em)
        trial.set_user_attr("top20", top20_em)
        trial.set_user_attr("mean_cer", mean_cer)
        trial.set_user_attr("char_acc", char_acc)
        trial.set_user_attr("mrr", mrr_val)
        trial.set_user_attr("mrr_pct", mrr_pct)
        trial.set_user_attr("cluster_inclusion_rate", cluster_inc)
        trial.set_user_attr("cos_sim_top1_max", cos_sim_top1)
        trial.set_user_attr(
            "mean_inclusion_margin", float(metrics.get("mean_inclusion_margin", -1.0))
        )
        trial.set_user_attr(
            "mean_gold_centroid_cosine_sim",
            float(metrics.get("mean_gold_centroid_cosine_sim", 0.0)),
        )
        trial.set_user_attr("val_loss", float(metrics.get("val_loss", 0.0)))

        print("\n" + "-" * 65)
        print(f" RISULTATI TRIAL ITHACA #{trial.number}:")
        print(f"   Top-1 Exact Match:          {top1_em:.2f}% (Top-5: {top5_em:.2f}%, Top-20: {top20_em:.2f}%)")
        print(f"   Character Accuracy (1-CER): {char_acc:.2f}% (CER: {mean_cer:.4f})")
        print(f"   Mean Reciprocal Rank (MRR): {mrr_val:.4f} ({mrr_pct:.2f}%)")
        print(f"   Cluster Inclusion Rate:     {cluster_inc:.2f}%")
        print(f"   CosSim Top-1 Max:           {cos_sim_top1:.2f}%")
        print("-" * 65 + "\n")

        return _format_objectives_tuple(
            top1_em=top1_em,
            cluster_inc=cluster_inc,
            cos_sim_top1=cos_sim_top1,
            char_acc=char_acc,
            mrr_pct=mrr_pct,
            objectives_mode=objectives_mode,
        )

    return objective


def _log_pareto_frontier_to_wandb(
    study: "optuna.Study",
    study_name: str,
    checkpoint: str,
    objectives_mode: str,
    html_out: str | None = None,
) -> None:
    """Registra una run di riepilogo su W&B contenente la tabella della Frontiera di Pareto."""
    try:
        import wandb
        from models.bert.finetuning import WANDB_PROJECT
    except ImportError:
        return

    pareto_trials = study.best_trials
    if not pareto_trials:
        return

    ckpt_short = checkpoint.split("/")[-1]
    try:
        wb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"pareto_summary_{study_name}",
            group=study_name,
            tags=[ckpt_short, "optuna", "nsga2", "pareto_frontier"],
            job_type="pareto_summary",
            config={
                "study_name": study_name,
                "checkpoint": checkpoint,
                "objectives_mode": objectives_mode,
                "total_trials": len(study.trials),
                "pareto_trials_count": len(pareto_trials),
            },
            reinit=True,
        )
        columns = [
            "Trial",
            "Top-1 EM (%)",
            "Top-5 EM (%)",
            "Cluster Inc (%)",
            "Char Acc 1-CER (%)",
            "CER",
            "MRR",
            "CosSim Max @1 (%)",
            "LR",
            "Epochs",
            "Batch Size",
            "Freeze Layers",
            "Weight Decay",
            "Dropout",
            "Strategy",
            "Temperature",
        ]
        table = wandb.Table(columns=columns)
        for t in sorted(pareto_trials, key=lambda x: x.values[0], reverse=True):
            ua = t.user_attrs
            p = t.params
            table.add_data(
                t.number,
                float(ua.get("top1", t.values[0] if t.values else 0.0)),
                float(ua.get("top5", 0.0)),
                float(ua.get("cluster_inclusion_rate", t.values[1] if len(t.values) > 1 else 0.0)),
                float(ua.get("char_acc", 0.0)),
                float(ua.get("mean_cer", 0.0)),
                float(ua.get("mrr", 0.0)),
                float(ua.get("cos_sim_top1_max", 0.0)),
                float(p.get("lr", 0.0)),
                int(p.get("epochs", 0)),
                int(p.get("batch_size", 0)),
                int(p.get("num_layers_to_freeze", 0)),
                float(p.get("weight_decay", 0.0)),
                float(p.get("dropout_rate", 0.1)),
                str(p.get("strategy", "default")),
                float(p.get("temperature", 1.0)),
            )

        log_payload: dict[str, object] = {"pareto_frontier_table": table}
        if html_out and os.path.exists(html_out):
            log_payload["pareto_front_plot"] = wandb.Html(open(html_out, "r", encoding="utf-8").read())

        wandb.log(log_payload)
        wb_run.finish()
    except Exception as e:
        print(f"[Avviso W&B] Impossibile registrare la tabella riepilogativa di Pareto: {e}")


def main():
    allowed_checkpoints = list(ModelRegistry().configs.keys()) + sorted(
        ITHACA_CHECKPOINT_ALIASES
    )
    parser = argparse.ArgumentParser(
        description="Ottimizzazione Multi-Obiettivo NSGA-II per Modelli BERT e Ithaca con Optuna"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="CNR-ILC/gs-GreBerta",
        choices=allowed_checkpoints,
        help="Checkpoint target per il finetuning (es. CNR-ILC/gs-GreBerta o CNR-ILC/gs-ithaca-tlg)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=25,
        help="Numero totale di trial da eseguire",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=8,
        help="Dimensione della popolazione per NSGA-II",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CNR-ILC/gs-dataset-tlg-uncased",
        help="Dataset di training da utilizzare",
    )
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default=None,
        help="Dataset di validazione reale (opzionale, altrimenti sintetico 2D)",
    )
    parser.add_argument(
        "--ithaca_base_ckpt",
        type=str,
        default="checkpoints/ithaca/checkpoint_v1.pkl",
        help="Percorso al checkpoint base di Ithaca (usato quando --checkpoint è Ithaca)",
    )
    parser.add_argument(
        "--ithaca_train_path",
        type=str,
        default="data/ithaca/tlg_train.jsonl",
        help="Percorso al file JSONL di training per Ithaca",
    )
    parser.add_argument(
        "--ithaca_val_path",
        type=str,
        default="data/ithaca/tlg_val.jsonl",
        help="Percorso al file JSONL di validazione per Ithaca",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=20,
        help="Beam size HCB per la valutazione di Ithaca durante i trial (default: 20)",
    )
    parser.add_argument(
        "--max_eval_cases",
        type=int,
        default=300,
        help="Numero di casi da valutare nel dev set ad ogni trial",
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="2d",
        choices=["2d", "3d", "em_cer", "em_cer_mrr"],
        help=(
            "2d: (Top-1 EM, Cluster Inclusion Rate) | "
            "3d: (+ CosSim Max) | "
            "em_cer: (Top-1 EM, Character Accuracy 1-CER) | "
            "em_cer_mrr: (Top-1 EM, 1-CER, MRR)"
        ),
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_nsga_studies.db",
        help="Database SQLite per la persistenza dello studio Optuna",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Nome identificativo dello studio Optuna (default: basato su checkpoint)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disabilita il logging su Weights & Biases",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed casuale per la riproducibilità di NSGA-II",
    )
    args = parser.parse_args()

    if optuna is None:
        raise ImportError(
            "Optuna non è installato nell'ambiente corrente. "
            "Installa le dipendenze con: uv add optuna plotly (oppure pip install optuna plotly)"
        )

    # Login servizi esterni
    if HF_TOKEN:
        login(token=HF_TOKEN)
    if WANDB_API_KEY and not args.no_wandb:
        wandb_login()

    checkpoint = args.checkpoint
    is_ithaca = is_ithaca_checkpoint(checkpoint)
    if is_ithaca:
        checkpoint = "CNR-ILC/gs-ithaca-tlg"

    ckpt_slug = checkpoint.split("/")[-1].lower().replace("-", "_")
    study_name = args.study_name or f"nsga_{ckpt_slug}_{args.objectives}"

    # 1. Configurazione del campionatore genetico NSGA-II
    sampler = NSGAIISampler(
        population_size=args.population_size,
        crossover_prob=0.9,
        mutation_prob=None,  # Default: 1 / n_params
        seed=args.seed,
    )

    # Direzioni degli obiettivi (tutti espressi come percentuali da massimizzare)
    if args.objectives in ("3d", "em_cer_mrr"):
        directions = ["maximize", "maximize", "maximize"]
    else:
        directions = ["maximize", "maximize"]

    # 2. Creazione o caricamento dello studio persistente
    print(f"\nInizializzazione studio Optuna: '{study_name}' (Storage: {args.storage})")
    print(
        f"Modello: {checkpoint} | Algoritmo: NSGA-II "
        f"(Population Size: {args.population_size}, Obiettivi: {args.objectives.upper()})"
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        sampler=sampler,
        directions=directions,
        load_if_exists=True,
    )

    # 3. Funzione obiettivo (Ithaca JAX/Flax oppure BERT PyTorch)
    if is_ithaca:
        obj_fn = create_ithaca_objective(
            checkpoint_path=args.ithaca_base_ckpt,
            train_path=args.ithaca_train_path,
            val_path=args.ithaca_val_path,
            dataset_name=args.dataset_name,
            max_eval_cases=args.max_eval_cases,
            beam_size=args.beam_size,
            objectives_mode=args.objectives,
            study_name=study_name,
            log_to_wandb=not args.no_wandb,
        )
    else:
        config = ModelRegistry().get_config(checkpoint)
        base_model = ModelRegistry().base_model_map.get(checkpoint, checkpoint)
        obj_fn = create_objective(
            checkpoint=checkpoint,
            base_model=base_model,
            model_default_config=config,
            dataset_name=args.dataset_name,
            eval_dataset_name=args.eval_dataset_name,
            max_eval_cases=args.max_eval_cases,
            objectives_mode=args.objectives,
            study_name=study_name,
        )

    # 4. Avvio dell'ottimizzazione multi-obiettivo
    study.optimize(obj_fn, n_trials=args.n_trials, gc_after_trial=True)

    # 5. Analisi e stampa della Frontiera di Pareto
    print("\n" + "=" * 100)
    print("                     FRONTIERA DI PARETO (SOLUZIONI NON DOMINATE)")
    print("=" * 100)

    pareto_trials = study.best_trials
    print(f"Trovate {len(pareto_trials)} soluzioni non dominate su {len(study.trials)} trial eseguiti.\n")

    obj_labels_map = {
        "2d": ("Top-1 EM (%)", "Cluster Inc (%)"),
        "3d": ("Top-1 EM (%)", "Cluster Inc (%)", "CosSim Max (%)"),
        "em_cer": ("Top-1 EM (%)", "Char Acc 1-CER (%)"),
        "em_cer_mrr": ("Top-1 EM (%)", "Char Acc 1-CER (%)", "MRR (%)"),
    }
    target_names = list(obj_labels_map[args.objectives])

    if len(target_names) == 2:
        print(
            f"{'Trial':<7} | {target_names[0]:<14} | {target_names[1]:<18} | "
            f"{'LR':<10} | {'Freeze':<8} | {'BS':<5} | {'Extra':<20}"
        )
        print("-" * 100)
        for t in pareto_trials:
            vals = t.values
            p = t.params
            extra_str = (
                f"drop={p.get('dropout_rate', 0.1)}, {p.get('strategy', 'hcb')[:7]}"
                if is_ithaca
                else f"chunk={p.get('chunk_size', 0)}"
            )
            print(
                f"#{t.number:<6} | {vals[0]:>12.2f}% | {vals[1]:>16.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6} | "
                f"{p.get('batch_size', 0):>3} | {extra_str:<20}"
            )
    else:
        print(
            f"{'Trial':<7} | {target_names[0]:<12} | {target_names[1]:<16} | "
            f"{target_names[2]:<14} | {'LR':<10} | {'Freeze':<8}"
        )
        print("-" * 100)
        for t in pareto_trials:
            vals = t.values
            p = t.params
            print(
                f"#{t.number:<6} | {vals[0]:>10.2f}% | {vals[1]:>14.2f}% | "
                f"{vals[2]:>12.2f}% | {p.get('lr', 0):>8.2e} | "
                f"{p.get('num_layers_to_freeze', 0):>6}"
            )
    print("=" * 100 + "\n")

    # 6. Salvataggio del grafico interattivo della frontiera di Pareto
    html_out = f"pareto_front_{study_name}.html"
    try:
        from optuna.visualization import plot_pareto_front

        fig = plot_pareto_front(study, target_names=target_names)
        fig.write_html(html_out)
        print(f"Grafico interattivo della frontiera di Pareto esportato in: {html_out}")
    except Exception as e:
        print(f"[Avviso] Impossibile generare il grafico della frontiera di Pareto: {e}")
        html_out = None

    # 7. Pubblicazione della tabella della Frontiera di Pareto su W&B
    if WANDB_API_KEY and not args.no_wandb:
        _log_pareto_frontier_to_wandb(
            study=study,
            study_name=study_name,
            checkpoint=checkpoint,
            objectives_mode=args.objectives,
            html_out=html_out,
        )


if __name__ == "__main__":
    main()

