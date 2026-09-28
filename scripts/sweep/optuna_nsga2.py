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


def create_objective(
    checkpoint: str,
    base_model: str,
    model_default_config: dict,
    dataset_name: str,
    eval_dataset_name: str | None,
    max_eval_cases: int,
    objectives_mode: str = "2d",  # "2d" (top1, cluster_inclusion) o "3d" (+ cos_sim)
    study_name: str = "optuna_nsga2",
):
    """
    Crea la funzione obiettivo multi-obiettivo da passare ad Optuna.
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

        trial_desc = (
            f"[Trial {trial.number}] lr={lr:.2e}, ep={epochs}, bs={batch_size}, "
            f"chunk={chunk_size}, freeze={num_layers_to_freeze}, wd={weight_decay}"
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
            # Se fallisce per CUDA OOM o altro, restituiamo valori minimi per non bloccare lo studio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if objectives_mode == "3d":
                return (0.0, 0.0, 0.0)
            return (0.0, 0.0)

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 3. Estrazione delle metriche multi-obiettivo
        # Obiettivo 1: Fedeltà lessicale (Top-1 Exact Match)
        top1_em = float(metrics.get("eval_top1", 0.0))
        # Obiettivo 2: Plausibilità semantica e coesione del cluster denso
        cluster_inc = float(metrics.get("eval_cluster_inclusion_rate", 0.0))
        # Obiettivo 3 (opzionale): Similarità contestuale massima
        cos_sim_top1 = float(metrics.get("eval_cos_sim_top1_max", 0.0))

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

        if objectives_mode == "3d":
            return (top1_em, cluster_inc, cos_sim_top1)
        return (top1_em, cluster_inc)

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Ottimizzazione Multi-Obiettivo NSGA-II per Modelli BERT con Optuna"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="CNR-ILC/gs-GreBerta",
        choices=list(ModelRegistry().configs.keys()),
        help="Checkpoint target per il finetuning",
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
        "--max_eval_cases",
        type=int,
        default=300,
        help="Numero di casi da valutare nel dev set ad ogni epoca",
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="2d: (Top-1 EM, Cluster Inclusion Rate) | 3d: (+ CosSim Max)",
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
    if WANDB_API_KEY:
        wandb_login()

    checkpoint = args.checkpoint
    config = ModelRegistry().get_config(checkpoint)
    base_model = ModelRegistry().base_model_map.get(checkpoint, checkpoint)

    ckpt_slug = checkpoint.split("/")[-1].lower().replace("-", "_")
    study_name = args.study_name or f"nsga_{ckpt_slug}_{args.objectives}"

    # 1. Configurazione del campionatore genetico NSGA-II
    sampler = NSGAIISampler(
        population_size=args.population_size,
        crossover_prob=0.9,
        mutation_prob=None,  # Default: 1 / n_params
        seed=args.seed,
    )

    # Direzioni degli obiettivi (tutti da massimizzare)
    if args.objectives == "3d":
        directions = ["maximize", "maximize", "maximize"]
    else:
        directions = ["maximize", "maximize"]

    # 2. Creazione o caricamento dello studio persistente
    print(f"\nInizializzazione studio Optuna: '{study_name}' (Storage: {args.storage})")
    print(f"Algoritmo: NSGA-II (Population Size: {args.population_size}, Obiettivi: {args.objectives.upper()})")

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        sampler=sampler,
        directions=directions,
        load_if_exists=True,
    )

    # 3. Funzione obiettivo
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
    print("\n" + "=" * 90)
    print("                     FRONTIERA DI PARETO (SOLUZIONI NON DOMINATE)")
    print("=" * 90)

    pareto_trials = study.best_trials
    print(f"Trovate {len(pareto_trials)} soluzioni non dominate su {len(study.trials)} trial eseguiti.\n")

    if args.objectives == "2d":
        print(f"{'Trial':<7} | {'Top-1 EM (%)':<14} | {'Cluster Inc (%)':<16} | {'LR':<10} | {'Freeze':<8} | {'Chunk':<7} | {'BS':<5}")
        print("-" * 90)
        for t in pareto_trials:
            vals = t.values
            p = t.params
            print(
                f"#{t.number:<6} | {vals[0]:>12.2f}% | {vals[1]:>14.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6} | "
                f"{p.get('chunk_size', 0):>5} | {p.get('batch_size', 0):>3}"
            )
    else:
        print(f"{'Trial':<7} | {'Top-1 EM':<10} | {'Cluster Inc':<12} | {'CosSim Max':<12} | {'LR':<10} | {'Freeze':<8}")
        print("-" * 90)
        for t in pareto_trials:
            vals = t.values
            p = t.params
            print(
                f"#{t.number:<6} | {vals[0]:>8.2f}% | {vals[1]:>10.2f}% | {vals[2]:>10.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6}"
            )
    print("=" * 90 + "\n")

    # 6. Salvataggio del grafico interattivo della frontiera di Pareto
    try:
        from optuna.visualization import plot_pareto_front

        fig = plot_pareto_front(
            study,
            target_names=(
                ["Top-1 Exact Match (%)", "Cluster Inclusion Rate (%)"]
                if args.objectives == "2d"
                else ["Top-1 EM (%)", "Cluster Inc. (%)", "CosSim Max (%)"]
            ),
        )
        html_out = f"pareto_front_{study_name}.html"
        fig.write_html(html_out)
        print(f"Grafico interattivo della frontiera di Pareto esportato in: {html_out}")
    except Exception as e:
        print(f"[Avviso] Impossibile generare il grafico della frontiera di Pareto: {e}")


if __name__ == "__main__":
    main()
