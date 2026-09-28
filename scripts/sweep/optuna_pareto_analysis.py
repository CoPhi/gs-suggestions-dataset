import os
import sys
import argparse
import numpy as np

# Assicura che la root del progetto sia nel sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import optuna
except ImportError:
    optuna = None

from models.bert.finetuning import ModelRegistry
from models.bert.finetuning.pipeline import pipeline_finetuning


def find_knee_point(trials: list["optuna.trial.FrozenTrial"]) -> "optuna.trial.FrozenTrial":
    """
    Identifica il 'Knee Point' (il punto di massimo compromesso) sulla frontiera di Pareto
    calcolando la minima distanza euclidea dal punto ideale (Utopia Point) nello spazio normalizzato [0, 1].
    """
    if not trials:
        raise ValueError("Lista di trial vuota.")
    if len(trials) == 1:
        return trials[0]

    matrix = np.array([t.values for t in trials], dtype=float)
    # Min-Max normalization per colonna/obiettivo
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm_matrix = (matrix - mins) / ranges

    # L'Utopia Point è [1.0, 1.0, ...]
    utopia = np.ones(norm_matrix.shape[1])
    distances = np.linalg.norm(norm_matrix - utopia, axis=1)
    best_idx = int(np.argmin(distances))
    return trials[best_idx]


def main():
    parser = argparse.ArgumentParser(
        description="Analisi della Frontiera di Pareto da Studio Optuna NSGA-II"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Nome dello studio Optuna salvato nel database",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_nsga_studies.db",
        help="Percorso al database SQLite di Optuna",
    )
    parser.add_argument(
        "--train_selected",
        action="store_true",
        help="Se specificato, avvia il training finale del modello con push su HuggingFace",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="knee",
        choices=["knee", "max_em", "max_cluster"],
        help="Criterio di selezione del modello: 'knee' (miglior compromesso), 'max_em' (massimo EM), 'max_cluster' (massima plausibilità)",
    )
    args = parser.parse_args()

    if optuna is None:
        raise ImportError("Optuna non è installato. Esegui: uv add optuna")

    print(f"\nCaricamento dello studio '{args.study_name}' da {args.storage}...")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    pareto_trials = study.best_trials
    if not pareto_trials:
        print("Nessun trial valido trovato nello studio.")
        return

    n_obj = len(pareto_trials[0].values)

    print("\n" + "=" * 95)
    print(f"       FRONTIERA DI PARETO: {len(pareto_trials)} CONFIGURAZIONI NON DOMINATE (STUDIO: {args.study_name})")
    print("=" * 95)

    if n_obj == 2:
        print(f"{'#':<4} | {'Top-1 EM':<12} | {'Cluster Inc':<14} | {'LR':<10} | {'Freeze':<8} | {'Chunk':<7} | {'Epochs':<7}")
        print("-" * 95)
        for t in sorted(pareto_trials, key=lambda x: x.values[0], reverse=True):
            p = t.params
            print(
                f"#{t.number:<3} | {t.values[0]:>10.2f}% | {t.values[1]:>12.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6} | "
                f"{p.get('chunk_size', 0):>5} | {p.get('epochs', 0):>5}"
            )
    else:
        print(f"{'#':<4} | {'Top-1 EM':<10} | {'Cluster Inc':<12} | {'CosSim Max':<12} | {'LR':<10} | {'Freeze':<8}")
        print("-" * 95)
        for t in sorted(pareto_trials, key=lambda x: x.values[0], reverse=True):
            p = t.params
            print(
                f"#{t.number:<3} | {t.values[0]:>8.2f}% | {t.values[1]:>10.2f}% | {t.values[2]:>10.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6}"
            )

    print("=" * 95)

    # Identificazione delle configurazioni speciali
    max_em_trial = max(pareto_trials, key=lambda t: t.values[0])
    max_cluster_trial = max(pareto_trials, key=lambda t: t.values[1])
    knee_trial = find_knee_point(pareto_trials)

    print("\nCONFIGURAZIONI CHIAVE SULLA FRONTIERA:")
    print(f"  1. Massima Precisione Lessicale (Max EM): Trial #{max_em_trial.number} (Top-1: {max_em_trial.values[0]:.2f}%, Cluster Inc: {max_em_trial.values[1]:.2f}%)")
    print(f"  2. Massima Plausibilità Semantica (Max Cluster): Trial #{max_cluster_trial.number} (Top-1: {max_cluster_trial.values[0]:.2f}%, Cluster Inc: {max_cluster_trial.values[1]:.2f}%)")
    print(f"  3. Miglior Compromesso Pareto (Knee Point): Trial #{knee_trial.number} (Top-1: {knee_trial.values[0]:.2f}%, Cluster Inc: {knee_trial.values[1]:.2f}%)")

    selected = knee_trial
    if args.selection_strategy == "max_em":
        selected = max_em_trial
    elif args.selection_strategy == "max_cluster":
        selected = max_cluster_trial

    print(f"\nConfigurazione selezionata ({args.selection_strategy.upper()}): Trial #{selected.number}")
    print("-" * 60)
    for k, v in sorted(selected.params.items()):
        print(f"  {k:<25}: {v}")
    print("-" * 60)

    if args.train_selected:
        print("\nAvvio dell'addestramento finale per la configurazione selezionata...")
        # Risaliamo al checkpoint dal nome dello studio o chiediamo all'utente
        # Esempio: studio 'nsga_gs_greberta_2d' -> checkpoint 'CNR-ILC/gs-GreBerta'
        ckpt_candidate = None
        for ckpt in ModelRegistry().configs.keys():
            if ckpt.split("/")[-1].lower().replace("-", "_") in args.study_name.lower():
                ckpt_candidate = ckpt
                break

        if not ckpt_candidate:
            print("Impossibile dedurre il checkpoint dal nome dello studio. Specifica manualmente con --checkpoint.")
            return

        base_model = ModelRegistry().base_model_map.get(ckpt_candidate)
        p = selected.params

        pipeline_finetuning(
            checkpoint=ckpt_candidate,
            base_model=base_model,
            lr=p["lr"],
            epochs=p.get("epochs", 3),
            batch_size=p.get("batch_size", 128),
            chunk_size=p.get("chunk_size", 128),
            num_layers_to_freeze=p.get("num_layers_to_freeze", 0),
            weight_decay=p.get("weight_decay", 0.01),
            warmup_ratio=p.get("warmup_ratio", 0.1),
            mlm_probability=p.get("mlm_probability", 0.15),
            max_span_length=p.get("max_span_length", 3),
            lr_scheduler_type=p.get("lr_scheduler_type", "cosine"),
            push_to_hub=True,
            evaluate_on_test=True,
        )


if __name__ == "__main__":
    main()
