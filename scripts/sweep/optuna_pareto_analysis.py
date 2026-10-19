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

from pathlib import Path
from models.bert.finetuning import ModelRegistry
from models.bert.finetuning.pipeline import pipeline_finetuning
from scripts.sweep.optuna_nsga2 import is_ithaca_checkpoint


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
        description="Analisi della Frontiera di Pareto da Studio Optuna NSGA-II (BERT & Ithaca)"
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
        help="Se specificato, avvia il training finale del modello selezionato con valutazione su test set e log W&B",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="knee",
        choices=["knee", "max_em", "max_cluster", "max_char_acc"],
        help=(
            "Criterio di selezione del modello: "
            "'knee' (miglior compromesso), "
            "'max_em' (massimo Top-1 EM), "
            "'max_cluster' (massima plausibilità semantica), "
            "'max_char_acc' (massima accuratezza carattere 1-CER)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint esplicito (opzionale se deducibile dal nome dello studio)",
    )
    parser.add_argument(
        "--no_push_to_hub",
        action="store_true",
        help="Se specificato con --train_selected, non esegue il push su Hugging Face Hub",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disabilita il logging su Weights & Biases durante --train_selected",
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

    is_ithaca = is_ithaca_checkpoint(args.study_name) or (
        bool(args.checkpoint) and is_ithaca_checkpoint(args.checkpoint)
    )
    n_obj = len(pareto_trials[0].values)

    print("\n" + "=" * 105)
    print(f"       FRONTIERA DI PARETO: {len(pareto_trials)} CONFIGURAZIONI NON DOMINATE (STUDIO: {args.study_name})")
    print("=" * 105)

    if is_ithaca:
        print(
            f"{'#':<4} | {'Top-1 EM':<10} | {'Obj 2':<12} | {'1-CER':<10} | "
            f"{'Cluster':<10} | {'LR':<10} | {'Freeze':<6} | {'Drop':<6} | {'Strategy':<18} | {'Temp':<5}"
        )
        print("-" * 105)
        for t in sorted(pareto_trials, key=lambda x: x.values[0], reverse=True):
            p = t.params
            ua = t.user_attrs
            char_acc_v = float(ua.get("char_acc", t.values[1] if n_obj > 1 else 0.0))
            cluster_v = float(ua.get("cluster_inclusion_rate", t.values[1] if n_obj > 1 else 0.0))
            print(
                f"#{t.number:<3} | {t.values[0]:>8.2f}% | {t.values[1]:>10.2f}% | "
                f"{char_acc_v:>8.2f}% | {cluster_v:>8.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6} | "
                f"{p.get('dropout_rate', 0.1):>6.2f} | {str(p.get('strategy', 'hcb_best_to_worst')):<18} | "
                f"{p.get('temperature', 1.0):>4.2f}"
            )
    elif n_obj == 2:
        print(f"{'#':<4} | {'Top-1 EM':<12} | {'Cluster Inc':<14} | {'LR':<10} | {'Freeze':<8} | {'Chunk':<7} | {'Epochs':<7}")
        print("-" * 105)
        for t in sorted(pareto_trials, key=lambda x: x.values[0], reverse=True):
            p = t.params
            print(
                f"#{t.number:<3} | {t.values[0]:>10.2f}% | {t.values[1]:>12.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6} | "
                f"{p.get('chunk_size', 0):>5} | {p.get('epochs', 0):>5}"
            )
    else:
        print(f"{'#':<4} | {'Top-1 EM':<10} | {'Obj 2':<12} | {'Obj 3':<12} | {'LR':<10} | {'Freeze':<8}")
        print("-" * 105)
        for t in sorted(pareto_trials, key=lambda x: x.values[0], reverse=True):
            p = t.params
            print(
                f"#{t.number:<3} | {t.values[0]:>8.2f}% | {t.values[1]:>10.2f}% | {t.values[2]:>10.2f}% | "
                f"{p.get('lr', 0):>8.2e} | {p.get('num_layers_to_freeze', 0):>6}"
            )

    print("=" * 105)

    # Identificazione delle configurazioni chiave
    max_em_trial = max(pareto_trials, key=lambda t: t.values[0])
    max_cluster_trial = max(
        pareto_trials,
        key=lambda t: float(t.user_attrs.get("cluster_inclusion_rate", t.values[1] if len(t.values) > 1 else 0.0)),
    )
    max_char_trial = max(
        pareto_trials,
        key=lambda t: float(t.user_attrs.get("char_acc", t.values[1] if len(t.values) > 1 else 0.0)),
    )
    knee_trial = find_knee_point(pareto_trials)

    print("\nCONFIGURAZIONI CHIAVE SULLA FRONTIERA:")
    print(f"  1. Massima Precisione Lessicale (Max EM):       Trial #{max_em_trial.number} (Obj1: {max_em_trial.values[0]:.2f}%, Obj2: {max_em_trial.values[1]:.2f}%)")
    print(f"  2. Massima Plausibilità Semantica (Max Cluster): Trial #{max_cluster_trial.number} (Obj1: {max_cluster_trial.values[0]:.2f}%, Obj2: {max_cluster_trial.values[1]:.2f}%)")
    if is_ithaca:
        print(f"  3. Massima Accuratezza Carattere (Max 1-CER):   Trial #{max_char_trial.number} (Obj1: {max_char_trial.values[0]:.2f}%, 1-CER: {float(max_char_trial.user_attrs.get('char_acc', max_char_trial.values[1])):.2f}%)")
    print(f"  4. Miglior Compromesso Pareto (Knee Point):     Trial #{knee_trial.number} (Obj1: {knee_trial.values[0]:.2f}%, Obj2: {knee_trial.values[1]:.2f}%)")

    selected = knee_trial
    if args.selection_strategy == "max_em":
        selected = max_em_trial
    elif args.selection_strategy == "max_cluster":
        selected = max_cluster_trial
    elif args.selection_strategy == "max_char_acc":
        selected = max_char_trial

    selected_params = selected.params
    print(f"\nConfigurazione selezionata ({args.selection_strategy.upper()}): Trial #{selected.number}")
    print("-" * 60)
    for k, v in sorted(selected_params.items()):
        print(f"  {k:<25}: {v}")
    print("-" * 60)

    if args.train_selected:
        print("\nAvvio dell'addestramento finale per la configurazione selezionata...")
        if is_ithaca:
            import json
            from models.ithaca.finetuning.train import pipeline_ithaca_finetuning
            from models.ithaca.evaluation.compare import (
                render_comparison_table,
                extract_comparison_rows,
            )

            res = pipeline_ithaca_finetuning(
                checkpoint_path="checkpoints/ithaca/checkpoint_v1.pkl",
                output_dir="checkpoints/ithaca",
                train_path="data/ithaca/tlg_train.jsonl",
                val_path="data/ithaca/tlg_val.jsonl",
                test_path="data/ithaca/tlg_test.jsonl",
                epochs=int(selected_params.get("epochs", 3)),
                batch_size=int(selected_params.get("batch_size", 8)),
                lr=float(selected_params.get("lr", 2e-5)),
                warmup_ratio=float(selected_params.get("warmup_ratio", 0.1)),
                weight_decay=float(selected_params.get("weight_decay", 1e-4)),
                lr_scheduler_type=str(selected_params.get("lr_scheduler_type", "cosine")),
                num_layers_to_freeze=int(selected_params.get("num_layers_to_freeze", 0)),
                dropout_rate=float(selected_params.get("dropout_rate", 0.1)),
                strategy=str(selected_params.get("strategy", "hcb_best_to_worst")),
                temperature=float(selected_params.get("temperature", 1.0)),
                beam_size=20,
                evaluate_hcb=True,
                evaluate_on_test=True,
                save_checkpoint=True,
                log_to_wandb=not args.no_wandb,
                run_name=f"pareto_selected_gs-ithaca-tlg_{args.selection_strategy}",
                wandb_group=args.study_name,
                wandb_tags=["gs-ithaca-tlg", "ithaca", "pareto_selected", args.selection_strategy, "production"],
                wandb_job_type="pareto_production_train",
                extra_config={
                    "optuna_study": args.study_name,
                    "selected_trial": selected.number,
                    "selection_strategy": args.selection_strategy,
                },
            )

            pre_m = res.get("pre_metrics", {})
            post_m = res.get("post_metrics", {})
            out_json = Path("eval/results/eval_results_ithaca_comparison.json")
            if pre_m and post_m:
                out_json.parent.mkdir(parents=True, exist_ok=True)
                table_md = render_comparison_table(pre_m, post_m)
                rows = extract_comparison_rows(pre_m, post_m)
                report = {
                    "pre_checkpoint": "checkpoints/ithaca/checkpoint_v1.pkl",
                    "post_checkpoint": "checkpoints/ithaca/checkpoint_tlg.pkl",
                    "test_dataset": "data/ithaca/tlg_test.jsonl",
                    "strategy": str(selected_params.get("strategy", "hcb_best_to_worst")),
                    "temperature": float(selected_params.get("temperature", 1.0)),
                    "beam_size": 20,
                    "selected_trial": selected.number,
                    "selection_strategy": args.selection_strategy,
                    "hyperparameters": selected_params,
                    "pre_metrics": pre_m,
                    "post_metrics": post_m,
                    "comparison": rows,
                    "markdown_table": table_md,
                }
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"Report comparativo finale salvato in: {out_json}")

            if not args.no_push_to_hub:
                from scripts.publish_ithaca_hub import publish_ithaca_to_hub

                target_repo_id = (
                    args.checkpoint
                    if (args.checkpoint and "/" in args.checkpoint)
                    else "CNR-ILC/gs-ithaca-tlg"
                )
                publish_ithaca_to_hub(
                    repo_id=target_repo_id,
                    checkpoint_path="checkpoints/ithaca/checkpoint_tlg.pkl",
                    config_path="checkpoints/ithaca/config.json",
                    eval_json_path=str(out_json),
                    hyperparameters=selected_params,
                )
            return

        ckpt_candidate = args.checkpoint
        if not ckpt_candidate:
            for ckpt in ModelRegistry().configs.keys():
                if ckpt.split("/")[-1].lower().replace("-", "_") in args.study_name.lower():
                    ckpt_candidate = ckpt
                    break

        if not ckpt_candidate:
            print("Impossibile dedurre il checkpoint dal nome dello studio. Specifica manualmente con --checkpoint.")
            return

        base_model = ModelRegistry().base_model_map.get(ckpt_candidate)
        ckpt_short = ckpt_candidate.split("/")[-1]
        pipeline_finetuning(
            checkpoint=ckpt_candidate,
            base_model=base_model,
            lr=selected_params["lr"],
            epochs=selected_params.get("epochs", 3),
            batch_size=selected_params.get("batch_size", 128),
            chunk_size=selected_params.get("chunk_size", 128),
            num_layers_to_freeze=selected_params.get("num_layers_to_freeze", 0),
            weight_decay=selected_params.get("weight_decay", 0.01),
            warmup_ratio=selected_params.get("warmup_ratio", 0.1),
            mlm_probability=selected_params.get("mlm_probability", 0.15),
            max_span_length=selected_params.get("max_span_length", 3),
            lr_scheduler_type=selected_params.get("lr_scheduler_type", "cosine"),
            push_to_hub=not args.no_push_to_hub,
            evaluate_on_test=True,
            run_name=f"pareto_selected_{ckpt_short}_{args.selection_strategy}",
            wandb_group=args.study_name,
            wandb_tags=[ckpt_short, "pareto_selected", args.selection_strategy, "production"],
            wandb_job_type="pareto_production_train",
            extra_config={
                "optuna_study": args.study_name,
                "selected_trial": selected.number,
                "selection_strategy": args.selection_strategy,
            },
        )


if __name__ == "__main__":
    main()

