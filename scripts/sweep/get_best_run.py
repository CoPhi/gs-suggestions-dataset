import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import wandb
from dotenv import load_dotenv

# Carica le variabili d'ambiente (.env)
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Recupera e stampa i migliori iperparametri da uno sweep W&B"
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        required=True,
        help="ID dello sweep nel formato: 'entity/project/sweep_id' oppure solo 'sweep_id' se project/entity sono deducibili dal contesto",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Nome del progetto WandB (di default letto da models.bert.finetuning.WANDB_PROJECT)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Entità/Username di WandB (di default ricavato dall'API)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval/top1",
        help="Metrica target su cui selezionare la miglior run (default: 'eval/top1')",
    )
    args = parser.parse_args()

    # Inizializza l'API di Wandb
    api = wandb.Api()

    # Per default recuperiamo dal progetto locale
    from models.bert.finetuning import WANDB_PROJECT

    project = args.project or WANDB_PROJECT
    entity = args.entity

    print(f"Recupero dati dallo sweep: {args.sweep_id}...")
    
    sweep = None
    runs = None
    errors = []

    # 1. Tentativo con il path fornito così com'è
    try:
        sweep = api.sweep(args.sweep_id)
        runs = sweep.runs
    except Exception as e:
        errors.append(f"Path diretto '{args.sweep_id}': {e}")

    # 2. Se non è andato a buon fine, proviamo a costruire il path
    if sweep is None:
        candidate_paths = []
        
        if entity:
            candidate_paths.append(f"{entity}/{project}/{args.sweep_id}")
        else:
            # Prova prima solo project/sweep_id (wandb autoderiva l'entità)
            candidate_paths.append(f"{project}/{args.sweep_id}")
            
            # Prova con default_entity
            try:
                def_entity = api.default_entity
                if def_entity:
                    candidate_paths.append(f"{def_entity}/{project}/{args.sweep_id}")
            except Exception:
                pass
                
            # Prova con viewer.username
            try:
                username = api.viewer.username
                if username:
                    candidate_paths.append(f"{username}/{project}/{args.sweep_id}")
            except Exception:
                pass

        # Rimuove duplicati mantenendo l'ordine
        seen = set()
        candidate_paths = [x for x in candidate_paths if not (x in seen or seen.add(x))]

        for path in candidate_paths:
            print(f"Tentativo con path: {path}...")
            try:
                sweep = api.sweep(path)
                runs = sweep.runs
                break
            except Exception as err:
                errors.append(f"Path '{path}': {err}")

    if sweep is None:
        print("\nErrore nel recupero dello sweep. Dettagli dei tentativi:")
        for err in errors:
            print(f" - {err}")
        return

    # Determina la chiave della metrica da estrarre
    target_metric = args.metric
    candidate_keys = [
        target_metric,
        f"eval/{target_metric.replace('eval/', '')}",
        f"eval_{target_metric.replace('eval/', '')}",
        "eval/top1",
        "eval_top1",
        "eval/cluster_inclusion_rate",
        "eval_composite_score",
    ]

    score_key = None
    valid_runs = []
    for k in candidate_keys:
        v_runs = [r for r in runs if r.state == "finished" and k in r.summary]
        if v_runs:
            score_key = k
            valid_runs = v_runs
            break

    if not valid_runs:
        print(f"Nessuna run completata con successo contenente la metrica '{target_metric}' trovata.")
        return

    # Ordina le run per trovare la migliore (punteggio più alto)
    best_run = max(valid_runs, key=lambda r: r.summary[score_key])
    best_score = best_run.summary[score_key]

    print("\n" + "=" * 60)
    print(f" RUN MIGLIORE TROVATA: {best_run.name} ({best_run.id})")
    print(f" Metrica ({score_key}): {best_score:.2f}%")
    print("=" * 60)

    # Estrae la configurazione
    config = best_run.config

    # Separa metadati generali dagli iperparametri effettivi
    meta_keys = ["checkpoint", "base_model", "gap_token"]
    hyperparams = {k: v for k, v in config.items() if k not in meta_keys}
    metadata = {k: v for k, v in config.items() if k in meta_keys}

    print("\nIPERPARAMETRI OTTIMALI:")
    print("-" * 60)
    print(f"{'Iperparametro':<25} | {'Valore':<30}")
    print("-" * 60)
    for k, v in sorted(hyperparams.items()):
        print(f"{k:<25} | {str(v):<30}")
    print("-" * 60)

    print("\nMETADATI MODELLO:")
    print("-" * 60)
    for k, v in sorted(metadata.items()):
        print(f"{k:<25} | {str(v):<30}")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
