import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import wandb


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
    args = parser.parse_args()

    # Inizializza l'API di Wandb
    api = wandb.Api()

    print(f"Recupero dati dallo sweep: {args.sweep_id}...")
    try:
        sweep = api.sweep(args.sweep_id)
        runs = sweep.runs
    except Exception as e:
        # Tenta di aggiungere l'entity e project di default se l'utente ha passato solo l'id
        try:
            # Per default recuperiamo dal progetto locale
            from models.bert.finetuning import WANDB_PROJECT

            # Recupera il nome dell'utente attualmente loggato
            username = api.viewer.username
            full_path = f"{username}/{WANDB_PROJECT}/{args.sweep_id}"
            print(f"Tentativo con path completo: {full_path}...")
            sweep = api.sweep(full_path)
            runs = sweep.runs
        except Exception as err:
            print(f"Errore nel recupero dello sweep: {e} (Tentativo alternativo fallito: {err})")
            return

    # Filtra le run completate con successo che hanno calcolato il composite score
    valid_runs = [
        r
        for r in runs
        if r.state == "finished" and "eval/composite_score" in r.summary
    ]

    if not valid_runs:
        # Prova a cercare la chiave senza prefisso "eval/" per retrocompatibilità
        valid_runs = [
            r
            for r in runs
            if r.state == "finished" and "eval_composite_score" in r.summary
        ]
        score_key = "eval_composite_score"
    else:
        score_key = "eval/composite_score"

    if not valid_runs:
        print("Nessuna run completata con successo contenente 'composite_score' trovata.")
        return

    # Ordina le run per trovare la migliore (composite score più alto)
    best_run = max(valid_runs, key=lambda r: r.summary[score_key])
    best_score = best_run.summary[score_key]

    print("\n" + "=" * 60)
    print(f" RUN MIGLIORE TROVATA: {best_run.name} ({best_run.id})")
    print(f" Composite Score: {best_score:.2f}%")
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
