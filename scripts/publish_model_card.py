"""
Script standalone per generare e pubblicare la Model Card su Hugging Face Hub.

Supporta:
1. Creazione della Model Card a partire da un file JSON generato da evaluate_comparison.py
2. Generazione diretta per un checkpoint registrato con metriche fornite da riga di comando o default
3. Push opzionale su Hugging Face Hub tramite HfApi
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Assicura che la root del progetto sia nel sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bert.finetuning import ModelRegistry, HF_TOKEN, get_model_config
from models.bert.finetuning.model_card import (
    generate_model_card,
    save_model_card,
    push_model_card_to_hub,
)

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generazione e pubblicazione su Hugging Face Hub della Model Card arricchita"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="CNR-ILC/gs-GreBerta",
        help="Checkpoint o nome del repository su Hugging Face (es. CNR-ILC/gs-GreBerta)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Checkpoint del modello base. Se omesso, viene dedotto da ModelRegistry",
    )
    parser.add_argument(
        "--from_json",
        type=str,
        default=None,
        help="Percorso di un file JSON prodotto da evaluate_comparison.py",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CNR-ILC/gs-dataset-tlg-uncased",
        help="Nome del dataset di addestramento / valutazione",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Percorso del file dove salvare la Model Card (default: README_<model>.md)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Se specificato, esegue il caricamento su Hugging Face Hub del repository del checkpoint",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint
    base_model = args.base_model
    if not base_model:
        base_model = ModelRegistry().base_model_map.get(checkpoint, checkpoint)

    pre_ft_metrics = None
    post_ft_metrics = None
    dataset_name = args.dataset_name

    if args.from_json:
        print(f"Caricamento dati di valutazione dal file JSON: {args.from_json}")
        with open(args.from_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        pre_ft_metrics = data.get("pre_ft_metrics")
        post_ft_metrics = data.get("post_ft_metrics")
        if "eval_dataset" in data:
            dataset_name = data["eval_dataset"]
        if "checkpoint" in data:
            checkpoint = data["checkpoint"]
        if "base_model" in data:
            base_model = data["base_model"]

    model_cfg = get_model_config(checkpoint)
    raw_cfg = ModelRegistry().configs.get(checkpoint, {})
    hyperparams = raw_cfg.get("hyperparameters", {})

    card_content = generate_model_card(
        checkpoint=checkpoint,
        base_model=base_model,
        dataset_name=dataset_name,
        pre_ft_metrics=pre_ft_metrics,
        post_ft_metrics=post_ft_metrics,
        hyperparameters=hyperparams,
        preprocessing_config=model_cfg,
    )

    out_path = args.output or f"README_{checkpoint.split('/')[-1]}.md"
    save_model_card(card_content, out_path)

    if args.push_to_hub:
        print(f"\nCaricamento su Hugging Face Hub per il modello [{checkpoint}]...")
        push_model_card_to_hub(
            repo_id=checkpoint,
            card_content=card_content,
            token=HF_TOKEN,
        )
        print(f"Model Card pubblicata con successo: https://huggingface.co/{checkpoint}")


if __name__ == "__main__":
    main()
