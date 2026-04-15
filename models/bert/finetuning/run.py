"""
Entrypoint CLI per il finetuning MLM di modelli BERT su testi in greco antico.

Uso:
    poetry run python -m models.bert.finetuning.run --checkpoint CNR-ILC/gs-GreBerta
    poetry run python -m models.bert.finetuning.run --checkpoint CNR-ILC/gs-Logion --epochs 5
    poetry run python -m models.bert.finetuning.run --checkpoint CNR-ILC/gs-aristoBERTo --push_to_hub
"""

import argparse
import torch

from models.bert.finetuning import BERT_MODEL_CONFIG, BASE_MODEL_MAP
from models.bert.finetuning.pipeline import pipeline_finetuning


def main():
    parser = argparse.ArgumentParser(description="Finetuning MLM per greco antico")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="CNR-ILC/gs-GreBerta",
        choices=list(BERT_MODEL_CONFIG.keys()),
        help="Checkpoint fine-tuned target (determina la normalizzazione del testo)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Se specificato, carica il modello su HuggingFace Hub al termine",
    )
    args = parser.parse_args()

    torch.cuda.empty_cache()

    checkpoint = args.checkpoint
    base_model = BASE_MODEL_MAP.get(checkpoint, checkpoint)

    pipeline_finetuning(
        checkpoint=checkpoint,
        base_model=base_model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        lr=args.lr,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()