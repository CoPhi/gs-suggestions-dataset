import os
import argparse
import torch
from huggingface_hub import login

from models.bert.finetuning import BERT_MODEL_CONFIG, BASE_MODEL_MAP, HF_TOKEN, WANDB_API_KEY, wandb_login
from models.bert.finetuning.pipeline import pipeline_finetuning

"""

    Esempio:

    

    uv run python -m models.bert.finetuning.run --checkpoint "CNR-ILC/gs-Logion"
    uv run python -m models.bert.finetuning.run --checkpoint "CNR-ILC/gs-GreBerta"
    uv run python -m models.bert.finetuning.run --checkpoint "CNR-ILC/gs-aristoBERTo"
"""

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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=5000, help="Frequenza di log (steps) per la loss su wandb")
    parser.add_argument(
        "--no_push_to_hub",
        action="store_false",
        dest="push_to_hub",
        help="Disabilita il caricamento del modello su HuggingFace Hub al termine",
    )

    args = parser.parse_args()

    torch.cuda.empty_cache()

    if HF_TOKEN:
        login(token=HF_TOKEN)

    if WANDB_API_KEY:
        wandb_login()

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
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()