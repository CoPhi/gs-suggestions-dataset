import argparse
import torch
from huggingface_hub import login

from models.bert.finetuning import (
    ModelRegistry,
    HF_TOKEN,
    WANDB_API_KEY,
    wandb_login,
)
from models.bert.finetuning.pipeline import pipeline_finetuning

"""
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
        choices=list(ModelRegistry().configs.keys()),
        help="Checkpoint fine-tuned target (determina la normalizzazione del testo)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--num_layers_to_freeze",
        type=int,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--max_span_length",
        type=int,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        help="Sovrascrive il default del ModelRegistry",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Frequenza di log (steps) per la loss su wandb",
    )
    parser.add_argument(
        "--no_push_to_hub",
        action="store_false",
        dest="push_to_hub",
        help="Disabilita il caricamento del modello su HuggingFace Hub al termine",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CNR-ILC/gs-dataset-tlg-uncased",
        help="Dataset da usare per il finetuning",
    )

    args = parser.parse_args()

    torch.cuda.empty_cache()

    if HF_TOKEN:
        login(token=HF_TOKEN)

    if WANDB_API_KEY:
        wandb_login()

    checkpoint = args.checkpoint
    config = ModelRegistry().get_config(checkpoint)
    base_model = ModelRegistry().base_model_map.get(checkpoint, checkpoint)

    epochs = args.epochs if args.epochs is not None else config.get("epochs", 10)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 128)
    )
    lr = args.lr if args.lr is not None else config.get("lr", 5e-6)
    chunk_size = (
        args.chunk_size
        if args.chunk_size is not None
        else config.get("chunk_size", 256)
    )
    num_layers_to_freeze = (
        args.num_layers_to_freeze
        if args.num_layers_to_freeze is not None
        else config.get("num_layers_to_freeze", 6)
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else config.get("weight_decay", 0.01)
    )
    warmup_ratio = (
        args.warmup_ratio
        if args.warmup_ratio is not None
        else config.get("warmup_ratio", 0.1)
    )
    mlm_probability = (
        args.mlm_probability
        if args.mlm_probability is not None
        else config.get("mlm_probability", 0.15)
    )
    max_span_length = (
        args.max_span_length
        if args.max_span_length is not None
        else config.get("max_span_length", 3)
    )
    lr_scheduler_type = (
        args.lr_scheduler_type
        if args.lr_scheduler_type is not None
        else config.get("lr_scheduler_type", "linear")
    )

    dataset_name = (
        args.dataset_name
        if args.dataset_name is not None
        else "CNR-ILC/gs-dataset-tlg-uncased"
    )

    pipeline_finetuning(
        checkpoint=checkpoint,
        base_model=base_model,
        batch_size=batch_size,
        chunk_size=chunk_size,
        epochs=epochs,
        lr=lr,
        num_layers_to_freeze=num_layers_to_freeze,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        mlm_probability=mlm_probability,
        max_span_length=max_span_length,
        lr_scheduler_type=lr_scheduler_type,
        push_to_hub=args.push_to_hub,
        logging_steps=args.logging_steps,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()
