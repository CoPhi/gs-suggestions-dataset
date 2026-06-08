import wandb
import argparse
from models.bert.finetuning.pipeline import pipeline_finetuning
from models.bert.finetuning import ModelRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for BERT Models"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Target fine-tuned checkpoint name (e.g., CNR-ILC/gs-GreBerta)",
    )
    args = parser.parse_args()

    # Inizializza la run (Wandb agent si occuperà di iniettare la config definita in sweep.yaml)
    wandb.init()
    config = wandb.config

    checkpoint = args.checkpoint
    base_model = ModelRegistry().base_model_map.get(checkpoint)

    if not base_model:
        raise ValueError(f"Base model per {checkpoint} non trovato.")

    pipeline_finetuning(
        checkpoint=checkpoint,
        base_model=base_model,
        lr=config.get("lr"),
        epochs=config.get("epochs"),
        batch_size=config.get("batch_size"),
        chunk_size=config.get("chunk_size"),
        num_layers_to_freeze=config.get("num_layers_to_freeze"),
        weight_decay=config.get("weight_decay"),
        warmup_ratio=config.get("warmup_ratio"),
        mlm_probability=config.get("mlm_probability"),
        max_span_length=config.get("max_span_length"),
        lr_scheduler_type=config.get("lr_scheduler_type"),
        push_to_hub=False,
        evaluate_on_test=False,
    )


if __name__ == "__main__":
    main()
