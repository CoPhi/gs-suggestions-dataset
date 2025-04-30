"""
Script per rendere disponibile l'insieme delle frasi pre-processsate del MAAT corpus su hugging face

access token: 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
nome: hf-maat-upload

token dell'organizzazione CNR-ILC: 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

Questo dataset serve per fare il finetuning dei modelli BERT per il greco antico sul task MLM

Il train_set è composto da frasi pulite in greco antico senza la presenza di token sconosciuti, in cui è stato applicato il case folding.
Il test_set è composto da frasi pulite in greco antico con la presenza di token mascherati, in cui è stato applicato il case folding.

Le gold label sono usate per confrontare l'accuracy del modello BERT sul test_set rispetto ad altri modelli.
"""

from datasets import DatasetDict
from finetuning import (
    load_and_split_sentences,
    get_processed_sentences,
)
from finetuning import TRAIN_DATASET_CHECKPOINT, TEST_DATASET_CHECKPOINT


def push_trainset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:

    dataset.push_to_hub(TRAIN_DATASET_CHECKPOINT, commit_message=message)


def push_testset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub(TEST_DATASET_CHECKPOINT, commit_message=message)


def push_set_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub("CNR-ILC/gs-maat-corpus", commit_message=message)


def main():

    train_abs, test_abs = load_and_split_sentences()
    dataset = DatasetDict(
        {
            "train": get_processed_sentences(train_abs),
            "test": get_processed_sentences(test_abs),
        }
    )

    push_set_to_huggingface_hub(dataset, "Unlabeled ancient greek sentences for fill-mask task, folded to uppercase, version without dev set")

if __name__ == "__main__":
    main()
