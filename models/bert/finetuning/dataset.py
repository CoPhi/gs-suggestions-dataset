"""
Script per rendere disponibile l'insieme delle frasi pre-processsate del MAAT corpus su hugging face
Questo dataset serve per fare il finetuning dei modelli BERT per il greco antico sul task MLM

Il train_set è composto da frasi pulite in greco antico senza la presenza di token sconosciuti, in cui è stato applicato il case folding.
Il test_set è composto da frasi pulite in greco antico con la presenza di token mascherati, in cui è stato applicato il case folding.

Le gold label sono usate per la valutazione del modello BERT sul test_set rispetto ad altri modelli.
"""

from datasets import DatasetDict
from backend.core.preprocess import clean_tokens, process_editorial_marks
from models.bert.finetuning import (
    load_train_and_dev_set,
)

from backend.core.cleaner import load_test_set, get_sentences
from models.bert.finetuning import (
    TRAIN_DATASET_CHECKPOINT,
    TEST_DATASET_CHECKPOINT,
    get_cast_unk_tokens_text,
    get_sent_from_tokens,
)
from models.bert.finetuning.utils import build_raw_dataset

def push_trainset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub(TRAIN_DATASET_CHECKPOINT, commit_message=message)


def push_testset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub(TEST_DATASET_CHECKPOINT, commit_message=message)


def push_set_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub("CNR-ILC/gs-maat-corpus", commit_message=message)

def main():
    train_abs, dev_abs = load_train_and_dev_set(test_size=0.2)
    test_abs = load_test_set()
    dataset = DatasetDict(
        {
            "train": build_raw_dataset(train_abs),
            "dev": build_raw_dataset(dev_abs),
            "test": build_raw_dataset(test_abs),
        }
    )

    push_set_to_huggingface_hub(
        dataset,
        "Raw dataset with clean sentences: no strip_diacritics, no case folding, no punctuation removal, no token masking",
    )
    
if __name__ == "__main__":
    main()