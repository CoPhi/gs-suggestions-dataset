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

from datasets import DatasetDict, Dataset
from finetuning import (
    load_and_split_sentences,
    get_processed_sentences,
)
from finetuning.utils import get_filtered_processed_sentences
from train import load_test_abs, get_sentences
from finetuning import TRAIN_DATASET_CHECKPOINT, TEST_DATASET_CHECKPOINT, get_cast_unk_tokens_text, get_sent_from_tokens
from nltk.lm.preprocessing import flatten

from train.cleaner import load_abs, load_specific_domain_abs, split_abs

def push_trainset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:

    dataset.push_to_hub(TRAIN_DATASET_CHECKPOINT, commit_message=message)


def push_testset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub(TEST_DATASET_CHECKPOINT, commit_message=message)


def push_set_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub("CNR-ILC/gs-maat-corpus", commit_message=message)


def main():

    train_abs, dev_abs = load_and_split_sentences()
    test_abs = load_test_abs()
    dataset = DatasetDict(
        {
            "train": get_processed_sentences(train_abs),
            "dev": get_processed_sentences(dev_abs),
            "test": Dataset.from_dict({
                "text": [get_cast_unk_tokens_text(get_sent_from_tokens(sent_tkns)) for sent_tkns in get_sentences(test_abs, case_folding=True, remove_punct=True)],
                })
        }
    )

    push_set_to_huggingface_hub(dataset, "Unlabeled ancient greek sentences for fill-mask task, folded to uppercase, remove punct, with dev & test set")

if __name__ == "__main__":
    #main()
    train_abs = load_abs()
    domain_abs = load_specific_domain_abs(abs=train_abs)
    train_domain_abs, dev_domain_abs = split_abs(domain_abs, test_size=0.2)
    test_abs = load_test_abs()

    def count_tokens(sentences):
        return len(list(flatten(get_filtered_processed_sentences(sentences))))

    def count_sentences(sentences):
        return len(get_filtered_processed_sentences(sentences))

    train_set = [ab for ab in train_abs if ab not in dev_domain_abs]
    dev_set = domain_abs

    print(f"Total anonymous blocks loaded (training set): {len(train_set)}")
    print(f"Total anonymous blocks loaded (dev set): {len(dev_domain_abs)}")
    print(f"Total anonymous blocks loaded (test set): {len(test_abs)}")

    print(f"Numero di token in training set: {count_tokens(train_set)}")
    print(f"Numero di token in dev set: {count_tokens(dev_set)}")
    print(f"Numero di token in test set: {count_tokens(test_abs)}")

    print(f"Numero di frasi in training set: {count_sentences(train_set)}")
    print(f"Numero di frasi in dev set: {count_sentences(dev_set)}")
    print(f"Numero di frasi in test set: {count_sentences(test_abs)}")
