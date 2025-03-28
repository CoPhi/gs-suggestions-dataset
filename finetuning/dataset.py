"""
Script per rendere disponibile l'insieme delle frasi pre-processsate del MAAT corpus su hugging face

access token: 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
nome: hf-maat-upload

Questo dataset serve per fare il finetuning dei modelli BERT per il greco antico sul task MLM

Il train_set è composto da frasi pulite in greco antico senza la presenza di token sconosciuti, in cui è stato applicato il case folding.
Il test_set è composto da frasi pulite in greco antico con la presenza di token mascherati, in cui è stato applicato il case folding.

Le gold label sono usate per confrontare l'accuracy del modello BERT sul test_set rispetto ad altri modelli.
"""

from train import load_abs, split_abs, get_sentences
from datasets import Dataset, DatasetDict
from finetuning import CHUNK_SIZE
from finetuning.utils import get_test_cases_from_abs


def get_sent_from_tokens(tokens: list[str]):
    return " ".join(tokens)


def push_to_huggingface_hub(dataset: DatasetDict):
    dataset.push_to_hub(
        "GabrieleGiannessi/maat-corpus", commit_message="v6: sentences chunked, add puntuaction"
    )
    pass


def chunk_sentences(sentences: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    chunked = []
    for sentence in sentences:
        tokens = sentence.split()
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i : i + chunk_size]
            if len(chunk) == chunk_size:  # Ensure all chunks are of the same length
                chunked.append(" ".join(chunk))
    return chunked


def main():
    abs = load_abs()
    temp_abs, test_abs = split_abs(abs, 0.1)
    train_abs, dev_abs = split_abs(temp_abs, 0.1)

    test_set = chunk_sentences(
        [
            get_sent_from_tokens(tokens)
            for tokens in get_sentences(test_abs, remove_punct=False)
            if tokens and "<UNK>" not in tokens
        ],
    )
    train_set = chunk_sentences(
        [
            get_sent_from_tokens(tokens)
            for tokens in get_sentences(train_abs, remove_punct=False)
            if tokens and "<UNK>" not in tokens
        ],
    )
    dev_set = chunk_sentences(
        [
            get_sent_from_tokens(tokens)
            for tokens in get_sentences(dev_abs, remove_punct=False)
            if tokens and "<UNK>" not in tokens
        ],
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": train_set,
                }
            ),
            "dev": Dataset.from_dict(
                {
                    "text": dev_set,
                }
            ),
            "test": Dataset.from_dict({"text": test_set}),
        }
    )

    push_to_huggingface_hub(dataset)


if __name__ == "__main__":
    main()
