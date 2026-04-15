"""
Pipeline per la pubblicazione del corpus MAAT su HuggingFace Hub.

Flusso consigliato
------------------
raw   = build_raw_dataset(abs_)          # da train_set.py
tok   = prepare_dataset_for_model(raw, checkpoint)
"""

from __future__ import annotations

from functools import partial

from cltk.alphabet.grc.grc import normalize_grc
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from backend.core.cleaner import load_abs, load_test_set, split_abs_herc_dev
from backend.core.preprocess import remove_punctuation, strip_diacritics
from models.bert.dataset import MAAT_CORPUS_CHECKPOINT, MAAT_EVAL_CHECKPOINT
from models.bert.dataset.dev_set import DevCase, build_dev_case, build_dev_set
from models.bert.dataset.train_set import build_train_set
from models.bert.finetuning import (
    BERT_MAX_SEQ_LENGTH,
    BERT_UNK_TOKEN,
    CHUNK_SIZE,
    MIN_SENT_TOKEN_TRESHOLD,
    get_model_config,
)

# Hub


def get_db() -> DatasetDict:
    """Carica il dataset MAAT dal HuggingFace Hub."""
    return load_dataset("GabrieleGiannessi/maat-corpus")


def push_to_hub(dataset: DatasetDict, checkpoint: str, message: str) -> None:
    """Pubblica *dataset* sul repository HF *checkpoint*."""
    dataset.push_to_hub(checkpoint, commit_message=message)


# Caricamento e split del corpus


def load_train_and_dev_set(test_size: float = 0.1) -> tuple[list, list]:
    """
    Carica i blocchi anonimi e li suddivide in train e dev set.

    Il dev set contiene esclusivamente blocchi P.Herc.

    Returns:
        (train_abs, dev_abs)
    """
    return split_abs_herc_dev(load_abs(), test_size)


def dev_set_to_hf_dataset(dev_set: list[DevCase]) -> Dataset:
    """
    Converte una lista di DevCase in un Dataset HuggingFace.
    Le liste (campo `y`) sono supportate nativamente come feature Sequence.
    """
    records = [
        {
            "x": case.x,
            "y": case.y,
            "gap_length": case.gap_length,
            "corpus_id": case.corpus_id,
            "file_id": case.file_id,
        }
        for case in dev_set
    ]
    return Dataset.from_list(records)


# Chunking


def chunk_sentences(sentences: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Divide le frasi in blocchi di esattamente *chunk_size* word token."""
    return [
        " ".join(chunk)
        for sentence in sentences
        for i in range(0, len(sentence.split()), chunk_size)
        if len(chunk := sentence.split()[i : i + chunk_size]) == chunk_size
    ]


def chunk_for_bert(
    sentences: list[str],
    tokenizer,
    max_length: int = BERT_MAX_SEQ_LENGTH,
) -> list[str]:
    """Aggrega frasi rispettando il limite di sub-word token di BERT."""
    chunks: list[str] = []
    current_tokens: list[str] = []
    current_length = 0

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)

        if len(sent_tokens) > max_length:
            if current_tokens:
                chunks.append(tokenizer.convert_tokens_to_string(current_tokens))
                current_tokens, current_length = [], 0
            chunks.append(tokenizer.convert_tokens_to_string(sent_tokens[:max_length]))
            continue

        if current_length + len(sent_tokens) > max_length:
            chunks.append(tokenizer.convert_tokens_to_string(current_tokens))
            current_tokens, current_length = sent_tokens, len(sent_tokens)
        else:
            current_tokens.extend(sent_tokens)
            current_length += len(sent_tokens)

    if current_tokens:
        chunks.append(tokenizer.convert_tokens_to_string(current_tokens))

    return chunks


# Normalizzazione e tokenizzazione model-specific


def _quality_filter_subword(
    example: dict,
    tokenizer,
    unk_ratio_threshold: float = 0.1,
) -> bool:
    """Filtra su sub-word token reali del tokenizer specifico."""
    tokens = tokenizer.tokenize(example["text"])
    if len(tokens) < MIN_SENT_TOKEN_TRESHOLD:
        return False
    return tokens.count(tokenizer.unk_token) / max(len(tokens), 1) < unk_ratio_threshold


def _normalize_example(example: dict, config: dict) -> dict:
    text = normalize_grc(example["text"])
    if config["strip_diacritics"]:
        text = strip_diacritics(text)
    if config["remove_punct"]:
        text = remove_punctuation(text)

    case_folding = config.get("case_folding")
    if case_folding == "upper":
        text = text.upper()
    elif case_folding == "lower":
        text = text.lower()

    return {"text": text.replace("<UNK>", BERT_UNK_TOKEN)}


def _tokenize_example(example: dict, tokenizer) -> dict:
    return tokenizer(example["text"], truncation=True, max_length=512, padding=False)


def prepare_dataset_for_model(
    raw_dataset: Dataset,
    checkpoint: str,
    num_proc: int = 4,
) -> Dataset:
    """
    Pipeline model-specific: normalizza → filtra → tokenizza.

    Args:
        raw_dataset: Dataset grezzo da `train_set.build_raw_dataset()`.
        checkpoint:  Es. "CNR-ILC/gs-aristoBERTo".
        num_proc:    Processi paralleli per Dataset.map().
    """
    config = get_model_config(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return (
        raw_dataset.map(
            partial(_normalize_example, config=config), #applica normalizzazione
            desc=f"Normalizing [{checkpoint}]",
            num_proc=num_proc,
        )
        .filter(
            partial(_quality_filter_subword, tokenizer=tokenizer),
            desc=f"Filtering [{checkpoint}]",
            num_proc=num_proc,
        )
        .map(
            partial(_tokenize_example, tokenizer=tokenizer),
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing [{checkpoint}]",
            num_proc=num_proc,
        )
    )


def main() -> None:
    train_abs, dev_abs = load_train_and_dev_set(test_size=0.2)
    test_abs = load_test_set()

    train_dataset = DatasetDict(
        {
            "train": build_train_set(train_abs),
            "dev": build_train_set(dev_abs),
        }
    )
    push_to_hub(
        train_dataset,
        MAAT_CORPUS_CHECKPOINT,
        "Raw dataset: no strip_diacritics, no case folding, no punctuation removal",
    )

    eval_dataset = DatasetDict(
        {
            "dev": dev_set_to_hf_dataset(build_dev_set(dev_abs)),
            "test": dev_set_to_hf_dataset(build_dev_set(test_abs)),
        }
    )
    push_to_hub(
        eval_dataset,
        MAAT_EVAL_CHECKPOINT,
        "Evaluation dataset with gold labels",
    )


if __name__ == "__main__":
    main()
