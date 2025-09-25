from datasets import (
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
    Dataset,
)

from utils import SUPPLEMENTS_REGEX
from utils.preprocess import strip_diacritics
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from typing import Optional, Tuple, Union


def get_BERT_model(model_checkpoint: str):
    return AutoModelForMaskedLM.from_pretrained(model_checkpoint)


def get_tokenizer(model_checkpoint: str):
    return AutoTokenizer.from_pretrained(model_checkpoint)


def get_masker(model: AutoModelForMaskedLM, tokenizer: AutoTokenizer):
    return pipeline("fill-mask", model=model, tokenizer=tokenizer)


def get_dataset(
    data_checkpoint: str,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    return load_dataset(data_checkpoint)


def convert_lacuna_to_masks(text: str, mask_token: str) -> Optional[Tuple[str, int]]:
    """
    Converte le lacune nel testo in token mascherati.

    Args:
        text (str): Il testo contenente lacune da convertire.
        mask_token (str): Il token mascherato da utilizzare.

    Returns:
        Optional[Tuple[str, int]]: Una tupla contenente il testo con la lacuna sostituita dal token mascherato e la lunghezza della lacuna, oppure None se non viene trovata una singola lacuna.
    """

    gap_matches = list(SUPPLEMENTS_REGEX.finditer(text))
    if len(gap_matches) != 1:
        return

    seq = gap_matches[0]

    start, end = seq.start(), seq.end()

    return (
        strip_diacritics(text.replace(text[start:end], mask_token)),
        end - start - 2,
    )
