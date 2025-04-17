import re
TRAIN_DATASET_CHECKPOINT = "CNR-ILC/maat-corpus-train"
TEST_DATASET_CHECKPOINT = "CNR-ILC/maat-corpus-test"
BERT_TOKENS_PER_WORD = 1.77

CHUNK_SIZE = 50  # Define the desired chunk size
BERT_UNK_TOKEN = "[UNK]"

#Valori soglia per la considerazione delle frasi 
MAX_UNK_TOKEN_TRESHOLD = 5
MAX_MASK_TOKEN_TRESHOLD=10
MIN_MASK_TOKEN_TRESHOLD=1

LACUNAE_REGEX = re.compile(r"\[([. ]+)\]")

from .utils import (
    load_and_split_sentences,
    get_processed_sentences,
    get_model,
    get_tokenizer,
    get_dataset,
    convert_lacuna_to_masks,
)

from.accuracy import (
    hcb_beam_search,
)