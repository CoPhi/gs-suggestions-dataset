from backend.core import GreekRegexSentenceTokenizer, sentence_tokenizer

FALLBACK_LOSS = 10000
MIN_BEAM_SIZE = 10
MAX_BEAM_SIZE = 100

_SPECIAL_TOKENS = {"<s>", "</s>", "<UNK>"}
_LANGUAGE = "grc"

