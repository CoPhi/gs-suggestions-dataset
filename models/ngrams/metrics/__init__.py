from cltk.sentence.grc import GreekRegexSentenceTokenizer
# Tokenizer testo -> frasi (str -> list[str])
sentence_tokenizer = GreekRegexSentenceTokenizer()
FALLBACK_LOSS = 10000
MIN_BEAM_SIZE = 10
MAX_BEAM_SIZE = 100

_SPECIAL_TOKENS = {"<s>", "</s>", "<UNK>"}
_LANGUAGE = "grc"

