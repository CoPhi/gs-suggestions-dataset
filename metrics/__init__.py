from cltk.sentence.grc import GreekRegexSentenceTokenizer

# Tokenizer testo -> frasi (str -> list[str])
sentence_tokenizer = GreekRegexSentenceTokenizer()
FALLBACK_LOSS = 10000

from .accuracy import get_topK_accuracy
from .pp import perplexity