from cltk.sentence.grc import GreekRegexSentenceTokenizer

# Tokenizer testo -> frasi (str -> list[str])
sentence_tokenizer = GreekRegexSentenceTokenizer()

from .accuracy import get_topK_accuracy
from .pp import perplexity