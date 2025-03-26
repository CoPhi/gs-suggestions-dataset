from cltk.sentence.grc import GreekRegexSentenceTokenizer

# Tokenizer testo -> frasi (str -> list[str])
sentence_tokenizer = GreekRegexSentenceTokenizer()

from .cleaner import *
from .training import * 
