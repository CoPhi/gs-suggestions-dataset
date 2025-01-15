from pathlib import Path
import json

import spacy
from spacy.tokens import DocBin

from sklearn.model_selection import train_test_split


class TrigramModel:
    def __init__(self, data_path="data/"):

        self.data_path = Path(data_path)
        self.nlp = spacy.load(
            "grc_perseus_lg"
        )  # LM per il greco antico: la uso per tokenizzare i testi del dataset
        self.docs = DocBin()  # DocBin per la serializzazione dei documenti
        
        def get_abs(self):
            pass

        def split_ab(self) -> None:
            pass

        def get_train_sentences(self, ab) -> list:
            pass

        def get_test_sentences(self) -> list:
            pass

        def filter_vocab(self, vocab, min_freq):
            pass

        def train_lm(self, gamma, ab):
            pass

        def select_best_lm(self):
            pass

        def pipeline_train(self) -> None:
            pass

        def save_lm(self, model_path="bigram_lm.pkl") -> None:
            pass

        def load_lm(self, model_path="bigram_lm.pkl") -> None:
            pass

        def generate_words(self, context, num_words):
            pass

        def evaluate(self) -> float:
            pass

        def accuracy(self, abs) -> float:
            pass
            

if __name__ == "__main__":

    model = TrigramModel()
    model.pipeline()
