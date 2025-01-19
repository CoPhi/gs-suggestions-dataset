from pathlib import Path
import json
import re
import unicodedata

import spacy
from spacy.tokens import DocBin


from sklearn.model_selection import train_test_split, KFold
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import (
    padded_everygram_pipeline,
    pad_both_ends,
    flatten,
)

class TrigramModel:
    def __init__(self, data_path="data/"):

        self.data_path = Path(data_path)
        self.nlp = spacy.load(
            "grc_proiel_trf"
        )  # LM per il greco antico: la uso per tokenizzare i testi del dataset
        self.docs = DocBin()  # DocBin per la serializzazione dei documenti
        self.ab = None
        self.train_ab = None
        self.test_ab = None
        
    def get_abs(self):
        for file_path in self.data_path.glob("*.json"):
            print(f"Processing file: {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
            self.ab = tuple([obj for obj in data])
        
    def split_ab(self) -> None:
        """
        Divide gli anonymous block in set di addestramento, validazione e test
        """
        if not self.ab:
            raise ValueError("AB set empty. Cannot split data.")
        self.train_ab, self.test_ab = train_test_split(self.ab, test_size=0.1)
        self.kfold = KFold(
            n_splits=9, shuffle=True
        )  # uso il 10% del dataset per il dev set
        
    def contains_lacunae(self, token: str) -> bool: 
    
        if token == '.': 
            return False
    
        if "." in token and not token.isalpha():
        # Identifica sequenze di puntini come lacune
        # Se c'è una parte vuota (o solo puntini), considerala lacuna
            return True
        return False   
    
    def greek_case_folding(self, text):
        return unicodedata.normalize("NFC",  unicodedata.normalize("NFD", text).lower())

    def clean_lacunae(self, token: str) -> str:
        # Rimuovi i puntini
        return token.replace('.', '')

    def clean_text(self, text: str) -> str:
        
        cleaned_tokens = []
        for token in self.nlp(self.greek_case_folding(text=text)):
            if self.contains_lacunae(token.text):
                cleaned_token = self.clean_lacunae(token.text)
                cleaned_tokens.append(cleaned_token)
            else:
                cleaned_tokens.append(token.text)
        
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text

    def get_train_sentences(self, ab) -> list:
        """
    Estrae e processa le frasi di addestramento da una lista di blocchi anonimi fornita.
    Questo metodo filtra e processa il 'training_text' da ciascun oggetto nella lista di input 'ab'.
    Include solo i testi in cui la 'language' è 'grc' ed esclude i token che corrispondono al 
    pattern dei token non validi. Le frasi risultanti sono imbottite su entrambi i lati con un token di padding.
    Args:
        ab (list): Una lista di oggetti, ciascuno contenente le chiavi 'training_text' e 'language'.
    Returns:
        list: Una lista di frasi di addestramento processate e imbottite.
    """
        invalid_token_pattern = re.compile(r"[<>[\]gap\b]")
        train_sentences = []
        for obj in ab:
            if obj["training_text"] and obj["language"] == "grc":
                tokens = []
                for token in self.nlp(self.clean_text(obj["training_text"])):
                    if not invalid_token_pattern.match(token.text):
                        tokens.append(token.text)
                if tokens:
                    train_sentences.append(pad_both_ends(tokens, n=2))
                    
        return train_sentences

    def get_test_sentences(self) -> list:
        pass
    def filter_vocab(self, vocab, min_freq):
        pass
    def train_lm(self, gamma, ab):
        pass
    def select_best_lm(self):
        pass
    def pipeline_train(self) -> None:
        self.get_abs()
        self.split_ab()
        print (self.get_train_sentences(self.train_ab))
        
        
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
    model.pipeline_train()
