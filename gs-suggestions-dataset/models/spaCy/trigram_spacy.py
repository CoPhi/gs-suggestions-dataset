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
        self.ab = []  # insieme degli Anonymous Block "ab" (oggetti MAAT)
        self.train_ab = None
        self.test_ab = None
        
    def get_abs(self):
        for file_path in self.data_path.glob("*.json"):
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
                self.ab.extend([obj for obj in data])
        
    def split_ab(self) -> None:
        """
        Divide gli anonymous block in set di addestramento, validazione e test
        """
        if not self.ab:
            raise ValueError("AB set empty. Cannot split data.")

        self.train_ab, self.test_ab = train_test_split(self.ab, test_size=0.1)
        self.kfold = KFold(n_splits=5, shuffle=True)
 
    def contains_lacunae(self, token: str) -> bool: 
        """
            Verifica se un dato token contiene lacune (gap o parti mancanti).

            Args:
                token (str): Il token da verificare.

            Returns:
                bool: True se il token contiene lacune, False altrimenti.
            """

        if token == ".":
            return False

        if "." in token and not token.isalpha():
            return True

        if re.compile(r"(<|>|]|\[|gap|/)").search(token):
            return True

        return False

    def greek_case_folding(self, text):
        """
        Esegue il case folding per il greco sul testo di input.

        Questa funzione normalizza il testo di input utilizzando la Form C di Normalizzazione Unicode (NFC)
        dopo averlo convertito in minuscolo utilizzando la Form D di Normalizzazione Unicode (NFD).

        Args:
            text (str): Il testo di input da normalizzare.

        Returns:
            str: Il testo normalizzato.
        """
        return unicodedata.normalize("NFC",  unicodedata.normalize("NFD", text).lower())

    def clean_lacunae(self, token: str) -> str:
        """
        Pulisce il token dato rimuovendo specifici caratteri indesiderati.

        Args:
            token (str): Il token da pulire.

        Returns:
            str: Il token pulito.

        La funzione esegue i seguenti passaggi di pulizia:
        1. Se il token contiene un punto (.) e non è interamente alfabetico, rimuove tutti i punti.
        2. Se il token contiene uno qualsiasi dei caratteri '<', '>', ']', '[', 'gap', o '/', li rimuove.

        Esempi:
            >>> clean_lacunae("γέ.δουσιν")
            'γέδουσιν'
            >>> clean_lacunae("<")
            ''
        """
        if "." in token and not token.isalpha():
            return token.replace(".", "")  # Rimuovo i puntini
        if re.compile(r"(<|>|]|\[|gap|/)").search(token):
            return re.sub(
                r"(<|>|]|\[|gap|/)", "", token
            )  # Rimuovo i caratteri specificati

        return token
    
    def clean_text(self, text: str) -> str:
        """
        Pulisce il testo di input eseguendo il case folding per il greco, la tokenizzazione e la gestione delle lacune.

        Args:
            text (str): Il testo di input da pulire.

        Returns:
            str: Il testo pulito con i token uniti da spazi.
        """

        cleaned_tokens = []
        for token in word_tokenize(text=self.greek_case_folding(text)):
            if self.contains_lacunae(token):
                cleaned_token = self.clean_lacunae(token)
                cleaned_tokens.append(cleaned_token)
            else:
                cleaned_tokens.append(token)

        cleaned_text = " ".join(cleaned_tokens)
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
