from pathlib import Path
import json
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from nltk.tokenize.punkt import PunktLanguageVars
from nltk.util import bigrams, trigrams
from nltk.lm.api import Smoothing
from nltk.probability import FreqDist

class BigramModel:
    def __init__(self, data_path="data/", smoothing_k=1):
        """
        Modello N-gram.
        
        Args:
            data_path (str): Percorso alla cartella contenente i file JSON.
            smoothing_k (int): Valore di k per lo Add-k smoothing.
        """
        self.data_path = Path(data_path)
        self.smoothing_k = smoothing_k
        self.train_tokens = []
        self.dev_tokens = []
        self.test_tokens = []
        self.bigram_counter = Counter()
        self.trigram_counter = Counter()
        self.probabilities = defaultdict(dict)
    
    def tokenizer(self) -> list :
        """
        Estrae i token da tutti i file JSON nella cartella specificata.
        
        Returns:
            list: Una lista di token estratti dai file JSON.
        """
        tokens = []
        tokenizer = PunktLanguageVars()
        
        for file_path in self.data_path.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                for obj in data:
                    tokens.extend([
                        token.replace(",", "") 
                        for token in tokenizer.word_tokenize(obj["training_text"])
                    ])
        
        return tokens

    def split_data(self, tokens) -> None:
        """
        Divide i token in train, dev e test set.
        
        Args:
            tokens (list): Lista di token.
        """
        train_tokens, temp_tokens = train_test_split(tokens, test_size=0.1, random_state=42)
        self.dev_tokens, self.test_tokens = train_test_split(temp_tokens, test_size=0.5, random_state=42)
        self.train_tokens = train_tokens

    def build_ngram_counters(self):
        """
        Costruisce i conteggi di bigrammi e trigrammi sul training set.
        """
        self.bigram_counter = Counter(list(bigrams(self.train_tokens)))
        self.trigram_counter = Counter(list(trigrams(self.train_tokens)))

    def calculate_probabilities(self) -> None:
        """
        Calcola le distribuzioni di probabilità condizionali per ogni bigramma
        applicando lo Add-k smoothing.
        """
        vocabulary_size = len(set(self.train_tokens))
        
        for (w1, w2, w3), trigram_count in self.trigram_counter.items():
            # Applicazione dello Add-k smoothing
            smoothed_prob = (trigram_count) / (self.bigram_counter[(w1, w2)])
            print (smoothed_prob)   
            self.probabilities[(w1, w2)][w3] = smoothed_prob
        """
        # Gestione degli zeri 
        for (w1, w2), bigram_count in self.bigram_counter.items():
            if (w1, w2) not in self.probabilities:
                self.probabilities[(w1, w2)] = defaultdict(
                    lambda: self.smoothing_k / (bigram_count + (self.smoothing_k * vocabulary_size))
                )
        """
            

    def get_next_word_distribution(self, bigram):
        """
        Restituisce la distribuzione di probabilità della parola successiva a un bigramma.
        
        Args:
            bigram (tuple): Bigramma (w1, w2).
        
        Returns:
            dict: Probabilità condizionali delle parole successive.
        """
        return self.probabilities.get(bigram, {})
    
    def train(self):
        """
        Pipeline per addestrare il modello: tokenizzazione, divisione dati, 
        conteggio n-grammi e calcolo delle probabilità.
        """
        tokens = self.tokenizer()
        self.split_data(tokens)
        self.build_ngram_counters()
        self.calculate_probabilities()


# Esempio d'uso
if __name__ == "__main__":
    model = BigramModel(data_path="data/", smoothing_k=1)
    model.train()
    
    # Esempio: distribuzione per un bigramma specifico
    bigram = ('ὡς', 'αἱ')
    print(f"Distribuzione di probabilità per il bigramma {bigram}:")
    print(model.get_next_word_distribution(bigram))
    
