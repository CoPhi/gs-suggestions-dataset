from pathlib import Path
import json
import random 
from sklearn.model_selection import train_test_split

from nltk.util import bigrams, trigrams
from nltk.tokenize.punkt import PunktLanguageVars
from nltk.probability import FreqDist, KneserNeyProbDist
from collections import defaultdict

class TrigramModel:
    def __init__(self, data_path="data/"):
        """
        Modello N-gram (N=3).
        
        Args:
            data_path (str): Percorso alla cartella contenente i file JSON.
            smoothing_k (int): Valore di k per lo Add-k smoothing.
        """
        self.data_path = Path(data_path)
        self.train_tokens = []
        self.dev_tokens = []
        self.test_tokens = []
        self.trigram_fdist = FreqDist()
        self.probabilities = None
        
    def tokenizer(self) -> list:
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

    def build_ngram_fdist(self):
        """
        Costruisce le distribuzioni di frequenza dei trigrammi sul training set.
        """
        self.trigram_fdist.update(trigrams(self.train_tokens))

    def calculate_probabilities(self) -> None:
        """
        Calcola le distribuzioni di probabilità condizionali per ogni trigramma
        Si usa la seguente distribuzione di probabilità di NLTK: KneserNeyProbDist 
        """
        
        self.probabilities = KneserNeyProbDist(self.trigram_fdist)

    def get_next_word_distribution(self, context):
        """
        Restituisce la distribuzione di probabilità della parola successiva a un bigramma.
        Usiamo la distribuzione di probabilità dei trigrammi per calcolarla. 
        Args: 
        Returns: dict -> Probabilità condizionali delle parole successive.
        
        """
        
        prob = defaultdict (dict)
        b1, b2 = context
        
        for (t1, t2, t3) in self.probabilities.samples(): 
            if t1 == b1 and t2 == b2:
                prob.update({
                    t3 : self.probabilities.prob((t1, t2, t3))
                })
        
        return prob
        
        
        
            
    
    def generate_next_word(self, context):
        """
        Genera la parola successiva dato un bigramma.
        """
        next_word_dist = self.get_next_word_distribution(context)
        
        # Seleziona una parola casuale basata sulla distribuzione di probabilità
        next_word = random.choices(list(next_word_dist.keys()), weights=next_word_dist.values())[0]
        
        return next_word

    
    def train(self):
        """
        Pipeline per addestrare il modello: tokenizzazione, divisione dati, 
        conteggio n-grammi e calcolo delle probabilità. 
        """
        tokens = self.tokenizer()
        self.split_data(tokens)
        self.build_ngram_fdist()
        self.calculate_probabilities()
        
    


if __name__ == "__main__":
    model = TrigramModel(data_path="data/")
    model.train()
    
    print (model.generate_next_word(('<gap/>','[')))
    
    
    