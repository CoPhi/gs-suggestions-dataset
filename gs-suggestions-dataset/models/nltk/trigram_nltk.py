from pathlib import Path
import json
import pickle
import argparse
import numpy as np

from sklearn.model_selection import train_test_split, KFold

from nltk.tokenize import word_tokenize
from nltk.lm.models import Lidstone
from nltk.lm.preprocessing import (
    padded_everygram_pipeline,
    pad_both_ends,
    padded_everygrams,
)


class TrigramModel:
    def __init__(self, data_path="data/"):
        """
        Modello N-gram (N=3).

        Args:
            data_path (str): Percorso alla cartella contenente i file JSON.
        """
        self.data_path = Path(data_path)
        self.tokenized_sentences = []
        self.train_sentences = []
        self.test_sentences = []
        self.lm = None

    def tokenize(self) -> None:
        """
        Estrae le frasi di addestramento da tutti i file JSON nella cartella specificata.
        Inserisce i tag <s> e </s> all'inizio e alla fine di ogni frase.
        """

        for file_path in self.data_path.glob("*.json"):
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
                for obj in data:
                    if obj["language"] == "grc":
                        self.tokenized_sentences.extend(
                            [
                                list(
                                    pad_both_ends(
                                        word_tokenize(obj["training_text"]), n=2
                                    )
                                )
                            ]
                        )

    def split_data(self) -> None:
        """
        Divide i token in train, dev e test set.
        Utilizza KFold per creare i set di validazione.
        """
        if not self.tokenized_sentences:
            raise ValueError("Tokenized sentences are empty. Cannot split data.")

        self.train_sentences, self.test_sentences = train_test_split(
            self.tokenized_sentences, test_size=0.1
        )

        self.kfold = KFold(n_splits=5, shuffle=False)

    def train_lm(self, gamma, train_sentences) -> None:
        """
        Addestra il modello sulle frasi di addestramento.
        """
        train_ngrams, vocab = padded_everygram_pipeline(order=3, text=train_sentences)
        self.lm = Lidstone(order=3, gamma=gamma)
        self.lm.fit(train_ngrams, vocab)

    def select_best_lm(self):
        """
        Seleziona il miglior modello utilizzando Kfold e ottimizza il parametro gamma.
        """
        best_perplexity = float("inf")
        best_lm = None

        for gamma in np.linspace(1, 1000000, 5):
            print(f"Testing gamma: {gamma}")
            fold_perplexities = []

            for train_index, val_index in self.kfold.split(self.train_sentences):

                self.train_lm(
                    gamma=gamma,
                    train_sentences=[self.train_sentences[i] for i in train_index],
                )
                val_fold_ngrams = list(
                    padded_everygrams(
                        order=3, sentence=[self.train_sentences[i] for i in val_index]
                    )
                )

                perplexity = self.lm.perplexity(val_fold_ngrams)
                fold_perplexities.append(perplexity)
                print(f"Perplexity for current fold with gamma {gamma}: {perplexity}")

            avg_perplexity = sum(fold_perplexities) / len(fold_perplexities)
            print(f"Average perplexity for gamma {gamma}: {avg_perplexity}")

            if avg_perplexity < best_perplexity:
                best_perplexity = avg_perplexity
                best_lm = self.lm

        self.lm = best_lm
        print(f"Best model selected with better perplexity: {best_perplexity}")
        self.save_lm()

    def pipeline_train(self) -> None:
        self.tokenize()
        print("Tokenization complete. Starting data split...")
        self.split_data()
        print("Data split complete. Starting model selection...")
        self.select_best_lm()

    def save_lm(self, model_path="trigram_lm.pkl") -> None:
        """
        Salva il modello linguistico su disco.

        Args:
            model_path (str): Percorso per salvare il modello.
        """
        with open(model_path, "wb") as f:
            pickle.dump(self.lm, f)
            print("Language model saved.")

    def load_lm(self, model_path="trigram_lm.pkl") -> None:
        """
        Carica solo il modello linguistico da disco.

        Args:
            model_path (str): Percorso da cui caricare il modello.
        """
        with open(model_path, "rb") as f:
            self.lm = pickle.load(f)
            print("Language model loaded.")

    def generate_words(self, context, num_words):
        """
        Genera un numero di parole dato un contesto.
        """
        if not self.lm:
            raise ValueError("Il modello non è stato caricato correttamente.")

        return self.lm.generate(num_words=num_words, text_seed=list(context))

    def evaluate(self):
        """
        Funzione di valutazione del modello.
        Calcola la perplessità su dei dati di valutazione o sul test set.

        Returns:
            float: Perplessità.
        """

        test_ngrams = list(padded_everygrams(order=3, sentence=self.test_sentences))

        if not self.lm.vocab:
            raise ValueError("Il modello non è stato addestrato correttamente.")
        try:
            return self.lm.perplexity(test_ngrams)
        except ZeroDivisionError:
            raise RuntimeError(
                "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
            )


if __name__ == "__main__":
    """
    Questo script permette di addestrare un modello trigramma o generare parole utilizzando un modello pre-addestrato.
    Modalità di utilizzo:

        1. Addestramento del modello:
            Esegui lo script con l'argomento "train" per addestrare il modello sui dati presenti nella cartella specificata (data/).
            Esempio: python trigram_lm.py train

        2. Valutazione del modello:
            Esegui lo script con l'argomento "eval" per calcolare la perplessità del modello sui dati di test.
            Esempio: python trigram_lm.py eval

        3. Generazione di parole:
            Esegui lo script con l'argomento "infer" per generare parole utilizzando un modello pre-addestrato.
            È necessario specificare il contesto e il numero di parole da generare.
            Esempio: python trigram_lm.py infer --context "parole di esempio" --num_words 5

    Argomenti:

        - mode: Modalità di esecuzione dello script ("train" per addestrare, "infer" per generare parole).
        - context: Contesto per la generazione di parole (richiesto in modalità "infer").
        - num_words: Numero di parole da generare (richiesto in modalità "infer").
    """
    parser = argparse.ArgumentParser(
        description="Train or infer using the trigram model."
    )
    parser.add_argument(
        "mode",
        choices=["train", "infer", "eval"],
        help="Mode to run: train, infer or eval",
    )
    parser.add_argument(
        "--context", type=str, help="Context for word generation (for infer mode)"
    )
    parser.add_argument(
        "--num_words", type=int, help="Number of words to generate (for infer mode)"
    )
    args = parser.parse_args()

    model = TrigramModel()

    if args.mode == "train":
        model.pipeline_train()
    elif args.mode == "infer":
        model.load_lm("trigram_lm.pkl")
        if args.context and args.num_words:
            context = args.context.split()
            generated_words = model.generate_words(context, args.num_words)
            print("Generated words:", ", ".join(generated_words))
        else:
            print("Please provide context and num_words for inference.")
    elif args.mode == "eval":
        model.load_lm("trigram_lm.pkl")
        print("Perplexity:", model.evaluate())
    