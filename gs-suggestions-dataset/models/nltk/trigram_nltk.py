from pathlib import Path
import json
import pickle
import argparse
import re
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
        self.ab = None  # insieme degli Anonymous Block "ab" (oggetti MAAT)
        self.train_ab = None
        self.test_ab = None
        self.lm = Lidstone(order=3, gamma=100000)

    def get_ab(self) -> None:
        """
        Estrae gli anonymous block da tutti i file JSON nella cartella specificata.
        """

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

        self.kfold = KFold(n_splits=5, shuffle=False)

    def train_lm(self) -> None:
        """
        Addestra il modello sulle frasi di addestramento prelevate dall'insieme train_ab.
        Le frasi subiscono un controllo per rimuovere token non validi nella fase di addestramento.
        """

        invalid_token_pattern = re.compile(r"(<|>|]|\[|g|a|p)")
        train_sentences = []

        for obj in self.train_ab:
            if obj["training_text"] and obj["language"] == "grc":
                train_sentences.extend(
                    [
                        list(
                            pad_both_ends(
                                [
                                    token
                                    for token in obj["training_text"]
                                    if not invalid_token_pattern.search(token)
                                ],
                                n=2,
                            )
                        )
                    ]
                )

        train_ngrams, vocab = padded_everygram_pipeline(order=3, text=train_sentences)
        self.lm.fit(train_ngrams, vocab)
        
    def select_best_lm(self):
        """
        Seleziona il miglior modello utilizzando Kfold e ottimizza il parametro gamma.
        """
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
        """
        self.train_lm()
        self.save_lm()

    def pipeline_train(self) -> None:
        self.get_ab()
        print("Tokenization complete. Starting data split...")
        self.split_ab()
        print("Data split complete. Starting model selection...")
        self.select_best_lm()

    def save_lm(self, model_path="trigram_lm.pkl") -> None:
        """
        Salva il modello linguistico su disco.

        Args:
            model_path (str): Percorso per salvare il modello.
        """
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
            print("Language model saved.")

    def load_lm(self, model_path="trigram_lm.pkl") -> None:
        """
        Carica solo il modello linguistico da disco.

        Args:
            model_path (str): Percorso da cui caricare il modello.
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            self.__dict__.update(model.__dict__) 
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
        invalid_token_pattern = re.compile(r"(<|>|]|\[|g|a|p)")
        test_sentences = []

        for obj in self.test_ab:
            if obj["training_text"] and obj["language"] == "grc":
                test_sentences.extend(
                    [
                        list(
                            pad_both_ends(
                                [
                                    token
                                    for token in obj["training_text"]
                                    if not invalid_token_pattern.search(token)
                                ],
                                n=2,
                            )
                        )
                    ]
                )

        test_ngrams = list(padded_everygrams(order=3, sentence=test_sentences))

        if not self.lm.vocab:
            raise ValueError("Il modello non è stato addestrato correttamente.")
        try:
            return self.lm.perplexity(test_ngrams)
        except ZeroDivisionError:
            raise RuntimeError(
                "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
            )
            
    def accuracy(self) -> float:
        """
        Calcola l'accuratezza del modello sui casi di test forniti dal test_ab.
        L'accuratezza viene calcolata come il rapporto tra il numero di predizioni corrette e il numero totale di predizioni sul test set.

        Returns:
            float: L'accuratezza del modello come numero in virgola mobile.
        """
        correct_predictions = 0
        total_predictions = 0

        for ab in self.test_ab:
            if ab["language"] == "grc":
                restored = re.findall(
                    r"\[([^\]]+)\]", ab["training_text"]
                )  # restauri dentro il training_text
                for i, obj in enumerate(ab["test_cases"]):
                    test_case = obj["test_case"]
                    lacuna = (
                        re.search(r"\[([^\]]+)\]", test_case).group(1)
                        if re.search(r"\[([^\]]+)\]", test_case)
                        else ""
                    )
                    context = test_case.split("[")[0]  # contesto fino alla parola da predire
                    if self.generate_words(context, len(lacuna)) == restored[i]:
                        correct_predictions += 1

                    total_predictions += 1

        return correct_predictions / total_predictions




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
        print ("Accuracy:", model.accuracy(), '%')
    