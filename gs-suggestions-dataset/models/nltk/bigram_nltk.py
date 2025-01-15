from pathlib import Path
import json
import pickle
import argparse
import re
import numpy as np

from sklearn.model_selection import train_test_split, KFold

from nltk.lm import Vocabulary
from nltk.lm.models import Lidstone
from nltk.lm.preprocessing import (
    padded_everygram_pipeline,
    pad_both_ends,
    padded_everygrams,
)


class BigramModel:
    def __init__(self, data_path="data/"):
        """
        Modello N-gram (N=2).

        Args:
            data_path (str): Percorso alla cartella contenente i file JSON.
        """
        self.data_path = Path(data_path)
        self.ab = None  # insieme degli Anonymous Block "ab" (oggetti MAAT)
        self.train_ab = None
        self.test_ab = None

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
        self.kfold = KFold(
            n_splits=9, shuffle=True
        )  # uso il 10% del dataset per il dev set

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
        invalid_token_pattern = re.compile(r"(<|>|]|\[|g|a|p)")
        train_sentences = []

        for obj in ab:
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
        return train_sentences

    def get_test_sentences(self) -> list:
        """
        Estrae e processa le frasi di test dall'insieme dei blocchi anonimi di test test_ab.

        Questo metodo filtra e processa il testo di addestramento dagli oggetti 
        nell'attributo test_ab dove la lingua è "grc". Rimuove i token che corrispondono 
        al pattern invalid_token_pattern e aggiunge padding alle frasi su entrambi i lati 
        con un numero specificato di token di padding.

        Returns:
            list: Una lista di frasi di test processate, dove ogni frase è una lista 
            di token con padding su entrambi i lati.
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
        return test_sentences

    def filter_vocab(self, vocab, min_freq):
        """
        Filtra il vocabolario rimuovendo i token con frequenza inferiore a min_freq.

        Args:
            vocab (Vocabulary): Il vocabolario da filtrare.
            min_freq (int): La frequenza minima richiesta per mantenere un token nel vocabolario.

        Returns:
            Vocabulary: Il vocabolario filtrato.
        """
        return Vocabulary(vocab.counts, unk_cutoff=min_freq)

    def train_lm(self, gamma, ab):
        """
        Addestra il modello sulle frasi di addestramento prelevate dall'insieme train_ab.
        Le frasi subiscono un controllo per rimuovere token non validi nella fase di addestramento.

        Args:
            gamma (float): Il parametro di smoothing per il modello Lidstone.
            ab (list): L'insieme di anonymous block per l'addestramento.
        """
        self.lm = Lidstone(order=2, gamma=gamma)
        train_ngrams, vocab = padded_everygram_pipeline(
            order=2, text=self.get_train_sentences(ab)
        )
        self.lm.fit(train_ngrams, self.filter_vocab(Vocabulary(vocab), min_freq=5))

    def select_best_lm(self):
        """
        Seleziona il miglior modello, secondo la metrica dell'accuratezza, utilizzando Kfold e ottimizzando il parametro gamma.
        """
        best_accuracy = 0
        best_lm = None
        for gamma in [10, 100, 1000, 10000, 100000]:
            print(f"Testing gamma: {gamma}")

            for train_ab_index, val_ab_index in self.kfold.split(self.train_ab):

                self.train_lm(
                    gamma=gamma,
                    ab=[self.train_ab[i] for i in train_ab_index],
                )

                val_fold_ngrams = list(
                    padded_everygrams(
                        order=2,
                        sentence=self.get_train_sentences(
                            [self.train_ab[i] for i in val_ab_index]
                        ),
                    )
                )

                acc = self.accuracy([self.train_ab[i] for i in val_ab_index])
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_lm = self.lm

        self.lm = best_lm
        print(f"Best model selected with better accuracy: {best_accuracy}")

        self.save_lm()

    def pipeline_train(self) -> None:
        self.get_ab()
        print("Tokenization complete. Starting data split...")
        self.split_ab()
        print("Data split complete. Starting model selection...")
        self.select_best_lm()

    def save_lm(self, model_path="bigram_lm.pkl") -> None:
        """
        Salva il modello linguistico su disco.

        Args:
            model_path (str): Percorso per salvare il modello.
        """
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
            print("Language model saved.")

    def load_lm(self, model_path="bigram_lm.pkl") -> None:
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
            raise ValueError("Model not loaded. Cannot generate words.")

        return self.lm.generate(num_words=num_words, text_seed=list(context))

    def evaluate(self) -> float:
        """
        Funzione di valutazione del modello.
        Calcola la perplessità su dei dati di valutazione o sul test set.

        Returns:
            float: Perplessità.
        """
        test_ngrams = list(
            padded_everygrams(order=2, sentence=self.get_test_sentences())
        )

        if not self.lm.vocab:
            raise ValueError("Il modello non è stato addestrato correttamente.")
        try:
            return self.lm.perplexity(test_ngrams)
        except ZeroDivisionError:
            raise RuntimeError(
                "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
            )

    def accuracy(self, abs) -> float:
        """
        Calcola l'accuratezza del modello sui casi di test forniti dal test_ab.
        L'accuratezza viene calcolata come il rapporto tra il numero di predizioni corrette e il numero totale di predizioni sul test set.

        Returns:
            float: L'accuratezza del modello come numero in virgola mobile.
        """
        correct_predictions = 0
        total_predictions = 0

        for ab in abs:
            if ab["language"] == "grc":
                restored = re.findall(
                    r"\[([^\]]+)\]", ab["training_text"]
                )  # restauri dentro il training_text
                if not restored:
                    continue
                for i, obj in enumerate(ab["test_cases"]):
                    test_case = obj["test_case"]
                    lacuna = (
                        re.search(r"\[([^\]]+)\]", test_case).group(1)
                        if re.search(r"\[([^\]]+)\]", test_case)
                        else ""
                    )
                    context = test_case.split("[")[
                        0
                    ]  # contesto fino alla parola da predire
                    if self.generate_words(context, len(lacuna)) == restored[i]:
                        correct_predictions += 1

                    total_predictions += 1

        return correct_predictions / total_predictions


if __name__ == "__main__":
    """
    Questo script permette di addestrare un modello bigramma o generare parole utilizzando un modello pre-addestrato.
    Modalità di utilizzo:

        1. Addestramento del modello:
            Esegui lo script con l'argomento "train" per addestrare il modello sui dati presenti nella cartella specificata (data/).
            Esempio: python bigram_lm.py train

        2. Valutazione del modello:
            Esegui lo script con l'argomento "eval" per calcolare la perplessità del modello sui dati di test.
            Esempio: python bigram_lm.py eval

        3. Generazione di parole:
            Esegui lo script con l'argomento "infer" per generare parole utilizzando un modello pre-addestrato.
            È necessario specificare il contesto e il numero di parole da generare.
            Esempio: python bigram_lm.py infer --context "parole di esempio" --num_words 5

    Argomenti:

        - mode: Modalità di esecuzione dello script ("train" per addestrare, "infer" per generare parole).
        - context: Contesto per la generazione di parole (richiesto in modalità "infer").
        - num_words: Numero di parole da generare (richiesto in modalità "infer").
    """
    parser = argparse.ArgumentParser(
        description="Train or infer using the bigram model."
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

    model = BigramModel()

    if args.mode == "train":
        model.pipeline_train()
    elif args.mode == "infer":
        model.load_lm("bigram_lm.pkl")
        if args.context and args.num_words:
            context = args.context.split()
            generated_words = model.generate_words(context, args.num_words)
            print("Generated words:", ", ".join(generated_words))
        else:
            print("Please provide context and num_words for inference.")
    elif args.mode == "eval":
        model.load_lm("bigram_lm.pkl")
        print("Perplexity:", model.evaluate())
        print("Accuracy:", model.accuracy(model.test_ab), "%")
