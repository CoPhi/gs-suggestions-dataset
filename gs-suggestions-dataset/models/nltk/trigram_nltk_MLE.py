from pathlib import Path
import json
import pickle
import argparse
import re
import unicodedata

from sklearn.model_selection import train_test_split, KFold

from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.models import MLE
from nltk.lm.preprocessing import (
    padded_everygram_pipeline,
    pad_both_ends,
    flatten,
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
        self.lm = MLE(3)

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

    def contains_lacunae(self, token: str) -> bool:

        if token == ".":
            return False

        if "." in token and not token.isalpha():
            return True

        if re.compile(r"(<|>|]|\[|gap|/)").search(token):
            return True

        return False

    def clean_lacunae(self, token: str) -> str:
        if "." in token and not token.isalpha():
            return token.replace(".", "")  # Rimuovo i puntini
        if re.compile(r"(<|>|]|\[|gap|/)").search(token):
            return re.sub(
                r"(<|>|]|\[|gap|/)", "", token
            )  # Rimuovo i caratteri specificati

        return token

    def greek_case_folding(self, text):
        return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).lower())

    def clean_text(self, text: str) -> str:

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
        train_sentences = []

        for obj in ab:
            if obj["training_text"] and obj["language"] == "grc":
                train_sentences.extend(
                    [
                        list(
                            pad_both_ends(
                                [
                                    token
                                    for token in word_tokenize(
                                        self.clean_text(obj["training_text"])
                                    )
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
        test_sentences = []

        for obj in self.test_ab:
            if obj["training_text"] and obj["language"] == "grc":
                test_sentences.extend(
                    [
                        list(
                            pad_both_ends(
                                [
                                    token
                                    for token in word_tokenize(
                                        self.clean_text(obj["training_text"])
                                    )
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

    def train_lm(self, ab) -> None:
        """
        Addestra il modello sulle frasi di addestramento prelevate dall'insieme train_ab.
        Le frasi subiscono un controllo per rimuovere token non validi nella fase di addestramento.
        """

        train_ngrams, vocab = padded_everygram_pipeline(
            order=3, text=self.get_train_sentences(ab)
        )

        self.lm.fit(train_ngrams, vocab)

    def select_best_lm(self):
        """
        Seleziona il miglior modello utilizzando Kfold e ottimizza il parametro gamma.
        """
        best_accuracy = 0
        best_lm = None

        for train_ab_index, val_ab_index in self.kfold.split(self.train_ab):
            self.train_lm(
                ab=[self.train_ab[i] for i in train_ab_index],
            )
            """"
            val_fold_ngrams = list(
                padded_everygrams(
                    order=2,
                    sentence=self.get_train_sentences(
                        [self.train_ab[i] for i in val_ab_index]
                    ),
                )
            )
            """
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

    def save_lm(self, model_path="trigram_lm_MLE.pkl") -> None:
        """
        Salva il modello linguistico su disco.

        Args:
            model_path (str): Percorso per salvare il modello.
        """
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
            print("Language model saved.")

    def load_lm(self, model_path="trigram_lm_MLE.pkl") -> None:
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
        Genera parole utilizzando il modello linguistico addestrato.

        Args:
            context (str): Il contesto per la generazione delle parole.
            num_words (int): Il numero di parole da generare.

        Returns:
            list: Una lista di parole generate.
        """
        if not self.lm:
            raise ValueError("Il modello non è stato caricato correttamente.")

        return self.lm.generate(
            num_words=num_words, text_seed=list(context)
        )  # modificare in context[-1] (ultima parola)

    def evaluate(self):
        """
        Funzione di valutazione del modello.
        Calcola la perplessità su dei dati di valutazione o sul test set.

        Returns:
            float: Perplessità.
        """
        test_ngrams = []
        for sentence in self.get_test_sentences():
            test_ngrams.append(
                list(
                    ngrams(
                        sentence,
                        3,
                        pad_left=True,
                        pad_right=True,
                        left_pad_symbol="<s>",
                        right_pad_symbol="</s>",
                    )
                )
            )

        if not self.lm.vocab:
            raise ValueError("Il modello non è stato addestrato correttamente.")
        try:
            return self.lm.perplexity(list(flatten(test_ngrams)))
        except ZeroDivisionError:
            raise RuntimeError(
                "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
            )

    def accuracy(self, abs) -> float:
        """
        Calcola l'accuratezza del modello sui dati forniti (abs).

        Args:
            abs (list): Lista di anonymous block per il calcolo dell'accuratezza.

        Returns:
            float: Accuratezza del modello.
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
                    """
                    lacuna = (
                        word_tokenize(re.search(r"\[([^\]]+)\]", test_case).group(1))
                        if re.search(r"\[([^\]]+)\]", test_case)
                        else ""
                    )"""

                    context = test_case.split("[")[
                        0
                    ]  # contesto fino alla parola da predire

                    print(restored[i])

                    if len([e for e in restored[i].split(" ") if e != ""]) == 1:
                        # Una sola parola da predire
                        cleaned_context = [
                            e for e in self.clean_text(context).split(" ") if e != ""
                        ]
                        seed = (
                            cleaned_context[-2:] if len(cleaned_context) >= 2 else None
                        )  # prendo gli ultimi due token
                        if seed:
                            token = self.lm.generate(text_seed=seed, num_words=1)
                            if token == self.greek_case_folding(restored[i]):
                                correct_predictions += 1

                            total_predictions += 1
                    else:
                        # più parole da predire
                        prediction = []
                        cleaned_context = [
                            e for e in self.clean_text(context).split(" ") if e != ""
                        ]
                        seed = (
                            cleaned_context[-2:] if len(cleaned_context) >= 2 else None
                        )  # prendo l'ultimo bigramma
                        if seed:
                            for _ in range(
                                len([e for e in restored[i].split(" ") if e != ""])
                            ):
                                token = self.lm.generate(text_seed=seed, num_words=1)
                                seed = list(seed[-1:])
                                seed.append(token)
                                prediction.append(token)

                            if " ".join(prediction) == self.greek_case_folding(
                                restored[i]
                            ):
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
        model.load_lm("trigram_lm_MLE.pkl")
        if args.context and args.num_words:
            context = args.context.split()
            generated_words = model.generate_words(context, args.num_words)
            print("Generated words:", ", ".join(generated_words))
        else:
            print("Please provide context and num_words for inference.")
    elif args.mode == "eval":
        model.load_lm("trigram_lm_MLE.pkl")
        print("Perplexity:", model.evaluate())
        print("Accuracy:", model.accuracy(model.test_ab), "%")
