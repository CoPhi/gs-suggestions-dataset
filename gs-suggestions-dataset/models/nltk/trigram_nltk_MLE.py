from pathlib import Path
import json
import pickle
import argparse
import re
import unicodedata

from sklearn.model_selection import train_test_split, KFold

from collections import Counter
from cltk.sentence.grc import GreekRegexSentenceTokenizer
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
        self.ab = []  # insieme degli Anonymous Block "ab" (oggetti MAAT)
        self.train_ab = None
        self.test_ab = None
        self.lm = MLE(3)
        self.sentence_tokenizer = GreekRegexSentenceTokenizer()

    def get_ab(self) -> None:
        """
        Estrae gli anonymous block da tutti i file JSON nella cartella specificata.
        """

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
        return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).lower())

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
                if cleaned_token:
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
                        word_tokenize(sent)
                        for sent in self.sentence_tokenizer.tokenize(
                            self.clean_text(obj["training_text"])
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
                        word_tokenize(sent)
                        for sent in self.sentence_tokenizer.tokenize(
                            self.clean_text(obj["training_text"])
                        )
                    ]
                )
        return test_sentences

    def filter_vocab(self, vocab_tokens, min_freq):
        """
        Filtra il vocabolario rimuovendo i token con frequenza inferiore a min_freq.

        Args:
            vocab (Vocabulary): Il vocabolario da filtrare.
            min_freq (int): La frequenza minima richiesta per mantenere un token nel vocabolario.

        Returns:
            Vocabulary: Il vocabolario filtrato.
        """

        token_counts = Counter(vocab_tokens)
        return Vocabulary(
            [token for token, freq in token_counts.items() if freq >= min_freq]
        )

    def train_lm(self, ab) -> None:
        """
        Addestra il modello sulle frasi di addestramento prelevate dall'insieme train_ab.
        Le frasi subiscono un controllo per rimuovere token non validi nella fase di addestramento.
        """

        train_ngrams, vocab_tokens = padded_everygram_pipeline(
            order=3, text=self.get_train_sentences(ab)
        )

        self.lm.fit(train_ngrams, self.filter_vocab(vocab_tokens, 2))

    def select_best_lm(self):
        """
        Seleziona il miglior modello, secondo la metrica dell'accuratezza, utilizzando Kfold
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
        print("Data split complete. start training the model...")
        # self.select_best_lm()
        self.train_lm(self.train_ab)
        self.save_lm()

    def save_lm(self, model_path="trigram_lm_MLE.pkl") -> None:
        """
        Salva il modello linguistico su disco.

        Args:
            model_path (str): Percorso per salvare il modello.
        """

        model_data = {"lm": self.lm, "test_ab": self.test_ab}

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
            print("Language model saved.")

    def load_lm(self, model_path="trigram_lm_MLE.pkl") -> None:
        """
        Carica solo il modello linguistico da disco.

        Args:
            model_path (str): Percorso da cui caricare il modello.
        """
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            self.lm = model_data["lm"]
            self.test_ab = model_data["test_ab"]
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
            num_words=num_words,
            text_seed=word_tokenize(self.clean_text(context)),
        )

    def evaluate(self):
        """
        Funzione di valutazione del modello.
        Calcola la perplessità su dei dati di valutazione o sul test set.

        Returns:
            float: Perplessità.
        """
        test_ngrams = []
        for sentence in self.get_test_sentences():
            test_ngrams.extend(
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
            return self.lm.perplexity(test_ngrams)
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
                for obj in ab["test_cases"]:
                    test_case = obj["test_case"]
                    alternatives = obj["alternatives"]

                    if not alternatives:
                        continue

                    context = list(
                        flatten(
                        [
                            list(
                                pad_both_ends(
                                    word_tokenize(sent),
                                    n=3,
                                )
                            )
                            for sent in self.sentence_tokenizer.tokenize(
                                self.clean_text(test_case.split("[")[0])
                            )
                        ]
                    ))[:-2]
                    
                    for alt in alternatives:
                        alt_words = word_tokenize(self.clean_text(alt))

                        prediction = []
                        while len(prediction) < len(alt_words):
                            token = self.lm.generate(text_seed=context, num_words=1)
                            prediction.append(token)
                            context.append(token)

                        if " ".join(prediction) == " ".join(alt_words):
                            correct_predictions += 1
                            break 
                            # se una delle alternative è corretta, passa al prossimo test case

                    total_predictions += 1
                    print(correct_predictions, "/", total_predictions)

        return round((correct_predictions / total_predictions) * 100, 2)


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
