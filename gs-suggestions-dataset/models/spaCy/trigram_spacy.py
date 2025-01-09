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
        self.db = DocBin()  # DocBin per la serializzazione dei documenti
        self.dataset = []  # Lista di documenti tokenizzati
        self.train_data = []
        self.test_data = []
        self.dev_data = []

    def tokenize(self):
        """
        Tokenizza il testo presente nei dataset in singoli token usando spaCy.
        """

        for file_path in self.data_path.glob("*.json"):
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
                for obj in data:
                    if obj["language"] == "grc":
                        #self.dataset.append(self.nlp(obj["training_text"]))
                        for token in self.nlp(obj["training_text"]): 
                            print ('token : ', token.text)
                            

        print(self.dataset)

    def split_data(self) -> None:
        """
        Splits the dataset into training and testing sets.

        This method does not take any parameters and does not return any values.
        It modifies the dataset attribute of the class instance by splitting it
        into training and testing subsets.
        """

        if not self.dataset:
            raise ValueError("Dataset is empty. Cannot split data.")

        self.train_data, temp_data = train_test_split(self.dataset, test_size=0.1)
        self.dev_data, self.test_data = train_test_split(temp_data, test_size=0.5)

        for doc in self.train_data:
            self.db.add(doc)

        self.db.to_disk(
            "./train.spacy"
        )  # salvo i dati di training in un file binario (.spacy)

    def pipeline(self):
        self.tokenize()
        self.split_data()


if __name__ == "__main__":

    model = TrigramModel()
    model.pipeline()
