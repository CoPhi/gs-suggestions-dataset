from pathlib import Path
import json
import spacy

from spacy.tokens import DocBin

spacy.prefer_gpu()
class TrigramModel:
    def __init__(self, data_path="data/"):
        
        #Qui definisco le proprietà del modello 
        self.data_path = Path(data_path)
        self.nlp = spacy.load("el_core_news_sm") #LM per il greco antico
        self.doc_bin = DocBin() # DocBin per la serializzazione dei documenti
        pass
    
      
    def load_doc (self): 
        """
        Converte i dati presenti nei file JSON in oggetti spaCy Doc.

        Questo metodo elabora i dati di testo e li serializza in un oggetto DocBin,
        che può essere utilizzato per allenare il modello.


        """
        
        for file_path in self.data_path.glob("*.json"):
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
                for obj in data:
                    if obj['language'] == 'grc': 
                        self.doc_bin.add(self.nlp(obj["training_text"]))
                        
        self.doc_bin.to_disk("training.spacy")
        print("Data serialized successfully.")
        
if __name__ == "__main__":
    
    model = TrigramModel()
    model.load_doc()
    