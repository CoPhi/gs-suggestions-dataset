from pathlib import Path
import json
import spacy

spacy.prefer_gpu()
class TrigramModel:
    def __init__(self, data_path="data/"):
        
        #Qui definisco le proprietà del modello 
        self.data_path = Path(data_path)
        self.docs = [] # List[Doc]
        self.nlp = spacy.load("grc_perseus_trf") #LM per il greco antico
        pass
    
      
    def load_doc (self): 
        """
        Converte i dati presenti nei file JSON in oggetti spaCy Doc.

        Questo metodo elabora i dati di testo e li restituisce come un oggetto spaCy Doc,
        che può essere utilizzato per ulteriori attività di elaborazione del linguaggio naturale.

        Ritorna:
            spacy.tokens.Doc: Il testo elaborato come oggetto spaCy Doc.
        """
        
        for file_path in self.data_path.glob("*.json"):
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
                for obj in data:
                    if obj['language'] == 'grc': 
                        self.docs.append(self.nlp(obj["training_text"]))
        
if __name__ == "__main__":
    
    model = TrigramModel()
    model.load_doc()
    for doc in model.docs:
        print(doc.text)