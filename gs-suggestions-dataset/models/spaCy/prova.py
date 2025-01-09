from pathlib import Path
import spacy 
import json

nlp = spacy.blank('xx') # modello vuoto
data_path = Path("data/")

for file_path in data_path.glob("*.json"):
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
                for obj in data:
                    if obj["language"] == "grc":
                        for token in  (nlp(obj["training_text"])): 
                            print ('token : ', token.text)