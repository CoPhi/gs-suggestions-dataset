import pickle
import base64
from typing import Optional
from models.training import pipeline_train
from config.settings import MONGO_URI  # Aggiungi la tua connessione MongoDB
from pymongo import MongoClient

# Connessione a MongoDB
client = MongoClient(MONGO_URI)
db = client["models_db"]  # Il nome del database
collection = db["models"]  # La raccolta (collezione) che contiene i modelli

class ModelService:
    def __init__(self):
        self._model = None  # Modello attualmente selezionato
        self.models = []  # Lista dei modelli caricati in memoria

    def save_model_to_db(self, model_id: str, model_obj):
        """ Salva il modello in MongoDB serializzandolo con pickle """
        try:
            serialized_model = pickle.dumps(model_obj)

            # Si controlla se il modello esiste già
            existing_model = collection.find_one({"model_id": model_id})
            if existing_model:
                collection.update_one({"model_id": model_id}, {"$set": {"data": serialized_model}}) # Se il modello esiste già, aggiorna i dati
            else:
                collection.insert_one({"model_id": model_id, "data": serialized_model})

            print(f"✅ Modello {model_id} salvato in MongoDB")
        except Exception as e:
            print(f"❌ Errore nella serializzazione del modello: {e}")

    def load_model_from_db(self, model_id: str):
        """ Recupera e deserializza il modello da MongoDB """
        try:
            model_record = collection.find_one({"model_id": model_id})

            if model_record:
                # Deserializza il modello
                model_data = pickle.loads(model_record["data"])
                print(f"✅ Modello {model_id} caricato da MongoDB")
                return model_data
            else:
                print(f"⚠️ Modello {model_id} non trovato in MongoDB")
                return None
        except Exception as e:
            print(f"❌ Errore nella deserializzazione: {e}")
            return None

    def load_ngram_model(self, k_pred: int, lm_type: str, ngrams_order: int, test_size: float, gamma: Optional[float] = None):
        """ Carica un modello esistente o ne crea uno nuovo """
        model_id = f"{lm_type}_{ngrams_order}_{test_size}_{gamma}_{k_pred}"

        # 1️⃣ Cerca il modello in memoria
        for model in self.models:
            if model["model_id"] == model_id:
                self._model = model
                return model["lm"]

        # 2️⃣ Prova a caricarlo da MongoDB
        model_data = self.load_model_from_db(model_id)
        if model_data:
            self.models.append(model_data)
            self._model = model_data
            return model_data["lm"]

        # 3️⃣ Se non esiste, lo crea e lo salva
        lm, _ = pipeline_train(
            lm_type=lm_type,
            gamma=gamma,
            n=ngrams_order,
            test_size=test_size,
        )

        model_info = {
            "model_id": model_id,
            "lm": lm,
            "model": "ngrams",
            "k_pred": k_pred,
            "lm_type": lm_type,
            "gamma": gamma,
            "ngrams_order": ngrams_order,
            "test_size": test_size,
        }

        self.models.append(model_info)
        self._model = model_info
        self.save_model_to_db(model_id, model_info)
        return lm
