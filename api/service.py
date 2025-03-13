import requests
import pickle
import base64
from typing import Optional
from models.training import pipeline_train
from config.settings import LM_TYPES, GAMMAS, K_PREDICTIONS, TEST_SIZES, DIMENSIONS

EXISTDB_URL = "http://localhost:8080/exist/rest/db/models/"
EXISTDB_USER = "admin"
EXISTDB_PASS = "admin"

class ModelService:
    def __init__(self):
        self._model = None  # Modello attualmente selezionato
        self.models = []  # Lista dei modelli caricati in memoria

    def save_model_to_existdb(self, model_id: str, model_obj):
        """ Salva il modello in eXistDB serializzandolo con pickle """
        try:
            serialized_model = base64.b64encode(pickle.dumps(model_obj)).decode()
            xml_data = f"""<model>
                <id>{model_id}</id>
                <data>{serialized_model}</data>
            </model>"""

            response = requests.put(
                f"{EXISTDB_URL}{model_id}.xml",
                auth=(EXISTDB_USER, EXISTDB_PASS),
                data=xml_data,
                headers={"Content-Type": "application/xml"},
            )

            if response.status_code in [200, 201]:
                print(f"✅ Modello {model_id} salvato in eXistDB")
            else:
                print(f"❌ Errore nel salvataggio: {response.text}")

        except Exception as e:
            print(f"❌ Errore nella serializzazione del modello: {e}")

    def load_model_from_existdb(self, model_id: str):
        """ Recupera e deserializza il modello da eXistDB """
        response = requests.get(
            f"{EXISTDB_URL}{model_id}.xml",
            auth=(EXISTDB_USER, EXISTDB_PASS),
        )

        if response.status_code == 200:
            try:
                xml_content = response.text
                serialized_model = xml_content.split("<data>")[1].split("</data>")[0]
                model_data = pickle.loads(base64.b64decode(serialized_model.encode()))
                print(f"✅ Modello {model_id} caricato da eXistDB")
                return model_data
            except Exception as e:
                print(f"❌ Errore nella deserializzazione: {e}")
                return None
        else:
            print(f"⚠️ Modello {model_id} non trovato in eXistDB")
            return None

    def load_ngram_model(self, k_pred: int, lm_type: str, ngrams_order: int, test_size: float, gamma: Optional[float] = None):
        """ Carica un modello esistente o ne crea uno nuovo """
        model_id = f"{lm_type}_{ngrams_order}_{test_size}_{gamma}_{k_pred}"

        # 1️⃣ Cerca il modello in memoria
        for model in self.models:
            if model["model_id"] == model_id:
                self._model = model
                return model["lm"]

        # 2️⃣ Prova a caricarlo da eXistDB
        model_data = self.load_model_from_existdb(model_id)
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
        self.save_model_to_existdb(model_id, model_info)
        return lm
