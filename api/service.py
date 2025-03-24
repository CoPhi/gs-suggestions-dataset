import pickle
from models.training import pipeline_train
from config.settings import MONGO_URI  # Aggiungi la tua connessione MongoDB
from pymongo import MongoClient
from uuid import uuid4

"""
    password mongoDB: Tl1E99fOZYAZh5PQ
    username: gabrielegiannessi
    
    mongodb+srv://gabrielegiannessi:Tl1E99fOZYAZh5PQ@cluster0.u21iw.mongodb.net/
"""

# Connessione a MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client["models_db"]  # Il nome del database
    collection = db["models"]  # La raccolta (collezione) che contiene i modelli     
except Exception as e:
    print(f"❌ Errore nella connessione a MongoDB: {e}")
    
def get_model(id: int):
    """
            Restituisce il modello identificato dall'ID passato per parametro.
            Codici stato dell'operazione: 
            
            200 -> Il modello viene restituito correttamente (payload -> modello) 
            404 -> Bad request, parametri non corretti nella richiesta del client (per esempio se non si immette un numero intero)
            500 -> Errore lato server, il modello non è presente nel DB 
    """
    model = collection.find_one({
        "model_id": id
    })
    print (model)
    
def get_models():
    pass
    
def save_model_to_db(
        k_pred: int, lm_type: str, gamma: float, test_size: float, n: int
    ):
        """Salva il modello in MongoDB serializzandolo con pickle"""
        try:
            model, _ = pipeline_train(
                lm_type=lm_type, gamma=gamma, n=n, test_size=test_size
            )
            serialized_model = pickle.dumps(model)
            model_id = str(uuid4())

            # Si controlla se il modello esiste già
            if not collection.find_one({"model_id": model_id}):
                collection.insert_one(
                    {
                        "model_id": model_id,
                        "model": serialized_model,
                        "k_pred": k_pred,
                        "type": "ngrams",
                        "lm_score": lm_type,
                        "gamma": gamma,
                        "test_size": test_size,
                        "order": n,
                    }
                )

            print(f"✅ Modello {model_id} salvato in MongoDB")
        except Exception as e:
            print(f"❌ Errore nella serializzazione del modello: {e}")
            
            
