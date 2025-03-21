from fastapi import FastAPI, HTTPException
from models.infer import generate_k_suggests
from api.service import model_service
from config.settings import K_PREDICTIONS, LM_TYPES, GAMMA, TEST_SIZE, N

app = FastAPI()

@app.get("/models/{id}")
def get_model(id: int):
    """
        Restituisce il modello identificato dall'ID passato per parametro
    """
    return {
        "id": id
    } 

@app.get("/models")
def get_models(): 
    """
        Restituisce l'insieme dei modelli di default presenti nel DB
    """
    pass

@app.post("/models") #usata per caricare i modelli
def create_models(): 
    """
        Chiamata usata per creare i modelli di default in modo persistente
    """
    #Carico i modelli di default 
    """
        MLE: 10, 20 predizioni 
        LIDSTONE: 10, 20 predizioni
    """
    
    for k_pred, lm_type in zip(K_PREDICTIONS, LM_TYPES): 
        try: 
            model_service.save_model_to_db(k_pred=k_pred, lm_type=lm_type, gamma=GAMMA, test_size=TEST_SIZE, n=N)
        except: 
            return {
                "status": 409
            }
    
    return {
        "status" : 201 
    }

@app.get("/predictions")
def restore(context: str, num_words: int, id: int):
    """
        Prelevo il modello che ha il seguente id, performa l'operazione e la restituisce
    """
    
    return {
        "context": context, 
        "num_words": num_words, 
        "id" : id
    } 