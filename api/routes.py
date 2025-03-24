from fastapi import FastAPI, HTTPException
from models.infer import generate_k_suggests
from api.service import save_model_to_db, get_model, get_models
from config.settings import K_PREDICTIONS, LM_TYPES, GAMMA, TEST_SIZE, N

app = FastAPI()


@app.get("/models/{id}")
def get_model_by_id(id: int):
    """
    Restituisce il modello identificato dall'ID passato per parametro.
    Codici stato dell'operazione:

    200 -> Il modello viene restituito correttamente (payload -> modello)
    400 -> Bad request, il modello non è presente nel DB
    404 -> Errore nei parametri della richiesta del client (per esempio se non si immette un numero intero)
    """
    
    try:
        model = get_model(id)
        if not model:
            raise HTTPException(status_code=400, detail="Bad request")
        return {"status": "200", "model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str("Server error"))


@app.get("/models")
def get_all_models():
    """
    Restituisce l'insieme dei modelli di default presenti nel DB
    200 -> i modelli vengono restituiti correttamente
            payload -> lista dei modelli
    500 -> il server non riesce a trovare i modelli perché non sono stati creati
    """
    models = get_models()
    if not models:
        raise HTTPException(status_code=500, detail="Models not found")

    return {"status": "200", "models": models}


@app.post("/models")  # usata per caricare i modelli
def create_models():
    """
    Chiamata usata per creare i modelli di default in modo persistente
    """
    try:
        for k_pred, lm_type in zip(K_PREDICTIONS, LM_TYPES):
            save_model_to_db(
                k_pred=k_pred, lm_type=lm_type, gamma=GAMMA, test_size=TEST_SIZE, n=N
            )
        return {"status": "201", "message": "Models created successfully"}
    except Exception as e:
        raise HTTPException(status_code=409, detail="Conflict: " + str(e))


@app.get("/predictions")
def restore(context: str, num_words: int, id: int):
    """
    Preleva il modello che ha il seguente id, esegue la predizione e la restituisce.
    """
    try:
        model = get_model(id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Genera le predizioni (dovresti avere una funzione per farlo)
        predictions = generate_k_suggests(
            context=context, num_words=num_words, model=model
        )
        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
