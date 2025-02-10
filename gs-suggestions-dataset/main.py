from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.infer import generate_words
from models.training import load_lm

""""
    For test 
    
    uvicorn nome_file_api:app --reload (avvio server fastAPI)
    curl -X POST "http://127.0.0.1:8000/restore" \
    -H "Content-Type: application/json" \
    -d '{"context": "Questo è un esempio di", "num_words": 3}' (test richiesta all'API)

"""

app = FastAPI()
model_path = "trigram_lm_MLE.pkl"
try:
    lm, _ = load_lm(model_path)
    print("Modello caricato con successo.")
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    model = None


class RestoreRequest(BaseModel):
    context: str
    num_words: int
    
class RestoreResponse(BaseModel):
    restored_text: str

@app.get("/restore", response_model=RestoreResponse)
def restore(request: RestoreRequest):
    try: 
        restored = generate_words(lm = lm, context=request.context, num_words=request.num_words)
        return {"restored_text": " ".join(restored)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))