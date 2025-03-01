from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.lm.models import LanguageModel

from models.infer import generate_words
from models.training import pipeline_train

""""
    For test 
    
    uvicorn nome_file_api:app --reload (avvio server fastAPI)
    curl -X POST "http://127.0.0.1:8000/restore" \
    -H "Content-Type: application/json" \
    -d '{"context": "Questo è un esempio di", "num_words": 3}' (test richiesta all'API)

"""

app = FastAPI()


class ModelRequest(BaseModel):
    k_pred: int
    ngrams_order: int
    lm_type: str
    gamma: Optional[str]
    test_size: float


class ModelResponse(BaseModel):
    lm: LanguageModel


class RestoreRequest(BaseModel):
    context: str
    num_words: int


class RestoreResponse(BaseModel):
    k_predictions : list[str]


@app.get("/model", response_model=ModelResponse)
def get_model(request: ModelRequest):
    try:
        lm, _ = pipeline_train(
            lm_type=request.lm_type,
            gamma=request.gamma,
            n=request.ngrams_order,
            test_size=request.test_size,
        )
        return {"lm": lm}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/restore", response_model=RestoreResponse)
def restore(request: RestoreRequest):
    try:
        restored = generate_words(
            lm=get_model, context=request.context, num_words=request.num_words
        )
        return {"restored_text": " ".join(restored)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
