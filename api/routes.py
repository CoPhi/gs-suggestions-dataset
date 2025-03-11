from fastapi import FastAPI, HTTPException
from models.infer import generate_k_suggests

from api.schemas import NGramModelRequest, NGramModelResponse, RestoreRequest, RestoreResponse
from api.service import model_service

app = FastAPI()

@app.get("/model")
def get_ngram_model():
    try: 
        return model_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model", response_model=NGramModelResponse)
def get_ngram_model(request: NGramModelRequest):
    try:
        lm = model_service.get_ngram_model(k_pred=request.k_pred, lm_type=request.lm_type, n=request.ngrams_order, test_size=request.test_size, gamma=request.gamma)
        if lm is None: 
             raise HTTPException(status_code=400, detail="Parametri non corretti")
         
        return NGramModelResponse(status=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restore", response_model=RestoreResponse)
def restore(request: RestoreRequest):
    try:
        lm = model_service.get_model()
        if lm is None:
            raise HTTPException(status_code=500, detail="Model not found")
          
        return RestoreResponse(
                    k_predictions=generate_k_suggests(
                    lm=lm, context=request.context, num_words=request.num_words
                    )
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
