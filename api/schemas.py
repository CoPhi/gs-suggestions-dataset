from pydantic import BaseModel
from typing import Optional

class NGramModelRequest(BaseModel):
    k_pred: int
    ngrams_order: int
    lm_type: str #MLE o Lidstone
    gamma: Optional[float] #in caso in cui il modello sia Lidstone
    test_size: float

class NGramModelResponse(BaseModel):
    status: int  #creo il modello
    
class GreekBERTModelRequest(BaseModel):
    k_pred: int
    model: str
    
class GreekBERTModelResponse(BaseModel):
    status: int  #creo il modello

class RestoreRequest(BaseModel):
    context: str
    num_words: int

class RestoreResponse(BaseModel):
    k_predictions : list[str]