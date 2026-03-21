from typing import List, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum, IntEnum

# Model
class NgramModel(BaseModel):
    LM_SCORE: Literal["MLE", "LIDSTONE"] = Field(
        description="Tipo di score adottato dal modello linguistico (sono disponibili MLE e LIDSTONE)"
    )
    GAMMA: float | None = Field(
        default=None, description="Parametro di smoothing per il modello LIDSTONE"
    )
    N: int = Field(description="Dimensionalità massima degli n-grammi del modello")
    CORPUS_NAMES: list[str] | None = Field(
        default=None,
        description="Lista dei nomi dei corpora da utilizzare per il training del modello, se assente si utilizzano i corpus di default",
    )
    TYPE: Literal["Ngrams"]

class BERTModel(BaseModel):
    CHECKPOINT: str = Field(description="Nome del modello BERT da utilizzare")
    TYPE: Literal["BERT"]
    
Model = Union[NgramModel, BERTModel]

class ModelsResponse(BaseModel): 
    models: List[Model]

class PredictionCount(IntEnum):
    ONE = 1
    FIVE = 5
    TEN = 10
    TWENTY = 20
class Prediction(BaseModel):
    sentence: str
    token_str: str
    score: float

class PredictionsResponse(BaseModel):
    predictions: List[Prediction]
    
class ModelType(str, Enum):
    NGRAMS = "Ngrams"
    BERT = "BERT"
