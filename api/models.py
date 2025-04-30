from typing import Union, Literal
from pydantic import BaseModel, Field
from enum import IntEnum

# Model
class NgramModel(BaseModel):
    LM_SCORE: Literal["MLE", "LIDSTONE"] = Field(
        description="Tipo di score adottato dal modello linguistico (sono disponibili MLE e LIDSTONE)"
    )
    GAMMA: float | None = Field(
        default=None, description="Parametro di smoothing per il modello LIDSTONE"
    )
    MIN_FREQ: int = Field(
        description="Frequenza minima per la considerazione di un n-gramma dal vocabolario"
    )
    N: int = Field(description="Dimensionalità massima degli n-grammi del modello")
    CORPUS_NAMES: list[str] | None = Field(
        default=None,
        description="Lista dei nomi dei corpora da utilizzare per il training del modello, se assente si utilizzano i corpus di default",
    )
    TYPE: Literal["Ngrams"]

class BERTModel(BaseModel):
    MODEL: str = Field(description="Nome del modello BERT da utilizzare")
    TOKENIZER: str = Field(description="Nome del tokenizer da utilizzare")
    TYPE: Literal["BERT"]
    
class PredictionCount(IntEnum):
    ONE = 1
    FIVE = 5
    TEN = 10
    TWENTY = 20

Model = Union[NgramModel, BERTModel]
