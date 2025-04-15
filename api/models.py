from typing import Union, Literal
from pydantic import BaseModel, Field


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
    K_PRED: int = Field(description="Numero di predizioni che restituisce il modello")
    TEST_SIZE: float = Field(
        description="Percentuale di dataset da utilizzare per il formare il test set"
    )
    N: int = Field(description="Dimensionalità massima degli n-grammi del modello")
    CORPUS_NAMES: list[str] | None = Field(
        default=None,
        description="Lista dei nomi dei corpora da utilizzare per il training del modello, se assente si utilizzano i corpus di default",
    )
    TYPE: Literal["Ngrams"]


# BERT model (da implementare)
class BERTModel(BaseModel):
    MODEL: str = Field(
        description="Nome del modello BERT da utilizzare"
    )
    TOKENIZER: str = Field(
        description="Nome del tokenizer da utilizzare"
    )
    TYPE: Literal["BERT"]


Model = Union[NgramModel, BERTModel]