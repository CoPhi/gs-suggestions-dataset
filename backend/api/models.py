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

class IthacaModel(BaseModel):
    CHECKPOINT: str = Field(description="Nome del checkpoint o repo Hugging Face di Ithaca")
    TYPE: Literal["Ithaca"]


class RAGModel(BaseModel):
    CHECKPOINT: str = Field(
        description="Nome del modello LLM su Hugging Face (es. 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct')"
    )
    BACKEND: Literal["auto", "hf_api", "local", "openai_compatible"] = Field(
        default="auto",
        description="Modalità di esecuzione dell'LLM: 'auto', 'hf_api', 'local' (Transformers GPU) o 'openai_compatible' (vLLM/Ollama)",
    )
    TOP_K_PASSAGES: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Numero di passi paralleli (loci similes) da recuperare dai corpora TLG/MAAT",
    )
    COMPILED_PROGRAM_PATH: str | None = Field(
        default=None,
        description="Percorso opzionale al programma DSPy compilato (.json)",
    )
    TYPE: Literal["RAG"]
    
Model = Union[NgramModel, BERTModel, IthacaModel, RAGModel]

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
    ITHACA = "Ithaca"
    RAG = "RAG"
