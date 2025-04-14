from pydantic import BaseModel

#Model
class NgramModel(BaseModel): 
    LM_SCORE: str #lidstone o mle
    GAMMA: float = None
    MIN_FREQ: int
    K_PRED: int
    TEST_SIZE: float
    N: int
    CORPUS_NAMES: list[str] = None
    TYPE: str = "Ngrams"
    
    
#BERT model (da implementare)
class BERTModel(BaseModel): 
    model : str
    tokenizer : str
    TYPE: str = "BERT"