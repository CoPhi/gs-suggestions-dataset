from pydantic import BaseModel

#Model
class NgramModel(BaseModel): 
    LM_SCORE: str #lidstone o mle
    GAMMA: float = None
    K_PRED: int
    TEST_SIZE: float
    N: int
    CORPUS_NAMES: list[str] = None
    
    
#BERT model (da implementare)