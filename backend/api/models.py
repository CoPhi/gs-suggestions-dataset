from typing import List, Union, Literal, Any, Dict
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
        description="Nome del modello LLM o BERT su Hugging Face (es. 'Qwen/Qwen2.5-72B-Instruct', 'CNR-ILC/gs-GreBerta')"
    )
    BACKEND: Literal["auto", "hf_api", "local", "openai_compatible", "neuro_lexical"] = Field(
        default="auto",
        description="Modalità di esecuzione: 'auto', 'hf_api', 'local' (Transformers GPU), 'openai_compatible' (vLLM/Ollama) o 'neuro_lexical' (BERT + RAG in-process)",
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
class CorpusAttestation(BaseModel):
    """Riferimento al documento del corpus in cui è attestato il suggerimento."""

    corpus_id: str = Field(
        default="unknown",
        description="Identificativo della collezione di documenti nel corpus (es. 'tlg', 'DCLP', 'DDbDP', 'PHerc.')",
    )
    source_id: str = Field(
        default="",
        description="Titolo dell'opera, papiro o luogo di attestazione nel corpus",
    )
    attested_form: str = Field(
        default="",
        description="Parola o forma greca attestata nel documento",
    )
    passage: str = Field(
        default="",
        description="Estratto testuale del documento che attesta il suggerimento",
    )
    highlighted_passage: str = Field(
        default="",
        description="Estratto testuale del documento con l'attestazione evidenziata in grassetto (**...**)",
    )
    doc_freq: int = Field(
        default=1,
        description="Numero di documenti del corpus in cui la forma è attestata",
    )
    source_type: str = Field(
        default="retrieved_passage",
        description="Origine dell'attestazione ('retrieved_passage' o 'corpus_index')",
    )


class Prediction(BaseModel):
    sentence: str
    token_str: str
    score: float
    tlg_attested: bool = Field(
        default=False,
        description="True se la forma ottenuta tramite boundary reconstruction col contesto è attestata nel TLG",
    )
    reconstructed_word: str | None = Field(
        default=None,
        description="Parola greca completa ricostruita concatenando prefisso, supplemento e suffisso adiacenti alla lacuna",
    )
    philological_reasoning: str | None = None
    candidate_reasoning: str | None = None
    retrieved_passages: List[str] | None = None
    all_retrieved_passages: List[str] | None = None
    attestations: List[CorpusAttestation] | None = None

class PredictionsResponse(BaseModel):
    predictions: List[Prediction]


class ModelPredictionsItem(BaseModel):
    """Risultato delle predizioni di un singolo modello all'interno di un confronto parallelo."""

    model_id: str = Field(description="ID del modello consultato")
    model_label: str = Field(description="Etichetta leggibile del modello (es. 'gs-GreBerta + RAG')")
    model_type: str = Field(description="Famiglia architetturale ('RAG', 'BERT', 'Ithaca', 'Ngrams')")
    checkpoint: str | None = Field(default=None, description="Checkpoint o identificativo del modello")
    use_rag: bool = Field(default=False, description="True se il sistema RAG di recupero informazioni è attivo")
    predictions: List[Prediction] = Field(default_factory=list, description="Suggerimenti generati dal modello")
    error: str | None = Field(default=None, description="Messaggio d'errore eventuale in caso di fallimento isolato")


class BatchPredictionsResponse(BaseModel):
    """Risposta aggregata al completamento della barriera di esecuzione parallela multi-modello."""

    results: List[ModelPredictionsItem]


class AlternativeCandidate(BaseModel):
    """Candidato alternativo generato nello stesso turno, per la costruzione di preference pairs per RLHF/DPO."""

    token_str: str = Field(description="Supplemento alternativo proposto")
    score: float | None = Field(default=None, description="Punteggio o confidenza assegnata dal modello")
    rank: int | None = Field(default=None, description="Posizione di ranking (1-based) nella lista delle predizioni")
    tlg_attested: bool | None = Field(default=None, description="True se la forma è attestata nel TLG")
    reconstructed_word: str | None = Field(default=None, description="Parola greca ricostruita")


class FeedbackCreateRequest(BaseModel):
    """Payload inviato dal filologo per validare o scartare un supplemento proposto (HITL Fase 1 e RLHF)."""

    model_id: str = Field(description="ID del modello che ha generato il supplemento")
    context: str = Field(description="Frammento greco con lacuna [...] sottoposto a infilling")
    token_str: str = Field(description="Supplemento proposto oggetto di valutazione")
    sentence: str | None = Field(
        default=None,
        description="Contesto completo con la lacuna colmata dal supplemento",
    )
    is_accepted: bool = Field(
        description="True se il filologo valida il supplemento come plausibile/corretto, False se lo scarta"
    )
    notes: str | None = Field(
        default=None,
        description="Note filologiche o motivazione opzionale fornita dallo studioso",
    )

    # Dati intrinseci del suggerimento proposto
    score: float | None = Field(
        default=None,
        description="Punteggio o probabilità di confidenza assegnata dal modello generatore",
    )
    rank: int | None = Field(
        default=None,
        description="Posizione ordinale (1-based) del suggerimento all'interno della lista dei top-k",
    )
    reconstructed_word: str | None = Field(
        default=None,
        description="Parola greca completa ricostruita col contesto adiacente alla lacuna",
    )
    tlg_attested: bool | None = Field(
        default=None,
        description="True se la parola ricostruita è presente nel corpus TLG",
    )
    candidate_reasoning: str | None = Field(
        default=None,
        description="Chain-of-Thought filologica di 2° livello specifica per il candidato",
    )
    philological_reasoning: str | None = Field(
        default=None,
        description="Chain-of-Thought filologica di 1° livello sull'intero passo",
    )
    attestations: List[CorpusAttestation] | None = Field(
        default=None,
        description="Attestazioni puntuali del supplemento nei corpora storici",
    )

    # Metadati estesi sul modello generatore (Policy)
    model_label: str | None = Field(
        default=None,
        description="Etichetta descrittiva del modello (es. 'gs-GreBerta + RAG')",
    )
    model_type: str | None = Field(
        default=None,
        description="Famiglia architetturale del modello ('BERT', 'RAG', 'Ithaca', 'Ngrams')",
    )
    checkpoint: str | None = Field(
        default=None,
        description="Identificativo del checkpoint Hugging Face o percorso pesi",
    )
    use_rag: bool | None = Field(
        default=None,
        description="True se la predizione è stata potenziata con retrieval RAG dai corpora",
    )
    rag_backend: str | None = Field(
        default=None,
        description="Backend di esecuzione RAG ('neuro_lexical', 'local', 'openai_compatible', 'hf_api')",
    )
    top_k: int | None = Field(
        default=None,
        description="Numero totale di candidati richiesti durante la generazione",
    )

    # Predisposizione per Reinforcement Learning from Human Feedback (RLHF / DPO / KTO)
    alternative_candidates: List[AlternativeCandidate] | List[dict[str, Any]] | None = Field(
        default=None,
        description="Candidati alternativi generati nello stesso turno per supportare la creazione di coppie di preferenza per DPO/RLHF",
    )
    session_id: str | None = Field(
        default=None,
        description="Identificativo univoco di sessione di lavoro del filologo",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Dizionario per metadati addizionali o futuri iperparametri di inferenza",
    )


class FeedbackResponse(BaseModel):
    id: str
    status: str = "saved"


class ModelType(str, Enum):
    NGRAMS = "Ngrams"
    BERT = "BERT"
    ITHACA = "Ithaca"
    RAG = "RAG"
