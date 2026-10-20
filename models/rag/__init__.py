"""
Package `models.rag`: Sistema di Text-Infilling basato su Retrieval-Augmented Generation (RAG)
orchestrato tramite il framework DSPy 3.4+ con supporto a LLM Hugging Face (Qwen, Llama)
e retrieval ibrido (BM25 + Dense Embeddings BERT) sui corpora in greco antico (TLG e MAAT).
"""

from models.rag.inference import fill_mask_rag
from models.rag.lm import HuggingFaceLocalLM, TransformersLocalEngine, build_dspy_lm
from models.rag.modules import (
    GreekRAGInfiller,
    extract_lacuna_metadata,
    sanitize_greek_supplement,
)
from models.rag.optimizer import (
    compile_greek_rag_infiller,
    dev_cases_to_dspy_examples,
    greek_infilling_dspy_metric,
)
from models.rag.retriever import (
    GreekBertEmbedder,
    GreekHybridRetriever,
    GreekLexicalBM25Index,
    build_rag_corpus_from_data,
    get_or_build_default_retriever,
)
from models.rag.signatures import (
    ConstraintRefinementSignature,
    GreekCandidate,
    GreekInfillingSignature,
    format_gap_constraints,
)

__all__ = [
    "ConstraintRefinementSignature",
    "GreekBertEmbedder",
    "GreekCandidate",
    "GreekHybridRetriever",
    "GreekInfillingSignature",
    "GreekLexicalBM25Index",
    "GreekRAGInfiller",
    "HuggingFaceLocalLM",
    "TransformersLocalEngine",
    "build_dspy_lm",
    "build_rag_corpus_from_data",
    "compile_greek_rag_infiller",
    "dev_cases_to_dspy_examples",
    "extract_lacuna_metadata",
    "fill_mask_rag",
    "format_gap_constraints",
    "get_or_build_default_retriever",
    "greek_infilling_dspy_metric",
    "sanitize_greek_supplement",
]
