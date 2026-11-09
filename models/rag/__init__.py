"""
Package `models.rag`: Sistema di Text-Infilling basato su Retrieval-Augmented Generation (RAG)
orchestrato tramite il framework DSPy 3.4+ con supporto a LLM Hugging Face (Qwen, Llama)
e retrieval ibrido (BM25 + Dense Embeddings BERT) sui corpora in greco antico (TLG e MAAT).

Gli export pubblici sono risolti in modo lazy (PEP 562): importare un sottomodulo leggero
come `models.rag.inference` non deve trascinare dentro `optimizer`/`evaluation`, che
dipendono dai package di training (`models.bert.dataset`, `models.bert.evaluation`)
esclusi dal contesto Docker dell'API. Questo mantiene l'immagine `gs-api` slim e
disaccoppiata dagli strumenti di training/HPO (vedi `.dockerignore`).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.rag.inference import fill_mask_rag
    from models.rag.lm import (
        HuggingFaceLocalLM,
        TransformersLocalEngine,
        build_dspy_lm,
    )
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

# Mappa simbolo pubblico -> sottomodulo che lo definisce. La risoluzione effettiva
# avviene solo al primo accesso, evitando import eager di optimizer/evaluation.
_LAZY_IMPORTS: dict[str, str] = {
    "ConstraintRefinementSignature": "models.rag.signatures",
    "GreekBertEmbedder": "models.rag.retriever",
    "GreekCandidate": "models.rag.signatures",
    "GreekHybridRetriever": "models.rag.retriever",
    "GreekInfillingSignature": "models.rag.signatures",
    "GreekLexicalBM25Index": "models.rag.retriever",
    "GreekRAGInfiller": "models.rag.modules",
    "HuggingFaceLocalLM": "models.rag.lm",
    "TransformersLocalEngine": "models.rag.lm",
    "build_dspy_lm": "models.rag.lm",
    "build_rag_corpus_from_data": "models.rag.retriever",
    "compile_greek_rag_infiller": "models.rag.optimizer",
    "dev_cases_to_dspy_examples": "models.rag.optimizer",
    "extract_lacuna_metadata": "models.rag.modules",
    "fill_mask_rag": "models.rag.inference",
    "format_gap_constraints": "models.rag.signatures",
    "get_or_build_default_retriever": "models.rag.retriever",
    "greek_infilling_dspy_metric": "models.rag.optimizer",
    "sanitize_greek_supplement": "models.rag.modules",
}


def __getattr__(name: str) -> Any:
    """Risolve gli export pubblici al primo accesso, con cache in `globals()`."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
