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
    from models.rag.categories import (
        DEFAULT_TEXT_CATEGORY_PROMPT,
        REASONING_CATEGORIES,
        TLG_CATEGORIES,
        TLG_CATEGORY_BY_CODE,
        TLGCategory,
        extract_cla_categories_from_canon,
        format_text_category_for_prompt,
        get_reasoning_category_for_cla,
        get_reasoning_profile,
        list_tlg_categories,
        load_tlg_work_categories_from_canon,
        normalize_text_categories,
        normalize_text_category,
    )
    from models.rag.conversation_logging import (
        ConversationTraceRecorder,
        DSPyConversationCallback,
        format_conversation_record,
        is_conversation_logging_enabled,
        load_conversation_records,
    )
    from models.rag.inference import fill_mask_rag
    from models.rag.llm_logging import (
        HuggingFaceCallLogger,
        configure_llm_logging,
        install_llm_call_logging,
    )
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
        build_mixed_optimization_cases,
        build_synthetic_tlg_category_cases,
        compile_greek_rag_infiller,
        dev_cases_to_dspy_examples,
        greek_infilling_dspy_metric,
        greek_infilling_gepa_metric,
        score_philological_reasoning,
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
    "ConversationTraceRecorder",
    "DEFAULT_TEXT_CATEGORY_PROMPT",
    "DSPyConversationCallback",
    "GreekBertEmbedder",
    "GreekCandidate",
    "GreekHybridRetriever",
    "GreekInfillingSignature",
    "GreekLexicalBM25Index",
    "GreekRAGInfiller",
    "HuggingFaceCallLogger",
    "HuggingFaceLocalLM",
    "REASONING_CATEGORIES",
    "TLGCategory",
    "TLG_CATEGORIES",
    "TLG_CATEGORY_BY_CODE",
    "TransformersLocalEngine",
    "build_dspy_lm",
    "build_mixed_optimization_cases",
    "build_rag_corpus_from_data",
    "build_synthetic_tlg_category_cases",
    "compile_greek_rag_infiller",
    "configure_llm_logging",
    "dev_cases_to_dspy_examples",
    "extract_cla_categories_from_canon",
    "extract_lacuna_metadata",
    "fill_mask_rag",
    "format_conversation_record",
    "format_gap_constraints",
    "format_text_category_for_prompt",
    "get_or_build_default_retriever",
    "get_reasoning_category_for_cla",
    "get_reasoning_profile",
    "greek_infilling_dspy_metric",
    "greek_infilling_gepa_metric",
    "install_llm_call_logging",
    "is_conversation_logging_enabled",
    "list_tlg_categories",
    "load_conversation_records",
    "load_tlg_work_categories_from_canon",
    "normalize_text_categories",
    "normalize_text_category",
    "sanitize_greek_supplement",
    "score_philological_reasoning",
]

# Mappa simbolo pubblico -> sottomodulo che lo definisce. La risoluzione effettiva
# avviene solo al primo accesso, evitando import eager di optimizer/evaluation.
_LAZY_IMPORTS: dict[str, str] = {
    "ConstraintRefinementSignature": "models.rag.signatures",
    "ConversationTraceRecorder": "models.rag.conversation_logging",
    "DEFAULT_TEXT_CATEGORY_PROMPT": "models.rag.categories",
    "DSPyConversationCallback": "models.rag.conversation_logging",
    "GreekBertEmbedder": "models.rag.retriever",
    "GreekCandidate": "models.rag.signatures",
    "GreekHybridRetriever": "models.rag.retriever",
    "GreekInfillingSignature": "models.rag.signatures",
    "GreekLexicalBM25Index": "models.rag.retriever",
    "GreekRAGInfiller": "models.rag.modules",
    "HuggingFaceCallLogger": "models.rag.llm_logging",
    "HuggingFaceLocalLM": "models.rag.lm",
    "REASONING_CATEGORIES": "models.rag.categories",
    "TLGCategory": "models.rag.categories",
    "TLG_CATEGORIES": "models.rag.categories",
    "TLG_CATEGORY_BY_CODE": "models.rag.categories",
    "TransformersLocalEngine": "models.rag.lm",
    "build_dspy_lm": "models.rag.lm",
    "build_mixed_optimization_cases": "models.rag.optimizer",
    "build_rag_corpus_from_data": "models.rag.retriever",
    "build_synthetic_tlg_category_cases": "models.rag.optimizer",
    "compile_greek_rag_infiller": "models.rag.optimizer",
    "configure_llm_logging": "models.rag.llm_logging",
    "dev_cases_to_dspy_examples": "models.rag.optimizer",
    "extract_cla_categories_from_canon": "models.rag.categories",
    "extract_lacuna_metadata": "models.rag.modules",
    "fill_mask_rag": "models.rag.inference",
    "format_conversation_record": "models.rag.conversation_logging",
    "format_gap_constraints": "models.rag.signatures",
    "format_text_category_for_prompt": "models.rag.categories",
    "get_or_build_default_retriever": "models.rag.retriever",
    "get_reasoning_category_for_cla": "models.rag.categories",
    "get_reasoning_profile": "models.rag.categories",
    "greek_infilling_dspy_metric": "models.rag.optimizer",
    "greek_infilling_gepa_metric": "models.rag.optimizer",
    "install_llm_call_logging": "models.rag.llm_logging",
    "is_conversation_logging_enabled": "models.rag.conversation_logging",
    "list_tlg_categories": "models.rag.categories",
    "load_conversation_records": "models.rag.conversation_logging",
    "load_tlg_work_categories_from_canon": "models.rag.categories",
    "normalize_text_categories": "models.rag.categories",
    "normalize_text_category": "models.rag.categories",
    "sanitize_greek_supplement": "models.rag.modules",
    "score_philological_reasoning": "models.rag.optimizer",
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
