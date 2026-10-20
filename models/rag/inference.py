"""
Pipeline di Inferenza Unificata per il Text-Infilling tramite DSPy RAG (`fill_mask_rag`).

Espone una firma compatibile con `fill_mask` (BERT) e `fill_mask_ithaca` (Ithaca),
restituendo una lista ordinata di tuple `(supplemento_normalizzato, score)` e
supportando opzionalmente la restituzione dei dettagli di ragionamento filologico
e dei passi paralleli recuperati.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Literal

import dspy

from backend.config.settings import (
    RAG_DEFAULT_BACKEND,
    RAG_DEFAULT_EMBEDDER,
    RAG_DEFAULT_LM,
    RAG_INDEX_DIR,
    RAG_TOP_K_PASSAGES,
)
from backend.core.preprocess import (
    normalize_greek,
    remove_punctuation,
    strip_diacritics,
)
from models.bert.inference.vocab import get_normalized_vocab
from models.rag.lm import build_dspy_lm
from models.rag.modules import GreekRAGInfiller, extract_lacuna_metadata
from models.rag.retriever import GreekHybridRetriever, get_or_build_default_retriever

_COMPILED_MODULE_CACHE: dict[str, GreekRAGInfiller] = {}


def _load_or_create_infiller(
    retriever: GreekHybridRetriever | None,
    top_k_passages: int = RAG_TOP_K_PASSAGES,
    compiled_program_path: str | Path | None = None,
    enable_refinement: bool = True,
) -> GreekRAGInfiller:
    infiller = GreekRAGInfiller(
        retriever=retriever,
        top_k_passages=top_k_passages,
        enable_refinement=enable_refinement,
    )
    if compiled_program_path:
        ckpt_path = Path(compiled_program_path)
        if ckpt_path.exists():
            infiller.load(str(ckpt_path))
    return infiller


def fill_mask_rag(
    text: str,
    checkpoint: str = RAG_DEFAULT_LM,
    lm: dspy.BaseLM | None = None,
    backend: Literal["auto", "hf_api", "local", "openai_compatible"] = RAG_DEFAULT_BACKEND,
    retriever: GreekHybridRetriever | None = None,
    use_default_retriever: bool = True,
    index_dir: str | Path = RAG_INDEX_DIR,
    embedder_checkpoint: str = RAG_DEFAULT_EMBEDDER,
    top_k_passages: int = RAG_TOP_K_PASSAGES,
    compiled_program_path: str | Path | None = None,
    n_chars: int | tuple[int, int] | None = None,
    char_tolerance: int = 0,
    K: int = 20,
    case_folding: Literal["upper", "lower", "fold", "none"] = "fold",
    strip_diacritics_flag: bool = True,
    normalize_probs: bool = True,
    use_vocab_filter: bool = False,
    vocab_path: str = "grc.wl",
    exclude_query_overlap: bool = False,
    enable_refinement: bool = True,
    return_details: bool = False,
) -> list[tuple[str, float]] | tuple[list[tuple[str, float]], dict[str, Any]]:
    """
    Esegue il text-infilling su un testo greco contenente una lacuna `[...]` o `[---]`
    utilizzando il modulo DSPy `GreekRAGInfiller`.

    Args:
        text: Frammento in greco antico con almeno una lacuna `[...]` o `[---]`.
        checkpoint: Modello LLM di Hugging Face (es. `'Qwen/Qwen2.5-7B-Instruct'`,
                    `'meta-llama/Llama-3.1-8B-Instruct'`).
        lm: Istanza opzionale `dspy.BaseLM` già configurata (es. per test con `DummyLM`).
        backend: Modalità di esecuzione dell'LLM (`'auto'`, `'hf_api'`, `'local'`, `'openai_compatible'`).
        retriever: Istanza opzionale di `GreekHybridRetriever`.
        use_default_retriever: Se True e `retriever` è None, carica l'indice di default da `index_dir` o `data/`.
        index_dir: Percorso all'indice ibrido su disco.
        embedder_checkpoint: Checkpoint BERT greco per il retriever denso.
        top_k_passages: Numero di passi paralleli da recuperare.
        compiled_program_path: Percorso opzionale al file `.json` prodotto dall'ottimizzatore DSPy.
        n_chars: Lunghezza attesa della lacuna (intero o tupla `(min, max)`). Se None, dedotta dai puntini.
        char_tolerance: Tolleranza in caratteri attorno a `n_chars`.
        K: Numero massimo di suggerimenti da restituire.
        case_folding: Modalità di normalizzazione del case sull'output (`'fold'`, `'lower'`, `'upper'`, `'none'`).
        strip_diacritics_flag: Se True, rimuove i diacritici dai candidati restituiti.
        normalize_probs: Se True, normalizza i punteggi finali con softmax affinché sommino a 1.0.
        use_vocab_filter: Se True, filtra le parole ricostruite tramite il dizionario greco `vocab_path`.
        vocab_path: Percorso alla wordlist greca (`grc.wl`).
        exclude_query_overlap: Se True, attiva il filtro anti-data-leakage nel retriever (utile nei benchmark).
        enable_refinement: Se True, attiva il passo di correzione `ConstraintRefinementSignature`.
        return_details: Se True, restituisce `(candidates, details_dict)` con `philological_reasoning` e `retrieved_passages`.

    Returns:
        Lista di tuple `(supplemento, score)` (oppure tupla `(lista_candidati, dettagli)` se `return_details=True`).
    """
    meta = extract_lacuna_metadata(text, n_chars=n_chars, char_tolerance=char_tolerance)
    min_chars = meta["min_chars"]
    max_chars = meta["max_chars"]
    center_chars = meta["center_chars"]
    word_prefix = meta["word_prefix"]
    word_suffix = meta["word_suffix"]

    active_retriever = retriever
    if active_retriever is None and use_default_retriever and top_k_passages > 0:
        try:
            active_retriever = get_or_build_default_retriever(
                index_dir=index_dir,
                embedder_checkpoint=embedder_checkpoint,
            )
        except Exception:
            active_retriever = None

    active_lm = lm or dspy.settings.lm
    if active_lm is None:
        active_lm = build_dspy_lm(checkpoint=checkpoint, backend=backend)

    infiller = _load_or_create_infiller(
        retriever=active_retriever,
        top_k_passages=top_k_passages,
        compiled_program_path=compiled_program_path,
        enable_refinement=enable_refinement,
    )

    with dspy.context(lm=active_lm):
        pred = infiller(
            context_with_gap=meta["context_with_gap"],
            min_chars=min_chars,
            max_chars=max_chars,
            word_prefix=word_prefix,
            word_suffix=word_suffix,
            num_candidates=K,
            exclude_query_overlap=exclude_query_overlap,
        )

    vocab = get_normalized_vocab(vocab_path) if use_vocab_filter else None
    if not vocab:
        use_vocab_filter = False

    # Uniamo candidati validi (priorità True) e candidati fallback (priorità False)
    scored_pool: list[tuple[str, float, bool]] = []
    for idx, cand in enumerate(getattr(pred, "candidates", []) or []):
        base_len = len(strip_diacritics(cand.supplement))
        len_penalty = -0.05 * abs(base_len - center_chars) if center_chars > 0 else 0.0
        rank_Decay = -0.02 * idx
        score = float(cand.confidence) + len_penalty + rank_Decay
        scored_pool.append((cand.supplement, score, True))

    for idx, cand in enumerate(getattr(pred, "fallback_candidates", []) or []):
        base_len = len(strip_diacritics(cand.supplement))
        len_penalty = -0.15 * abs(base_len - center_chars) if center_chars > 0 else 0.0
        rank_decay = -0.03 * idx
        score = float(cand.confidence) + len_penalty + rank_decay - 0.5
        scored_pool.append((cand.supplement, score, False))

    scored_pool.sort(key=lambda x: (x[2], x[1]), reverse=True)

    seen: set[str] = set()
    unique_candidates: list[tuple[str, float]] = []

    for raw_supp, score, _is_valid_len in scored_pool:
        norm_key = normalize_greek(
            raw_supp,
            case_folding=case_folding,
            strip_diacritics_flag=strip_diacritics_flag,
        )
        norm_key = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", norm_key).strip()
        if not norm_key:
            continue

        if use_vocab_filter and vocab is not None:
            reconstructed = f"{word_prefix}{raw_supp}{word_suffix}"
            norm_recon = normalize_greek(
                reconstructed,
                case_folding="fold",
                strip_diacritics_flag=True,
            )
            norm_recon = remove_punctuation(norm_recon)
            norm_recon = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", norm_recon).strip()
            if norm_recon not in vocab:
                continue

        if norm_key not in seen:
            seen.add(norm_key)
            unique_candidates.append((norm_key, float(score)))
        if len(unique_candidates) >= K:
            break

    if normalize_probs and unique_candidates:
        max_s = max(s for _, s in unique_candidates)
        unnorm = [math.exp(s - max_s) for _, s in unique_candidates]
        total = sum(unnorm) or 1.0
        unique_candidates = [
            (item, float(p / total))
            for (item, _), p in zip(unique_candidates, unnorm)
        ]

    if return_details:
        details = {
            "philological_reasoning": getattr(pred, "philological_reasoning", ""),
            "retrieved_passages": getattr(pred, "retrieved_passages", []),
            "gap_constraints": getattr(pred, "gap_constraints", ""),
            "min_chars": min_chars,
            "max_chars": max_chars,
            "word_prefix": word_prefix,
            "word_suffix": word_suffix,
        }
        return unique_candidates, details

    return unique_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferenza interattiva di Text-Infilling in Greco Antico tramite DSPy RAG."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Contesto in greco antico contenente una lacuna `[...]` (es. 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]')",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default=RAG_DEFAULT_LM,
        help=f"Modello LLM Hugging Face (default: {RAG_DEFAULT_LM})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=RAG_DEFAULT_BACKEND,
        choices=["auto", "hf_api", "local", "openai_compatible"],
        help="Backend di inferenza per l'LLM",
    )
    parser.add_argument(
        "--top-k-passages",
        type=int,
        default=RAG_TOP_K_PASSAGES,
        help="Numero di passi paralleli da recuperare dai corpora",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Numero di supplementi candidati da restituire",
    )
    parser.add_argument(
        "--compiled-path",
        type=str,
        default=None,
        help="Percorso opzionale al modulo DSPy compilato (.json)",
    )
    args = parser.parse_args()

    preds, details = fill_mask_rag(
        text=args.text,
        checkpoint=args.lm,
        backend=args.backend,
        top_k_passages=args.top_k_passages,
        compiled_program_path=args.compiled_path,
        K=args.k,
        return_details=True,
    )

    print("\n=== Vincoli Papirologici Estratti ===")
    print(details["gap_constraints"])

    print(f"\n=== Passi Paralleli Recuperati ({len(details['retrieved_passages'])}) ===")
    for idx, p in enumerate(details["retrieved_passages"], 1):
        print(f"  [{idx}] {p[:180]}...")

    if details["philological_reasoning"]:
        print("\n=== Ragionamento Filologico (Chain-of-Thought) ===")
        print(details["philological_reasoning"])

    print(f"\n=== Top-{len(preds)} Supplementi Proposti ===")
    for rank, (supp, score) in enumerate(preds, 1):
        print(f"  {rank:2d}. {supp:<15} (score: {score:.4f})")


if __name__ == "__main__":
    main()
