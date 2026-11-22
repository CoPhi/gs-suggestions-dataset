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
from backend.core.model_cache import shared_bert_cache
from backend.core.preprocess import (
    normalize_greek,
    remove_punctuation,
    strip_diacritics,
)
from models.bert.inference.predict import fill_mask
from models.bert.inference.vocab import (
    check_tlg_attestation,
    get_normalized_vocab,
    normalize_for_tlg_lookup,
)
from models.rag.conversation_logging import (
    ConversationTraceRecorder,
    is_conversation_logging_enabled,
)
from models.rag.llm_logging import (
    log_llm_fallback_event,
    log_llm_skipped_event,
)
from models.rag.lm import (
    build_dspy_lm,
    get_fallback_dspy_lms,
    is_hf_api_unavailable,
    is_hf_permission_or_auth_error,
    is_ollama_model_within_ram_budget,
    mark_hf_api_unavailable,
)
from models.rag.modules import (
    GreekRAGInfiller,
    extract_lacuna_metadata,
    extract_parallel_attestations,
    sanitize_greek_supplement,
)
from models.rag.retriever import (
    GreekHybridRetriever,
    get_or_build_default_retriever,
    resolve_candidate_attestations,
)
from models.rag.signatures import GreekCandidate, format_gap_constraints

_COMPILED_MODULE_CACHE: dict[str, GreekRAGInfiller] = {}
# Condivide la cache LRU unificata con SuggestionsService e GreekBertEmbedder per evitare copie duplicate in RAM
_LOCAL_MLM_CACHE = shared_bert_cache.cache
# Temperatura softmax calibrata per amplificare il distacco percentuale dei candidati con doppia attestazione
RAG_SOFTMAX_TEMPERATURE: float = 0.28


def calibrate_candidate_attestation_score(
    base_score: float,
    is_tlg_attested: bool,
    attestations: list[dict[str, Any]] | None,
    is_index_only: bool = False,
    apply_unattested_penalty: bool = True,
) -> tuple[float, int]:
    """
    Calibra lo score di un candidato combinando il punteggio contestuale del modello
    con una gerarchia di evidenza documentaria a 4 livelli:
      - Tier 1: doppia attestazione TLG (`tlg.wl.gz`) + passi paralleli contestuali (`retrieved_passage`).
      - Tier 2: doppia attestazione TLG + indice globale dei corpora (`corpus_index`, pesato per `doc_freq`),
                riducendo il delta se il candidato proviene solo da `match_affix_words`.
      - Tier 3: attestazione singola (solo corpus del retriever oppure solo lessico TLG).
      - Tier 4: nessuna attestazione (congettura non attestata).
    Restituisce `(calibrated_score, attestation_tier)`.
    """
    atts = attestations or []
    has_retriever_att = bool(atts)
    has_passage_att = any(
        str(a.get("source_type", "")) == "retrieved_passage" for a in atts
    )
    max_doc_freq = max(
        (int(a.get("doc_freq", 1) or 1) for a in atts),
        default=0,
    )
    doc_freq_bonus = (
        min(0.12, 0.03 * math.log1p(max_doc_freq)) if max_doc_freq > 0 else 0.0
    )

    if is_tlg_attested and has_passage_att:
        # Tier 1: convergenza tra lessico TLG e loci similes recuperati per il contesto
        delta = 0.35 + doc_freq_bonus
        tier = 1
    elif is_tlg_attested and has_retriever_att:
        # Tier 2: doppia attestazione TLG + indice globale del corpus; evita il doppio conteggio
        # per i candidati proposti solo per affisso senza conferma contestuale BERT/LLM.
        if is_index_only:
            delta = 0.06 + 0.5 * doc_freq_bonus
        else:
            delta = 0.22 + doc_freq_bonus
        tier = 2
    elif is_tlg_attested or has_retriever_att:
        # Tier 3: evidenza su un solo fronte (solo corpus papirologico o solo wordlist TLG)
        if has_retriever_att:
            delta = (0.15 if has_passage_att else 0.08) + 0.5 * doc_freq_bonus
        else:
            delta = 0.05
        tier = 3
    else:
        # Tier 4: forma priva di riscontri sia nel TLG sia nei corpora indicizzati
        delta = -0.20 if apply_unattested_penalty else 0.0
        tier = 4

    return float(base_score + delta), tier


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


def is_bert_checkpoint(checkpoint: str) -> bool:
    """
    Verifica se il checkpoint identifica un modello Masked Language Model BERT greco
    (es. 'CNR-ILC/gs-GreBerta', 'CNR-ILC/gs-aristoBERTo', 'CNR-ILC/gs-Logion',
    'bowphs/GreBerta', 'Jacobo/aristoBERTo', 'cabrooks/LOGION-50k_wordpiece',
    'pranaydeeps/Ancient-Greek-BERT'), distinguendolo dagli LLM causali/instruct.
    """
    clean = (checkpoint or "").strip().lower()
    for prefix in ("huggingface/", "local/"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
    if any(
        llm_kw in clean
        for llm_kw in ("qwen", "llama", "gemma", "deepseek", "mistral", "ithaca")
    ):
        return False
    return any(
        bert_kw in clean
        for bert_kw in (
            "greberta",
            "aristoberto",
            "logion",
            "ancient-greek-bert",
            "bert",
        )
    )


def _get_cached_mlm(checkpoint: str) -> tuple[Any, Any]:
    """
    Carica o recupera dalla cache condivisa in-process il modello Masked Language Model greco
    (es. `CNR-ILC/gs-GreBerta`) utilizzato per il motore neuro-lessicale locale.
    """
    return shared_bert_cache.get_or_load_mlm(checkpoint)


def _build_neuro_lexical_rag_prediction(
    context_with_gap: str,
    min_chars: int,
    max_chars: int,
    word_prefix: str,
    word_suffix: str,
    num_candidates: int,
    retriever: GreekHybridRetriever | None,
    top_k_passages: int,
    exclude_query_overlap: bool,
    embedder_checkpoint: str,
    fallback_reason: str = "",
) -> dspy.Prediction:
    """
    Motore di fallback RAG neuro-lessicale eseguito interamente in-process quando il
    backend LLM esterno non è raggiungibile (es. `HF_TOKEN` privo del permesso
    'Make calls to Inference Providers' oppure server locale Ollama non disponibile / in OOM).

    Combina:
    1. Il recupero ibrido dei passi paralleli (*loci similes*) tramite `GreekHybridRetriever`.
    2. L'estrazione delle attestazioni dirette dai passi recuperati e dall'indice BM25 del corpus.
    3. La decodifica condizionata HCB Beam Search del modello MLM greco (`CNR-ILC/gs-GreBerta`).
    """
    gap_constraints_str = format_gap_constraints(
        min_chars=min_chars,
        max_chars=max_chars,
        word_prefix=word_prefix,
        word_suffix=word_suffix,
    )

    retrieved_passages: list[str] = []
    all_retrieved_passages: list[str] = []
    if retriever is not None and top_k_passages > 0:
        try:
            ret_pred = retriever(
                query=context_with_gap,
                k=top_k_passages,
                word_prefix=word_prefix,
                word_suffix=word_suffix,
                exclude_query_overlap=exclude_query_overlap,
            )
            retrieved_passages = list(getattr(ret_pred, "passages", []) or [])
            all_retrieved_passages = list(
                getattr(ret_pred, "all_passages", None) or retrieved_passages
            )
        except Exception:
            retrieved_passages = []
            all_retrieved_passages = []

    # 1. Attestazioni dirette nei passi paralleli recuperati
    passage_attestations = extract_parallel_attestations(
        context_with_gap=context_with_gap,
        retrieved_passages=all_retrieved_passages or retrieved_passages,
        min_chars=min_chars,
        max_chars=max_chars,
        word_prefix=word_prefix,
        word_suffix=word_suffix,
    )

    # 2. Attestazioni lessicali dall'indice BM25 per lacune intra-parola (prefisso/suffisso)
    index_attestations: list[tuple[str, str, int]] = []
    if retriever is not None and hasattr(retriever, "lexical_index") and (word_prefix or word_suffix):
        try:
            index_attestations = retriever.lexical_index.match_affix_words(
                word_prefix=word_prefix,
                word_suffix=word_suffix,
                min_gap_chars=min_chars,
                max_gap_chars=max_chars,
                top_n=max(num_candidates, 10),
            )
        except Exception:
            index_attestations = []

    cand_scores: dict[str, float] = {}
    cand_rationales: dict[str, str] = {}
    attested_forms_info: list[str] = []
    passage_att_norms: set[str] = set()
    index_only_norms: set[str] = set()

    for att in passage_attestations:
        san = sanitize_greek_supplement(
            att.supplement,
            word_prefix=word_prefix,
            word_suffix=word_suffix,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if san:
            norm_san = normalize_greek(san, case_folding="fold", strip_diacritics_flag=True)
            cand_scores[san] = max(cand_scores.get(san, 0.0), float(att.confidence))
            passage_att_norms.add(norm_san)
            if att.rationale:
                cand_rationales[san] = att.rationale

    affix_specificity = len(strip_diacritics(word_prefix)) + len(strip_diacritics(word_suffix))
    for full_word, supp, doc_freq in index_attestations:
        san = sanitize_greek_supplement(
            supp,
            word_prefix=word_prefix,
            word_suffix=word_suffix,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if san:
            norm_san = normalize_greek(san, case_folding="fold", strip_diacritics_flag=True)
            # Manteniamo un prior lessicale moderato per le forme trovate solo per affisso
            # nell'indice BM25, così che non prevalgano sui candidati confermati dal modello BERT.
            if affix_specificity >= 3:
                freq_conf = min(0.68, 0.56 + 0.025 * math.log1p(doc_freq))
            else:
                freq_conf = min(0.62, 0.52 + 0.02 * math.log1p(doc_freq))
            cand_scores[san] = max(cand_scores.get(san, 0.0), freq_conf)
            if norm_san not in passage_att_norms:
                index_only_norms.add(norm_san)
            cand_rationales.setdefault(
                san,
                f"Ricostruisce la forma **{full_word}** attestata in {doc_freq} "
                f"{'documento' if doc_freq == 1 else 'documenti'} del corpus e rispetta il vincolo di {len(strip_diacritics(san))} lettere.",
            )
            if len(attested_forms_info) < 3:
                attested_forms_info.append(f"**{full_word}** (supplemento **{san}**, {doc_freq} occorrenze nel corpus)")

    # 3. Completamento contestuale tramite HCB Beam Search con il modello MLM greco
    bert_short_name = (
        embedder_checkpoint.split("/")[-1]
        if "/" in embedder_checkpoint
        else embedder_checkpoint
    )
    fallback_candidates: list[GreekCandidate] = []
    try:
        mlm_model, mlm_tokenizer = _get_cached_mlm(embedder_checkpoint)
        mlm_raw = fill_mask(
            text=context_with_gap,
            model=mlm_model,
            tokenizer=mlm_tokenizer,
            checkpoint=embedder_checkpoint,
            n_chars=(min_chars, max_chars),
            K=max(num_candidates * 2, 15),
            normalize_probs=False,
        )
        norm_to_cand_key: dict[str, str] = {
            normalize_greek(k, case_folding="fold", strip_diacritics_flag=True): k
            for k in cand_scores
        }
        valid_rank = 0
        fb_rank = 0
        for raw_supp, _log_p in mlm_raw:
            if not isinstance(raw_supp, str):
                continue
            san = sanitize_greek_supplement(
                raw_supp,
                word_prefix=word_prefix,
                word_suffix=word_suffix,
                min_chars=min_chars,
                max_chars=max_chars,
            )
            if not san:
                continue
            cand_len = len(strip_diacritics(san))
            if min_chars <= cand_len <= max_chars:
                mlm_conf = max(0.25, 0.84 - 0.05 * valid_rank)
                valid_rank += 1
                norm_san = normalize_greek(san, case_folding="fold", strip_diacritics_flag=True)
                existing_key = san if san in cand_scores else norm_to_cand_key.get(norm_san)
                if existing_key is not None:
                    # La conferma contestuale di BERT promuove il candidato fuori dal gruppo index-only
                    index_only_norms.discard(norm_san)
                    cand_scores[existing_key] = min(
                        0.96, max(cand_scores[existing_key], mlm_conf) + 0.08
                    )
                    if existing_key in cand_rationales:
                        cand_rationales[existing_key] = (
                            f"{cand_rationales[existing_key]} Confermato anche dalla distribuzione contestuale {bert_short_name}."
                        )
                else:
                    cand_scores[san] = mlm_conf
                    norm_to_cand_key[norm_san] = san
                    full_rec = f"{word_prefix}{san}{word_suffix}"
                    cand_rationales[san] = (
                        f"Completa la forma **{full_rec}** ({cand_len} lettere) con elevata probabilità "
                        f"morfologico-contestuale secondo il modello {bert_short_name}."
                    )
            else:
                fb_conf = max(0.10, 0.45 - 0.04 * fb_rank)
                fb_rank += 1
                fallback_candidates.append(GreekCandidate(supplement=san, confidence=fb_conf))
    except Exception:
        pass

    valid_candidates = [
        GreekCandidate(
            supplement=supp,
            confidence=conf,
            rationale=cand_rationales.get(supp, ""),
        )
        for supp, conf in sorted(cand_scores.items(), key=lambda item: item[1], reverse=True)
    ]

    # Costruzione del ragionamento filologico strutturato (CoT di 1° livello)
    len_desc = (
        f"{min_chars} caratteri"
        if min_chars == max_chars
        else f"{min_chars}-{max_chars} caratteri"
    )
    if word_prefix or word_suffix:
        morph_desc = (
            f"Lacuna intra-parola di {len_desc} con struttura morfologica "
            f"'{word_prefix}[{'·' * min_chars}]{word_suffix}'."
        )
    else:
        morph_desc = f"Lacuna inter-parola di {len_desc}."

    attestation_desc = (
        f" Attestazioni lessicali riscontrate nell'indice dei corpora antichi: {', '.join(attested_forms_info)}."
        if attested_forms_info
        else f" Recuperati {len(retrieved_passages)} passi paralleli (loci similes) prioritari su {len(all_retrieved_passages)} trovati nei corpora TLG/MAAT."
    )
    top_supp_preview = (
        ", ".join(
            f"**{word_prefix}{c.supplement}{word_suffix}** (**{c.supplement}**)"
            if (word_prefix or word_suffix)
            else f"**{c.supplement}**"
            for c in valid_candidates[:3]
        )
        if valid_candidates
        else "nessun candidato di lunghezza esatta"
    )
    reasoning = (
        f"{morph_desc}{attestation_desc} "
        f"Sintesi filologica eseguita in modalità neuro-lessicale locale "
        f"(Retrieval Ibrido BM25+{bert_short_name} + HCB Beam Search{f' per fallback automatico da {fallback_reason}' if fallback_reason else ''}): "
        f"supplementi prioritari {top_supp_preview}."
    )

    return dspy.Prediction(
        philological_reasoning=reasoning,
        candidates=valid_candidates,
        fallback_candidates=fallback_candidates,
        retrieved_passages=retrieved_passages,
        all_retrieved_passages=all_retrieved_passages,
        gap_constraints=gap_constraints_str,
        min_chars=min_chars,
        max_chars=max_chars,
        word_prefix=word_prefix,
        word_suffix=word_suffix,
        index_only_supplements=sorted(index_only_norms),
    )


def fill_mask_rag(
    text: str,
    checkpoint: str = RAG_DEFAULT_LM,
    lm: dspy.BaseLM | None = None,
    backend: Literal["auto", "hf_api", "local", "openai_compatible", "neuro_lexical"] = RAG_DEFAULT_BACKEND,
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
    vocab_path: str | None = None,
    exclude_query_overlap: bool = False,
    enable_refinement: bool = True,
    return_details: bool = False,
    allow_local_fallback: bool | None = None,
    human_feedback: str = "",
    rejected_candidates: list[str] | None = None,
    session_id: str | None = None,
    model_id: str | None = None,
) -> list[tuple[str, float]] | tuple[list[tuple[str, float]], dict[str, Any]]:
    """
    Esegue il text-infilling su un testo greco contenente una lacuna `[...]` o `[---]`
    utilizzando il modulo DSPy `GreekRAGInfiller` o il motore neuro-lessicale locale (BERT + RAG).

    Args:
        text: Frammento in greco antico con almeno una lacuna `[...]` o `[---]`.
        checkpoint: Modello LLM di Hugging Face (es. `'Qwen/Qwen2.5-7B-Instruct'`,
                    `'meta-llama/Llama-3.1-8B-Instruct'`) oppure modello BERT greco
                    (es. `'CNR-ILC/gs-GreBerta'`, `'CNR-ILC/gs-aristoBERTo'`, `'CNR-ILC/gs-Logion'`).
        lm: Istanza opzionale `dspy.BaseLM` già configurata (es. per test con `DummyLM`).
        backend: Modalità di esecuzione (`'auto'`, `'hf_api'`, `'local'`, `'openai_compatible'`, `'neuro_lexical'`).
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
        use_vocab_filter: Se True, filtra le parole ricostruite tramite il dizionario greco TLG (`tlg.wl.gz` o `vocab_path`).
        vocab_path: Percorso opzionale a una wordlist greca personalizzata.
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

    # Se il modello selezionato a priori è un checkpoint BERT (o il backend è 'neuro_lexical')
    # e non è stato iniettato un LM mock esplicito, esegue direttamente la pipeline
    # neuro-lessicale locale (Retrieval Ibrido BM25+GreBerta + HCB Beam Search con il modello BERT scelto)
    # senza tentare la connessione a provider LLM esterni.
    is_explicit_neuro_lexical = (lm is None) and (
        backend == "neuro_lexical" or is_bert_checkpoint(checkpoint)
    )
    trace_recorder: ConversationTraceRecorder | None = None
    effective_backend = backend
    effective_model = str(getattr(lm, "model", None) or checkpoint)
    trace_status = "ok"

    if is_explicit_neuro_lexical:
        target_mlm_ckpt = (
            checkpoint if is_bert_checkpoint(checkpoint) else embedder_checkpoint
        )
        pred = _build_neuro_lexical_rag_prediction(
            context_with_gap=meta["context_with_gap"],
            min_chars=min_chars,
            max_chars=max_chars,
            word_prefix=word_prefix,
            word_suffix=word_suffix,
            num_candidates=K,
            retriever=active_retriever,
            top_k_passages=top_k_passages,
            exclude_query_overlap=exclude_query_overlap,
            embedder_checkpoint=target_mlm_ckpt,
            fallback_reason="",
        )
    else:
        # Di default abilita il fallback resiliente quando non è stato iniettato un LM mock esplicito
        should_fallback = (
            (lm is None) if allow_local_fallback is None else bool(allow_local_fallback)
        )

        active_lm = lm or dspy.settings.lm
        if active_lm is None:
            active_lm = build_dspy_lm(checkpoint=checkpoint, backend=backend)

        effective_model = str(getattr(active_lm, "model", None) or checkpoint)
        if is_conversation_logging_enabled():
            # Il recorder viene istanziato solo per le esecuzioni DSPy con LLM
            # (escludendo i run neuro_lexical a priori) quando il flag opt-in è attivo.
            trace_recorder = ConversationTraceRecorder(
                session_id=session_id,
                model_id=model_id,
                checkpoint=checkpoint,
                requested_backend=backend,
                input_payload={
                    "raw_text": text,
                    "context_with_gap": meta["context_with_gap"],
                    "min_chars": min_chars,
                    "max_chars": max_chars,
                    "word_prefix": word_prefix,
                    "word_suffix": word_suffix,
                    "K": K,
                    "top_k_passages": top_k_passages,
                    "compiled_program_path": str(compiled_program_path) if compiled_program_path else None,
                    "enable_refinement": enable_refinement,
                    "use_vocab_filter": use_vocab_filter,
                    "exclude_query_overlap": exclude_query_overlap,
                    "human_feedback": human_feedback or "",
                    "rejected_candidates": list(rejected_candidates or []),
                },
            )

        infiller = _load_or_create_infiller(
            retriever=active_retriever,
            top_k_passages=top_k_passages,
            compiled_program_path=compiled_program_path,
            enable_refinement=enable_refinement,
        )

        infill_kwargs = {
            "context_with_gap": meta["context_with_gap"],
            "min_chars": min_chars,
            "max_chars": max_chars,
            "word_prefix": word_prefix,
            "word_suffix": word_suffix,
            "num_candidates": K,
            "exclude_query_overlap": exclude_query_overlap,
            "human_feedback": human_feedback or "",
            "rejected_candidates": rejected_candidates,
        }

        try:
            lm_model_name = str(getattr(active_lm, "model", "") or "")
            if (
                should_fallback
                and lm_model_name.startswith("huggingface/")
                and is_hf_api_unavailable()
            ):
                skip_reason = "HF_TOKEN non dispone dei permessi per Hugging Face Inference Providers (circuit-breaker attivo)"
                log_llm_skipped_event(lm_model_name, skip_reason)
                if trace_recorder is not None:
                    trace_recorder.record_fallback(lm_model_name, "fallback_check", skip_reason, kind="skipped")
                raise RuntimeError(skip_reason)
            if should_fallback and lm_model_name.startswith(("ollama/", "ollama_chat/")):
                ram_ok, ram_reason = is_ollama_model_within_ram_budget(lm_model_name)
                if not ram_ok:
                    log_llm_skipped_event(lm_model_name, ram_reason)
                    if trace_recorder is not None:
                        trace_recorder.record_fallback(lm_model_name, "fallback_check", ram_reason, kind="skipped")
                    raise RuntimeError(ram_reason)

            ctx_mgr = (
                trace_recorder.make_dspy_context(active_lm, attempt="primary")
                if trace_recorder is not None
                else dspy.context(lm=active_lm)
            )
            with ctx_mgr:
                pred = infiller(**infill_kwargs)
        except Exception as primary_exc:
            if is_hf_permission_or_auth_error(primary_exc):
                mark_hf_api_unavailable()

            if not should_fallback:
                if trace_recorder is not None:
                    trace_recorder.finalize_and_write(
                        effective_backend=effective_backend,
                        effective_model=effective_model,
                        status="error",
                        error=f"{type(primary_exc).__name__}: {primary_exc}",
                    )
                raise

            # 1. Tenta eventuali runtime LLM alternativi disponibili in locale (es. Ollama compatibile con la RAM)
            pred = None
            for fb_lm in get_fallback_dspy_lms(checkpoint=checkpoint, backend=backend):
                try:
                    fb_model_name = str(getattr(fb_lm, "model", "") or "ollama")
                    log_llm_fallback_event(checkpoint, fb_model_name, str(primary_exc))
                    if trace_recorder is not None:
                        trace_recorder.record_fallback(
                            checkpoint,
                            fb_model_name,
                            str(primary_exc),
                            kind="fallback_ollama",
                        )
                        fb_ctx = trace_recorder.make_dspy_context(
                            fb_lm,
                            attempt=f"fallback_ollama:{fb_model_name}",
                        )
                    else:
                        fb_ctx = dspy.context(lm=fb_lm)
                    with fb_ctx:
                        pred = infiller(**infill_kwargs)
                    effective_model = fb_model_name
                    effective_backend = "ollama"
                    trace_status = "fallback_ollama"
                    break
                except Exception as fb_exc:
                    if trace_recorder is not None:
                        trace_recorder.record_fallback(
                            str(getattr(fb_lm, "model", "") or "ollama"),
                            "next_fallback",
                            f"{type(fb_exc).__name__}: {fb_exc}",
                            kind="fallback_error",
                        )
                    continue

            # 2. Se nessun LLM generativo esterno è raggiungibile/autorizzato, degrada al motore
            # neuro-lessicale locale (Retrieval Ibrido BM25+GreBerta + HCB Beam Search in-process)
            if pred is None:
                if is_hf_permission_or_auth_error(primary_exc):
                    reason_label = f"{checkpoint} [permessi HF_TOKEN Inference Providers assenti]"
                elif "richiede più RAM" in str(primary_exc):
                    reason_label = f"{checkpoint} [{primary_exc}]"
                else:
                    reason_label = f"{checkpoint} [runtime LLM non disponibile]"
                log_llm_fallback_event(
                    checkpoint,
                    f"neuro_lexical ({embedder_checkpoint})",
                    reason_label,
                )
                if trace_recorder is not None:
                    trace_recorder.record_fallback(
                        checkpoint,
                        f"neuro_lexical ({embedder_checkpoint})",
                        reason_label,
                        kind="fallback_neuro_lexical",
                    )
                effective_model = embedder_checkpoint
                effective_backend = "neuro_lexical"
                trace_status = "fallback_neuro_lexical"
                pred = _build_neuro_lexical_rag_prediction(
                    context_with_gap=meta["context_with_gap"],
                    min_chars=min_chars,
                    max_chars=max_chars,
                    word_prefix=word_prefix,
                    word_suffix=word_suffix,
                    num_candidates=K,
                    retriever=active_retriever,
                    top_k_passages=top_k_passages,
                    exclude_query_overlap=exclude_query_overlap,
                    embedder_checkpoint=embedder_checkpoint,
                    fallback_reason=reason_label,
                )

    vocab = get_normalized_vocab(vocab_path) if use_vocab_filter else None
    if not vocab:
        use_vocab_filter = False

    # Uniamo candidati validi (priorità True) e candidati fallback (priorità False),
    # preservando il rationale di secondo livello (CoT puntuale) fornito per ciascun supplemento.
    scored_pool: list[tuple[str, float, bool, str]] = []
    for idx, cand in enumerate(getattr(pred, "candidates", []) or []):
        base_len = len(strip_diacritics(cand.supplement))
        len_penalty = -0.05 * abs(base_len - center_chars) if center_chars > 0 else 0.0
        rank_Decay = -0.02 * idx
        score = float(cand.confidence) + len_penalty + rank_Decay
        cand_rationale = str(getattr(cand, "rationale", "") or "").strip()
        scored_pool.append((cand.supplement, score, True, cand_rationale))

    for idx, cand in enumerate(getattr(pred, "fallback_candidates", []) or []):
        base_len = len(strip_diacritics(cand.supplement))
        len_penalty = -0.15 * abs(base_len - center_chars) if center_chars > 0 else 0.0
        rank_decay = -0.03 * idx
        score = float(cand.confidence) + len_penalty + rank_decay - 0.5
        cand_rationale = str(getattr(cand, "rationale", "") or "").strip()
        scored_pool.append((cand.supplement, score, False, cand_rationale))

    scored_pool.sort(key=lambda x: (x[2], x[1]), reverse=True)

    # Costruisce l'insieme normalizzato dei candidati scartati dal filologo per escluderli
    # deterministicamente anche nel caso in cui intervenga il fallback neuro-lessicale locale.
    rejected_norm_set: set[str] = {
        normalize_greek(
            r.strip(),
            case_folding=case_folding,
            strip_diacritics_flag=strip_diacritics_flag,
        )
        for r in (rejected_candidates or [])
        if r and r.strip()
    }

    raw_index_only = getattr(pred, "index_only_supplements", None) or ()
    index_only_norm_set: set[str] = {
        re.sub(
            r"[\s\u200B-\u200D\uFEFF]",
            "",
            normalize_greek(
                str(s),
                case_folding=case_folding,
                strip_diacritics_flag=strip_diacritics_flag,
            ),
        ).strip()
        for s in raw_index_only
        if s
    }

    seen: set[str] = set()
    dedup_pool: list[tuple[str, float, bool]] = []
    raw_rationales_by_norm: dict[str, str] = {}

    for raw_supp, score, is_valid_len, raw_rationale in scored_pool:
        norm_key = normalize_greek(
            raw_supp,
            case_folding=case_folding,
            strip_diacritics_flag=strip_diacritics_flag,
        )
        norm_key = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", norm_key).strip()
        if not norm_key or norm_key in rejected_norm_set:
            continue

        if use_vocab_filter and vocab is not None:
            reconstructed = f"{word_prefix}{raw_supp}{word_suffix}"
            norm_recon = normalize_for_tlg_lookup(reconstructed)
            if norm_recon not in vocab:
                continue

        if raw_rationale and norm_key not in raw_rationales_by_norm:
            raw_rationales_by_norm[norm_key] = raw_rationale

        if norm_key not in seen:
            seen.add(norm_key)
            dedup_pool.append((norm_key, float(score), bool(is_valid_len)))

    retrieved_passages_list = list(getattr(pred, "retrieved_passages", []) or [])
    all_retrieved_passages_list = list(
        getattr(pred, "all_retrieved_passages", None) or retrieved_passages_list
    )
    raw_reasoning = str(getattr(pred, "philological_reasoning", "") or "").strip()

    # Calibriamo gli score in base alle attestazioni (TLG + Retriever) prima del taglio `[:K]`
    # e della softmax, così da promuovere i candidati doppiamente attestati nel ranking.
    passages_for_attestation = all_retrieved_passages_list or retrieved_passages_list
    has_retriever_context = active_retriever is not None or bool(passages_for_attestation)
    tlg_vocab = vocab if vocab is not None else get_normalized_vocab(vocab_path)

    precomputed_atts: dict[str, list[dict[str, Any]]] = {}
    precomputed_tlg: dict[str, dict[str, Any]] = {}
    calibrated_pool: list[tuple[str, float, bool]] = []

    for norm_key, base_score, is_valid_len in dedup_pool:
        atts = (
            resolve_candidate_attestations(
                supplement=norm_key,
                word_prefix=word_prefix,
                word_suffix=word_suffix,
                retrieved_passages=passages_for_attestation,
                retriever=active_retriever,
                max_attestations=3,
            )
            if has_retriever_context
            else []
        )

        # Intercetta l'attestazione TLG sia dal lookup lessicale sull'intero TLG sia dai documenti RAG recuperati
        lex_attested, recon_word = check_tlg_attestation(
            supplement=norm_key,
            word_prefix=word_prefix,
            word_suffix=word_suffix,
            vocab=tlg_vocab,
        )
        corpus_tlg_attested = any(
            str(a.get("corpus_id", "")).strip().lower().startswith("tlg")
            for a in atts
        )
        if (
            corpus_tlg_attested
            and atts
            and atts[0].get("attested_form")
            and normalize_for_tlg_lookup(str(atts[0]["attested_form"]))
            == normalize_for_tlg_lookup(recon_word)
        ):
            recon_word = str(atts[0]["attested_form"]).lower()

        is_tlg_attested = bool(lex_attested or corpus_tlg_attested)
        calibrated_score, att_tier = calibrate_candidate_attestation_score(
            base_score=base_score,
            is_tlg_attested=is_tlg_attested,
            attestations=atts,
            is_index_only=(norm_key in index_only_norm_set),
            apply_unattested_penalty=bool(tlg_vocab or has_retriever_context),
        )

        precomputed_atts[norm_key] = atts
        precomputed_tlg[norm_key] = {
            "tlg_attested": is_tlg_attested,
            "reconstructed_word": recon_word,
            "attestation_tier": att_tier,
        }
        calibrated_pool.append((norm_key, calibrated_score, is_valid_len))

    calibrated_pool.sort(key=lambda item: (item[2], item[1]), reverse=True)
    unique_candidates: list[tuple[str, float]] = [
        (norm_key, float(cal_score))
        for norm_key, cal_score, _is_valid_len in calibrated_pool[:K]
    ]

    if normalize_probs and unique_candidates:
        max_s = max(s for _, s in unique_candidates)
        temp = RAG_SOFTMAX_TEMPERATURE if RAG_SOFTMAX_TEMPERATURE > 0 else 1.0
        unnorm = [math.exp((s - max_s) / temp) for _, s in unique_candidates]
        total = sum(unnorm) or 1.0
        unique_candidates = [
            (item, float(p / total))
            for (item, _), p in zip(unique_candidates, unnorm)
        ]

    if return_details:
        # Risolve per ciascun suggerimento restituito:
        # 1) le attestazioni puntuali nella collezione di documenti (corpus) con la forma in grassetto (**...**);
        # 2) l'attestazione lessicale nell'intero TLG (riutilizzando i dati calcolati in pre-ranking);
        # 3) il ragionamento di 2° livello (CoT specifica per il singolo suggerimento).
        candidate_attestations: dict[str, list[dict[str, Any]]] = {}
        candidate_rationales: dict[str, str] = {}
        tlg_attestations: dict[str, dict[str, Any]] = {}

        for cand_supp, _cand_score in unique_candidates:
            atts = precomputed_atts.get(cand_supp, [])
            if atts:
                candidate_attestations[cand_supp] = atts

            tlg_info = precomputed_tlg.get(
                cand_supp,
                {
                    "tlg_attested": False,
                    "reconstructed_word": cand_supp.lower(),
                    "attestation_tier": 4,
                },
            )
            is_tlg_attested = bool(tlg_info.get("tlg_attested", False))
            tlg_attestations[cand_supp] = tlg_info
            tlg_attestations[cand_supp.lower()] = tlg_info

            base_rationale = raw_rationales_by_norm.get(cand_supp, "").strip()
            supp_len = len(strip_diacritics(cand_supp))
            reconstructed_word = f"{word_prefix}{cand_supp}{word_suffix}"
            if not base_rationale:
                if word_prefix or word_suffix:
                    base_rationale = (
                        f"Completa la parola greca **{reconstructed_word}** integrando "
                        f"la lacuna con **{cand_supp}** ({supp_len} {'lettera' if supp_len == 1 else 'lettere'})."
                    )
                else:
                    base_rationale = (
                        f"Supplemento **{cand_supp}** ({supp_len} {'lettera' if supp_len == 1 else 'lettere'}) "
                        f"plausibile per sintassi, morfologia e coerenza semantica col contesto."
                    )

            if atts:
                first_att = atts[0]
                loc_label = (
                    f"{first_att['corpus_id']} | {first_att['source_id']}"
                    if first_att.get("source_id")
                    else str(first_att.get("corpus_id", "corpus"))
                )
                att_form = first_att.get("attested_form") or reconstructed_word
                doc_freq = first_att.get("doc_freq", 1)
                if is_tlg_attested:
                    att_note = (
                        f"Forma **{att_form}** con doppia attestazione "
                        f"(lessico TLG e corpus [{loc_label}], "
                        f"{doc_freq} {'documento' if doc_freq == 1 else 'documenti'})."
                    )
                else:
                    att_note = (
                        f"Forma **{att_form}** attestata nel corpus [{loc_label}] "
                        f"({doc_freq} {'documento' if doc_freq == 1 else 'documenti'})."
                    )
                if loc_label not in base_rationale:
                    base_rationale = f"{base_rationale} {att_note}".strip()
                else:
                    if f"**{att_form}**" not in base_rationale and att_form in base_rationale:
                        base_rationale = base_rationale.replace(att_form, f"**{att_form}**", 1)
                    if is_tlg_attested and "doppia attestazione" not in base_rationale:
                        base_rationale = (
                            f"{base_rationale} Confermata anche dal lessico TLG (doppia attestazione)."
                        ).strip()

            candidate_rationales[cand_supp] = base_rationale

        # Se i suggerimenti principali sono attestati nel corpus, arricchisce anche la CoT di 1° livello
        # con il riferimento bibliografico esplicito e la forma attestata evidenziata in grassetto (**...**).
        if candidate_attestations:
            att_summaries: list[str] = []
            for cand_supp, _cand_score in unique_candidates[:3]:
                atts = candidate_attestations.get(cand_supp, [])
                if not atts:
                    continue
                first_att = atts[0]
                loc_label = (
                    f"{first_att['corpus_id']} | {first_att['source_id']}"
                    if first_att.get("source_id")
                    else str(first_att.get("corpus_id", "corpus"))
                )
                att_form = first_att.get("attested_form") or cand_supp
                att_summaries.append(
                    f"**{att_form}** (supplemento **{cand_supp}**) "
                    f"attestato in [{loc_label}] ({first_att.get('doc_freq', 1)} doc. nel corpus)"
                )
            if att_summaries:
                citation_note = f"Attestazioni nel corpus: {'; '.join(att_summaries)}."
                if "Attestazioni nel corpus:" not in raw_reasoning:
                    raw_reasoning = (
                        f"{raw_reasoning} {citation_note}".strip()
                        if raw_reasoning
                        else citation_note
                    )

        if trace_recorder is not None:
            trace_recorder.set_retrieval(retrieved_passages_list, all_retrieved_passages_list)
            trace_recorder.finalize_and_write(
                effective_backend=effective_backend,
                effective_model=effective_model,
                status=trace_status,
                candidates=unique_candidates,
                philological_reasoning=raw_reasoning,
            )

        details = {
            "philological_reasoning": raw_reasoning,
            "retrieved_passages": retrieved_passages_list,
            "all_retrieved_passages": all_retrieved_passages_list,
            "candidate_attestations": candidate_attestations,
            "candidate_rationales": candidate_rationales,
            "tlg_attestations": tlg_attestations,
            "gap_constraints": getattr(pred, "gap_constraints", ""),
            "min_chars": min_chars,
            "max_chars": max_chars,
            "word_prefix": word_prefix,
            "word_suffix": word_suffix,
        }
        return unique_candidates, details

    if trace_recorder is not None:
        trace_recorder.set_retrieval(retrieved_passages_list, all_retrieved_passages_list)
        trace_recorder.finalize_and_write(
            effective_backend=effective_backend,
            effective_model=effective_model,
            status=trace_status,
            candidates=unique_candidates,
            philological_reasoning=raw_reasoning,
        )

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
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Identificativo di sessione opzionale per correlare la traccia nei log JSONL",
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
        session_id=args.session_id,
    )

    print("\n=== Vincoli Papirologici Estratti ===")
    print(details["gap_constraints"])

    print(f"\n=== Passi Paralleli Recuperati ({len(details['retrieved_passages'])}) ===")
    for idx, p in enumerate(details["retrieved_passages"], 1):
        print(f"  [{idx}] {p[:180]}...")

    if details["philological_reasoning"]:
        print("\n=== Ragionamento Filologico (Chain-of-Thought) ===")
        print(details["philological_reasoning"])

    cand_atts = details.get("candidate_attestations") or {}
    print(f"\n=== Top-{len(preds)} Supplementi Proposti ===")
    for rank, (supp, score) in enumerate(preds, 1):
        atts = cand_atts.get(supp, [])
        if atts:
            loc = (
                f"{atts[0]['corpus_id']} | {atts[0]['source_id']}"
                if atts[0].get("source_id")
                else atts[0].get("corpus_id", "corpus")
            )
            att_str = f" | Attestato in [{loc}]: '{atts[0].get('attested_form', supp)}' ({atts[0].get('doc_freq', 1)} doc.)"
        else:
            att_str = ""
        print(f"  {rank:2d}. {supp:<15} (score: {score:.4f}){att_str}")


if __name__ == "__main__":
    main()
