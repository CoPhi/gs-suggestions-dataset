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
from models.bert.inference.predict import fill_mask
from models.bert.inference.vocab import (
    check_tlg_attestation,
    get_normalized_vocab,
    normalize_for_tlg_lookup,
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
_LOCAL_MLM_CACHE: dict[str, tuple[Any, Any]] = {}


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
    Carica o recupera dalla cache in-process il modello Masked Language Model greco
    (es. `CNR-ILC/gs-GreBerta`) utilizzato per il motore neuro-lessicale locale.
    """
    if checkpoint in _LOCAL_MLM_CACHE:
        return _LOCAL_MLM_CACHE[checkpoint]

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.eval()
    _LOCAL_MLM_CACHE[checkpoint] = (model, tokenizer)
    return model, tokenizer


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

    for att in passage_attestations:
        san = sanitize_greek_supplement(
            att.supplement,
            word_prefix=word_prefix,
            word_suffix=word_suffix,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if san:
            cand_scores[san] = max(cand_scores.get(san, 0.0), float(att.confidence))
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
            # Se la radice conservata (prefisso + suffisso) ha almeno 3 lettere, il vincolo
            # lessicale del corpus è altamente selettivo; per affissi brevi (1-2 lettere, es. -ν)
            # manteniamo il prior lessicale sotto 0.83 così da non sovrascrivere la predizione
            # contestuale Top-1 di GreBerta (0.84) a meno che entrambi i modelli concordino.
            if affix_specificity >= 3:
                freq_conf = min(0.95, 0.80 + 0.035 * math.log1p(doc_freq))
            else:
                freq_conf = min(0.82, 0.70 + 0.02 * math.log1p(doc_freq))
            cand_scores[san] = max(cand_scores.get(san, 0.0), freq_conf)
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
                if san in cand_scores:
                    # Bonus sinergico se il candidato è confermato sia dal corpus sia dal modello BERT
                    cand_scores[san] = min(0.99, max(cand_scores[san], mlm_conf) + 0.12)
                    if san in cand_rationales:
                        cand_rationales[san] = (
                            f"{cand_rationales[san]} Confermato anche dalla distribuzione contestuale {bert_short_name}."
                        )
                else:
                    cand_scores[san] = mlm_conf
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
                raise RuntimeError(
                    "HF_TOKEN non dispone dei permessi per Hugging Face Inference Providers"
                )
            if should_fallback and lm_model_name.startswith(("ollama/", "ollama_chat/")):
                ram_ok, ram_reason = is_ollama_model_within_ram_budget(lm_model_name)
                if not ram_ok:
                    raise RuntimeError(ram_reason)

            with dspy.context(lm=active_lm):
                pred = infiller(**infill_kwargs)
        except Exception as primary_exc:
            if is_hf_permission_or_auth_error(primary_exc):
                mark_hf_api_unavailable()

            if not should_fallback:
                raise

            # 1. Tenta eventuali runtime LLM alternativi disponibili in locale (es. Ollama compatibile con la RAM)
            pred = None
            for fb_lm in get_fallback_dspy_lms(checkpoint=checkpoint, backend=backend):
                try:
                    with dspy.context(lm=fb_lm):
                        pred = infiller(**infill_kwargs)
                    break
                except Exception:
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

    seen: set[str] = set()
    unique_candidates: list[tuple[str, float]] = []
    raw_rationales_by_norm: dict[str, str] = {}

    for raw_supp, score, _is_valid_len, raw_rationale in scored_pool:
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
        retrieved_passages_list = list(getattr(pred, "retrieved_passages", []) or [])
        all_retrieved_passages_list = list(
            getattr(pred, "all_retrieved_passages", None) or retrieved_passages_list
        )
        raw_reasoning = str(getattr(pred, "philological_reasoning", "") or "").strip()

        # Risolve per ciascun suggerimento restituito:
        # 1) le attestazioni puntuali nella collezione di documenti (corpus) con la forma in grassetto (**...**);
        # 2) l'attestazione lessicale nell'intero TLG (boundary reconstruction + lookup su tlg.wl.gz
        #    oppure intercettazione diretta delle attestazioni nel corpus TLG);
        # 3) il ragionamento di 2° livello (CoT specifica per il singolo suggerimento).
        candidate_attestations: dict[str, list[dict[str, Any]]] = {}
        candidate_rationales: dict[str, str] = {}
        tlg_attestations: dict[str, dict[str, Any]] = {}
        tlg_vocab = get_normalized_vocab(vocab_path)

        for cand_supp, _cand_score in unique_candidates:
            atts = resolve_candidate_attestations(
                supplement=cand_supp,
                word_prefix=word_prefix,
                word_suffix=word_suffix,
                retrieved_passages=all_retrieved_passages_list or retrieved_passages_list,
                retriever=active_retriever,
                max_attestations=3,
            )
            if atts:
                candidate_attestations[cand_supp] = atts

            # Intercetta l'attestazione TLG sia dal lookup lessicale sull'intero TLG sia dai documenti RAG recuperati
            lex_attested, recon_word = check_tlg_attestation(
                supplement=cand_supp,
                word_prefix=word_prefix,
                word_suffix=word_suffix,
                vocab=tlg_vocab,
            )
            corpus_tlg_attested = any(
                str(a.get("corpus_id", "")).strip().lower().startswith("tlg")
                for a in atts
            )
            if corpus_tlg_attested and atts and atts[0].get("attested_form"):
                recon_word = str(atts[0]["attested_form"]).lower()
            tlg_info = {
                "tlg_attested": bool(lex_attested or corpus_tlg_attested),
                "reconstructed_word": recon_word,
            }
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
                att_note = (
                    f"Forma **{att_form}** attestata nel corpus [{loc_label}] "
                    f"({doc_freq} {'documento' if doc_freq == 1 else 'documenti'})."
                )
                if loc_label not in base_rationale:
                    base_rationale = f"{base_rationale} {att_note}".strip()
                elif f"**{att_form}**" not in base_rationale and att_form in base_rationale:
                    base_rationale = base_rationale.replace(att_form, f"**{att_form}**", 1)

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
