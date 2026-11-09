"""
Modulo di Valutazione e Benchmark per il Sistema DSPy RAG (`GreekRAGInfiller`).

Valuta le prestazioni di text-infilling su casi di test stratificati (`default`, `word`, `suffix`)
calcolando:
- Top-1, Top-5, Top-10, Top-20 Exact Match Accuracy (%)
- Character Error Rate (CER) e Character Accuracy (1 - CER)
- Mean Reciprocal Rank (MRR)
- BERTScore F1 @K (opzionale)
Permette inoltre il confronto diretto tra Zero-Shot LLM (senza RAG) e RAG (compilato o base).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

import dspy
import numpy as np

from backend.config.settings import (
    RAG_DEFAULT_BACKEND,
    RAG_DEFAULT_EMBEDDER,
    RAG_DEFAULT_LM,
    RAG_INDEX_DIR,
    RAG_TOP_K_PASSAGES,
)
from backend.core.preprocess import get_expanded_supplement, normalize_greek
from models.bert.dataset.dev_set import DevCase, build_dev_cases
from models.bert.evaluation.metrics import (
    evaluate_bertscore_topk_text,
    evaluate_topK_text,
)
from models.rag.inference import fill_mask_rag
from models.rag.modules import extract_lacuna_metadata
from models.rag.optimizer import compute_cer
from models.rag.retriever import (
    GreekHybridRetriever,
    get_or_build_default_retriever,
    resolve_candidate_attestations,
)


def _strict_norm(text: str) -> str:
    norm = normalize_greek(text, case_folding="fold", strip_diacritics_flag=True)
    return re.sub(r"[\s\u200B-\u200D\uFEFF]", "", norm).strip()


def classify_case_gap_policy(context_with_gap: str) -> str:
    """
    Classifica la tipologia morfologica della lacuna target (`[...]`):
    - `'word'`: lacuna inter-parola (intera parola mancante tra spazi).
    - `'suffix'`: lacuna intra-parola finale (prefisso conservato a sinistra, suffisso vuoto).
    - `'default'`: lacuna intra-parola iniziale o centrale (suffisso o entrambi gli affissi presenti).
    """
    try:
        meta = extract_lacuna_metadata(context_with_gap)
    except Exception:
        return "default"

    if not meta["is_partial_word"]:
        return "word"
    if meta["word_prefix"] and not meta["word_suffix"]:
        return "suffix"
    return "default"


def load_cases_from_test_abs_csv(
    csv_path: str | Path = "test_abs.csv",
    max_cases: int | None = None,
    policy: str = "all",
    min_gap_length: int = 1,
    max_gap_length: int | None = None,
) -> list[DevCase]:
    """
    Carica i casi di valutazione papirologici direttamente dal file `test_abs.csv`,
    allineando `testo con supplemento` e `contesto ampio` (con fallback a `testo con supplemento`
    quando `contesto ampio` è assente o presenta varianti editoriali non sovrapponibili)
    e classificando ciascun caso secondo la policy morfologica (`default`, `word`, `suffix`).
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File CSV di valutazione non trovato: '{path}'.")

    df = pd.read_csv(path)
    cases: list[DevCase] = []

    for row_idx, row in df.iterrows():
        raw_id = row.get("ID")
        case_id = str(raw_id).strip() if pd.notna(raw_id) and str(raw_id).strip() else f"R_row_{row_idx + 1}"
        papyrus = str(row.get("papiro herc") or "PHerc.").strip()
        luogo = str(row.get("luogo") or "").strip() if pd.notna(row.get("luogo")) else ""
        suppl_text = str(row.get("testo con supplemento") or "").strip() if pd.notna(row.get("testo con supplemento")) else ""
        context = str(row.get("contesto ampio") or "").strip() if pd.notna(row.get("contesto ampio")) else ""

        if not suppl_text:
            continue

        # Allinea i supplementi tra quadre di `testo con supplemento` dentro `contesto ampio`
        aligned_text = context if context else suppl_text
        if context:
            supplements = list(re.finditer(r"\[([^\[\]]+)\]", suppl_text))
            for suppl in supplements:
                start, end = suppl.span()
                expand = get_expanded_supplement(suppl_text, start, end)
                expand_no_brackets = expand.replace("[", "").replace("]", "")
                matches = list(re.finditer(re.escape(expand_no_brackets), aligned_text))
                if matches:
                    aligned_text = aligned_text.replace(matches[0].group(0), expand, 1)

        ab = {
            "corpus_id": papyrus,
            "file_id": f"{case_id} | {luogo}" if luogo else case_id,
            "title": luogo,
            "training_text": aligned_text,
        }
        extracted = build_dev_cases(ab, normalize=False)
        # Fallback diretto su `testo con supplemento` se `contesto ampio` non ha recepito le parentesi quadre
        if not extracted and context and aligned_text != suppl_text:
            ab["training_text"] = suppl_text
            extracted = build_dev_cases(ab, normalize=False)

        for c in extracted:
            if c.gap_length < min_gap_length:
                continue
            if max_gap_length is not None and c.gap_length > max_gap_length:
                continue
            c.gap_type = classify_case_gap_policy(c.x)
            if policy not in ("all", c.gap_type):
                continue
            cases.append(c)
            if max_cases is not None and len(cases) >= max_cases:
                return cases

    return cases


def evaluate_rag_on_cases(
    cases: Sequence[DevCase | dict[str, Any]],
    checkpoint: str = RAG_DEFAULT_LM,
    lm: dspy.BaseLM | None = None,
    backend: Literal["auto", "hf_api", "local", "openai_compatible"] = RAG_DEFAULT_BACKEND,
    retriever: GreekHybridRetriever | None = None,
    top_k_passages: int = RAG_TOP_K_PASSAGES,
    compiled_program_path: str | Path | None = None,
    K: int = 20,
    compute_bertscore: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Valuta il modulo DSPy RAG su una sequenza di casi di test, aggregando le metriche
    per policy (`default`, `word`, `suffix`) e globalmente (`overall`).
    """
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "preds": [],
            "golds": [],
            "contexts": [],
            "cers": [],
            "mrrs": [],
        }
    )

    for case in cases:
        if isinstance(case, DevCase):
            x_text = case.x
            gold_raw = " ".join(case.y) if isinstance(case.y, list) else str(case.y)
            gap_len = case.gap_length
            policy = getattr(case, "gap_type", "default") or "default"
        else:
            x_text = str(case.get("x") or case.get("text") or case.get("context_with_gap") or "")
            raw_y = case.get("y") or case.get("gold_target") or case.get("gold_supplement") or ""
            gold_raw = " ".join(raw_y) if isinstance(raw_y, list) else str(raw_y)
            gap_len = case.get("gap_length")
            policy = str(case.get("policy") or case.get("gap_type") or "default")

        if not x_text or not gold_raw:
            continue

        gold_norm = _strict_norm(gold_raw)
        if not gold_norm:
            continue

        try:
            preds = fill_mask_rag(
                text=x_text,
                checkpoint=checkpoint,
                lm=lm,
                backend=backend,
                retriever=retriever,
                use_default_retriever=retriever is not None or top_k_passages > 0,
                top_k_passages=top_k_passages,
                compiled_program_path=compiled_program_path,
                n_chars=gap_len,
                K=K,
                exclude_query_overlap=True,
                return_details=False,
            )
        except Exception:
            preds = []

        top1_str = preds[0][0] if preds else ""
        cer_val = compute_cer(top1_str, gold_norm) if top1_str else 1.0

        rr = 0.0
        for rank_idx, (cand_str, _score) in enumerate(preds, 1):
            if _strict_norm(cand_str) == gold_norm:
                rr = 1.0 / rank_idx
                break

        for bucket in (policy, "overall"):
            grouped[bucket]["preds"].append(preds)
            grouped[bucket]["golds"].append(gold_norm)
            grouped[bucket]["contexts"].append(x_text)
            grouped[bucket]["cers"].append(cer_val)
            grouped[bucket]["mrrs"].append(rr)

    metrics_by_policy: dict[str, dict[str, float]] = {}
    for pol, data in grouped.items():
        if any(len(p) > 0 for p in data["preds"]):
            topk_res = evaluate_topK_text(data["preds"], data["golds"])
        else:
            topk_res = {"top1": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0}
        mean_cer = float(np.mean(data["cers"])) if data["cers"] else 1.0
        mean_mrr = float(np.mean(data["mrrs"])) if data["mrrs"] else 0.0

        pol_metrics: dict[str, float] = {
            "num_cases": float(len(data["golds"])),
            "top1": float(topk_res.get("top1", 0.0)),
            "top5": float(topk_res.get("top5", 0.0)),
            "top10": float(topk_res.get("top10", 0.0)),
            "top20": float(topk_res.get("top20", 0.0)),
            "cer": mean_cer,
            "char_accuracy": max(0.0, (1.0 - mean_cer) * 100.0),
            "mrr": mean_mrr * 100.0,
        }

        if compute_bertscore and data["preds"]:
            bs = evaluate_bertscore_topk_text(
                data["preds"],
                data["golds"],
                contexts=data["contexts"],
                k_values=[1, 5, 10, 20],
            )
            pol_metrics.update(bs)

        metrics_by_policy[pol] = pol_metrics

    return metrics_by_policy


def _compute_bucket_metrics(
    preds_list: list[list[tuple[str, float]]],
    golds_list: list[str],
    cers_list: list[float],
    mrrs_list: list[float],
) -> dict[str, float]:
    """Calcola le metriche aggregate Top-K EM, CER, Char Accuracy e MRR per un bucket di casi."""
    if not golds_list:
        return {
            "num_cases": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "top20": 0.0,
            "cer": 1.0,
            "char_accuracy": 0.0,
            "mrr": 0.0,
        }
    if any(len(p) > 0 for p in preds_list):
        topk_res = evaluate_topK_text(preds_list, golds_list)
    else:
        topk_res = {"top1": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0}
    mean_cer = float(np.mean(cers_list)) if cers_list else 1.0
    mean_mrr = float(np.mean(mrrs_list)) if mrrs_list else 0.0
    return {
        "num_cases": float(len(golds_list)),
        "top1": float(topk_res.get("top1", 0.0)),
        "top5": float(topk_res.get("top5", 0.0)),
        "top10": float(topk_res.get("top10", 0.0)),
        "top20": float(topk_res.get("top20", 0.0)),
        "cer": mean_cer,
        "char_accuracy": max(0.0, (1.0 - mean_cer) * 100.0),
        "mrr": mean_mrr * 100.0,
    }


def _compute_delta_metrics(
    no_rag_m: dict[str, float],
    with_rag_m: dict[str, float],
) -> dict[str, float]:
    """Calcola il delta (With-RAG meno No-RAG) per ciascuna metrica numerica."""
    delta: dict[str, float] = {"num_cases": float(with_rag_m.get("num_cases", 0.0))}
    for k in ("top1", "top5", "top10", "top20", "cer", "char_accuracy", "mrr"):
        if k in no_rag_m or k in with_rag_m:
            delta[k] = round(float(with_rag_m.get(k, 0.0)) - float(no_rag_m.get(k, 0.0)), 4)
    return delta


def evaluate_rag_impact_on_test_abs(
    cases: Sequence[DevCase | dict[str, Any]] | None = None,
    csv_path: str | Path = "test_abs.csv",
    checkpoint: str = RAG_DEFAULT_LM,
    lm: dspy.BaseLM | None = None,
    backend: Literal["auto", "hf_api", "local", "openai_compatible"] = RAG_DEFAULT_BACKEND,
    retriever: GreekHybridRetriever | None = None,
    top_k_passages: int = RAG_TOP_K_PASSAGES,
    compiled_program_path: str | Path | None = None,
    K: int = 20,
    max_cases: int | None = None,
    policy: str = "all",
) -> dict[str, Any]:
    """
    Valuta su `test_abs.csv` l'impatto dell'estensione delle capacità dell'LLM tramite
    l'aggiunta di documenti rilevanti (RAG), confrontando direttamente:
    1. **Baseline Zero-Shot (`no_rag`)**: inferenza senza documenti recuperati (`top_k_passages=0`, `retriever=None`).
    2. **Sistema Esteso con Corpus (`with_rag`)**: inferenza assistita da `GreekHybridRetriever`
       e attestazioni nella collezione di documenti definita.

    Restituisce un report completo con:
    - `no_rag`, `with_rag` e `delta` per policy (`overall`, `default`, `word`, `suffix`);
    - `by_papyrus`: disaggregazione per papiro ercolanese (`PHerc. 1004`, `PHerc. 1021`, `PHerc. 1020`, ecc.);
    - `corpus_attestation_analysis`: tasso di copertura del gold nel corpus e performance
      condizionata sui casi attestati nel corpus vs non attestati;
    - `case_records`: tracciato per singolo caso con CoT (`philological_reasoning`) e
      attestazioni puntuali nella collezione di documenti (`attestations`).
    """
    eval_cases = (
        list(cases)
        if cases is not None
        else load_cases_from_test_abs_csv(csv_path=csv_path, max_cases=max_cases, policy=policy)
    )

    active_retriever = retriever
    if active_retriever is None and top_k_passages > 0:
        active_retriever = get_or_build_default_retriever(
            index_dir=RAG_INDEX_DIR,
            embedder_checkpoint=RAG_DEFAULT_EMBEDDER,
        )

    policy_buckets_no_rag: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"preds": [], "golds": [], "cers": [], "mrrs": []}
    )
    policy_buckets_with_rag: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"preds": [], "golds": [], "cers": [], "mrrs": []}
    )
    papyrus_buckets_no_rag: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"preds": [], "golds": [], "cers": [], "mrrs": []}
    )
    papyrus_buckets_with_rag: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"preds": [], "golds": [], "cers": [], "mrrs": []}
    )
    attestation_buckets: dict[str, dict[str, dict[str, list[Any]]]] = {
        "attested": {
            "no_rag": {"preds": [], "golds": [], "cers": [], "mrrs": []},
            "with_rag": {"preds": [], "golds": [], "cers": [], "mrrs": []},
        },
        "unattested": {
            "no_rag": {"preds": [], "golds": [], "cers": [], "mrrs": []},
            "with_rag": {"preds": [], "golds": [], "cers": [], "mrrs": []},
        },
    }

    case_records: list[dict[str, Any]] = []
    gold_attested_count = 0

    for idx, case in enumerate(eval_cases):
        if isinstance(case, DevCase):
            x_text = case.x
            gold_raw = " ".join(case.y) if isinstance(case.y, list) else str(case.y)
            gap_len = case.gap_length
            pol = getattr(case, "gap_type", "") or classify_case_gap_policy(x_text)
            papyrus = case.corpus_id or "PHerc."
            file_id = case.file_id or f"case_{idx + 1}"
        else:
            x_text = str(case.get("x") or case.get("text") or case.get("context_with_gap") or "")
            raw_y = case.get("y") or case.get("gold_target") or case.get("gold_supplement") or ""
            gold_raw = " ".join(raw_y) if isinstance(raw_y, list) else str(raw_y)
            gap_len = case.get("gap_length")
            pol = str(case.get("policy") or case.get("gap_type") or classify_case_gap_policy(x_text))
            papyrus = str(case.get("corpus_id") or "PHerc.")
            file_id = str(case.get("file_id") or f"case_{idx + 1}")

        if not x_text or not gold_raw:
            continue
        gold_norm = _strict_norm(gold_raw)
        if not gold_norm:
            continue

        # 1. Valutazione Baseline Zero-Shot (senza RAG / senza documenti del corpus)
        try:
            preds_no_rag = fill_mask_rag(
                text=x_text,
                checkpoint=checkpoint,
                lm=lm,
                backend=backend,
                retriever=None,
                use_default_retriever=False,
                top_k_passages=0,
                compiled_program_path=compiled_program_path,
                n_chars=gap_len,
                K=K,
                exclude_query_overlap=True,
                return_details=False,
            )
        except Exception:
            preds_no_rag = []

        # 2. Valutazione RAG Esteso con Documenti Rilevanti e Attestazioni del Corpus (con CoT)
        try:
            rag_out = fill_mask_rag(
                text=x_text,
                checkpoint=checkpoint,
                lm=lm,
                backend=backend,
                retriever=active_retriever,
                use_default_retriever=active_retriever is not None or top_k_passages > 0,
                top_k_passages=top_k_passages,
                compiled_program_path=compiled_program_path,
                n_chars=gap_len,
                K=K,
                exclude_query_overlap=True,
                return_details=True,
            )
            if isinstance(rag_out, tuple) and len(rag_out) == 2:
                preds_with_rag, rag_details = rag_out
            else:
                preds_with_rag, rag_details = rag_out, {}
        except Exception:
            preds_with_rag, rag_details = [], {}

        top1_no_rag = preds_no_rag[0][0] if preds_no_rag else ""
        cer_no_rag = compute_cer(top1_no_rag, gold_norm) if top1_no_rag else 1.0
        rr_no_rag = 0.0
        for r_i, (c_str, _) in enumerate(preds_no_rag, 1):
            if _strict_norm(c_str) == gold_norm:
                rr_no_rag = 1.0 / r_i
                break

        top1_with_rag = preds_with_rag[0][0] if preds_with_rag else ""
        cer_with_rag = compute_cer(top1_with_rag, gold_norm) if top1_with_rag else 1.0
        rr_with_rag = 0.0
        for r_i, (c_str, _) in enumerate(preds_with_rag, 1):
            if _strict_norm(c_str) == gold_norm:
                rr_with_rag = 1.0 / r_i
                break

        # Verifica se il supplemento gold è attestato nella collezione di documenti (corpus)
        w_pref = str(rag_details.get("word_prefix") or "")
        w_suff = str(rag_details.get("word_suffix") or "")
        retrieved_passages = list(rag_details.get("retrieved_passages") or [])
        gold_atts = resolve_candidate_attestations(
            supplement=gold_norm,
            word_prefix=w_pref,
            word_suffix=w_suff,
            retrieved_passages=retrieved_passages,
            retriever=active_retriever,
            max_attestations=3,
        )
        is_gold_attested = len(gold_atts) > 0
        if is_gold_attested:
            gold_attested_count += 1

        att_key = "attested" if is_gold_attested else "unattested"
        for mode_key, p_list, c_val, r_val in (
            ("no_rag", preds_no_rag, cer_no_rag, rr_no_rag),
            ("with_rag", preds_with_rag, cer_with_rag, rr_with_rag),
        ):
            attestation_buckets[att_key][mode_key]["preds"].append(p_list)
            attestation_buckets[att_key][mode_key]["golds"].append(gold_norm)
            attestation_buckets[att_key][mode_key]["cers"].append(c_val)
            attestation_buckets[att_key][mode_key]["mrrs"].append(r_val)

        for bucket in (pol, "overall"):
            policy_buckets_no_rag[bucket]["preds"].append(preds_no_rag)
            policy_buckets_no_rag[bucket]["golds"].append(gold_norm)
            policy_buckets_no_rag[bucket]["cers"].append(cer_no_rag)
            policy_buckets_no_rag[bucket]["mrrs"].append(rr_no_rag)

            policy_buckets_with_rag[bucket]["preds"].append(preds_with_rag)
            policy_buckets_with_rag[bucket]["golds"].append(gold_norm)
            policy_buckets_with_rag[bucket]["cers"].append(cer_with_rag)
            policy_buckets_with_rag[bucket]["mrrs"].append(rr_with_rag)

        papyrus_buckets_no_rag[papyrus]["preds"].append(preds_no_rag)
        papyrus_buckets_no_rag[papyrus]["golds"].append(gold_norm)
        papyrus_buckets_no_rag[papyrus]["cers"].append(cer_no_rag)
        papyrus_buckets_no_rag[papyrus]["mrrs"].append(rr_no_rag)

        papyrus_buckets_with_rag[papyrus]["preds"].append(preds_with_rag)
        papyrus_buckets_with_rag[papyrus]["golds"].append(gold_norm)
        papyrus_buckets_with_rag[papyrus]["cers"].append(cer_with_rag)
        papyrus_buckets_with_rag[papyrus]["mrrs"].append(rr_with_rag)

        cand_atts_map = rag_details.get("candidate_attestations") or {}
        top1_atts = cand_atts_map.get(top1_with_rag, [])
        case_records.append(
            {
                "case_id": file_id,
                "papyrus": papyrus,
                "policy": pol,
                "context_with_gap": x_text,
                "word_prefix": w_pref,
                "word_suffix": w_suff,
                "gap_length": gap_len,
                "gold": gold_norm,
                "no_rag_top1": top1_no_rag,
                "no_rag_top5": [c for c, _ in preds_no_rag[:5]],
                "no_rag_hit_top1": _strict_norm(top1_no_rag) == gold_norm if top1_no_rag else False,
                "no_rag_hit_top5": any(_strict_norm(c) == gold_norm for c, _ in preds_no_rag[:5]),
                "with_rag_top1": top1_with_rag,
                "with_rag_top5": [c for c, _ in preds_with_rag[:5]],
                "with_rag_hit_top1": _strict_norm(top1_with_rag) == gold_norm if top1_with_rag else False,
                "with_rag_hit_top5": any(_strict_norm(c) == gold_norm for c, _ in preds_with_rag[:5]),
                "gold_attested_in_corpus": is_gold_attested,
                "gold_attestations": gold_atts,
                "top1_attestations": top1_atts,
                "philological_reasoning": str(rag_details.get("philological_reasoning") or ""),
                "retrieved_passages": retrieved_passages,
            }
        )

    no_rag_metrics: dict[str, dict[str, float]] = {}
    with_rag_metrics: dict[str, dict[str, float]] = {}
    delta_metrics: dict[str, dict[str, float]] = {}

    for pol_key in policy_buckets_with_rag.keys():
        m_no = _compute_bucket_metrics(
            policy_buckets_no_rag[pol_key]["preds"],
            policy_buckets_no_rag[pol_key]["golds"],
            policy_buckets_no_rag[pol_key]["cers"],
            policy_buckets_no_rag[pol_key]["mrrs"],
        )
        m_with = _compute_bucket_metrics(
            policy_buckets_with_rag[pol_key]["preds"],
            policy_buckets_with_rag[pol_key]["golds"],
            policy_buckets_with_rag[pol_key]["cers"],
            policy_buckets_with_rag[pol_key]["mrrs"],
        )
        no_rag_metrics[pol_key] = m_no
        with_rag_metrics[pol_key] = m_with
        delta_metrics[pol_key] = _compute_delta_metrics(m_no, m_with)

    by_papyrus: dict[str, dict[str, dict[str, float]]] = {}
    for pap_key in sorted(papyrus_buckets_with_rag.keys()):
        p_no = _compute_bucket_metrics(
            papyrus_buckets_no_rag[pap_key]["preds"],
            papyrus_buckets_no_rag[pap_key]["golds"],
            papyrus_buckets_no_rag[pap_key]["cers"],
            papyrus_buckets_no_rag[pap_key]["mrrs"],
        )
        p_with = _compute_bucket_metrics(
            papyrus_buckets_with_rag[pap_key]["preds"],
            papyrus_buckets_with_rag[pap_key]["golds"],
            papyrus_buckets_with_rag[pap_key]["cers"],
            papyrus_buckets_with_rag[pap_key]["mrrs"],
        )
        by_papyrus[pap_key] = {
            "no_rag": p_no,
            "with_rag": p_with,
            "delta": _compute_delta_metrics(p_no, p_with),
        }

    att_no = _compute_bucket_metrics(
        attestation_buckets["attested"]["no_rag"]["preds"],
        attestation_buckets["attested"]["no_rag"]["golds"],
        attestation_buckets["attested"]["no_rag"]["cers"],
        attestation_buckets["attested"]["no_rag"]["mrrs"],
    )
    att_with = _compute_bucket_metrics(
        attestation_buckets["attested"]["with_rag"]["preds"],
        attestation_buckets["attested"]["with_rag"]["golds"],
        attestation_buckets["attested"]["with_rag"]["cers"],
        attestation_buckets["attested"]["with_rag"]["mrrs"],
    )
    unatt_no = _compute_bucket_metrics(
        attestation_buckets["unattested"]["no_rag"]["preds"],
        attestation_buckets["unattested"]["no_rag"]["golds"],
        attestation_buckets["unattested"]["no_rag"]["cers"],
        attestation_buckets["unattested"]["no_rag"]["mrrs"],
    )
    unatt_with = _compute_bucket_metrics(
        attestation_buckets["unattested"]["with_rag"]["preds"],
        attestation_buckets["unattested"]["with_rag"]["golds"],
        attestation_buckets["unattested"]["with_rag"]["cers"],
        attestation_buckets["unattested"]["with_rag"]["mrrs"],
    )

    total_evaluated = len(case_records)
    return {
        "model": checkpoint,
        "dataset": str(csv_path),
        "top_k_passages": top_k_passages,
        "corpus_size": len(active_retriever.corpus) if active_retriever is not None else 0,
        "num_cases": total_evaluated,
        "no_rag": no_rag_metrics,
        "with_rag": with_rag_metrics,
        "delta": delta_metrics,
        "by_papyrus": by_papyrus,
        "corpus_attestation_analysis": {
            "gold_attested_count": gold_attested_count,
            "gold_attested_rate": round((gold_attested_count / total_evaluated) * 100.0, 2)
            if total_evaluated > 0
            else 0.0,
            "attested": {
                "no_rag": att_no,
                "with_rag": att_with,
                "delta": _compute_delta_metrics(att_no, att_with),
            },
            "unattested": {
                "no_rag": unatt_no,
                "with_rag": unatt_with,
                "delta": _compute_delta_metrics(unatt_no, unatt_with),
            },
        },
        "case_records": case_records,
    }


def format_rag_evaluation_markdown(
    results: dict[str, dict[str, float]],
    model_name: str = RAG_DEFAULT_LM,
) -> str:
    """Formatta i risultati di valutazione RAG in una tabella Markdown."""
    lines = [
        f"### Risultati Valutazione DSPy RAG (`{model_name}`)",
        "",
        "| Policy | Casi | Top-1 EM (%) | Top-5 EM (%) | Top-10 EM (%) | Top-20 EM (%) | Char Acc (%) | CER | MRR (%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pol in ("overall", "default", "word", "suffix"):
        if pol not in results:
            continue
        m = results[pol]
        lines.append(
            f"| **{pol}** | {int(m.get('num_cases', 0))} "
            f"| {m.get('top1', 0.0):.2f}% "
            f"| {m.get('top5', 0.0):.2f}% "
            f"| {m.get('top10', 0.0):.2f}% "
            f"| {m.get('top20', 0.0):.2f}% "
            f"| {m.get('char_accuracy', 0.0):.2f}% "
            f"| {m.get('cer', 1.0):.4f} "
            f"| {m.get('mrr', 0.0):.2f}% |"
        )
    return "\n".join(lines)


def format_rag_impact_markdown(
    report: dict[str, Any],
    model_name: str | None = None,
    max_qualitative_examples: int = 10,
) -> str:
    """
    Formatta in Markdown il report comparativo sull'impatto del RAG (`No-RAG` vs `With-RAG`)
    su `test_abs.csv`, includendo l'analisi delle attestazioni nel corpus, la disaggregazione
    per papiro ercolanese e gli esempi di Chain-of-Thought con citazione puntuale del corpus.
    """
    mdl = model_name or report.get("model", RAG_DEFAULT_LM)
    top_k = report.get("top_k_passages", RAG_TOP_K_PASSAGES)
    corpus_sz = report.get("corpus_size", 0)
    total_n = report.get("num_cases", 0)

    no_rag = report.get("no_rag", {})
    with_rag = report.get("with_rag", {})
    delta = report.get("delta", {})

    lines = [
        f"# Valutazione Impatto RAG su `{report.get('dataset', 'test_abs.csv')}` (`{mdl}`)",
        "",
        f"- **Casi papirologici valutati**: `{total_n}`",
        f"- **Documenti indicizzati nel Corpus RAG**: `{corpus_sz}` passaggi (`top_k={top_k}` *loci similes* per query)",
        "",
        "## 1. Confronto Zero-Shot (Senza RAG) vs RAG (Con Documenti Rilevanti)",
        "",
        "| Policy | Configurazione | Casi | Top-1 EM (%) | Top-5 EM (%) | Top-10 EM (%) | Top-20 EM (%) | Char Acc (%) | CER | MRR (%) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for pol in ("overall", "default", "word", "suffix"):
        if pol not in with_rag:
            continue
        m_no = no_rag.get(pol, {})
        m_with = with_rag.get(pol, {})
        m_d = delta.get(pol, {})
        n_c = int(m_with.get("num_cases", 0))
        lines.append(
            f"| **{pol}** | No-RAG ($k=0$) | {n_c} "
            f"| {m_no.get('top1', 0.0):.2f}% | {m_no.get('top5', 0.0):.2f}% "
            f"| {m_no.get('top10', 0.0):.2f}% | {m_no.get('top20', 0.0):.2f}% "
            f"| {m_no.get('char_accuracy', 0.0):.2f}% | {m_no.get('cer', 1.0):.4f} | {m_no.get('mrr', 0.0):.2f}% |"
        )
        lines.append(
            f"| **{pol}** | **With-RAG ($k={top_k}$)** | {n_c} "
            f"| **{m_with.get('top1', 0.0):.2f}%** | **{m_with.get('top5', 0.0):.2f}%** "
            f"| **{m_with.get('top10', 0.0):.2f}%** | **{m_with.get('top20', 0.0):.2f}%** "
            f"| **{m_with.get('char_accuracy', 0.0):.2f}%** | **{m_with.get('cer', 1.0):.4f}** | **{m_with.get('mrr', 0.0):.2f}%** |"
        )
        lines.append(
            f"| **{pol}** | *Delta ($\\Delta$)* | {n_c} "
            f"| {m_d.get('top1', 0.0):+.2f}% | {m_d.get('top5', 0.0):+.2f}% "
            f"| {m_d.get('top10', 0.0):+.2f}% | {m_d.get('top20', 0.0):+.2f}% "
            f"| {m_d.get('char_accuracy', 0.0):+.2f}% | {m_d.get('cer', 0.0):+.4f} | {m_d.get('mrr', 0.0):+.2f}% |"
        )

    att_analysis = report.get("corpus_attestation_analysis", {})
    if att_analysis:
        att_count = att_analysis.get("gold_attested_count", 0)
        att_rate = att_analysis.get("gold_attested_rate", 0.0)
        lines.extend(
            [
                "",
                "## 2. Impatto della Copertura del Corpus (Casi Attestati vs Non Attestati nel Corpus)",
                "",
                f"- **Copertura lessicale del Gold nel Corpus**: `{att_count}/{total_n}` casi (**{att_rate:.2f}%**)",
                "",
                "| Sottoinsieme Casi | Configurazione | Casi | Top-1 EM (%) | Top-5 EM (%) | Top-20 EM (%) | Char Acc (%) | MRR (%) |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for subset_key, subset_label in (
            ("attested", "Gold Attestato nel Corpus"),
            ("unattested", "Gold Non Attestato nel Corpus"),
        ):
            sub_data = att_analysis.get(subset_key, {})
            s_no = sub_data.get("no_rag", {})
            s_with = sub_data.get("with_rag", {})
            s_d = sub_data.get("delta", {})
            s_n = int(s_with.get("num_cases", 0))
            if s_n == 0:
                continue
            lines.append(
                f"| **{subset_label}** | No-RAG ($k=0$) | {s_n} "
                f"| {s_no.get('top1', 0.0):.2f}% | {s_no.get('top5', 0.0):.2f}% "
                f"| {s_no.get('top20', 0.0):.2f}% | {s_no.get('char_accuracy', 0.0):.2f}% | {s_no.get('mrr', 0.0):.2f}% |"
            )
            lines.append(
                f"| **{subset_label}** | **With-RAG ($k={top_k}$)** | {s_n} "
                f"| **{s_with.get('top1', 0.0):.2f}%** | **{s_with.get('top5', 0.0):.2f}%** "
                f"| **{s_with.get('top20', 0.0):.2f}%** | **{s_with.get('char_accuracy', 0.0):.2f}%** | **{s_with.get('mrr', 0.0):.2f}%** |"
            )
            lines.append(
                f"| **{subset_label}** | *Delta ($\\Delta$)* | {s_n} "
                f"| {s_d.get('top1', 0.0):+.2f}% | {s_d.get('top5', 0.0):+.2f}% "
                f"| {s_d.get('top20', 0.0):+.2f}% | {s_d.get('char_accuracy', 0.0):+.2f}% | {s_d.get('mrr', 0.0):+.2f}% |"
            )

    by_papyrus = report.get("by_papyrus", {})
    if by_papyrus:
        lines.extend(
            [
                "",
                "## 3. Performance per Papiro Ercolanese (`test_abs.csv`)",
                "",
                "| Papiro | Casi | No-RAG Top-1 (%) | With-RAG Top-1 (%) | $\\Delta$ Top-1 (%) | No-RAG Top-5 (%) | With-RAG Top-5 (%) | $\\Delta$ Top-5 (%) | With-RAG MRR (%) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for pap, p_data in by_papyrus.items():
            p_no = p_data.get("no_rag", {})
            p_with = p_data.get("with_rag", {})
            p_d = p_data.get("delta", {})
            p_n = int(p_with.get("num_cases", 0))
            lines.append(
                f"| **{pap}** | {p_n} "
                f"| {p_no.get('top1', 0.0):.2f}% | **{p_with.get('top1', 0.0):.2f}%** | {p_d.get('top1', 0.0):+.2f}% "
                f"| {p_no.get('top5', 0.0):.2f}% | **{p_with.get('top5', 0.0):.2f}%** | {p_d.get('top5', 0.0):+.2f}% "
                f"| {p_with.get('mrr', 0.0):.2f}% |"
            )

    case_records = report.get("case_records") or []
    # Seleziona prioritariamente i casi in cui il RAG migliora la predizione grazie alle attestazioni nel corpus
    improved_examples = [
        r
        for r in case_records
        if r.get("with_rag_hit_top1") and not r.get("no_rag_hit_top1") and r.get("top1_attestations")
    ]
    if len(improved_examples) < max_qualitative_examples:
        for r in case_records:
            if r not in improved_examples and r.get("with_rag_hit_top1") and r.get("top1_attestations"):
                improved_examples.append(r)
            if len(improved_examples) >= max_qualitative_examples:
                break

    if improved_examples:
        lines.extend(
            [
                "",
                "## 4. Consultazione Chain-of-Thought (CoT) e Attestazioni nel Corpus (Esempi Qualitativi)",
                "",
                "| Caso (`ID | Luogo`) | Papiro | Gold | No-RAG Top-1 | With-RAG Top-1 | Attestazione nel Corpus (`[Corpus | Documento]`) | Ragionamento Filologico (CoT) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for ex in improved_examples[:max_qualitative_examples]:
            atts = ex.get("top1_attestations") or ex.get("gold_attestations") or []
            if atts:
                a0 = atts[0]
                loc = f"{a0['corpus_id']} | {a0['source_id']}" if a0.get("source_id") else a0.get("corpus_id", "corpus")
                att_summary = f"`[{loc}]` forma **{a0.get('attested_form', ex['with_rag_top1'])}** ({a0.get('doc_freq', 1)} doc.)"
            else:
                att_summary = "—"
            cot_short = (ex.get("philological_reasoning") or "").replace("\n", " ").replace("|", "/")
            lines.append(
                f"| `{ex['case_id']}` | {ex['papyrus']} | **{ex['gold']}** | `{ex['no_rag_top1'] or '—'}` "
                f"| **{ex['with_rag_top1']}** | {att_summary} | {cot_short} |"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valutazione del sistema DSPy RAG su test_abs.csv e dataset di benchmark papirologici."
    )
    parser.add_argument(
        "--lm",
        type=str,
        default=RAG_DEFAULT_LM,
        help=f"Modello LLM Hugging Face da valutare (default: {RAG_DEFAULT_LM})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=RAG_DEFAULT_BACKEND,
        choices=["auto", "hf_api", "local", "openai_compatible"],
        help="Backend di esecuzione dell'LLM",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Percorso al file CSV papirologico da valutare direttamente (es. test_abs.csv)",
    )
    parser.add_argument(
        "--compare-no-rag",
        action="store_true",
        help="Confronta direttamente Zero-Shot senza RAG (k=0) vs RAG con documenti rilevanti sul dataset selezionato",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="CNR-ILC/gs-dataset-eval",
        help="Dataset Hugging Face di valutazione (default: CNR-ILC/gs-dataset-eval)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split da valutare (default: test)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="all",
        choices=["default", "word", "suffix", "all"],
        help="Policy di lacuna da valutare (default: all)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Numero massimo di casi da valutare (default: tutti per test_abs.csv, 100 per HF dataset)",
    )
    parser.add_argument(
        "--top-k-passages",
        type=int,
        default=RAG_TOP_K_PASSAGES,
        help="Numero di passi paralleli da recuperare (0 per valutare Zero-Shot senza RAG)",
    )
    parser.add_argument(
        "--compiled-path",
        type=str,
        default=None,
        help="Percorso opzionale al programma DSPy compilato (.json)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Percorso opzionale per salvare le metriche in formato JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Percorso opzionale per salvare il report comparativo in formato Markdown",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Percorso opzionale per salvare il dettaglio caso-per-caso (con CoT e attestazioni) in CSV",
    )
    args = parser.parse_args()

    retriever = (
        get_or_build_default_retriever(
            index_dir=RAG_INDEX_DIR,
            embedder_checkpoint=RAG_DEFAULT_EMBEDDER,
        )
        if args.top_k_passages > 0
        else None
    )

    if args.csv_path or args.compare_no_rag:
        csv_target = args.csv_path or "test_abs.csv"
        all_cases = load_cases_from_test_abs_csv(
            csv_path=csv_target,
            max_cases=args.max_cases,
            policy=args.policy,
        )
        report = evaluate_rag_impact_on_test_abs(
            cases=all_cases,
            csv_path=csv_target,
            checkpoint=args.lm,
            backend=args.backend,
            retriever=retriever,
            top_k_passages=max(1, args.top_k_passages),
            compiled_program_path=args.compiled_path,
        )
        md_report = format_rag_impact_markdown(report, model_name=args.lm)
        print("\n" + md_report + "\n")

        if args.output_md:
            md_path = Path(args.output_md)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md_report, encoding="utf-8")
            print(f"Report Markdown salvato in '{md_path}'.")

        if args.output_csv and report.get("case_records"):
            import pandas as pd

            csv_out_path = Path(args.output_csv)
            csv_out_path.parent.mkdir(parents=True, exist_ok=True)
            rows_for_csv = []
            for r in report["case_records"]:
                atts = r.get("top1_attestations") or []
                att_str = "; ".join(
                    f"[{a['corpus_id']}" + (f" | {a['source_id']}]" if a.get("source_id") else "]") + f" {a.get('attested_form', '')} ({a.get('doc_freq', 1)} doc.)"
                    for a in atts
                )
                rows_for_csv.append(
                    {
                        "case_id": r["case_id"],
                        "papyrus": r["papyrus"],
                        "policy": r["policy"],
                        "context_with_gap": r["context_with_gap"],
                        "gold": r["gold"],
                        "no_rag_top1": r["no_rag_top1"],
                        "no_rag_hit_top1": r["no_rag_hit_top1"],
                        "no_rag_hit_top5": r["no_rag_hit_top5"],
                        "with_rag_top1": r["with_rag_top1"],
                        "with_rag_hit_top1": r["with_rag_hit_top1"],
                        "with_rag_hit_top5": r["with_rag_hit_top5"],
                        "gold_attested_in_corpus": r["gold_attested_in_corpus"],
                        "corpus_attestations": att_str,
                        "philological_reasoning": r["philological_reasoning"],
                    }
                )
            pd.DataFrame(rows_for_csv).to_csv(csv_out_path, index=False)
            print(f"Predizioni e attestazioni salvate in '{csv_out_path}'.")

        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Risultati JSON salvati in '{out_path}'.")
        return

    from scripts.evaluate_comparison import load_cases_by_policy

    max_c = args.max_cases if args.max_cases is not None else 100
    cases_by_policy = load_cases_by_policy(
        dataset_name=args.eval_dataset,
        split=args.split,
        max_cases=max_c,
        policy=args.policy,
    )
    all_cases = []
    for pol, pol_cases in cases_by_policy.items():
        for c in pol_cases:
            c.gap_type = pol
            all_cases.append(c)

    results = evaluate_rag_on_cases(
        cases=all_cases,
        checkpoint=args.lm,
        backend=args.backend,
        retriever=retriever,
        top_k_passages=args.top_k_passages,
        compiled_program_path=args.compiled_path,
    )

    md_table = format_rag_evaluation_markdown(results, model_name=args.lm)
    print("\n" + md_table + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Risultati salvati in '{out_path}'.")


if __name__ == "__main__":
    main()
