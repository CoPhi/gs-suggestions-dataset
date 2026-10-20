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
from backend.core.preprocess import normalize_greek
from models.bert.dataset.dev_set import DevCase
from models.bert.evaluation.metrics import (
    evaluate_bertscore_topk_text,
    evaluate_topK_text,
)
from models.rag.inference import fill_mask_rag
from models.rag.optimizer import compute_cer
from models.rag.retriever import GreekHybridRetriever, get_or_build_default_retriever


def _strict_norm(text: str) -> str:
    norm = normalize_greek(text, case_folding="fold", strip_diacritics_flag=True)
    return re.sub(r"[\s\u200B-\u200D\uFEFF]", "", norm).strip()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valutazione del sistema DSPy RAG su dataset di benchmark papirologici e TLG."
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
        default="default",
        choices=["default", "word", "suffix", "all"],
        help="Policy di lacuna da valutare",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=100,
        help="Numero massimo di casi da valutare per policy (default: 100)",
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
    args = parser.parse_args()

    from scripts.evaluate_comparison import load_cases_by_policy

    cases_by_policy = load_cases_by_policy(
        dataset_name=args.eval_dataset,
        split=args.split,
        max_cases=args.max_cases,
        policy=args.policy,
    )
    all_cases: list[DevCase] = []
    for pol, pol_cases in cases_by_policy.items():
        for c in pol_cases:
            c.gap_type = pol
            all_cases.append(c)

    retriever = (
        get_or_build_default_retriever(
            index_dir=RAG_INDEX_DIR,
            embedder_checkpoint=RAG_DEFAULT_EMBEDDER,
        )
        if args.top_k_passages > 0
        else None
    )

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
