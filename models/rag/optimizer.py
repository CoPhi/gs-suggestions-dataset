"""
Ottimizzazione e Compilazione Automatica del Programma DSPy (`GreekRAGInfiller`).

Sfrutta i Teleprompters di DSPy (`BootstrapFewShot` e `MIPROv2`) per generare,
filtrare e congelare nel prompt le migliori dimostrazioni few-shot (`demos`) e le
istruzioni ottimali per gli LLM (Qwen, Llama) sui casi di validazione papirologici.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Literal, Sequence

import dspy

from backend.config.settings import (
    RAG_COMPILED_DIR,
    RAG_DEFAULT_BACKEND,
    RAG_DEFAULT_EMBEDDER,
    RAG_DEFAULT_LM,
    RAG_INDEX_DIR,
    RAG_TOP_K_PASSAGES,
)
from backend.core.cleaner import load_abs, split_abs_herc_dev
from backend.core.preprocess import normalize_greek, strip_diacritics
from models.bert.dataset.dev_set import DevCase, build_dev_set
from models.rag.lm import build_dspy_lm
from models.rag.modules import (
    GreekRAGInfiller,
    coerce_dspy_candidates,
    extract_lacuna_metadata,
)
from models.rag.retriever import GreekHybridRetriever, get_or_build_default_retriever


def compute_cer(pred: str, target: str) -> float:
    """Calcola il Character Error Rate (distanza di Levenshtein normalizzata sulla lunghezza target)."""
    p = normalize_greek(pred, case_folding="fold", strip_diacritics_flag=True)
    t = normalize_greek(target, case_folding="fold", strip_diacritics_flag=True)
    p = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", p)
    t = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", t)

    if not t:
        return 0.0 if not p else 1.0

    dp = [[0] * (len(t) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p) + 1):
        dp[i][0] = i
    for j in range(len(t) + 1):
        dp[0][j] = j

    for i in range(1, len(p) + 1):
        for j in range(1, len(t) + 1):
            cost = 0 if p[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[len(p)][len(t)] / len(t)


def dev_cases_to_dspy_examples(cases: Sequence[DevCase | dict[str, Any]]) -> list[dspy.Example]:
    """
    Converte una sequenza di `DevCase` (o dizionari con chiavi `x` e `y`) in
    oggetti `dspy.Example` pronti per gli ottimizzatori `BootstrapFewShot` / `MIPROv2`.
    """
    examples: list[dspy.Example] = []

    for case in cases:
        if isinstance(case, DevCase):
            x_text = case.x
            y_val = " ".join(case.y) if isinstance(case.y, list) else str(case.y)
            gap_len = case.gap_length
        else:
            x_text = str(case.get("x") or case.get("context_with_gap") or "")
            raw_y = case.get("y") or case.get("gold_supplement") or case.get("gold_target") or ""
            y_val = " ".join(raw_y) if isinstance(raw_y, list) else str(raw_y)
            gap_len = case.get("gap_length")

        if not x_text or not y_val:
            continue

        try:
            meta = extract_lacuna_metadata(x_text, n_chars=gap_len)
        except ValueError:
            continue

        gold_norm = normalize_greek(
            y_val,
            case_folding="fold",
            strip_diacritics_flag=True,
        )
        gold_norm = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", gold_norm).strip()
        if not gold_norm:
            continue

        ex = dspy.Example(
            context_with_gap=meta["context_with_gap"],
            min_chars=meta["min_chars"],
            max_chars=meta["max_chars"],
            word_prefix=meta["word_prefix"],
            word_suffix=meta["word_suffix"],
            gold_supplement=gold_norm,
        ).with_inputs(
            "context_with_gap",
            "min_chars",
            "max_chars",
            "word_prefix",
            "word_suffix",
        )
        examples.append(ex)

    return examples


def greek_infilling_dspy_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float | bool:
    """
    Metrica filologica multi-criterio per gli ottimizzatori DSPy:
    - Top-1 Exact Match (peso 0.50)
    - Top-5 Exact Match (peso 0.20)
    - Character Accuracy `1 - CER` sul candidato Top-1 (peso 0.20)
    - Rispetto della lunghezza della lacuna `min_chars..max_chars` (peso 0.10)

    Quando `trace is not None` (fase di filtraggio dimostrazioni in `BootstrapFewShot`),
    restituisce `True` solo se la predizione contiene la gold label nella Top-5 oppure
    raggiunge una Character Accuracy >= 0.60 rispettando la lunghezza della lacuna.
    """
    gold = getattr(example, "gold_supplement", "") or ""
    gold_norm = normalize_greek(gold, case_folding="fold", strip_diacritics_flag=True)
    gold_norm = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", gold_norm).strip()
    if not gold_norm:
        return False if trace is not None else 0.0

    raw_cands = coerce_dspy_candidates(getattr(pred, "candidates", []))
    if not raw_cands:
        return False if trace is not None else 0.0

    min_chars = int(getattr(example, "min_chars", len(gold_norm)))
    max_chars = int(getattr(example, "max_chars", len(gold_norm)))

    norm_supps: list[str] = []
    valid_len_count = 0
    for c in raw_cands:
        s_norm = normalize_greek(
            c.supplement,
            case_folding="fold",
            strip_diacritics_flag=True,
        )
        s_norm = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", s_norm).strip()
        if not s_norm:
            continue
        norm_supps.append(s_norm)
        if min_chars <= len(strip_diacritics(s_norm)) <= max_chars:
            valid_len_count += 1

    if not norm_supps:
        return False if trace is not None else 0.0

    top1_em = 1.0 if norm_supps[0] == gold_norm else 0.0
    top5_em = 1.0 if gold_norm in norm_supps[:5] else 0.0
    cer_top1 = compute_cer(norm_supps[0], gold_norm)
    char_acc = max(0.0, 1.0 - cer_top1)
    valid_len_ratio = valid_len_count / len(norm_supps)

    if trace is not None:
        return bool(top5_em > 0.0 or (char_acc >= 0.60 and valid_len_ratio >= 0.5))

    return float(
        0.50 * top1_em
        + 0.20 * top5_em
        + 0.20 * char_acc
        + 0.10 * valid_len_ratio
    )


def compile_greek_rag_infiller(
    trainset: Sequence[dspy.Example],
    lm: dspy.BaseLM,
    retriever: GreekHybridRetriever | None = None,
    top_k_passages: int = RAG_TOP_K_PASSAGES,
    optimizer_type: Literal["bootstrap", "miprov2"] = "bootstrap",
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    output_path: str | Path | None = None,
) -> GreekRAGInfiller:
    """
    Compila `GreekRAGInfiller` con `dspy.BootstrapFewShot` o `dspy.MIPROv2`
    e salva opzionalmente lo stato compilato in formato JSON su disco.
    """
    if not trainset:
        raise ValueError("Il trainset per la compilazione DSPy non può essere vuoto.")

    student = GreekRAGInfiller(
        retriever=retriever,
        top_k_passages=top_k_passages,
        enable_refinement=True,
    )

    with dspy.context(lm=lm):
        if optimizer_type == "miprov2":
            teleprompter = dspy.MIPROv2(
                metric=greek_infilling_dspy_metric,
                auto="light",
            )
            compiled_program = teleprompter.compile(
                student,
                trainset=list(trainset),
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
        else:
            teleprompter = dspy.BootstrapFewShot(
                metric=greek_infilling_dspy_metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=1,
            )
            compiled_program = teleprompter.compile(
                student,
                trainset=list(trainset),
            )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        compiled_program.save(str(out))

    return compiled_program


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compilazione e ottimizzazione del modulo DSPy GreekRAGInfiller."
    )
    parser.add_argument(
        "--lm",
        type=str,
        default=RAG_DEFAULT_LM,
        help=f"Modello LLM Hugging Face da ottimizzare (default: {RAG_DEFAULT_LM})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=RAG_DEFAULT_BACKEND,
        choices=["auto", "hf_api", "local", "openai_compatible"],
        help="Backend di inferenza per l'LLM",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "miprov2"],
        help="Teleprompter DSPy da utilizzare ('bootstrap' o 'miprov2')",
    )
    parser.add_argument(
        "--max-train-cases",
        type=int,
        default=40,
        help="Numero massimo di casi del dev set da usare per la compilazione (default: 40)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Percorso file JSON dove salvare il programma DSPy compilato",
    )
    args = parser.parse_args()

    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", args.lm.split("/")[-1]).lower()
    out_path = Path(args.output) if args.output else (RAG_COMPILED_DIR / f"compiled_rag_{slug}.json")

    print("[DSPy Optimizer] Caricamento casi di sviluppo da data/...")
    _, dev_abs = split_abs_herc_dev(load_abs(budget=25), test_size=0.2)
    dev_cases = build_dev_set(dev_abs, min_gap_length=1, max_gap_length=6)[: args.max_train_cases]
    trainset = dev_cases_to_dspy_examples(dev_cases)
    print(f"[DSPy Optimizer] Preparati {len(trainset)} esempi DSPy.")

    retriever = get_or_build_default_retriever(
        index_dir=RAG_INDEX_DIR,
        embedder_checkpoint=RAG_DEFAULT_EMBEDDER,
    )
    lm = build_dspy_lm(checkpoint=args.lm, backend=args.backend)

    print(f"[DSPy Optimizer] Avvio compilazione ({args.optimizer}) per '{args.lm}'...")
    compile_greek_rag_infiller(
        trainset=trainset,
        lm=lm,
        retriever=retriever,
        optimizer_type=args.optimizer,
        output_path=out_path,
    )
    print(f"[DSPy Optimizer] Programma compilato salvato in '{out_path}'.")


if __name__ == "__main__":
    main()
