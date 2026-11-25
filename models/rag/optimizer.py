"""
Ottimizzazione e Compilazione Automatica del Programma DSPy (`GreekRAGInfiller`).

Privilegia l'ottimizzazione zero-shot delle sole istruzioni di ragionamento filologico
(0 esempi few-shot `demos`, tramite `MIPROv2`, `COPRO` e `GEPA`) guidata dalle categorie
`<cla>` di `canon.xml` e dai relativi assi di ragionamento (`REASONING_CATEGORIES`),
mantenendo `BootstrapFewShot` come opzione legacy opt-in.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Sequence

import dspy

from backend.config.settings import (
    RAG_COMPILED_DIR,
    RAG_DEFAULT_BACKEND,
    RAG_DEFAULT_EMBEDDER,
    RAG_DEFAULT_LM,
    RAG_EXTRA_DATA_PATH,
    RAG_INDEX_DIR,
    RAG_TOP_K_PASSAGES,
)
from backend.core.cleaner import load_abs, split_abs_herc_dev
from backend.core.preprocess import normalize_greek, strip_diacritics, transpile
from models.bert.dataset.dev_set import DevCase, build_dev_set
from models.rag.categories import (
    TLG_CATEGORY_BY_CODE,
    get_reasoning_profile,
    load_tlg_work_categories_from_canon,
    normalize_text_categories,
)
from models.rag.lm import build_dspy_lm
from models.rag.modules import (
    GreekRAGInfiller,
    coerce_dspy_candidates,
    extract_lacuna_metadata,
)
from models.rag.retriever import (
    GreekHybridRetriever,
    get_or_build_default_retriever,
    load_rag_extra_blocks,
)

DEFAULT_CANON_XML_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "datasets"
    / "tlg-bulk"
    / "tlg"
    / "canon.xml"
)


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


def dev_cases_to_dspy_examples(
    cases: Sequence[DevCase | dict[str, Any]],
    default_categories: str | Sequence[str] | None = None,
) -> list[dspy.Example]:
    """
    Converte una sequenza di `DevCase` (o dizionari con chiavi `x` e `y`) in
    oggetti `dspy.Example` pronti per gli ottimizzatori `BootstrapFewShot` / `MIPROv2`,
    includendo il metadato di categoria TLG (`text_category`) negli input del programma.
    """
    examples: list[dspy.Example] = []

    for case in cases:
        if isinstance(case, DevCase):
            x_text = case.x
            y_val = " ".join(case.y) if isinstance(case.y, list) else str(case.y)
            gap_len = case.gap_length
            raw_cats = case.text_categories
        else:
            x_text = str(case.get("x") or case.get("context_with_gap") or "")
            raw_y = case.get("y") or case.get("gold_supplement") or case.get("gold_target") or ""
            y_val = " ".join(raw_y) if isinstance(raw_y, list) else str(raw_y)
            gap_len = case.get("gap_length")
            if "text_categories" in case:
                raw_cats = case.get("text_categories")
            elif "text_category" in case:
                raw_cats = case.get("text_category")
            else:
                raw_cats = case.get("cla")

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

        # Preserva una lista esplicita `[]` come esempio neutro (categoria non specificata)
        if raw_cats is None:
            effective_cats = normalize_text_categories(default_categories)
        else:
            effective_cats = normalize_text_categories(raw_cats)

        ex = dspy.Example(
            context_with_gap=meta["context_with_gap"],
            text_category=", ".join(effective_cats),
            min_chars=meta["min_chars"],
            max_chars=meta["max_chars"],
            word_prefix=meta["word_prefix"],
            word_suffix=meta["word_suffix"],
            gold_supplement=gold_norm,
        ).with_inputs(
            "context_with_gap",
            "text_category",
            "min_chars",
            "max_chars",
            "word_prefix",
            "word_suffix",
        )
        examples.append(ex)

    return examples


_GREEK_ALPHA_WORD_RE = re.compile(r"^[\u0370-\u03ff\u1f00-\u1fff]+$")


def _resolve_block_categories(
    ab: dict[str, Any],
    work_cat_map: dict[str, list[str]],
) -> list[str]:
    """Risolve i codici `<cla>` normalizzati per un blocco TLG tramite `canon.xml` o metadato locale."""
    file_id = str(ab.get("file_id") or "").strip()
    work_id = str(ab.get("work_id") or "").strip()
    if file_id and file_id in work_cat_map:
        return list(work_cat_map[file_id])
    if work_id and work_id in work_cat_map:
        return list(work_cat_map[work_id])

    raw_cla = ab.get("cla")
    if raw_cla:
        try:
            return normalize_text_categories(raw_cla)
        except ValueError:
            return []
    return []


def _synthesize_case_from_block(
    ab: dict[str, Any],
    categories: list[str],
    min_gap_length: int,
    max_gap_length: int,
    rng: random.Random,
    prefer_partial: bool = False,
) -> DevCase | None:
    """Genera una lacuna sintetica papirologica (inter- o intra-parola) da un blocco TLG."""
    raw_text = ab.get("training_text") or ab.get("text") or ""
    if not raw_text:
        return None

    clean_text = transpile(
        raw_text,
        case_folding="none",
        normalize=True,
        strip_diacritics=False,
    )
    clean_text = re.sub(r"<UNK>", " ", clean_text, flags=re.IGNORECASE)
    words = [w for w in clean_text.split() if w]
    if len(words) < 6:
        return None

    # Finestra contestuale compatta centrata su una porzione del blocco
    max_window = min(28, len(words))
    start_idx = rng.randint(0, max(0, len(words) - max_window))
    window = words[start_idx : start_idx + max_window]

    candidate_indices: list[tuple[int, str, str]] = []
    for idx in range(1, len(window) - 1):
        tok = window[idx]
        if not _GREEK_ALPHA_WORD_RE.match(tok):
            continue
        base_len = len(strip_diacritics(tok))
        if base_len < min_gap_length:
            continue
        candidate_indices.append((idx, tok, strip_diacritics(tok)))

    if not candidate_indices:
        return None

    rng.shuffle(candidate_indices)
    chosen = candidate_indices[0]
    if prefer_partial:
        for item in candidate_indices:
            if len(item[2]) > min_gap_length + 1:
                chosen = item
                break

    tok_idx, surface_tok, base_tok = chosen
    tok_len = len(base_tok)

    # Alterna lacune inter-parola (quando compatibile col range) e intra-parola (suffisso/centrale)
    can_full_word = min_gap_length <= tok_len <= max_gap_length
    if can_full_word and not (prefer_partial and tok_len >= min_gap_length + 2):
        gap_len = tok_len
        gold_supp = normalize_greek(surface_tok, case_folding="fold", strip_diacritics_flag=True)
        masked_tok = f"[{'.' * gap_len}]"
        gap_type = "word"
    else:
        # Maschera una porzione di lunghezza controllata preservando almeno 2 lettere di affisso
        max_mask = min(max_gap_length, max(min_gap_length, tok_len - 2))
        gap_len = rng.randint(min_gap_length, max(min_gap_length, max_mask))
        if tok_len - gap_len < 1:
            return None
        # Usa surface_tok solo se privo di diacritici complessi disallineati su indici base;
        # per stabilità degli indici caratteriali operiamo sulla forma NFC priva di diacritici combinatori
        clean_surface = strip_diacritics(surface_tok)
        if len(clean_surface) != tok_len:
            return None
        if rng.random() < 0.7 or tok_len - gap_len < 3:
            split_pos = tok_len - gap_len
            masked_tok = f"{clean_surface[:split_pos]}[{'.' * gap_len}]"
            gold_supp = normalize_greek(
                clean_surface[split_pos:], case_folding="fold", strip_diacritics_flag=True
            )
            gap_type = "suffix"
        else:
            prefix_len = rng.randint(1, tok_len - gap_len - 1)
            masked_tok = (
                f"{clean_surface[:prefix_len]}[{'.' * gap_len}]"
                f"{clean_surface[prefix_len + gap_len :]}"
            )
            gold_supp = normalize_greek(
                clean_surface[prefix_len : prefix_len + gap_len],
                case_folding="fold",
                strip_diacritics_flag=True,
            )
            gap_type = "default"

    if not gold_supp or len(strip_diacritics(gold_supp)) != gap_len:
        return None

    masked_window = list(window)
    masked_window[tok_idx] = masked_tok
    context_str = " ".join(masked_window)

    return DevCase(
        x=context_str,
        y=[gold_supp],
        gap_length=gap_len,
        corpus_id=str(ab.get("corpus_id") or "tlg"),
        file_id=str(ab.get("file_id") or ab.get("id") or ""),
        gap_type=gap_type,
        text_categories=list(categories),
    )


def build_synthetic_tlg_category_cases(
    max_cases: int = 24,
    min_gap_length: int = 2,
    max_gap_length: int = 6,
    canon_xml_path: str | Path | None = None,
    blocks_override: Sequence[dict[str, Any]] | None = None,
    extra_data_dirs: Sequence[Path | str] | None = None,
    seed: int = 42,
) -> list[DevCase]:
    """
    Campiona lacune sintetiche stratificate per categoria `<cla>` dai corpora TLG locali
    (`data/tlg_*.json` e `data/rag_corpora/`), mappando ciascun blocco ai codici `<cla>`
    di `canon.xml` così da esporre l'ottimizzatore DSPy a generi diversi.
    """
    if max_cases <= 0:
        return []

    canon_path = Path(canon_xml_path) if canon_xml_path is not None else DEFAULT_CANON_XML_PATH
    work_cat_map = load_tlg_work_categories_from_canon(canon_path) if canon_path.exists() else {}

    if blocks_override is not None:
        raw_blocks = list(blocks_override)
    else:
        raw_blocks = load_abs(corpus_set=["tlg"])
        resolved_extra = (
            list(extra_data_dirs)
            if extra_data_dirs is not None
            else ([RAG_EXTRA_DATA_PATH] if RAG_EXTRA_DATA_PATH.exists() else [])
        )
        raw_blocks.extend(load_rag_extra_blocks(resolved_extra, corpus_set=["tlg"]))

    rng = random.Random(seed)
    by_primary_cat: dict[str, list[tuple[dict[str, Any], list[str]]]] = defaultdict(list)
    for ab in raw_blocks:
        cats = _resolve_block_categories(ab, work_cat_map)
        if not cats:
            continue
        by_primary_cat[cats[0]].append((ab, cats))

    if not by_primary_cat:
        return []

    cat_keys = sorted(by_primary_cat.keys())
    rng.shuffle(cat_keys)
    for k in cat_keys:
        rng.shuffle(by_primary_cat[k])

    cases: list[DevCase] = []
    # Estrazione round-robin fra le categorie disponibili per massimizzare la copertura dei generi
    exhausted = False
    while len(cases) < max_cases and not exhausted:
        exhausted = True
        for k in cat_keys:
            if len(cases) >= max_cases:
                break
            bucket = by_primary_cat[k]
            while bucket:
                ab, cats = bucket.pop()
                case = _synthesize_case_from_block(
                    ab=ab,
                    categories=cats,
                    min_gap_length=min_gap_length,
                    max_gap_length=max_gap_length,
                    rng=rng,
                    prefer_partial=(len(cases) % 2 == 1),
                )
                if case is not None:
                    cases.append(case)
                    exhausted = False
                    break

    return cases


def build_mixed_optimization_cases(
    herc_cases: Sequence[DevCase],
    max_total_cases: int = 40,
    tlg_ratio: float = 0.6,
    herc_default_categories: str | Sequence[str] | None = ("phil.",),
    neutral_ratio: float = 0.15,
    canon_xml_path: str | Path | None = None,
    tlg_blocks_override: Sequence[dict[str, Any]] | None = None,
    extra_data_dirs: Sequence[Path | str] | None = None,
    seed: int = 42,
) -> list[DevCase]:
    """
    Costruisce un trainset misto stratificato per l'ottimizzatore DSPy combinando:
    - Casi papirologici reali (`P.Herc.`, tipicamente filosofici `phil.`).
    - Lacune sintetiche stratificate dai corpora TLG locali con `<cla>` da `canon.xml`.
    - Una quota di esempi neutri (`text_categories=[]`) per preservare l'affidabilità
      quando l'utente non specifica la categoria a runtime.
    """
    if max_total_cases <= 0:
        return []

    clamped_tlg_ratio = max(0.0, min(1.0, float(tlg_ratio)))
    target_tlg = int(round(max_total_cases * clamped_tlg_ratio))
    target_herc = max_total_cases - target_tlg

    default_herc_cats = normalize_text_categories(herc_default_categories)
    selected_herc: list[DevCase] = []
    for c in herc_cases[:target_herc]:
        cats = list(c.text_categories) if c.text_categories is not None else list(default_herc_cats)
        selected_herc.append(replace(c, text_categories=cats))

    tlg_cases = build_synthetic_tlg_category_cases(
        max_cases=target_tlg,
        canon_xml_path=canon_xml_path,
        blocks_override=tlg_blocks_override,
        extra_data_dirs=extra_data_dirs,
        seed=seed,
    )

    # Se i corpora TLG locali forniscono meno casi del target, compensa coi casi reali residui
    combined = list(selected_herc) + list(tlg_cases)
    if len(combined) < max_total_cases and len(herc_cases) > len(selected_herc):
        needed = max_total_cases - len(combined)
        for c in herc_cases[len(selected_herc) : len(selected_herc) + needed]:
            cats = list(c.text_categories) if c.text_categories is not None else list(default_herc_cats)
            combined.append(replace(c, text_categories=cats))

    rng = random.Random(seed)
    rng.shuffle(combined)

    clamped_neutral = max(0.0, min(1.0, float(neutral_ratio)))
    num_neutral = int(round(len(combined) * clamped_neutral))
    if num_neutral > 0 and combined:
        neutral_indices = set(rng.sample(range(len(combined)), min(num_neutral, len(combined))))
        combined = [
            replace(case, text_categories=[]) if idx in neutral_indices else case
            for idx, case in enumerate(combined)
        ]

    return combined


_GENERAL_PHILOLOGY_CUES = (
    "morfolog",
    "sintass",
    "contest",
    "lacun",
    "letter",
    "paralle",
    "loci",
    "attest",
    "prefiss",
    "suffiss",
    "lessic",
    "radic",
    "semantic",
    "caso",
    "verb",
)


def score_philological_reasoning(
    reasoning: str,
    text_categories: Sequence[str] | str | None = None,
    candidates: Sequence[Any] = (),
) -> tuple[float, list[str]]:
    """
    Valuta qualità filologica e allineamento alle categorie di ragionamento (`<cla>`)
    della Chain-of-Thought di 1° livello (`philological_reasoning`) e dei `rationale`
    di 2° livello. Restituisce `(score in [0.0, 1.0], note_di_feedback)`.
    """
    clean_reasoning = (reasoning or "").strip()
    lower_reasoning = clean_reasoning.lower()
    feedback_notes: list[str] = []

    # 1. Sostanza del ragionamento di 1° livello (peso 0.30)
    if len(clean_reasoning) >= 40:
        substance_score = 0.30
    elif len(clean_reasoning) >= 15:
        substance_score = 0.15
        feedback_notes.append(
            "Il ragionamento filologico di 1° livello è troppo sintetico (<40 caratteri)."
        )
    else:
        substance_score = 0.0
        feedback_notes.append(
            "Manca un ragionamento filologico articolato in `philological_reasoning`."
        )

    # 2. Ancoraggio metodologico generale (peso 0.25)
    general_hits = sum(1 for cue in _GENERAL_PHILOLOGY_CUES if cue in lower_reasoning)
    if general_hits >= 2:
        grounding_score = 0.25
    elif general_hits == 1:
        grounding_score = 0.15
        feedback_notes.append(
            "Esplicita più chiaramente i vincoli morfologici, sintattici e i passi paralleli."
        )
    else:
        grounding_score = 0.0
        feedback_notes.append(
            "Il ragionamento non richiama criteri filologici espliciti (morfologia, sintassi, contesto, loci similes)."
        )

    # 3. Allineamento all'asse di ragionamento della categoria <cla> (peso 0.30)
    norm_cats = normalize_text_categories(text_categories)
    rationale_texts = [
        str(getattr(c, "rationale", "") or "").strip()
        for c in (candidates or ())[:3]
        if str(getattr(c, "rationale", "") or "").strip()
    ]
    combined_reasoning = f"{lower_reasoning} {' '.join(rationale_texts).lower()}".strip()

    if norm_cats:
        profile = get_reasoning_profile(norm_cats)
        cat_cues = list(profile["cues"])
        for code in norm_cats:
            cat_info = TLG_CATEGORY_BY_CODE.get(code)
            if cat_info:
                cat_cues.append(cat_info.label.lower().split()[0])
                cat_cues.append(code.rstrip("."))
        cat_hits = sum(1 for cue in set(cat_cues) if cue and cue in combined_reasoning)
        if cat_hits >= 2:
            category_score = 0.30
        elif cat_hits == 1:
            category_score = 0.20
        else:
            category_score = 0.0
            labels_str = ", ".join(profile["labels"])
            focus_str = "; ".join(profile["focuses"])
            feedback_notes.append(
                f"Il ragionamento non applica i criteri della categoria '{labels_str}' ({focus_str})."
            )
    else:
        # Per esempi neutri senza categoria assegnata premia sostanza + ancoraggio filologico
        category_score = 0.30 if (substance_score >= 0.15 and grounding_score >= 0.15) else 0.15

    # 4. Copertura della CoT di 2° livello nei candidati Top-3 (peso 0.15)
    top_slice = list((candidates or ())[:3])
    if top_slice:
        rationale_ratio = sum(
            1 for c in top_slice if str(getattr(c, "rationale", "") or "").strip()
        ) / len(top_slice)
        rationale_score = 0.15 * rationale_ratio
        if rationale_ratio < 1.0:
            feedback_notes.append(
                "Alcuni candidati principali sono privi di `rationale` (CoT di 2° livello)."
            )
    else:
        rationale_score = 0.0

    total = min(1.0, max(0.0, substance_score + grounding_score + category_score + rationale_score))
    return float(total), feedback_notes


def _evaluate_prediction_components(
    example: dspy.Example,
    pred: dspy.Prediction,
) -> dict[str, Any]:
    """Calcola tutte le sotto-metriche filologiche e di ragionamento per `example`/`pred`."""
    gold = getattr(example, "gold_supplement", "") or ""
    gold_norm = normalize_greek(gold, case_folding="fold", strip_diacritics_flag=True)
    gold_norm = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", gold_norm).strip()
    if not gold_norm:
        return {"valid": False, "score": 0.0, "gold_norm": ""}

    raw_cands = coerce_dspy_candidates(getattr(pred, "candidates", []))
    if not raw_cands:
        return {"valid": False, "score": 0.0, "gold_norm": gold_norm, "no_cands": True}

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
        return {"valid": False, "score": 0.0, "gold_norm": gold_norm, "no_cands": True}

    top1_em = 1.0 if norm_supps[0] == gold_norm else 0.0
    top5_em = 1.0 if gold_norm in norm_supps[:5] else 0.0
    cer_top1 = compute_cer(norm_supps[0], gold_norm)
    char_acc = max(0.0, 1.0 - cer_top1)
    valid_len_ratio = valid_len_count / len(norm_supps)

    reasoning_text = str(getattr(pred, "philological_reasoning", "") or "")
    example_cats = normalize_text_categories(getattr(example, "text_category", None))
    reasoning_score, reasoning_notes = score_philological_reasoning(
        reasoning=reasoning_text,
        text_categories=example_cats,
        candidates=raw_cands,
    )

    total_score = float(
        0.40 * top1_em
        + 0.15 * top5_em
        + 0.15 * char_acc
        + 0.10 * valid_len_ratio
        + 0.20 * reasoning_score
    )
    return {
        "valid": True,
        "score": total_score,
        "gold_norm": gold_norm,
        "top1_supp": norm_supps[0],
        "top1_em": top1_em,
        "top5_em": top5_em,
        "char_acc": char_acc,
        "valid_len_ratio": valid_len_ratio,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "example_cats": example_cats,
        "reasoning_score": reasoning_score,
        "reasoning_notes": reasoning_notes,
    }


def greek_infilling_dspy_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float | bool:
    """
    Metrica filologica multi-criterio per l'ottimizzazione delle istruzioni DSPy:
    - Top-1 Exact Match (peso 0.40)
    - Top-5 Exact Match (peso 0.15)
    - Character Accuracy `1 - CER` sul candidato Top-1 (peso 0.15)
    - Rispetto della lunghezza della lacuna `min_chars..max_chars` (peso 0.10)
    - Qualità del ragionamento filologico e aderenza alla categoria `<cla>` (peso 0.20)

    Quando `trace is not None` (opzione legacy `BootstrapFewShot`), restituisce `True` solo se
    la predizione contiene la gold label nella Top-5 oppure raggiunge Character Accuracy >= 0.60
    rispettando la lunghezza della lacuna.
    """
    comp = _evaluate_prediction_components(example, pred)
    if not comp["valid"]:
        return False if trace is not None else 0.0

    if trace is not None:
        return bool(
            comp["top5_em"] > 0.0
            or (comp["char_acc"] >= 0.60 and comp["valid_len_ratio"] >= 0.5)
        )

    return float(comp["score"])


def greek_infilling_gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """
    Metrica con feedback testuale strutturato per `dspy.GEPA`, pensata per raffinare
    le istruzioni di ragionamento filologico rispetto alla categoria `<cla>`.
    """
    comp = _evaluate_prediction_components(gold, pred)
    if not comp["valid"]:
        reason = (
            "Nessun candidato greco valido generato per la lacuna."
            if comp.get("no_cands")
            else "Esempio privo di gold_supplement valida."
        )
        return dspy.Prediction(score=0.0, feedback=reason)

    feedback_parts: list[str] = []
    if comp["top1_em"] >= 1.0:
        feedback_parts.append(f"Top-1 esatto ('{comp['gold_norm']}').")
    elif comp["top5_em"] >= 1.0:
        feedback_parts.append(
            f"Target '{comp['gold_norm']}' presente in Top-5 ma non in prima posizione "
            f"(Top-1 proposto: '{comp['top1_supp']}', char_acc={comp['char_acc']:.2f})."
        )
    else:
        feedback_parts.append(
            f"Target '{comp['gold_norm']}' mancato (Top-1: '{comp['top1_supp']}', "
            f"char_acc={comp['char_acc']:.2f})."
        )

    if comp["valid_len_ratio"] < 1.0:
        feedback_parts.append(
            f"Rispetto lunghezza [{comp['min_chars']}..{comp['max_chars']}] parziale "
            f"({comp['valid_len_ratio']:.0%})."
        )

    if comp["reasoning_notes"]:
        feedback_parts.extend(comp["reasoning_notes"])
    elif comp["example_cats"]:
        profile = get_reasoning_profile(comp["example_cats"])
        feedback_parts.append(
            f"Ragionamento ben allineato alla categoria '{', '.join(profile['labels'])}' "
            f"(score={comp['reasoning_score']:.2f})."
        )

    return dspy.Prediction(
        score=float(comp["score"]),
        feedback=" ".join(feedback_parts),
    )


def _clear_predictor_demos(program: dspy.Module) -> None:
    """Garantisce un programma puramente zero-shot azzerando eventuali `demos` residui."""
    for predictor in program.predictors():
        if hasattr(predictor, "demos"):
            predictor.demos = []


def compile_greek_rag_infiller(
    trainset: Sequence[dspy.Example],
    lm: dspy.BaseLM,
    retriever: GreekHybridRetriever | None = None,
    top_k_passages: int = RAG_TOP_K_PASSAGES,
    optimizer_type: Literal["miprov2", "copro", "gepa", "bootstrap"] = "miprov2",
    max_bootstrapped_demos: int = 0,
    max_labeled_demos: int = 0,
    output_path: str | Path | None = None,
    prompt_lm: dspy.BaseLM | None = None,
    mipro_auto: Literal["light", "medium", "heavy"] = "light",
    copro_breadth: int = 6,
    copro_depth: int = 2,
) -> GreekRAGInfiller:
    """
    Compila `GreekRAGInfiller` ottimizzando in modalità zero-shot le istruzioni di
    ragionamento filologico (`miprov2` con 0 demos di default, `copro` o `gepa`;
    `bootstrap` resta disponibile solo come opzione legacy opt-in) e salva su disco il JSON.
    """
    if not trainset:
        raise ValueError("Il trainset per la compilazione DSPy non può essere vuoto.")

    # Attiva sempre il filtro anti-data-leakage durante la compilazione per impedire che
    # eventuali lacune sintetiche TLG recuperino il proprio passo sorgente non mascherato.
    student = GreekRAGInfiller(
        retriever=retriever,
        top_k_passages=top_k_passages,
        enable_refinement=True,
        default_exclude_query_overlap=True,
    )

    instruction_lm = prompt_lm or lm
    with dspy.context(lm=lm):
        if optimizer_type == "miprov2":
            zeroshot_only = (max_bootstrapped_demos == 0 and max_labeled_demos == 0)
            teleprompter = dspy.MIPROv2(
                metric=greek_infilling_dspy_metric,
                prompt_model=instruction_lm,
                task_model=lm,
                auto=mipro_auto,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
            compiled_program = teleprompter.compile(
                student,
                trainset=list(trainset),
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                minibatch=len(trainset) >= 35,
                fewshot_aware_proposer=not zeroshot_only,
            )
            if zeroshot_only:
                _clear_predictor_demos(compiled_program)
        elif optimizer_type == "copro":
            teleprompter = dspy.COPRO(
                metric=greek_infilling_dspy_metric,
                prompt_model=instruction_lm,
                breadth=copro_breadth,
                depth=copro_depth,
            )
            compiled_program = teleprompter.compile(
                student,
                trainset=list(trainset),
                eval_kwargs={"num_threads": 1, "display_progress": False},
            )
            _clear_predictor_demos(compiled_program)
        elif optimizer_type == "gepa":
            teleprompter = dspy.GEPA(
                metric=greek_infilling_gepa_metric,
                auto=mipro_auto,
                reflection_lm=instruction_lm,
            )
            compiled_program = teleprompter.compile(
                student,
                trainset=list(trainset),
            )
            _clear_predictor_demos(compiled_program)
        else:
            # Opzione legacy few-shot (esplicitamente richiesta con optimizer_type='bootstrap')
            boot_demos = max_bootstrapped_demos if max_bootstrapped_demos > 0 else 3
            label_demos = max_labeled_demos if max_labeled_demos > 0 else 3
            teleprompter = dspy.BootstrapFewShot(
                metric=greek_infilling_dspy_metric,
                max_bootstrapped_demos=boot_demos,
                max_labeled_demos=label_demos,
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
        default="miprov2",
        choices=["miprov2", "copro", "gepa", "bootstrap"],
        help="Teleprompter DSPy da utilizzare (default: 'miprov2' zero-shot sulle istruzioni; supporta 'copro', 'gepa' e legacy 'bootstrap')",
    )
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=0,
        help="Numero di demo bootstrapped nel prompt (default: 0 = ottimizzazione zero-shot delle sole istruzioni)",
    )
    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=0,
        help="Numero di demo etichettate nel prompt (default: 0 = ottimizzazione zero-shot delle sole istruzioni)",
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
    parser.add_argument(
        "--canon-xml",
        type=str,
        default=str(DEFAULT_CANON_XML_PATH),
        help="Percorso al file canon.xml del TLG per mappare i tag <cla> delle opere",
    )
    parser.add_argument(
        "--tlg-ratio",
        type=float,
        default=0.6,
        help="Quota (0.0-1.0) di esempi sintetici TLG stratificati per <cla> nel trainset misto (default: 0.6)",
    )
    parser.add_argument(
        "--herc-default-category",
        type=str,
        default="phil.",
        help="Categoria TLG <cla> di default da associare ai casi P.Herc. (default: 'phil.'; usare 'none' per neutro)",
    )
    parser.add_argument(
        "--neutral-ratio",
        type=float,
        default=0.15,
        help="Quota (0.0-1.0) di esempi senza categoria specificata per mantenere robustezza neutra (default: 0.15)",
    )
    args = parser.parse_args()

    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", args.lm.split("/")[-1]).lower()
    out_path = Path(args.output) if args.output else (RAG_COMPILED_DIR / f"compiled_rag_{slug}.json")

    print("[DSPy Optimizer] Caricamento casi di sviluppo P.Herc. e corpora TLG...")
    _, dev_abs = split_abs_herc_dev(load_abs(budget=25), test_size=0.2)
    herc_cases = build_dev_set(dev_abs, min_gap_length=1, max_gap_length=6)
    herc_cats = (
        None
        if str(args.herc_default_category).strip().lower() in {"", "none", "null"}
        else args.herc_default_category
    )
    mixed_cases = build_mixed_optimization_cases(
        herc_cases=herc_cases,
        max_total_cases=args.max_train_cases,
        tlg_ratio=args.tlg_ratio,
        herc_default_categories=herc_cats,
        neutral_ratio=args.neutral_ratio,
        canon_xml_path=args.canon_xml,
    )
    trainset = dev_cases_to_dspy_examples(mixed_cases)
    cat_dist = Counter(
        (ex.text_category or "<neutra>") for ex in trainset
    )
    rc_dist = Counter(
        rc
        for ex in trainset
        for rc in get_reasoning_profile(ex.text_category or None)["reasoning_categories"]
    )
    print(
        f"[DSPy Optimizer] Preparati {len(trainset)} esempi DSPy "
        f"(categorie={dict(cat_dist.most_common(10))}, "
        f"assi_ragionamento={dict(rc_dist)})."
    )

    retriever = get_or_build_default_retriever(
        index_dir=RAG_INDEX_DIR,
        embedder_checkpoint=RAG_DEFAULT_EMBEDDER,
    )
    lm = build_dspy_lm(checkpoint=args.lm, backend=args.backend)

    print(
        f"[DSPy Optimizer] Avvio compilazione ({args.optimizer}, "
        f"bootstrapped_demos={args.max_bootstrapped_demos}, "
        f"labeled_demos={args.max_labeled_demos}) per '{args.lm}'..."
    )
    compile_greek_rag_infiller(
        trainset=trainset,
        lm=lm,
        retriever=retriever,
        optimizer_type=args.optimizer,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        output_path=out_path,
    )
    print(f"[DSPy Optimizer] Programma compilato salvato in '{out_path}'.")


if __name__ == "__main__":
    main()
