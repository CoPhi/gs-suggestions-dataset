"""
Moduli componibili DSPy (`dspy.Module`) per il Text-Infilling in Greco Antico tramite RAG.

Implementa `GreekRAGInfiller`, che orchestra:
1. L'estrazione dei vincoli papirologici della lacuna (`min_chars`, `max_chars`,
   `word_prefix` e/o `word_suffix`).
2. Il recupero ibrido dei passi paralleli (*loci similes*) tramite `GreekHybridRetriever`.
3. La generazione ragionata (`dspy.ChainOfThought(GreekInfillingSignature)`).
4. La verifica deterministica dei vincoli di lunghezza e alfabeto greco, con attivazione
   condizionata del predittore di correzione (`dspy.Predict(ConstraintRefinementSignature)`).
"""

from __future__ import annotations

import re
from typing import Any

import dspy

from backend.core import UNK_TOKEN
from backend.core.preprocess import (
    normalize_greek,
    process_editorial_marks,
    strip_diacritics,
)
from models.rag.retriever import GreekHybridRetriever, _strip_passage_header
from models.rag.signatures import (
    ConstraintRefinementSignature,
    GreekCandidate,
    GreekInfillingSignature,
    format_gap_constraints,
)

TARGET_LACUNA_REGEX = re.compile(r"\[(\.+|\-+)\]")
AFFIX_LACUNA_REGEX = re.compile(r"([^\W\d_]*)\[(\.+|\-+)\]([^\W\d_]*)")


def extract_lacuna_metadata(
    text: str,
    n_chars: int | tuple[int, int] | None = None,
    char_tolerance: int = 0,
) -> dict[str, Any]:
    """
    Analizza un contesto con lacuna `[...]` o `[---]`:
    - Isola la prima lacuna come target e converte eventuali lacune secondarie in `<UNK>`.
    - Estrae `min_chars`, `max_chars`, `center_chars`.
    - Estrae `word_prefix` e/o `word_suffix` se la lacuna è interna a una parola.
    """
    parts = re.split(r"(\[.*?\])", text)
    cleaned_loose_dots = "".join(
        re.sub(r"\.{2,}", f" {UNK_TOKEN} ", part) if i % 2 == 0 else part
        for i, part in enumerate(parts)
    )
    processed = process_editorial_marks(cleaned_loose_dots, preserve_lacunae=True)

    match = TARGET_LACUNA_REGEX.search(processed)
    if not match:
        raise ValueError("Nessuna lacuna target `[...]` o `[---]` trovata nel testo.")

    extracted_len = len(match.group(1))
    if n_chars is None:
        min_chars = max(1, extracted_len - char_tolerance)
        max_chars = extracted_len + char_tolerance
        center_chars = extracted_len
    elif isinstance(n_chars, (tuple, list)):
        min_chars = max(1, int(n_chars[0]))
        max_chars = max(min_chars, int(n_chars[1]))
        center_chars = (min_chars + max_chars) // 2
    else:
        min_chars = max(1, int(n_chars) - char_tolerance)
        max_chars = int(n_chars) + char_tolerance
        center_chars = int(n_chars)

    # Estrazione prefisso e/o suffisso adiacenti alla prima lacuna
    word_prefix = ""
    word_suffix = ""
    affix_match = AFFIX_LACUNA_REGEX.search(processed)
    if affix_match and affix_match.start(2) == match.start(1):
        word_prefix = affix_match.group(1)
        word_suffix = affix_match.group(3)
        if word_prefix.upper() == "UNK" or UNK_TOKEN in word_prefix:
            word_prefix = ""
        if word_suffix.upper() == "UNK" or UNK_TOKEN in word_suffix:
            word_suffix = ""

    # Normalizzazione delle lacune secondarie a <UNK>, preservando la prima lacuna target
    prefix_ctx = processed[: match.start()]
    target_repr = f"[{'.' * center_chars}]"
    suffix_ctx = processed[match.end() :]

    suffix_ctx = TARGET_LACUNA_REGEX.sub(f" {UNK_TOKEN} ", suffix_ctx)
    suffix_ctx = re.sub(r"\.{2,}", f" {UNK_TOKEN} ", suffix_ctx)
    normalized_context = f"{prefix_ctx}{target_repr}{suffix_ctx}"
    normalized_context = re.sub(r"\s+", " ", normalized_context).strip()

    return {
        "context_with_gap": normalized_context,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "center_chars": center_chars,
        "word_prefix": word_prefix,
        "word_suffix": word_suffix,
        "is_partial_word": bool(word_prefix or word_suffix),
    }


def sanitize_greek_supplement(
    raw_supplement: str,
    word_prefix: str = "",
    word_suffix: str = "",
    min_chars: int = 1,
    max_chars: int = 20,
) -> str:
    """
    Pulisce un supplemento proposto dall'LLM:
    - Rimuove parentesi quadre, virgolette, punteggiatura e caratteri non alfabetici.
    - Scarta candidati con caratteri latini o numerici.
    - Se l'LLM ha accidentalmente restituito l'intera parola ricostruita includendo
      `word_prefix` e/o `word_suffix`, rimuove gli affissi ridondanti per isolare la lacuna.
    """
    if not raw_supplement or not isinstance(raw_supplement, str):
        return ""

    s = raw_supplement.strip()
    s = re.sub(r"^[\[\(\{«\"'`]+|[\]\)\}»\"'`]+$", "", s).strip()
    s = s.replace("##", "").replace("Ġ", "").replace("Ċ", "").strip()

    # Se contiene più parole o spiegazioni dopo un trattino/due punti, prendi il primo token greco
    if " " in s or ":" in s or "-" in s:
        s = re.split(r"[\s:\-—,;]+", s)[0].strip()

    # Rimuove punteggiatura residua
    s = re.sub(r"[.,;:!?'\"()\[\]{}|—_`~«»<>\\/*+&^%$#@=0-9]", "", s).strip()

    # Scarta intrusioni di caratteri latini
    if re.search(r"[a-zA-Z]", s):
        return ""

    if not s or not any(c.isalpha() for c in s):
        return ""

    # Gestione rimozione affissi se l'LLM ha restituito la parola intera invece del solo supplemento
    base_s = normalize_greek(s, case_folding="fold", strip_diacritics_flag=True)
    base_prefix = (
        normalize_greek(word_prefix, case_folding="fold", strip_diacritics_flag=True)
        if word_prefix
        else ""
    )
    base_suffix = (
        normalize_greek(word_suffix, case_folding="fold", strip_diacritics_flag=True)
        if word_suffix
        else ""
    )

    if len(base_s) > max_chars and (base_prefix or base_suffix):
        # Allineiamo la rimozione sui caratteri base privi di diacritici
        s_nobase = strip_diacritics(s)
        start_cut = 0
        end_cut = len(s_nobase)
        if base_prefix and base_s.startswith(base_prefix):
            start_cut = len(base_prefix)
        if base_suffix and base_s.endswith(base_suffix) and (end_cut - start_cut) > len(base_suffix):
            end_cut = len(s_nobase) - len(base_suffix)
        trimmed = s_nobase[start_cut:end_cut]
        if min_chars <= len(strip_diacritics(trimmed)) <= max_chars:
            s = trimmed

    return s


def coerce_dspy_candidates(raw_candidates: Any) -> list[GreekCandidate]:
    """
    Converte in modo resiliente l'output `candidates` del predittore DSPy
    (che può essere `list[GreekCandidate]`, `list[dict]`, `list[str]` o `str`)
    in una lista tipizzata di `GreekCandidate`.
    """
    if not raw_candidates:
        return []

    if isinstance(raw_candidates, str):
        # Fallback se un modello piccolo restituisce una lista separata da virgole o righe
        tokens = [
            t.strip()
            for t in re.split(r"[,;\n]+", raw_candidates)
            if t.strip()
        ]
        return [
            GreekCandidate(
                supplement=t,
                confidence=max(0.1, 1.0 - idx * 0.05),
            )
            for idx, t in enumerate(tokens)
        ]

    coerced: list[GreekCandidate] = []
    if isinstance(raw_candidates, (list, tuple)):
        for idx, item in enumerate(raw_candidates):
            default_conf = max(0.05, 1.0 - idx * 0.04)
            if isinstance(item, GreekCandidate):
                coerced.append(item)
            elif isinstance(item, dict):
                supp = str(
                    item.get("supplement")
                    or item.get("token_str")
                    or item.get("candidate")
                    or item.get("text")
                    or ""
                )
                conf = float(item.get("confidence", item.get("score", default_conf)))
                conf = min(1.0, max(0.0, conf))
                if supp:
                    coerced.append(GreekCandidate(supplement=supp, confidence=conf))
            elif isinstance(item, str):
                coerced.append(GreekCandidate(supplement=item, confidence=default_conf))
    return coerced


def extract_parallel_attestations(
    context_with_gap: str,
    retrieved_passages: list[str],
    min_chars: int,
    max_chars: int,
    word_prefix: str = "",
    word_suffix: str = "",
) -> list[GreekCandidate]:
    """
    Ispeziona i passi paralleli (*loci similes*) recuperati da TLG/MAAT per estrarre
    attestazioni lessicali dirette che soddisfano i vincoli morfologici (`word_prefix`
    e/o `word_suffix` o parole adiacenti alla lacuna) e la lunghezza `[min_chars, max_chars]`.
    """
    if not retrieved_passages:
        return []

    base_prefix = (
        normalize_greek(word_prefix, case_folding="fold", strip_diacritics_flag=True)
        if word_prefix
        else ""
    )
    base_suffix = (
        normalize_greek(word_suffix, case_folding="fold", strip_diacritics_flag=True)
        if word_suffix
        else ""
    )

    # Individua la parola immediatamente a sinistra e a destra della lacuna nel contesto
    gap_match = TARGET_LACUNA_REGEX.search(context_with_gap)
    left_neighbor = ""
    right_neighbor = ""
    if gap_match:
        left_part = context_with_gap[: gap_match.start()]
        right_part = context_with_gap[gap_match.end() :]
        if word_prefix and left_part.endswith(word_prefix):
            left_part = left_part[: -len(word_prefix)]
        if word_suffix and right_part.startswith(word_suffix):
            right_part = right_part[len(word_suffix) :]

        left_words = [
            normalize_greek(w, case_folding="fold", strip_diacritics_flag=True)
            for w in re.findall(r"[^\W\d_]+", left_part)
            if w.upper() != "UNK"
        ]
        right_words = [
            normalize_greek(w, case_folding="fold", strip_diacritics_flag=True)
            for w in re.findall(r"[^\W\d_]+", right_part)
            if w.upper() != "UNK"
        ]
        if left_words:
            left_neighbor = left_words[-1]
        if right_words:
            right_neighbor = right_words[0]

    extracted: list[GreekCandidate] = []
    seen_supps: set[str] = set()

    for p_idx, raw_passage in enumerate(retrieved_passages):
        clean_p = _strip_passage_header(raw_passage)
        words = [
            w for w in re.findall(r"[^\W\d_]+", clean_p)
            if not re.search(r"[a-zA-Z0-9]", w)
        ]
        norm_words = [
            normalize_greek(w, case_folding="fold", strip_diacritics_flag=True)
            for w in words
        ]

        for w_idx, norm_w in enumerate(norm_words):
            # Caso 1: Lacuna intra-parola con prefisso e/o suffisso
            if base_prefix or base_suffix:
                if base_prefix and not norm_w.startswith(base_prefix):
                    continue
                if base_suffix and not norm_w.endswith(base_suffix):
                    continue
                start_i = len(base_prefix)
                end_i = len(norm_w) - len(base_suffix) if base_suffix else len(norm_w)
                if end_i <= start_i:
                    continue
                supp = norm_w[start_i:end_i]
                if min_chars <= len(supp) <= max_chars and supp not in seen_supps:
                    seen_supps.add(supp)
                    # Bonus se anche i vicini contestuali coincidono
                    neighbor_bonus = 0.0
                    if left_neighbor and w_idx > 0 and norm_words[w_idx - 1] == left_neighbor:
                        neighbor_bonus += 0.15
                    if right_neighbor and w_idx + 1 < len(norm_words) and norm_words[w_idx + 1] == right_neighbor:
                        neighbor_bonus += 0.15
                    conf = min(0.95, max(0.35, 0.75 - p_idx * 0.05 + neighbor_bonus))
                    extracted.append(GreekCandidate(supplement=supp, confidence=conf))

            # Caso 2: Lacuna inter-parola (parola intera affiancata da left_neighbor o right_neighbor)
            else:
                if not (min_chars <= len(norm_w) <= max_chars):
                    continue
                left_match = bool(left_neighbor and w_idx > 0 and norm_words[w_idx - 1] == left_neighbor)
                right_match = bool(
                    right_neighbor and w_idx + 1 < len(norm_words) and norm_words[w_idx + 1] == right_neighbor
                )
                if (left_match or right_match) and norm_w not in seen_supps:
                    seen_supps.add(norm_w)
                    conf = 0.85 if (left_match and right_match) else max(0.30, 0.65 - p_idx * 0.05)
                    extracted.append(GreekCandidate(supplement=norm_w, confidence=conf))

    return extracted


class GreekRAGInfiller(dspy.Module):
    """
    Modulo DSPy principale per il Text-Infilling in Greco Antico assistito da RAG.

    Nel costruttore `__init__` istanzia e registra i predittori basati sulle Signature
    (`self.generate_infill` e `self.refine_infill`), permettendo agli ottimizzatori DSPy
    (`BootstrapFewShot`, `MIPROv2`) di compilarne automaticamente le istruzioni e gli
    esempi few-shot (`demos`).
    """

    def __init__(
        self,
        retriever: GreekHybridRetriever | None = None,
        top_k_passages: int = 5,
        enable_refinement: bool = True,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.top_k_passages = top_k_passages
        self.enable_refinement = enable_refinement

        # Predittori dichiarativi gestiti dal modulo DSPy:
        # GreekInfillingSignature include già `philological_reasoning` come primo OutputField
        # (Chain-of-Thought esplicito seguito da `candidates`).
        self.generate_infill = dspy.Predict(GreekInfillingSignature)
        self.refine_infill = dspy.Predict(ConstraintRefinementSignature)

    def forward(
        self,
        context_with_gap: str,
        min_chars: int | None = None,
        max_chars: int | None = None,
        word_prefix: str | None = None,
        word_suffix: str | None = None,
        num_candidates: int = 20,
        exclude_query_overlap: bool = False,
    ) -> dspy.Prediction:
        # 1. Estrazione automatica dei vincoli se non specificati esplicitamente
        meta = extract_lacuna_metadata(context_with_gap)
        clean_context = meta["context_with_gap"]
        req_min = meta["min_chars"] if min_chars is None else int(min_chars)
        req_max = meta["max_chars"] if max_chars is None else int(max_chars)
        w_prefix = meta["word_prefix"] if word_prefix is None else word_prefix
        w_suffix = meta["word_suffix"] if word_suffix is None else word_suffix

        gap_constraints_str = format_gap_constraints(
            min_chars=req_min,
            max_chars=req_max,
            word_prefix=w_prefix,
            word_suffix=w_suffix,
        )

        # 2. Retrieval Ibrido dei passi paralleli (loci similes) da TLG e MAAT
        retrieved_passages: list[str] = []
        if self.retriever is not None:
            ret_pred = self.retriever(
                query=clean_context,
                k=self.top_k_passages,
                word_prefix=w_prefix,
                word_suffix=w_suffix,
                exclude_query_overlap=exclude_query_overlap,
            )
            retrieved_passages = list(getattr(ret_pred, "passages", []) or [])

        # 3. Generazione Chain-of-Thought tramite GreekInfillingSignature
        gen_pred = self.generate_infill(
            context_with_gap=clean_context,
            gap_constraints=gap_constraints_str,
            retrieved_passages=retrieved_passages,
            num_candidates=num_candidates,
        )

        raw_candidates = coerce_dspy_candidates(getattr(gen_pred, "candidates", []))
        reasoning = getattr(gen_pred, "philological_reasoning", "") or ""

        valid_candidates: list[GreekCandidate] = []
        invalid_candidates_desc: list[str] = []
        fallback_candidates: list[GreekCandidate] = []

        for cand in raw_candidates:
            sanitized = sanitize_greek_supplement(
                cand.supplement,
                word_prefix=w_prefix,
                word_suffix=w_suffix,
                min_chars=req_min,
                max_chars=req_max,
            )
            if not sanitized:
                continue
            cand_len = len(strip_diacritics(sanitized))
            cleaned_cand = GreekCandidate(supplement=sanitized, confidence=cand.confidence)
            if req_min <= cand_len <= req_max:
                valid_candidates.append(cleaned_cand)
            else:
                invalid_candidates_desc.append(f"{sanitized} ({cand_len} lettere)")
                fallback_candidates.append(cleaned_cand)

        # 4. Self-Refinement condizionato se nessun candidato (o troppo pochi) rispetta la lunghezza
        if (
            self.enable_refinement
            and invalid_candidates_desc
            and len(valid_candidates) < max(1, min(3, num_candidates))
        ):
            try:
                ref_pred = self.refine_infill(
                    context_with_gap=clean_context,
                    gap_constraints=gap_constraints_str,
                    retrieved_passages=retrieved_passages,
                    previous_invalid_candidates=invalid_candidates_desc[:10],
                    num_candidates=num_candidates,
                )
                refined_raw = coerce_dspy_candidates(getattr(ref_pred, "candidates", []))
                for cand in refined_raw:
                    sanitized = sanitize_greek_supplement(
                        cand.supplement,
                        word_prefix=w_prefix,
                        word_suffix=w_suffix,
                        min_chars=req_min,
                        max_chars=req_max,
                    )
                    if not sanitized:
                        continue
                    cand_len = len(strip_diacritics(sanitized))
                    cleaned_cand = GreekCandidate(supplement=sanitized, confidence=cand.confidence)
                    if req_min <= cand_len <= req_max:
                        valid_candidates.append(cleaned_cand)
                    else:
                        fallback_candidates.append(cleaned_cand)
            except Exception:
                pass

        # 5. Integrazione con attestazioni dirette dai passi paralleli recuperati
        corpus_attestations = extract_parallel_attestations(
            context_with_gap=clean_context,
            retrieved_passages=retrieved_passages,
            min_chars=req_min,
            max_chars=req_max,
            word_prefix=w_prefix,
            word_suffix=w_suffix,
        )
        valid_candidates.extend(corpus_attestations)

        return dspy.Prediction(
            philological_reasoning=reasoning,
            candidates=valid_candidates,
            fallback_candidates=fallback_candidates,
            retrieved_passages=retrieved_passages,
            gap_constraints=gap_constraints_str,
            min_chars=req_min,
            max_chars=req_max,
            word_prefix=w_prefix,
            word_suffix=w_suffix,
        )
