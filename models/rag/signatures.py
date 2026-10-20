"""
Signature dichiarative di DSPy per il Text-Infilling in Greco Antico.

Definisce i contratti di Input/Output (`dspy.Signature`) e gli schemi Pydantic
che guidano gli LLM (es. Qwen, Llama) a integrare le lacune papirologiche (`[...]`)
sfruttando i passi paralleli (*loci similes*) recuperati da TLG e MAAT e rispettando
i vincoli di lunghezza in caratteri e gli eventuali prefissi e/o suffissi intra-parola.
"""

from __future__ import annotations

import dspy
from pydantic import BaseModel, Field


class GreekCandidate(BaseModel):
    """Singolo supplemento candidato proposto dall'LLM per colmare la lacuna `[...]`."""

    supplement: str = Field(
        description=(
            "Solo le lettere greche che colmano la lacuna [...], "
            "senza parentesi quadre, senza punteggiatura e senza ripetere prefisso o suffisso."
        )
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Grado di confidenza filologica, grammaticale e contestuale tra 0.0 e 1.0.",
    )


class GreekInfillingSignature(dspy.Signature):
    """Sei un esperto di papirologia greca antica (Papiri di Ercolano) e filologia classica (TLG).
    Il tuo compito è proporre supplementi in greco antico per colmare la lacuna indicata da `[...]`
    nel frammento fornito, sfruttando i passi paralleli (loci similes) recuperati dai corpora antichi
    e rispettando rigorosamente i vincoli di lunghezza in caratteri e la morfologia della parola.
    Restituisci esclusivamente i caratteri greci che sostituiscono `[...]`."""

    context_with_gap: str = dspy.InputField(
        desc="Frammento di testo in greco antico contenente la lacuna target indicata da `[...]`."
    )
    gap_constraints: str = dspy.InputField(
        desc=(
            "Vincoli papirologici: lunghezza attesa della lacuna in caratteri greci (es. esattamente 4 lettere), "
            "ed eventuale prefisso (word_prefix) e/o suffisso (word_suffix) se la lacuna è interna a una parola "
            "(in tal caso `word_prefix + supplement + word_suffix` deve formare una parola greca valida)."
        )
    )
    retrieved_passages: list[str] = dspy.InputField(
        desc="Passi paralleli (loci similes) recuperati dal Thesaurus Linguae Graecae (TLG) e dai papiri MAAT."
    )
    num_candidates: int = dspy.InputField(
        desc="Numero K di supplementi candidati distinti da generare, ordinati dal più probabile al meno probabile."
    )

    philological_reasoning: str = dspy.OutputField(
        desc=(
            "Analisi sintetica grammaticale, sintattica e lessicale del contesto e dei passi paralleli "
            "per giustificare l'integrazione e verificare il numero di lettere."
        )
    )
    candidates: list[GreekCandidate] = dspy.OutputField(
        desc="Lista ordinata di K supplementi candidati in greco antico che rispettano la lunghezza della lacuna."
    )


class ConstraintRefinementSignature(dspy.Signature):
    """Alcuni supplementi proposti in precedenza non rispettavano il numero esatto di lettere greche
    richiesto dalla lacuna papirologica. Correggi e genera nuovi supplementi in greco antico che
    rispettino rigorosamente il conteggio di caratteri specificato in `gap_constraints`."""

    context_with_gap: str = dspy.InputField(
        desc="Frammento in greco antico con lacuna `[...]`."
    )
    gap_constraints: str = dspy.InputField(
        desc="Vincoli rigorosi sul numero di caratteri e prefisso e/o suffisso."
    )
    retrieved_passages: list[str] = dspy.InputField(
        desc="Passi paralleli di riferimento."
    )
    previous_invalid_candidates: list[str] = dspy.InputField(
        desc="Candidati precedenti scartati perché di lunghezza errata (con indicazione della loro lunghezza)."
    )
    num_candidates: int = dspy.InputField(
        desc="Numero K di candidati validi richiesti."
    )

    candidates: list[GreekCandidate] = dspy.OutputField(
        desc="Lista corretta di supplementi in greco antico della lunghezza richiesta."
    )


def format_gap_constraints(
    min_chars: int,
    max_chars: int,
    word_prefix: str = "",
    word_suffix: str = "",
) -> str:
    """
    Formatta in una stringa esplicita per l'LLM i vincoli papirologici della lacuna
    (conteggio caratteri ed eventuale prefisso e/o suffisso intra-parola).
    """
    if min_chars == max_chars:
        len_desc = f"esattamente {min_chars} lettere greche (senza contare spiriti o accenti)"
    else:
        len_desc = f"tra {min_chars} e {max_chars} lettere greche (senza contare spiriti o accenti)"

    if word_prefix and word_suffix:
        morph_desc = (
            f"Lacuna INTERNA a una parola: prefisso conservato a sinistra = '{word_prefix}', "
            f"suffisso conservato a destra = '{word_suffix}'. "
            f"La sequenza completa `{word_prefix}<supplemento>{word_suffix}` deve formare una parola greca valida; "
            f"restituisci SOLO `<supplemento>`."
        )
    elif word_prefix:
        morph_desc = (
            f"Lacuna INTRA-PAROLA finale: prefisso conservato a sinistra = '{word_prefix}'. "
            f"La sequenza `{word_prefix}<supplemento>` deve completare la parola greca; "
            f"restituisci SOLO `<supplemento>` (non ripetere '{word_prefix}')."
        )
    elif word_suffix:
        morph_desc = (
            f"Lacuna INTRA-PAROLA iniziale: suffisso conservato a destra = '{word_suffix}'. "
            f"La sequenza `<supplemento>{word_suffix}` deve completare la parola greca; "
            f"restituisci SOLO `<supplemento>` (non ripetere '{word_suffix}')."
        )
    else:
        morph_desc = (
            "Lacuna INTER-PAROLA (parola intera mancante separata da spazi)."
        )

    return f"Lunghezza richiesta: {len_desc}. Tipo di lacuna: {morph_desc}"
