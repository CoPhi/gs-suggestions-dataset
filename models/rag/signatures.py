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


class CorpusAttestation(BaseModel):
    """Riferimento puntuale al documento del corpus in cui è attestato un supplemento."""

    corpus_id: str = Field(
        default="unknown",
        description="Identificativo della collezione/corpus (es. 'tlg', 'DCLP', 'DDbDP', 'PHerc.')",
    )
    source_id: str = Field(
        default="",
        description="Titolo dell'opera, papiro o luogo di attestazione nel corpus",
    )
    attested_form: str = Field(
        default="",
        description="Parola o forma greca completa attestata nel documento",
    )
    passage: str = Field(
        default="",
        description="Estratto testuale del documento in cui compare l'attestazione",
    )
    highlighted_passage: str = Field(
        default="",
        description="Estratto testuale del documento con la forma attestata messa in evidenza in grassetto (**...**)",
    )
    doc_freq: int = Field(
        default=1,
        ge=1,
        description="Numero totale di documenti nel corpus che attestano la forma",
    )
    source_type: str = Field(
        default="retrieved_passage",
        description="Origine dell'attestazione ('retrieved_passage' dai loci similes o 'corpus_index' dall'indice del corpus)",
    )


class GreekCandidate(BaseModel):
    """Singolo supplemento candidato proposto dall'LLM o dal retriever per colmare la lacuna `[...]`."""

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
    rationale: str = Field(
        default="",
        description=(
            "CoT di secondo livello (1 frase breve, max 15-20 parole): spiega perché questo "
            "specifico supplemento è plausibile per morfologia, sintassi, semantica o attestazione nei passi paralleli."
        ),
    )
    attestations: list[CorpusAttestation] = Field(
        default_factory=list,
        description="Lista opzionale dei documenti del corpus che attestano direttamente il supplemento.",
    )


class GreekInfillingSignature(dspy.Signature):
    """Sei un esperto di papirologia greca antica (Papiri di Ercolano) e filologia classica (TLG).
    Il tuo compito è proporre supplementi in greco antico per colmare la lacuna indicata da `[...]`
    nel frammento fornito, sfruttando i passi paralleli (loci similes) recuperati dai corpora antichi
    e rispettando rigorosamente i vincoli di lunghezza in caratteri e la morfologia della parola.
    Articola il ragionamento su due livelli:
    1) `philological_reasoning` (CoT di 1° livello): analisi logica e filologica di come interpreti il contesto
       e le informazioni estratte dai documenti rilevanti del retriever per inquadrare la lacuna;
    2) `rationale` in ciascun `GreekCandidate` (CoT di 2° livello): spiegazione puntuale e concisa del perché
       quella singola integrazione è plausibile per colmare la lacuna.
    Restituisci in `supplement` esclusivamente i caratteri greci che sostituiscono `[...]`."""

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
        desc="Passi paralleli (loci similes) recuperati dal Thesaurus Linguae Graecae (TLG) e dai blocchi anonimi MAAT."
    )
    num_candidates: int = dspy.InputField(
        desc="Numero K di supplementi candidati distinti da generare, ordinati dal più probabile al meno probabile."
    )

    philological_reasoning: str = dspy.OutputField(
        desc=(
            "CoT di 1° livello (2-3 frasi): ragionamento logico e filologico su come vengono interpretate "
            "le informazioni dei documenti rilevanti restituiti dal retriever e la sintassi del passo per colmare la lacuna."
        )
    )
    candidates: list[GreekCandidate] = dspy.OutputField(
        desc=(
            "Lista ordinata di K supplementi candidati in greco antico che rispettano la lunghezza della lacuna, "
            "ciascuno corredato dal proprio `rationale` (CoT di 2° livello)."
        )
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
    human_feedback: str = "",
    rejected_candidates: list[str] | None = None,
) -> str:
    """
    Formatta in una stringa esplicita per l'LLM i vincoli papirologici della lacuna
    (conteggio caratteri, eventuale prefisso e/o suffisso intra-parola e vincoli
    Human-in-the-loop forniti dal filologo nella sessione corrente).
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

    base_constraints = f"Lunghezza richiesta: {len_desc}. Tipo di lacuna: {morph_desc}"

    # Inietta nel campo gap_constraints eventuali indicazioni filologiche o esclusioni
    # esplicite (Human-in-the-loop) senza alterare la firma compilata del modulo DSPy.
    hitl_parts: list[str] = []
    if human_feedback and human_feedback.strip():
        hitl_parts.append(f"Indicazione del filologo: {human_feedback.strip()}.")
    if rejected_candidates:
        cleaned_rejected = [c.strip() for c in rejected_candidates if c and c.strip()]
        if cleaned_rejected:
            hitl_parts.append(
                f"Supplementi già scartati dal filologo (NON proporli): {', '.join(cleaned_rejected)}."
            )

    if hitl_parts:
        return f"{base_constraints} {' '.join(hitl_parts)}"
    return base_constraints
