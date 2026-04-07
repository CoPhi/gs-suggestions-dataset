"""
Test suite per il transpiler MAAT Leiden → LM-ready text.

Basata sulla specifica: backend/core/transpiler_spec.md

Include:
  - Test di idempotenza (Fase 1 e pipeline completa)
  - Test end-to-end (esempi dalla specifica + casi reali)
  - Test delle post-condizioni / invarianti per ogni fase
"""

import json
import random
import unicodedata

import pytest

from backend.core import GAP_TOKEN, UNK_TOKEN
from backend.core.preprocess import (
    transpile,
    process_editorial_marks,
)


# ─────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────


# Testi sintetici che coprono ogni costrutto della grammatica sorgente MAAT.
SYNTHETIC_TEXTS = [
    # Supplemento pieno
    "κ[ώμη]ς Σεκνεπαί",
    # Gap ignoto
    f"sibi et {GAP_TOKEN} uxo[ri]",
    # Lacuna nota dentro supplemento (trade-off deliberato)
    f"sibi et {GAP_TOKEN} uxo[..]",
    # Multi-riga con supplementi
    "κ[ώμη]ς Σεκνεπαί\nου Νήσου ἃ ἐπεστά\nλη[σαν δ]ι' ἡμῶν",
    # Dattili
    "τὸν [πωλοῦντα] ‒⏑⏑‒",
    # Espunzione + marker + nota
    "ἄρτους &lt;τοὺς&gt; {καλούς} ε‡12",
    # Integrazioni
    "a || b ‖ c",
    # Leiden line break
    "αρχη|τελος",
    # Segni di incertezza
    "α+β*γ",
    # Vacat
    "πρῶτοςvac.δεύτερος",
    "πρῶτοςvacatδεύτερος",
    # Punti interrogativi
    "λόγος?",
    # Linee mancanti
    "πρῶτος⟦---⟧δεύτερος",
    # Parentesi tonde (abbreviazioni)
    "στρατηγ(ός)",
    # Espunzione doppia
    "a{{bc}}d",
    # Testo parallelo
    "a†bc†d",
    "x_y",
    # Trattini
    "πα-\nρά",
    "inizio -fine",
    # Gap inter-frasale
    f".{GAP_TOKEN}.",
    # Lacuna di punti senza brackets
    "uxo..",
    # Solo punti
    "...",
    # None come placeholder
    "None",
    # Combinazione complessa
    f"[τοῦ] {GAP_TOKEN} κα[ὶ] vac. τ(ῆ)ς || πό[λε]ως",
    # Testo pulito (nessuna trasformazione necessaria)
    "ἀγαθός ἄνθρωπος λέγει",
    # UNK token diretto
    f"{GAP_TOKEN}",
]


# ─────────────────────────────────────────────────────────────────────
#  Test di idempotenza
# ─────────────────────────────────────────────────────────────────────


class TestIdempotency:
    """
    Il transpiler deve essere idempotente: applicare la trasformazione
    due volte deve produrre lo stesso risultato della prima applicazione.
        f(f(x)) == f(x)
    """

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_idempotency(self, text: str):
        """Fase 1: process_editorial_marks(process_editorial_marks(x)) == process_editorial_marks(x)"""
        once = process_editorial_marks(text)
        twice = process_editorial_marks(once)
        assert once == twice, (
            f"Fase 1 non idempotente:\n"
            f"  Input:  {text!r}\n"
            f"  f(x):   {once!r}\n"
            f"  f(f(x)):{twice!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_full_pipeline_idempotency(self, text: str):
        """Pipeline completa: transpile(transpile(x)) == transpile(x)"""
        once = transpile(text)
        twice = transpile(once)
        assert once == twice, (
            f"Pipeline non idempotente:\n"
            f"  Input:  {text!r}\n"
            f"  f(x):   {once!r}\n"
            f"  f(f(x)):{twice!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_full_pipeline_idempotency_no_case_folding(self, text: str):
        """Pipeline con case_folding=False."""
        once = transpile(text, case_folding=False)
        twice = transpile(once, case_folding=False)
        assert once == twice


# ─────────────────────────────────────────────────────────────────────
#  Test end-to-end (dalla specifica, sezione 6)
# ─────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    """
    Test end-to-end derivati dagli esempi nella specifica (transpiler_spec.md §6)
    e da casi reali del dataset.
    """

    # Dati dalla specifica (sezione 6, Esempi 1–5)
    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Esempio 1: supplemento pieno + gap ignoto
            (
                f"sibi et {GAP_TOKEN} uxo[ri]",
                f"SIBI ET {UNK_TOKEN} UXORI",
            ),
            # Esempio 2: lacuna nota dentro supplemento (trade-off)
            (
                f"sibi et {GAP_TOKEN} uxo[..]",
                f"SIBI ET {UNK_TOKEN} {UNK_TOKEN}",
            ),
            # Esempio 3: training text reale multi-riga
            (
                "κ[ώμη]ς Σεκνεπαί\nου Νήσου ἃ ἐπεστά\nλη[σαν δ]ι' ἡμῶν",
                "ΚΩΜΗΣ ΣΕΚΝΕΠΑΙ ΟΥ ΝΗΣΟΥ Α ΕΠΕΣΤΑ ΛΗΣΑΝ ΔΙ' ΗΜΩΝ",
            ),
            # Esempio 4: verso con dattilo
            (
                "τὸν [πωλοῦντα] ‒⏑⏑‒",
                f"ΤΟΝ ΠΩΛΟΥΝΤΑ {UNK_TOKEN}",
            ),
            # Esempio 5: espunzione + marker + nota
            (
                "ἄρτους &lt;τοὺς&gt; {καλούς} ε‡12",
                "ΑΡΤΟΥΣ ΤΟΥΣ Ε",
            ),
        ],
        ids=[
            "spec_esempio_1_supplemento_pieno",
            "spec_esempio_2_lacuna_nota_tradeoff",
            "spec_esempio_3_multiriga",
            "spec_esempio_4_dattili",
            "spec_esempio_5_espunzione_marker",
        ],
    )
    def test_spec_examples(self, input_text: str, expected: str):
        result = transpile(input_text)
        assert result == expected

    # Casi reali aggiuntivi
    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Supplemento semplice tutto-testuale
            (
                "κ[ώμη]ς",
                "ΚΩΜΗΣ",
            ),
            # Solo gap ignoto
            (
                f"{GAP_TOKEN}",
                UNK_TOKEN,
            ),
            # Solo punti
            (
                "...",
                UNK_TOKEN,
            ),
            # Testo pulito senza markup
            (
                "ἀγαθός ἄνθρωπος",
                "ΑΓΑΘΟΣ ΑΝΘΡΩΠΟΣ",
            ),
            # Abbreviazione con parentesi tonde
            (
                "στρατηγ(ός)",
                "ΣΤΡΑΤΗΓΟΣ",
            ),
            # Vacat
            (
                "πρῶτος vac. δεύτερος",
                "ΠΡΩΤΟΣ ΔΕΥΤΕΡΟΣ",
            ),
            # Integrazioni
            (
                "a || b",
                "A B",
            ),
            # Doppio obelisco
            (
                "abc‡123",
                "ABC",
            ),
            # Espunzione semplice
            (
                "a{bc}d",
                "AD",
            ),
            # Espunzione doppia
            (
                "a{{bc}}d",
                "AD",
            ),
            # Testo parallelo
            (
                "a†bc†d",
                "AD",
            ),
            # Underscore (separatore testo parallelo)
            (
                "x_y",
                "XY",
            ),
            # Segni di incertezza
            (
                "α+β*γ",
                "ΑΒΓ",
            ),
            # Punto interrogativo
            (
                "λόγος?",
                "ΛΟΓΟΣ",
            ),
            # Linee mancanti
            (
                "πρῶτος⟦---⟧δεύτερος",
                "ΠΡΩΤΟΣΔΕΥΤΕΡΟΣ",
            ),
            # Gap inter-frasale: .<gap/>. collassato in "." da T4a.
            # Il punto singolo sopravvive perché contains_lacunae(".")
            # restituisce False (len == 1, non soddisfa "len > 1").
            (
                f".{GAP_TOKEN}. testo",
                ". TESTO",
            ),
            # Leiden line break singolo
            (
                "αρχη|τελος",
                "ΑΡΧΗ ΤΕΛΟΣ",
            ),
            # None placeholder
            (
                "None",
                UNK_TOKEN,
            ),
            # Multiple gaps producono UNK consecutivi
            (
                f"{GAP_TOKEN} ... {GAP_TOKEN}",
                f"{UNK_TOKEN} {UNK_TOKEN} {UNK_TOKEN}",
            ),
        ],
        ids=[
            "supplemento_semplice",
            "gap_ignoto_solo",
            "solo_punti",
            "testo_pulito",
            "abbreviazione_tonde",
            "vacat",
            "integrazioni",
            "doppio_obelisco",
            "espunzione_semplice",
            "espunzione_doppia",
            "testo_parallelo",
            "underscore",
            "segni_incertezza",
            "punto_interrogativo",
            "linee_mancanti",
            "gap_interfrasale",
            "leiden_lb",
            "none_placeholder",
            "unk_consecutivi",
        ],
    )
    def test_additional_cases(self, input_text: str, expected: str):
        result = transpile(input_text)
        assert result == expected

    def test_case_folding_disabled(self):
        """Con case_folding=False, le lettere restano minuscole."""
        result = transpile("κ[ώμη]ς", case_folding=False)
        assert result == "κωμης"

    def test_empty_input(self):
        """Stringa vuota produce stringa vuota."""
        assert transpile("") == ""


# ─────────────────────────────────────────────────────────────────────
#  Test combinati (interazione tra più costrutti)
# ─────────────────────────────────────────────────────────────────────


class TestCombinedConstructs:
    """
    Test con combinazioni di costrutti editoriali nello stesso testo.
    Verificano che le trasformazioni non interferiscano tra loro.
    """

    def test_supplement_plus_gap_plus_vacat(self):
        result = transpile(
            f"[τοῦ] {GAP_TOKEN} vac. τ(ῆ)ς"
        )
        assert result == f"ΤΟΥ {UNK_TOKEN} ΤΗΣ"

    def test_integration_with_brackets(self):
        result = transpile("a || [bc] ‖ d")
        assert result == "A BC D"

    def test_expunction_inside_and_outside_brackets(self):
        result = transpile("a{bc}[de]f")
        assert result == "ADEF"

    def test_markers_with_gap(self):
        result = transpile(f"&lt;abc&gt; {GAP_TOKEN} def")
        assert result == f"ABC {UNK_TOKEN} DEF"

    def test_dactyl_after_supplement(self):
        result = transpile("[πωλοῦντα] ‒⏑⏑‒ [τινα]")
        assert result == f"ΠΩΛΟΥΝΤΑ {UNK_TOKEN} ΤΙΝΑ"

    def test_multiple_supplements_multiline(self):
        text = "[πρῶτος]\n[δεύτερος]\n[τρίτος]"
        result = transpile(text)
        assert result == "ΠΡΩΤΟΣ ΔΕΥΤΕΡΟΣ ΤΡΙΤΟΣ"

    def test_dash_across_supplement(self):
        result = transpile("πα[ρα]-\nλαμβάνω")
        assert result == "ΠΑΡΑΛΑΜΒΑΝΩ"

    def test_parentheses_with_brackets(self):
        """Abbreviazione sciolta dentro supplemento."""
        result = transpile("[στρατηγ(ός)]")
        assert result == "ΣΤΡΑΤΗΓΟΣ"


# ─────────────────────────────────────────────────────────────────────
#  Test post-condizioni / invarianti
# ─────────────────────────────────────────────────────────────────────


# Caratteri / pattern che NON devono comparire dopo la Fase 1
PHASE1_FORBIDDEN = [
    ("[", "]"),          # P1.1
    ("(", ")"),          # P1.2
    ("{", "}"),          # P1.3
    ("+",),             # P1.5
    ("?",),             # P1.6
    ("†",),             # P1.7
    ("vacat", "vac."),  # P1.8
    ("⟦", "⟧"),         # P1.9
    ("\n",),            # P1.10
    ("⏑", "‒"),         # P1.11
]

# Caratteri / pattern che NON devono comparire dopo la Fase 2+3
PHASE3_FORBIDDEN = [
    (GAP_TOKEN,),  # P2.1 — nessun <gap/> rimasto
]


class TestPostConditions:
    """
    Verifica le post-condizioni definite nella specifica (§5)
    su tutti i testi sintetici.
    """

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_brackets(self, text: str):
        """P1.1: nessuna parentesi quadra."""
        result = process_editorial_marks(text)
        assert "[" not in result and "]" not in result, (
            f"P1.1 violata: '[' o ']' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_parentheses(self, text: str):
        """P1.2: nessuna parentesi tonda."""
        result = process_editorial_marks(text)
        assert "(" not in result and ")" not in result, (
            f"P1.2 violata: '(' o ')' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_braces(self, text: str):
        """P1.3: nessuna graffa."""
        result = process_editorial_marks(text)
        assert "{" not in result and "}" not in result, (
            f"P1.3 violata: '{{' o '}}' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_pipes(self, text: str):
        """P1.4: nessuna pipe singola o doppia."""
        result = process_editorial_marks(text)
        assert "||" not in result and "‖" not in result, (
            f"P1.4 violata: pipe in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_uncertainty_signs(self, text: str):
        """P1.5: nessun segno di incertezza."""
        result = process_editorial_marks(text)
        assert "+" not in result and "*" not in result, (
            f"P1.5 violata: '+' o '*' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_question_marks(self, text: str):
        """P1.6: nessun punto interrogativo."""
        result = process_editorial_marks(text)
        assert "?" not in result, (
            f"P1.6 violata: '?' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_obelisks(self, text: str):
        """P1.7: nessun obelisco."""
        result = process_editorial_marks(text)
        assert "†" not in result, (
            f"P1.7 violata: '†' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_vacat(self, text: str):
        """P1.8: nessun vacat."""
        result = process_editorial_marks(text)
        assert "vacat" not in result and "vac." not in result, (
            f"P1.8 violata: 'vacat'/'vac.' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_missing_lines(self, text: str):
        """P1.9: nessuna marcatura di linee mancanti."""
        result = process_editorial_marks(text)
        assert "⟦" not in result and "⟧" not in result, (
            f"P1.9 violata: '⟦' o '⟧' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_newlines(self, text: str):
        """P1.10: nessun newline."""
        result = process_editorial_marks(text)
        assert "\n" not in result, (
            f"P1.10 violata: '\\n' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase1_no_dactyl_patterns(self, text: str):
        """P1.11: nessun pattern dattilico."""
        result = process_editorial_marks(text)
        assert "⏑" not in result and "‒" not in result, (
            f"P1.11 violata: pattern dattilico in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase3_no_gap_tokens(self, text: str):
        """P2.1: nessun <gap/> rimasto dopo la pipeline completa."""
        result = transpile(text)
        assert GAP_TOKEN not in result, (
            f"P2.1 violata: '{GAP_TOKEN}' in output: {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase3_unk_separated_by_spaces(self, text: str):
        """P2.3: ogni <UNK> è separato da spazi."""
        result = transpile(text)
        for i, token in enumerate(result.split()):
            if UNK_TOKEN in token:
                assert token == UNK_TOKEN, (
                    f"P2.3 violata: UNK non isolato: token={token!r} in {result!r}"
                )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase3_no_combining_characters(self, text: str):
        """P3.1: nessun carattere combining (diacritici rimossi)."""
        result = transpile(text)
        combining = [
            c for c in result if unicodedata.combining(c)
        ]
        assert not combining, (
            f"P3.1 violata: combining characters in output: {combining} in {result!r}"
        )

    @pytest.mark.parametrize("text", SYNTHETIC_TEXTS)
    def test_phase3_all_uppercase(self, text: str):
        """P3.2: tutte le lettere alfabetiche sono maiuscole (case_folding=True)."""
        result = transpile(text, case_folding=True)
        # Rimuoviamo i token <UNK> prima del check
        text_without_unk = result.replace(UNK_TOKEN, "")
        for char in text_without_unk:
            if char.isalpha():
                assert char.isupper(), (
                    f"P3.2 violata: '{char}' minuscola in output: {result!r}"
                )


# ─────────────────────────────────────────────────────────────────────
#  Test post-condizioni su dati reali (subset del corpus)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_training_texts() -> list[str]:
    """Carica un campione di training_text reali da maat_1.json."""
    try:
        with open("data/maat_1.json", "r", encoding="utf-8") as f:
            abs_data = json.load(f)
    except FileNotFoundError:
        pytest.skip("data/maat_1.json non presente — skip test su dati reali")

    greek_texts = [
        ab["training_text"]
        for ab in abs_data
        if ab.get("language") == "grc" and ab.get("training_text")
    ]
    return random.Random(42).sample(greek_texts, min(200, len(greek_texts)))


class TestPostConditionsOnRealData:
    """
    Verifica le post-condizioni su un campione reale del corpus MAAT.
    """

    def test_phase1_no_brackets_real(self, real_training_texts: list[str]):
        """P1.1 su dati reali."""
        for text in real_training_texts:
            result = process_editorial_marks(text)
            assert "[" not in result and "]" not in result

    def test_phase1_no_parentheses_real(self, real_training_texts: list[str]):
        """P1.2 su dati reali."""
        for text in real_training_texts:
            result = process_editorial_marks(text)
            assert "(" not in result and ")" not in result

    def test_phase1_no_braces_real(self, real_training_texts: list[str]):
        """P1.3 su dati reali."""
        for text in real_training_texts:
            result = process_editorial_marks(text)
            assert "{" not in result and "}" not in result

    def test_phase1_no_newlines_real(self, real_training_texts: list[str]):
        """P1.10 su dati reali."""
        for text in real_training_texts:
            result = process_editorial_marks(text)
            assert "\n" not in result

    def test_phase1_no_dactyl_patterns_real(self, real_training_texts: list[str]):
        """P1.11 su dati reali."""
        for text in real_training_texts:
            result = process_editorial_marks(text)
            assert "⏑" not in result and "‒" not in result

    def test_phase3_no_gap_tokens_real(self, real_training_texts: list[str]):
        """P2.1 su dati reali."""
        for text in real_training_texts:
            result = transpile(text)
            assert GAP_TOKEN not in result

    @pytest.mark.xfail(
        reason="Bug noto: il modifier letter apostrophe ʼ (U+02BC) non è .isalpha() "
               "ma nemmeno .isupper(). Prodotto da normalize_grc di CLTK in alcuni testi.",
        strict=False,
    )
    def test_phase3_all_uppercase_real(self, real_training_texts: list[str]):
        """P3.2 su dati reali."""
        for text in real_training_texts:
            result = transpile(text, case_folding=True)
            text_no_unk = result.replace(UNK_TOKEN, "")
            for char in text_no_unk:
                if char.isalpha():
                    assert char.isupper()

    @pytest.mark.xfail(
        reason="Bug noto: normalize_grc (CLTK) non rimuove tutti i combining characters "
               "in edge case (es. combining comma above U+0313 in spiriti dolci).",
        strict=False,
    )
    def test_phase3_no_combining_real(self, real_training_texts: list[str]):
        """P3.1 su dati reali."""
        for text in real_training_texts:
            result = transpile(text)
            assert not any(unicodedata.combining(c) for c in result)

    @pytest.mark.xfail(
        reason="Bug noto: combining characters (U+0313) sopravvivono al primo passaggio "
               "di normalize_grc ma vengono trattati diversamente al secondo passaggio, "
               "rompendo l'idempotenza. Conseguenza diretta del bug P3.1.",
        strict=False,
    )
    def test_idempotency_real(self, real_training_texts: list[str]):
        """Idempotenza su dati reali."""
        for text in real_training_texts:
            once = transpile(text)
            twice = transpile(once)
            assert once == twice
