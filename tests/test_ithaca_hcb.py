"""
Unit test per la pipeline di Ithaca e l'algoritmo di decodifica HCB Beam Search.
Valida la normalizzazione, la tokenizzazione con SOS, la generazione multi-policy
delle lacune e la logica di correzione probabilistica HCB.
"""

from __future__ import annotations

import re
import numpy as np
import pytest

from models.ithaca.dataset.prepare_tlg import (
    clean_to_ithaca_alphabet,
    chunk_text,
    inject_lacuna,
    GREEK_CHARS,
)
from models.ithaca.finetuning.checkpoint import load_ithaca_checkpoint
from models.ithaca.finetuning.train import encode_sequence
from models.ithaca.inference.predict import decode_gap_hcb_beam_search


class DummyAlphabet:
    """Alfabeto fittizio per test unitari offline."""

    def __init__(self):
        self.pad = "#"
        self.sos = "<"
        self.unk = "^"
        self.space = " "
        self.missing = "-"
        greek_chars = list("αβγδεζηθικλμνξοπρςστυφχψωϙϛ")
        numerals = list("0")
        punctuation = list(".")
        self.idx2char = np.array(
            [self.pad, self.sos, self.unk, self.space, self.missing]
            + greek_chars
            + numerals
            + punctuation
        )
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}
        self.idx2word = np.array([self.pad, self.sos, self.unk, "εδοξεν", "βουληι", "δημος"])
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


def test_clean_to_ithaca_alphabet():
    """Verifica che il testo venga pulito mantenendo caratteri greci minuscoli e marcatori ammessi."""
    raw = "Ἔδοξεν τῇ βουλῇ! 123 -- text"
    cleaned = clean_to_ithaca_alphabet(raw.lower())
    assert "123" in cleaned
    assert "--" in cleaned
    # I caratteri alfabetici non conformi o punteggiatura non ammessa vengono rimossi o normalizzati
    for char in cleaned:
        assert char in GREEK_CHARS or char in "0123456789 .-"


def test_inject_lacuna_policies():
    """Verifica la generazione controllata delle lacune per policy 'default', 'word' e 'suffix'."""
    text = "εδοξεν τηι βουληι και τωι δημωι"

    for policy in ["default", "word", "suffix"]:
        case = inject_lacuna(text, policy=policy, min_gap=1, max_gap=4)
        assert case is not None
        assert case["policy"] == policy
        assert case["gap_length"] >= 1

        # Formato Ithaca: trattini puri senza parentesi quadre
        assert "[" not in case["text_ithaca"]
        assert "]" not in case["text_ithaca"]
        assert "-" in case["text_ithaca"]

        # Formato Leiden: puntini racchiusi da parentesi quadre
        assert "[" in case["text_leiden"]
        assert "]" in case["text_leiden"]
        assert "." in case["text_leiden"]

        # La lunghezza dei trattini deve corrispondere a gap_length
        hyphens_count = case["text_ithaca"].count("-")
        assert hyphens_count == case["gap_length"]


def test_encode_sequence_sos_and_words():
    """Verifica che encode_sequence prefigga il token SOS e allinei correttamente char_ids e word_ids."""
    alphabet = DummyAlphabet()
    text = "εδοξεν ---- δημωι"

    char_ids, word_ids, mask_pos = encode_sequence(text, alphabet, max_len=30)

    # 1. Il primo carattere deve essere SOS ('<')
    assert char_ids[0] == alphabet.char2idx[alphabet.sos]

    # 2. Le posizioni della lacuna devono essere segnalate in mask_pos
    assert np.sum(mask_pos) == 4
    mask_indices = np.where(mask_pos == 1.0)[0]
    for idx in mask_indices:
        assert char_ids[idx] == alphabet.char2idx[alphabet.missing]

    # 3. La parola conosciuta 'εδοξεν' deve avere il suo ID assegnato
    edoxen_id = alphabet.word2idx["εδοξεν"]
    # Troviamo dove 'εδοξεν' è presente (dopo il token SOS)
    assert char_ids[1] == alphabet.char2idx["ε"]
    assert word_ids[1] == edoxen_id
    assert word_ids[6] == edoxen_id


def test_decode_gap_hcb_beam_search_synthetic():
    """Verifica l'algoritmo HCB beam search con una mock forward pass."""
    alphabet = DummyAlphabet()
    max_len = 20
    char_ids = np.full((1, max_len), alphabet.char2idx[alphabet.pad], dtype=np.int32)
    word_ids = np.full((1, max_len), alphabet.word2idx[alphabet.unk], dtype=np.int32)

    # Inseriamo una lacuna di 2 caratteri alle posizioni 2 e 3
    char_ids[0, 0] = alphabet.char2idx[alphabet.sos]
    char_ids[0, 1] = alphabet.char2idx["ε"]
    char_ids[0, 2] = alphabet.char2idx[alphabet.missing]
    char_ids[0, 3] = alphabet.char2idx[alphabet.missing]
    char_ids[0, 4] = alphabet.char2idx["α"]

    mask_indices = [2, 3]

    # Mock forward function che assegna alta probabilità a 'β' in pos 2 e 'γ' in pos 3
    def mock_forward(params, text_char, text_word):
        batch_size = text_char.shape[0]
        vocab_size = len(alphabet.idx2char)
        # Inizializza log-prob uniformi basse
        log_probs = np.full((batch_size, max_len, vocab_size), -10.0, dtype=np.float32)

        beta_idx = alphabet.char2idx["β"]
        gamma_idx = alphabet.char2idx["γ"]
        mask_idx = alphabet.char2idx[alphabet.missing]

        # Logit per il pivot della maschera
        log_probs[:, :, mask_idx] = -3.0

        for b in range(batch_size):
            # Posizione 2 predilige 'β'
            log_probs[b, 2, beta_idx] = 1.0
            # Posizione 3 predilige 'γ' se pos 2 è già 'β'
            if text_char[b, 2] == beta_idx:
                log_probs[b, 3, gamma_idx] = 2.0
            else:
                log_probs[b, 3, gamma_idx] = -1.0

        return log_probs

    # Eseguiamo il beam search HCB
    suggestions = decode_gap_hcb_beam_search(
        forward_fn=mock_forward,
        params=None,
        char_ids_initial=char_ids,
        word_ids_initial=word_ids,
        mask_indices=mask_indices,
        alphabet=alphabet,
        K=5,
        beam_width=10,
        strategy="hcb_best_to_worst",
    )

    assert len(suggestions) > 0
    # La predizione top-1 deve essere 'βγ'
    top_cand, top_score = suggestions[0]
    assert top_cand == "βγ"
    assert 0.0 <= top_score <= 1.0

    # La somma delle probabilità dei top suggerimenti deve essere circa 1.0
    total_prob = sum(s for _, s in suggestions)
    assert pytest.approx(total_prob, 0.01) == 1.0


def test_fill_mask_ithaca_interface(monkeypatch):
    """Verifica che fill_mask_ithaca parsi sia Leiden che trattini e restituisca sempre minuscolo."""
    from models.ithaca.inference.predict import fill_mask_ithaca
    import models.ithaca.inference.predict as pred_module

    alphabet = DummyAlphabet()

    def mock_get_or_load(ckpt):
        def dummy_forward(params, text_char, text_word):
            batch_size = text_char.shape[0]
            max_len = text_char.shape[1]
            vocab_size = len(alphabet.idx2char)
            # Logit alti su 'α' per semplicità
            log_p = np.full((batch_size, max_len, vocab_size), -5.0, dtype=np.float32)
            alpha_idx = alphabet.char2idx["α"]
            log_p[:, :, alpha_idx] = 2.0
            return log_p

        return (None, None, alphabet, {"max_len": 64}, dummy_forward)

    monkeypatch.setattr(pred_module, "get_or_load_ithaca", mock_get_or_load)

    # 1. Test input tutto maiuscolo con convenzione Leiden [...]: output sempre minuscolo
    upper_input = "ΕΔΟΞΕΝ [..] ΔΗΜΩΙ"
    res_upper = fill_mask_ithaca(upper_input, checkpoint="dummy", K=2)
    assert len(res_upper) > 0
    assert res_upper[0][0] == "αα"

    # 2. Test input con maiuscola iniziale / casing misto: output sempre minuscolo
    mixed_input = "Ἔδοξεν τῇ βουλῇ [..] δήμῳ"
    res_mixed = fill_mask_ithaca(mixed_input, checkpoint="dummy", K=2)
    assert len(res_mixed) > 0
    assert res_mixed[0][0] == "αα"

    # 3. Test input minuscolo con trattini
    lower_input = "εδοξεν [..] δημωι"
    res_lower = fill_mask_ithaca(lower_input, checkpoint="dummy", K=2)
    assert len(res_lower) > 0
    assert res_lower[0][0] == "αα"

    # 4. Test eccezione su input senza lacuna
    with pytest.raises(ValueError):
        fill_mask_ithaca("εδοξεν χωρις κενου δημωι", checkpoint="dummy")


def test_decode_gap_excludes_digits_punctuation_and_archaic_chars():
    """Verifica l'esclusione di '0', '.', 'ϙ', 'ϛ' e token di controllo in entrambe le strategie HCB."""
    from models.ithaca.inference.predict import STANDARD_GREEK_LOWER_CHARS

    alphabet = DummyAlphabet()
    max_len = 16
    char_ids = np.full((1, max_len), alphabet.char2idx[alphabet.pad], dtype=np.int32)
    word_ids = np.full((1, max_len), alphabet.word2idx[alphabet.unk], dtype=np.int32)
    char_ids[0, 0] = alphabet.char2idx[alphabet.sos]
    char_ids[0, 1] = alphabet.char2idx["ε"]
    char_ids[0, 2] = alphabet.char2idx[alphabet.missing]
    char_ids[0, 3] = alphabet.char2idx[alphabet.missing]
    char_ids[0, 4] = alphabet.char2idx["α"]

    def mock_forward(params, text_char, text_word):
        batch_size = text_char.shape[0]
        vocab_size = len(alphabet.idx2char)
        log_probs = np.full((batch_size, max_len, vocab_size), -10.0, dtype=np.float32)
        for forbidden in ("0", ".", "ϙ", "ϛ", "#", "^", "-", " "):
            log_probs[:, :, alphabet.char2idx[forbidden]] = 10.0
        log_probs[:, :, alphabet.char2idx["α"]] = 1.0
        log_probs[:, :, alphabet.char2idx["β"]] = 0.5
        return log_probs

    for strategy in ("hcb_best_to_worst", "hcb_left_to_right"):
        suggestions = decode_gap_hcb_beam_search(
            forward_fn=mock_forward,
            params=None,
            char_ids_initial=char_ids,
            word_ids_initial=word_ids,
            mask_indices=[2, 3],
            alphabet=alphabet,
            K=5,
            beam_width=10,
            strategy=strategy,
        )
        assert len(suggestions) > 0
        assert suggestions[0][0] == "αα"
        assert pytest.approx(sum(s for _, s in suggestions), 1e-4) == 1.0
        for cand, _ in suggestions:
            assert all(ch in STANDARD_GREEK_LOWER_CHARS for ch in cand)


def test_decode_gap_well_formed_internal_spaces_only():
    """Verifica che gli spazi siano vietati ai bordi/lacune <= 2 e ammessi solo internamente senza doppi."""
    alphabet = DummyAlphabet()
    max_len = 20

    def _run_for_gap(gap_len: int):
        char_ids = np.full((1, max_len), alphabet.char2idx[alphabet.pad], dtype=np.int32)
        word_ids = np.full((1, max_len), alphabet.word2idx[alphabet.unk], dtype=np.int32)
        char_ids[0, 0] = alphabet.char2idx[alphabet.sos]
        char_ids[0, 1] = alphabet.char2idx["ε"]
        mask_indices = list(range(2, 2 + gap_len))
        for pos in mask_indices:
            char_ids[0, pos] = alphabet.char2idx[alphabet.missing]
        char_ids[0, 2 + gap_len] = alphabet.char2idx["α"]

        def mock_forward(params, text_char, text_word):
            batch_size = text_char.shape[0]
            vocab_size = len(alphabet.idx2char)
            log_probs = np.full((batch_size, max_len, vocab_size), -10.0, dtype=np.float32)
            log_probs[:, :, alphabet.char2idx[" "]] = 10.0
            log_probs[:, :, alphabet.char2idx["α"]] = 1.0
            log_probs[:, :, alphabet.char2idx["β"]] = 0.5
            return log_probs

        return decode_gap_hcb_beam_search(
            forward_fn=mock_forward,
            params=None,
            char_ids_initial=char_ids,
            word_ids_initial=word_ids,
            mask_indices=mask_indices,
            alphabet=alphabet,
            K=10,
            beam_width=16,
            strategy="hcb_best_to_worst",
        )

    # Lacune di 1 o 2 caratteri: nessuno spazio ammesso
    for short_len in (1, 2):
        short_suggs = _run_for_gap(short_len)
        assert len(short_suggs) > 0
        for cand, _ in short_suggs:
            assert len(cand) == short_len
            assert " " not in cand

    # Lacuna di 4 caratteri: ammessi solo spazi singoli strettamente interni
    long_suggs = _run_for_gap(4)
    assert len(long_suggs) > 0
    assert pytest.approx(sum(s for _, s in long_suggs), 1e-4) == 1.0
    for cand, _ in long_suggs:
        assert len(cand) == 4
        assert not cand.startswith(" ")
        assert not cand.endswith(" ")
        assert "  " not in cand
    assert any(" " in cand for cand, _ in long_suggs)


def test_decode_gap_contextual_sigma_normalization_and_logaddexp_merge():
    """Verifica la normalizzazione contestuale di 'σ'/'ς' e la fusione logaddexp dei duplicati."""
    alphabet = DummyAlphabet()
    max_len = 16

    # 1. Fine parola (seguito da spazio): 'τοσ' + 'τος' devono fondersi in 'τος' superando 'τον'
    char_ids_end = np.full((1, max_len), alphabet.char2idx[alphabet.pad], dtype=np.int32)
    word_ids = np.full((1, max_len), alphabet.word2idx[alphabet.unk], dtype=np.int32)
    char_ids_end[0, 0] = alphabet.char2idx[alphabet.sos]
    char_ids_end[0, 1] = alphabet.char2idx[" "]
    char_ids_end[0, 2] = alphabet.char2idx[alphabet.missing]
    char_ids_end[0, 3] = alphabet.char2idx[alphabet.missing]
    char_ids_end[0, 4] = alphabet.char2idx[alphabet.missing]
    char_ids_end[0, 5] = alphabet.char2idx[" "]

    def mock_forward_sigma(params, text_char, text_word):
        batch_size = text_char.shape[0]
        vocab_size = len(alphabet.idx2char)
        log_probs = np.full((batch_size, max_len, vocab_size), -20.0, dtype=np.float32)
        log_probs[:, :, alphabet.char2idx[alphabet.missing]] = -3.0
        log_probs[:, 2, alphabet.char2idx["τ"]] = 5.0
        log_probs[:, 3, alphabet.char2idx["ο"]] = 5.0
        log_probs[:, 4, alphabet.char2idx["ν"]] = 0.0
        log_probs[:, 4, alphabet.char2idx["σ"]] = -0.3
        log_probs[:, 4, alphabet.char2idx["ς"]] = -0.3
        return log_probs

    suggs_end = decode_gap_hcb_beam_search(
        forward_fn=mock_forward_sigma,
        params=None,
        char_ids_initial=char_ids_end,
        word_ids_initial=word_ids,
        mask_indices=[2, 3, 4],
        alphabet=alphabet,
        K=5,
        beam_width=10,
        strategy="hcb_left_to_right",
    )
    cands_end = dict(suggs_end)
    assert "τοσ" not in cands_end
    assert "τος" in cands_end
    assert "τον" in cands_end
    # La somma probabilistica logaddexp(-0.3, -0.3) supera 'τον' (0.0)
    assert suggs_end[0][0] == "τος"
    expected_tos_prob = float(
        np.exp(np.logaddexp(-0.3, -0.3))
        / (np.exp(np.logaddexp(-0.3, -0.3)) + np.exp(0.0))
    )
    assert pytest.approx(cands_end["τος"], 1e-4) == expected_tos_prob

    # 2. Prima di una lettera greca nel suffisso ('α'): il sigma resta mediano ('σ')
    char_ids_mid = char_ids_end.copy()
    char_ids_mid[0, 5] = alphabet.char2idx["α"]
    suggs_mid = decode_gap_hcb_beam_search(
        forward_fn=mock_forward_sigma,
        params=None,
        char_ids_initial=char_ids_mid,
        word_ids_initial=word_ids,
        mask_indices=[2, 3, 4],
        alphabet=alphabet,
        K=5,
        beam_width=10,
        strategy="hcb_left_to_right",
    )
    cands_mid = dict(suggs_mid)
    assert "τος" not in cands_mid
    assert "τοσ" in cands_mid
    assert suggs_mid[0][0] == "τοσ"

    # 3. Spazio interno alla lacuna: 'τοσ κα' deve diventare 'τος κα'
    char_ids_multi = np.full((1, max_len), alphabet.char2idx[alphabet.pad], dtype=np.int32)
    char_ids_multi[0, 0] = alphabet.char2idx[alphabet.sos]
    for pos in range(2, 8):
        char_ids_multi[0, pos] = alphabet.char2idx[alphabet.missing]

    def mock_forward_multi(params, text_char, text_word):
        batch_size = text_char.shape[0]
        vocab_size = len(alphabet.idx2char)
        log_probs = np.full((batch_size, max_len, vocab_size), -20.0, dtype=np.float32)
        log_probs[:, :, alphabet.char2idx[alphabet.missing]] = -3.0
        for pos, ch in zip(range(2, 8), ["τ", "ο", "σ", " ", "κ", "α"]):
            log_probs[:, pos, alphabet.char2idx[ch]] = 5.0
        return log_probs

    suggs_multi = decode_gap_hcb_beam_search(
        forward_fn=mock_forward_multi,
        params=None,
        char_ids_initial=char_ids_multi,
        word_ids_initial=word_ids,
        mask_indices=list(range(2, 8)),
        alphabet=alphabet,
        K=3,
        beam_width=5,
        strategy="hcb_left_to_right",
    )
    assert suggs_multi[0][0] == "τος κα"


def test_postprocess_ithaca_candidates_defensive_filter():
    """Verifica filtraggio difensivo, minuscolo, normalizzazione sigma e merge in postprocess_ithaca_candidates."""
    from models.ithaca.inference.predict import postprocess_ithaca_candidates

    raw = [
        ("ΑΓΑΘΟΣ", -1.0),
        ("α0β", -0.1),
        ("α.β", -0.1),
        ("αϙβ", -0.1),
        ("αϛβ", -0.1),
        (" και", -0.1),
        ("και ", -0.1),
        ("α  β", -0.1),
        ("   ", -0.1),
        ("τοσ", -2.0),
        ("τος", -2.0),
        ("τοσ και", -3.0),
    ]

    res = postprocess_ithaca_candidates(raw, next_char_after_gap=" ", K=3)
    assert [c for c, _ in res] == ["αγαθος", "τος", "τος και"]
    assert pytest.approx(sum(p for _, p in res), 1e-6) == 1.0

    # Verifica che il merge di 'τοσ' + 'τος' usi logaddexp(-2.0, -2.0) = -2.0 + ln(2)
    merged_log = float(np.logaddexp(-2.0, -2.0))
    exp_scores = np.array([np.exp(-1.0 - (-1.0)), np.exp(merged_log - (-1.0)), np.exp(-3.0 - (-1.0))])
    expected_probs = exp_scores / exp_scores.sum()
    for (_, actual_p), exp_p in zip(res, expected_probs):
        assert pytest.approx(actual_p, 1e-5) == float(exp_p)

    # Se il carattere dopo la lacuna è una lettera greca maiuscola/minuscola, sigma finale diventa 'σ'
    res_before_letter = postprocess_ithaca_candidates([("τος", -1.0)], next_char_after_gap="Α", K=5)
    assert len(res_before_letter) == 1
    assert res_before_letter[0][0] == "τοσ"
    assert pytest.approx(res_before_letter[0][1], 1e-6) == 1.0


def test_fill_mask_ithaca_default_beam_headroom(monkeypatch):
    """Verifica il margine automatico del beam (max(20, K + 4)) e il rispetto di beam_size esplicito."""
    from models.ithaca.inference.predict import fill_mask_ithaca
    import models.ithaca.inference.predict as pred_module

    alphabet = DummyAlphabet()
    captured_beam_widths: list[int] = []

    monkeypatch.setattr(
        pred_module,
        "get_or_load_ithaca",
        lambda ckpt: (None, None, alphabet, {"max_len": 64}, lambda *a, **kw: None),
    )

    def spy_decode(**kwargs):
        captured_beam_widths.append(kwargs["beam_width"])
        return [("αα", 1.0)]

    monkeypatch.setattr(pred_module, "decode_gap_hcb_beam_search", spy_decode)

    fill_mask_ithaca("εδοξεν [..] δημωι", checkpoint="dummy", K=20)
    fill_mask_ithaca("εδοξεν [..] δημωι", checkpoint="dummy", K=5)
    fill_mask_ithaca("εδοξεν [..] δημωι", checkpoint="dummy", K=20, beam_size=12)

    assert captured_beam_widths == [24, 20, 12]


def test_ithaca_compare_metrics_and_export(tmp_path):
    """Verifica il calcolo di CER, la formattazione della tabella e l'estrazione righe comparativa."""
    import json
    import csv
    from models.ithaca.evaluation.compare import (
        compute_cer,
        render_comparison_table,
        extract_comparison_rows,
    )

    # 1. Test CER
    assert compute_cer("AB", "AB") == 0.0
    assert compute_cer("A", "B") == 1.0
    assert compute_cer("", "") == 0.0

    # 2. Test render_comparison_table e extract_comparison_rows
    dummy_pre = {
        "overall": {"top1_acc": 0.30, "top5_acc": 0.50, "top20_acc": 0.70, "mrr": 0.40, "mean_cer": 0.50, "count": 10},
        "default": {"top1_acc": 0.35, "top5_acc": 0.55, "top20_acc": 0.75, "mrr": 0.45, "mean_cer": 0.45, "count": 10},
    }
    dummy_post = {
        "overall": {"top1_acc": 0.45, "top5_acc": 0.65, "top20_acc": 0.85, "mrr": 0.55, "mean_cer": 0.35, "count": 10},
        "default": {"top1_acc": 0.50, "top5_acc": 0.70, "top20_acc": 0.90, "mrr": 0.60, "mean_cer": 0.30, "count": 10},
    }

    table_md = render_comparison_table(dummy_pre, dummy_post)
    assert "| OVERALL | Top-1 Exact Match |" in table_md
    assert "30.00%" in table_md
    assert "45.00%" in table_md

    rows = extract_comparison_rows(dummy_pre, dummy_post)
    assert len(rows) > 0
    top1_overall = [r for r in rows if r["policy"] == "overall" and r["metric_key"] == "top1_acc"][0]
    assert pytest.approx(top1_overall["delta"], 0.001) == 0.15

    # 3. Test esportazione file JSON, CSV e Markdown
    json_path = tmp_path / "comparison.json"
    csv_path = tmp_path / "comparison.csv"
    md_path = tmp_path / "comparison.md"

    # JSON export
    report = {
        "pre_metrics": dummy_pre,
        "post_metrics": dummy_post,
        "comparison": rows,
        "markdown_table": table_md,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert "markdown_table" in loaded
        assert len(loaded["comparison"]) == len(rows)

    # CSV export
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "metric", "metric_key", "pre_ft", "post_ft", "delta"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ["policy", "metric", "metric_key", "pre_ft", "post_ft", "delta"]})
    assert csv_path.exists()

    # MD export
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    assert md_path.exists()


def test_create_optimizer_with_selective_layer_freezing():
    """Verifica che create_optimizer_with_freezing congeli i primi N encoderblock, gli embedding e le teste di attribuzione."""
    import jax
    import jax.numpy as jnp
    import optax
    from models.ithaca.finetuning.checkpoint import create_optimizer_with_freezing

    dummy_params = {
        "params": {
            "char_embeddings": {"embedding": jnp.ones((10, 4))},
            "word_embeddings": {"embedding": jnp.ones((20, 4))},
            "posembed_input": {"pe": jnp.ones((1, 8, 4))},
            "encoderblock_0": {"kernel": jnp.ones((4, 4))},
            "encoderblock_1": {"kernel": jnp.ones((4, 4))},
            "encoderblock_2": {"kernel": jnp.ones((4, 4))},
            "MlpBlock_0": {"kernel": jnp.ones((4, 4))},  # Testa di restauro (trainable)
            "MlpBlock_2": {"kernel": jnp.ones((4, 4))},  # Testa datazione (frozen)
            "MlpBlock_3": {"kernel": jnp.ones((4, 4))},  # Testa regione (frozen)
        }
    }

    opt = create_optimizer_with_freezing(
        learning_rate=1e-3,
        warmup_steps=1,
        total_steps=10,
        weight_decay=1e-2,
        lr_scheduler_type="linear",
        num_layers_to_freeze=2,
        freeze_attribution_heads=True,
        params=dummy_params,
    )
    opt_state = opt.init(dummy_params)

    # Gradiente unitario ovunque (eseguiamo 2 step per superare lo step 0 del warmup lineare)
    grads = jax.tree_util.tree_map(jnp.ones_like, dummy_params)
    updates, opt_state = opt.update(grads, opt_state, dummy_params)
    updates, opt_state = opt.update(grads, opt_state, dummy_params)

    # 1. Layer congelati (num_layers_to_freeze=2 -> encoderblock_0, encoderblock_1 + embeddings + teste attribuzione)
    assert jnp.all(updates["params"]["char_embeddings"]["embedding"] == 0.0)
    assert jnp.all(updates["params"]["word_embeddings"]["embedding"] == 0.0)
    assert jnp.all(updates["params"]["encoderblock_0"]["kernel"] == 0.0)
    assert jnp.all(updates["params"]["encoderblock_1"]["kernel"] == 0.0)
    assert jnp.all(updates["params"]["MlpBlock_2"]["kernel"] == 0.0)
    assert jnp.all(updates["params"]["MlpBlock_3"]["kernel"] == 0.0)

    # 2. Layer addestrabili (encoderblock_2 e MlpBlock_0)
    assert jnp.any(updates["params"]["encoderblock_2"]["kernel"] != 0.0)
    assert jnp.any(updates["params"]["MlpBlock_0"]["kernel"] != 0.0)


def test_get_ithaca_contextual_embeddings_and_cluster():
    """Verifica l'estrazione degli embedding contestuali 1D dal torso di Ithaca e il calcolo del cluster denso."""
    from models.ithaca.inference.predict import get_ithaca_contextual_embeddings
    from models.bert.evaluation.metrics import (
        evaluate_contextual_similarity,
        check_gold_in_dense_cluster,
    )

    alphabet = DummyAlphabet()

    def dummy_torso_fn(params, text_char, text_word):
        batch_size, max_len = text_char.shape
        # Restituisce un vettore torso deterministico dipendente dagli ID dei caratteri
        base = text_char.astype(np.float32)[..., None]
        return np.repeat(base + 1.0, repeats=16, axis=-1)

    text_with_gap = "εδοξεν [....] δημωι"
    candidates = ["βουλ", "δημο", "πολι", "βουλ"]  # l'ultimo è la gold label

    embs = get_ithaca_contextual_embeddings(
        text_with_gap=text_with_gap,
        candidates=candidates,
        params_override={},
        alphabet_override=alphabet,
        config_override={"max_len": 32},
        torso_fn_override=dummy_torso_fn,
    )

    assert len(embs) == 4
    for t in embs:
        assert t.ndim == 1
        assert t.shape[0] == 16

    sims = evaluate_contextual_similarity(embs[:-1], embs[-1])
    assert len(sims) == 3
    assert pytest.approx(sims[0], 1e-4) == 1.0

    cluster_res = check_gold_in_dense_cluster(embs[:-1], embs[-1])
    assert "is_inside_cluster" in cluster_res
    assert "inclusion_margin" in cluster_res


def test_log_ithaca_comparison_to_wandb_tables(monkeypatch):
    """Verifica che log_ithaca_comparison_to_wandb crei le tabelle confronto_pre_post_ft_{policy} e predictions_table."""
    import sys
    import types
    from models.ithaca.evaluation.compare import log_ithaca_comparison_to_wandb

    logged_data: list[dict] = []

    class MockTable:
        def __init__(self, columns=None, data=None):
            self.columns = columns or []
            self.data = list(data) if data else []

        def add_data(self, *row):
            self.data.append(list(row))

    mock_wandb = types.SimpleNamespace(
        run=object(),
        Table=MockTable,
        log=lambda payload: logged_data.append(payload),
    )
    monkeypatch.setitem(sys.modules, "wandb", mock_wandb)

    pre_m = {
        "overall": {"top1": 30.0, "top5": 50.0, "char_acc": 60.0, "mean_cer": 0.40, "mrr": 0.40, "cluster_inclusion_rate": 70.0},
        "default": {"top1": 35.0, "top5": 55.0, "char_acc": 65.0, "mean_cer": 0.35, "mrr": 0.45, "cluster_inclusion_rate": 75.0},
    }
    post_m = {
        "overall": {"top1": 45.0, "top5": 65.0, "char_acc": 75.0, "mean_cer": 0.25, "mrr": 0.55, "cluster_inclusion_rate": 85.0},
        "default": {"top1": 50.0, "top5": 70.0, "char_acc": 80.0, "mean_cer": 0.20, "mrr": 0.60, "cluster_inclusion_rate": 88.0},
    }
    preds_records = [
        {
            "policy": "default",
            "context": "ΕΔΟΞΕΝ [....] ΔΗΜΩΙ",
            "gold": "ΒΟΥΛ",
            "predictions": [("ΒΟΥΛ", 0.75), ("ΠΟΛΕ", 0.25)],
        }
    ]

    log_ithaca_comparison_to_wandb(
        pre_metrics_by_policy=pre_m,
        post_metrics_by_policy=post_m,
        predictions_records=preds_records,
        split_prefix="test",
    )

    merged_keys = {}
    for entry in logged_data:
        merged_keys.update(entry)

    assert "confronto_pre_post_ft_overall" in merged_keys
    assert "confronto_pre_post_ft_default" in merged_keys
    assert "test_post_ft_default/predictions_table" in merged_keys
    assert "test_post_ft_overall/predictions_table" in merged_keys

    overall_table = merged_keys["confronto_pre_post_ft_overall"]
    assert overall_table.columns == ["Metrica", "Pre-FT", "Post-FT", "Delta"]
    em1_row = [r for r in overall_table.data if r[0] == "Exact Match @1"][0]
    assert em1_row == ["Exact Match @1", 30.0, 45.0, 15.0]

    pred_table = merged_keys["test_post_ft_default/predictions_table"]
    assert "**ΒΟΥΛ** (0.7500)" in pred_table.data[0][2]


def test_nsga2_ithaca_helpers():
    """Verifica gli helper di optuna_nsga2 per il riconoscimento di Ithaca e la formattazione degli obiettivi."""
    from scripts.sweep.optuna_nsga2 import (
        is_ithaca_checkpoint,
        _format_objectives_tuple,
    )

    assert is_ithaca_checkpoint("CNR-ILC/gs-ithaca-tlg") is True
    assert is_ithaca_checkpoint("deepmind/ithaca") is True
    assert is_ithaca_checkpoint("CNR-ILC/gs-GreBerta") is False

    assert _format_objectives_tuple(40.0, 80.0, 75.0, 65.0, 50.0, "2d") == (40.0, 80.0)
    assert _format_objectives_tuple(40.0, 80.0, 75.0, 65.0, 50.0, "3d") == (40.0, 80.0, 75.0)
    assert _format_objectives_tuple(40.0, 80.0, 75.0, 65.0, 50.0, "em_cer") == (40.0, 65.0)
    assert _format_objectives_tuple(40.0, 80.0, 75.0, 65.0, 50.0, "em_cer_mrr") == (40.0, 65.0, 50.0)


@pytest.mark.asyncio
async def test_suggestions_service_predict_ithaca_lowercases_output(monkeypatch):
    """Verifica che SuggestionsService._predict_ithaca normalizzi in minuscolo token_str e reconstructed_word preservando il casing del contesto in sentence."""
    from unittest.mock import AsyncMock, MagicMock
    from bson import ObjectId

    from backend.api.models import ModelType, PredictionCount
    from backend.api.services.suggestions_service import SuggestionsService
    import models.ithaca.inference.predict as pred_module

    mock_collection = MagicMock()
    ithaca_id = ObjectId()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": ithaca_id,
            "CHECKPOINT": "CNR-ILC/gs-ithaca-tlg",
            "TYPE": ModelType.ITHACA,
        }
    )

    sugg_service = SuggestionsService(
        db_collection=mock_collection,
        gridfs=MagicMock(),
        feedback_db=MagicMock(),
    )

    monkeypatch.setattr(
        pred_module,
        "fill_mask_ithaca",
        lambda **kwargs: [("ΑΓΑΘΟΣ", 0.90), ("ΚαΛοΝ", 0.10)],
    )

    preds = await sugg_service.get_predictions(
        model_id=str(ithaca_id),
        context="Ἔδοξεν τῇ βουλῇ [......]",
        num_tokens=1,
        num_predictions=PredictionCount.FIVE,
    )

    assert len(preds) == 2
    assert preds[0]["token_str"] == "αγαθος"
    assert preds[0]["sentence"] == "Ἔδοξεν τῇ βουλῇ αγαθος"
    assert preds[0]["reconstructed_word"] == "αγαθος"
    assert preds[1]["token_str"] == "καλον"
    assert preds[1]["sentence"] == "Ἔδοξεν τῇ βουλῇ καλον"
    assert preds[1]["reconstructed_word"] == "καλον"


@pytest.mark.asyncio
async def test_suggestions_service_preserves_uppercase_context_around_partial_lacuna(monkeypatch):
    """Verifica che la ricostruzione e la frase restituita gestiscano prefissi/suffissi maiuscoli attorno alla lacuna."""
    from unittest.mock import AsyncMock, MagicMock
    from bson import ObjectId

    from backend.api.models import ModelType, PredictionCount
    from backend.api.services.suggestions_service import SuggestionsService
    import models.ithaca.inference.predict as pred_module

    mock_collection = MagicMock()
    ithaca_id = ObjectId()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": ithaca_id,
            "CHECKPOINT": "CNR-ILC/gs-ithaca-tlg",
            "TYPE": ModelType.ITHACA,
        }
    )

    sugg_service = SuggestionsService(
        db_collection=mock_collection,
        gridfs=MagicMock(),
        feedback_db=MagicMock(),
    )

    monkeypatch.setattr(
        pred_module,
        "fill_mask_ithaca",
        lambda **kwargs: [("ΛΗΤΑ", 0.95)],
    )

    preds = await sugg_service.get_predictions(
        model_id=str(ithaca_id),
        context="Ἆρες Ἆρες ΤΕΙΧΕΣΙΒ[....]",
        num_tokens=1,
        num_predictions=PredictionCount.FIVE,
    )

    assert len(preds) == 1
    assert preds[0]["token_str"] == "λητα"
    assert preds[0]["sentence"] == "Ἆρες Ἆρες ΤΕΙΧΕΣΙΒλητα"
    assert preds[0]["reconstructed_word"] == "τειχεσιβλητα"


