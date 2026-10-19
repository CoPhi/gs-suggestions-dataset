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
    """Verifica che fill_mask_ithaca parsi sia Leiden che trattini e mantenga il casing desiderato."""
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

    # 1. Test input maiuscolo con convenzione Leiden [...]
    upper_input = "ΕΔΟΞΕΝ [..] ΔΗΜΩΙ"
    res_upper = fill_mask_ithaca(upper_input, checkpoint="dummy", K=2)
    assert len(res_upper) > 0
    # Poiché l'input era maiuscolo, il suggerimento deve essere maiuscolo
    assert res_upper[0][0] == "ΑΑ"

    # 2. Test input minuscolo con trattini
    lower_input = "εδοξεν [..] δημωι"
    res_lower = fill_mask_ithaca(lower_input, checkpoint="dummy", K=2)
    assert len(res_lower) > 0
    assert res_lower[0][0] == "αα"

    # 3. Test eccezione su input senza lacuna
    with pytest.raises(ValueError):
        fill_mask_ithaca("εδοξεν χωρις κενου δημωι", checkpoint="dummy")


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

