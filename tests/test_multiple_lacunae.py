"""
Unit test per la gestione uniforme di lacune multiple nel contesto di inferenza.
Verifica che tutti i modelli (N-grammi, MLM/BERT, Ithaca) e il servizio API
gestiscano coerentemente la presenza di più lacune ([...], <UNK>, ....).
"""

import pytest
import numpy as np

from backend.core.preprocess import test_case_contains_lacuna as check_contains_lacuna
from models.ngrams.metrics.utils import get_context_from_test_case


class DummyAlphabet:
    """Alfabeto di prova per Ithaca con token speciali."""

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


def test_check_contains_lacuna_multiple():
    """Verifica che test_case_contains_lacuna restituisca la prima lacuna anche con lacune multiple."""
    assert check_contains_lacuna("testo senza lacuna") is None
    assert check_contains_lacuna("testo [....] singolo") == "[....]"
    assert check_contains_lacuna("prima [..] seconda [....] terza [...]") == "[..]"
    assert check_contains_lacuna("trattini [---] e punti [..]") == "[---]"


def test_ngrams_get_context_from_test_case_multiple():
    """Verifica che get_context_from_test_case gestisca lacune multiple e mantenga la prima come target."""
    context_text = "ΟΙ ΜΕΝ [..] ΚΑΙ ΟΙ [....] ΤΟΥ ΣΟΦΟΥ"
    context, head, tail, lacuna_len = get_context_from_test_case(context_text, n=3)

    assert lacuna_len == 2
    # Il contesto estratto a sinistra deve contenere i token precedenti alla prima lacuna
    assert len(context) > 0
    # Head e tail per lacuna inter-parola isolata devono essere vuoti o None
    assert not head
    assert not tail

    # Test con lacuna intra-parola e seconda lacuna nel contesto
    context_intra = "ΚΑΙ Φ[..]ΕΡΩΤΕΡΟΝ ΚΑΤΑ [....] ΤΟΥΤΩΝ"
    ctx_i, head_i, tail_i, len_i = get_context_from_test_case(context_intra, n=3)
    assert len_i == 2
    assert head_i == "Φ"
    assert tail_i == "ΕΡΩΤΕΡΟΝ"


def test_ithaca_multiple_lacunae_handling(monkeypatch):
    """Verifica che Ithaca ripristini solo la prima lacuna e tratti le successive come UNK."""
    from models.ithaca.inference.predict import fill_mask_ithaca
    import models.ithaca.inference.predict as pred_module

    alphabet = DummyAlphabet()

    def mock_get_or_load(ckpt):
        def dummy_forward(params, text_char, text_word):
            batch_size = text_char.shape[0]
            max_len = text_char.shape[1]
            vocab_size = len(alphabet.idx2char)
            # Logit alti su 'α'
            log_p = np.full((batch_size, max_len, vocab_size), -5.0, dtype=np.float32)
            alpha_idx = alphabet.char2idx["α"]
            log_p[:, :, alpha_idx] = 2.0
            return log_p

        return (None, None, alphabet, {"max_len": 64}, dummy_forward)

    monkeypatch.setattr(pred_module, "get_or_load_ithaca", mock_get_or_load)

    # Testo con due lacune: prima di 2 punti, seconda di 4 punti
    text_multiple = "ΕΔΟΞΕΝ [..] ΚΑΙ [....] ΔΗΜΩΙ"
    suggestions = fill_mask_ithaca(text_multiple, checkpoint="dummy", K=2)

    assert len(suggestions) > 0
    # Deve suggerire 2 caratteri per la prima lacuna ('ΑΑ'), non 6 caratteri concatenando la seconda
    cand_str, score = suggestions[0]
    assert len(cand_str) == 2
    assert cand_str == "ΑΑ"

    # Testo con lacuna target e token <UNK> / puntini liberi
    text_with_unk = "ΕΔΟΞΕΝ [..] ΤΗΙ <UNK> ΒΟΥΛΗΙ .... ΔΗΜΩΙ"
    suggestions_unk = fill_mask_ithaca(text_with_unk, checkpoint="dummy", K=2)
    assert len(suggestions_unk) > 0
    assert len(suggestions_unk[0][0]) == 2


def test_bert_multiple_lacunae_handling(monkeypatch):
    """Verifica che BERT fill_mask converta la prima lacuna in [MASK] e le secondarie in [UNK]."""
    import torch
    import models.bert.inference.predict as bert_predict

    captured_masked_texts = []

    def mock_decode(model, input_ids, attention_mask, beam_size, mask_id):
        return [[(0.0, 10, 11)]]

    monkeypatch.setattr(bert_predict, "decode_modified_BestToWorst_vectorized", mock_decode)

    class DummyTokenizer:
        mask_token = "[MASK]"
        mask_token_id = 103
        unk_token = "[UNK]"
        unk_token_id = 100
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0
        pad_token = "[PAD]"
        model_max_length = 512
        is_fast = False

        def __call__(self, text, **kwargs):
            captured_masked_texts.append(text)
            return {"input_ids": [103]}

        def decode(self, token_ids, **kwargs):
            return "αγαθος"

    class DummyModel:
        def eval(self):
            pass

        def parameters(self):
            yield torch.zeros(1)

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    text = "λογον [....] και [..] του σοφου .... τελος"
    res = bert_predict.fill_mask(
        text=text,
        model=dummy_model,
        tokenizer=dummy_tokenizer,
        checkpoint=None,
        n_chars=4,
        K=1,
    )

    assert len(res) > 0
    assert len(captured_masked_texts) > 0
    masked = captured_masked_texts[0]
    assert "[MASK]" in masked
    assert "[UNK]" in masked
    assert "[..]" not in masked

