"""
Unit test e test di parità numerica per le ottimizzazioni di memoria dell'ecosistema GreekSchools:
- Rimozione dipendenza runtime CLTK (`normalize_grc`, `GreekRegexSentenceTokenizer`)
- Cache BERT condivisa (`SharedBertCache`), lock anti-race ed eviction LRU/TTL
- Indice BM25 compatto CSR (`GreekLexicalBM25Index`) e migrazione da v1
- Vocabolario TLG compatto `mmap` (`CompactGreekVocab`)
- Storage SQLite compatto N-grammi (`CompactNgramLM` vs NLTK `MLE`/`Lidstone` + `ModelService`/`SuggestionsService`)
- Ithaca micro-batching fisso, lock di caricamento ed eviction TTL
"""

from __future__ import annotations

import pickle
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from bson import ObjectId
from nltk.lm import Lidstone, MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

from backend.api.models import ModelType, PredictionCount
from backend.api.services.model_service import ModelService
from backend.api.services.suggestions_service import SuggestionsService
from backend.core import GreekRegexSentenceTokenizer, normalize_grc
from backend.core.model_cache import SharedBertCache
from models.bert.inference.vocab import CompactGreekVocab, check_tlg_attestation
from models.ithaca.inference import predict as ithaca_predict
from models.ngrams.inference.suggests import generate_k_suggests
from models.ngrams.storage import (
    CompactNgramLM,
    export_nltk_models_to_sqlite,
    load_compact_ngram_pair,
)
from models.rag.retriever import GreekBertEmbedder, GreekLexicalBM25Index


def test_pure_python_greek_normalization_and_sentence_tokenizer():
    """Verifica che `normalize_grc` e `GreekRegexSentenceTokenizer` replichino esattamente CLTK."""
    raw = "ὅτι μὲν ὑμεῖς· ἄνδρες ά έ ή ί ό ύ ώ ‘λόγος’ “εἶπον”. τέρμα; τέλος"
    norm = normalize_grc(raw)
    # Virgolette tipografiche curve rimosse e normalizzazione Unicode NFKC applicata come in CLTK
    assert "‘" not in norm and "”" not in norm
    assert "λόγος εἶπον." in norm

    try:
        from cltk.alphabet.grc.grc import normalize_grc as cltk_normalize_grc
        from cltk.sentence.grc import GreekRegexSentenceTokenizer as CltkSentenceTokenizer

        assert norm == cltk_normalize_grc(raw)
        assert GreekRegexSentenceTokenizer().tokenize(raw) == CltkSentenceTokenizer().tokenize(raw)
    except ImportError:
        pass

    tokenizer = GreekRegexSentenceTokenizer()
    sents = tokenizer.tokenize("πρῶτος λόγος· δεύτερος λόγος. τρίτος; τέταρτος")
    assert sents == ["πρῶτος λόγος·", "δεύτερος λόγος.", "τρίτος;", "τέταρτος"]


def test_shared_bert_cache_concurrency_lru_and_ttl():
    """Verifica caricamento singolo sotto concorrenza, condivisione backbone, LRU e TTL."""
    cache = SharedBertCache(max_models=1)
    load_calls = 0
    call_lock = threading.Lock()
    barrier = threading.Barrier(4)

    class DummyMLM:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.base_model = MagicMock(name=f"base-{tag}")
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    def fake_loader(ckpt: str):
        nonlocal load_calls
        with call_lock:
            load_calls += 1
        time.sleep(0.02)
        return DummyMLM(ckpt), MagicMock(name=f"tok-{ckpt}")

    def worker():
        barrier.wait()
        return cache.get_or_load_mlm("ckpt-a", loader_fn=fake_loader)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: worker(), range(4)))

    assert load_calls == 1
    assert all(r[0] is results[0][0] for r in results)
    assert results[0][0].eval_called is True

    # get_encoder_backbone condivide lo stesso base_model senza ricaricare
    with patch("backend.core.model_cache.shared_bert_cache", cache):
        enc, _ = cache.get_encoder_backbone("ckpt-a")
        assert enc is results[0][0].base_model

    # LRU max_models=1: caricando ckpt-b viene rimosso ckpt-a
    cache.get_or_load_mlm("ckpt-b", loader_fn=fake_loader)
    assert "ckpt-a" not in cache.cache
    assert "ckpt-b" in cache.cache

    # TTL eviction
    cache._last_used["ckpt-b"] -= 120.0
    evicted = cache.evict_expired(ttl_seconds=60.0)
    assert evicted == ["ckpt-b"]
    assert len(cache.cache) == 0


def test_greek_bert_embedder_uses_shared_cache_without_strong_ref():
    """Verifica che GreekBertEmbedder ottenga il backbone dalla cache condivisa rilasciandolo dopo TTL."""
    dummy_base = MagicMock()
    dummy_tok = MagicMock()
    dummy_mlm = MagicMock()
    dummy_mlm.base_model = dummy_base

    test_cache = SharedBertCache(max_models=1)
    test_cache.cache["CNR-ILC/gs-GreBerta"] = (dummy_mlm, dummy_tok)

    embedder = GreekBertEmbedder(checkpoint="CNR-ILC/gs-GreBerta", device="cpu")
    with patch("models.rag.retriever.shared_bert_cache", test_cache):
        enc, tok = embedder._ensure_loaded()
        assert enc is dummy_base
        assert tok is dummy_tok
        assert embedder._model is None  # Nessun riferimento forte trattenuto sull'istanza


def test_bm25_csr_parity_and_v1_migration():
    """Verifica parità esatta dell'indice BM25 CSR NumPy rispetto a v1 e migrazione da v1."""
    corpus = [
        "[Corpus: tlg | Hom. Il. 5.31] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα",
        "[Corpus: DCLP | P.Oxy. 1] ἔδοξεν τῇ βουλῇ καὶ τῷ δήμῳ ἀγαθῇ τύχῃ",
        "[Corpus: tlg | Plat. Apol.] οὐθὲν ἡμῖν ἀμφοτέροις διοίσει περὶ τῆς πόλεως",
    ]
    idx = GreekLexicalBM25Index(corpus)

    res = idx.search("βροτολοιγε τειχεσιβλητα", top_k=5)
    assert res and res[0][0] == 0

    affix = idx.match_affix_words(word_prefix="τειχεσι", word_suffix="τα", min_gap_chars=1, max_gap_chars=6)
    assert any(full == "τειχεσιβλητα" and supp == "βλη" for full, supp, _ in affix)

    # Round-trip v2 CSR
    raw_v2 = idx.to_bytes()
    idx_v2 = GreekLexicalBM25Index.from_bytes(raw_v2)
    assert idx_v2.search("βροτολοιγε τειχεσιβλητα", top_k=5) == res
    assert idx_v2.get_postings("w:τειχεσιβλητα") == [(0, 1)]
    assert idx_v2.postings.get("w:τειχεσιβλητα") == [(0, 1)]

    # Migrazione trasparente da payload legacy v1 (`postings` dict di tuple)
    legacy_postings = {term: idx.get_postings(term) for term in idx.term_to_id}
    legacy_idf = {term: float(idx.idf_array[tid]) for term, tid in idx.term_to_id.items()}
    v1_bytes = zlib.compress(
        pickle.dumps(
            {
                "k1": idx.k1,
                "b": idx.b,
                "char_ngram_range": idx.char_ngram_range,
                "num_docs": idx.num_docs,
                "doc_lens": idx.doc_lens,
                "avgdl": idx.avgdl,
                "idf": legacy_idf,
                "postings": legacy_postings,
            }
        )
    )
    migrated = GreekLexicalBM25Index.from_bytes(v1_bytes)
    assert migrated.search("βροτολοιγε τειχεσιβλητα", top_k=5) == res


def test_compact_greek_vocab_in_memory_and_mmap(tmp_path: Path):
    """Verifica ricerca binaria, iterazione e `check_tlg_attestation` con `CompactGreekVocab`."""
    words = sorted(w.encode("utf-8") for w in ("λογοσ", "ο", "τειχεσιβλητα", "ανθρωπων"))
    vocab_mem = CompactGreekVocab(words, cache_file=None)
    mmap_file = tmp_path / "vocab.bin"
    vocab_mmap = CompactGreekVocab(words, cache_file=mmap_file)

    try:
        for vocab in (vocab_mem, vocab_mmap):
            assert len(vocab) == 4
            assert bool(vocab) is True
            assert "τειχεσιβλητα" in vocab
            assert "λογοσ" in vocab
            assert "δ" not in vocab
            assert "unknown" not in vocab
            assert set(vocab) == {"λογοσ", "ο", "τειχεσιβλητα", "ανθρωπων"}

            attested, recon = check_tlg_attestation(
                supplement="λητα",
                context_with_gap="τειχεσιβ[....]",
                vocab=vocab,
            )
            assert attested is True
            assert recon == "τειχεσιβλητα"
    finally:
        vocab_mmap.close()


def test_compact_ngram_sqlite_exact_parity_with_nltk(tmp_path: Path):
    """
    Verifica la parità matematica esatta (`abs < 1e-12`) tra `CompactNgramLM` su SQLite
    e i modelli NLTK `MLE` e `Lidstone` su `vocab.lookup`, `context_counts`, `score` e `generate_k_suggests`.
    """
    global_sents = [
        ["ΕΔΟΞΕΝ", "ΤΗΙ", "ΒΟΥΛΗΙ", "ΚΑΙ", "ΤΩΙ", "ΔΗΜΩΙ"],
        ["ΕΔΟΞΕΝ", "ΤΗΙ", "ΒΟΥΛΗΙ", "ΠΕΡΙ", "ΤΩΝ", "ΙΕΡΩΝ"],
        ["ΕΔΟΞΕΝ", "ΤΩΙ", "ΔΗΜΩΙ", "ΑΓΑΘΗΙ", "ΤΥΧΗΙ"],
        ["ΚΑΙ", "ΕΔΟΞΕΝ", "ΤΗΙ", "ΒΟΥΛΗΙ", "ΣΥΝΕΔΡΕΥΕΙΝ"],
        ["ΑΠΑΞ", "ΛΕΓΟΜΕΝΟΝ", "ΤΗΙ", "ΒΟΥΛΗΙ"],  # parole rare per testare <UNK> con unk_cutoff=2
    ]
    domain_sents = [
        ["ΕΔΟΞΕΝ", "ΤΗΙ", "ΒΟΥΛΗΙ", "ΑΓΑΘΗΙ", "ΤΥΧΗΙ"],
        ["ΤΗΙ", "ΒΟΥΛΗΙ", "ΚΑΙ", "ΤΩΙ", "ΔΗΜΩΙ"],
        ["ΣΠΑΝΙΟΝ", "ΤΗΙ", "ΒΟΥΛΗΙ"],
    ]

    def _fit(cls_lm, sents, *args):
        train_data, padded_vocab = padded_everygram_pipeline(3, sents)
        lm = cls_lm(*args)
        # unk_cutoff=2 sui globali per verificare la gestione di <UNK>
        from nltk.lm.vocabulary import Vocabulary

        vocab = Vocabulary(padded_vocab, unk_cutoff=2 if len(sents) > 3 else 1)
        lm = cls_lm(*args, vocabulary=vocab)
        lm.fit(train_data)
        return lm

    g_lid = _fit(Lidstone, global_sents, 0.001, 3)
    d_lid = _fit(Lidstone, domain_sents, 0.001, 3)
    g_mle = _fit(MLE, global_sents, 3)
    d_mle = _fit(MLE, domain_sents, 3)

    db_path = tmp_path / "ngram_n3.sqlite"
    export_nltk_models_to_sqlite(g_lid, d_lid, db_path, n=3)

    cg_lid, cd_lid = load_compact_ngram_pair(db_path, lm_type="LIDSTONE", gamma=0.001)
    cg_mle, cd_mle = load_compact_ngram_pair(db_path, lm_type="MLE", gamma=None)

    try:
        # 1. Parità vocab
        assert len(cg_lid.vocab) == len(g_lid.vocab)
        assert len(cd_lid.vocab) == len(d_lid.vocab)
        for probe in ("ΕΔΟΞΕΝ", "ΤΗΙ", "ΑΠΑΞ", "INESISTENTE", ["ΕΔΟΞΕΝ", "ΑΠΑΞ"]):
            assert cg_lid.vocab.lookup(probe) == g_lid.vocab.lookup(probe)
            assert cd_lid.vocab.lookup(probe) == d_lid.vocab.lookup(probe)

        # 2. Parità context_counts e score su MLE e Lidstone (inclusi unigrammi, bigrammi, trigrammi e contesti OOV)
        contexts = [
            None,
            (),
            ("<s>",),
            ("ΕΔΟΞΕΝ", "ΤΗΙ"),
            ("ΤΗΙ", "ΒΟΥΛΗΙ"),
            ("<UNK>", "ΤΗΙ"),
            ("NON_ESISTE", "MAI"),
        ]
        words = ["ΒΟΥΛΗΙ", "ΚΑΙ", "ΤΥΧΗΙ", "<UNK>", "OOV_WORD"]

        for nltk_lm, cmp_lm in ((g_lid, cg_lid), (d_lid, cd_lid), (g_mle, cg_mle), (d_mle, cd_mle)):
            for ctx in contexts:
                if ctx is not None:
                    assert dict(cmp_lm.context_counts(cmp_lm.vocab.lookup(ctx))) == dict(
                        nltk_lm.context_counts(nltk_lm.vocab.lookup(ctx))
                    )
                for w in words:
                    s_ref = nltk_lm.score(w, ctx)
                    s_cmp = cmp_lm.score(w, ctx)
                    assert s_cmp == pytest.approx(s_ref, abs=1e-12)

        # 3. Parità end-to-end su generate_k_suggests
        ctx_gap = "ΕΔΟΞΕΝ ΤΗΙ [......] ΚΑΙ ΤΩΙ ΔΗΜΩΙ"
        res_lid_ref = generate_k_suggests(g_lid, d_lid, ctx_gap, num_tokens=1, lm_type="LIDSTONE", n=3, k_pred=5)
        res_lid_cmp = generate_k_suggests(cg_lid, cd_lid, ctx_gap, num_tokens=1, lm_type="LIDSTONE", n=3, k_pred=5)
        assert [w for w, _ in res_lid_cmp] == [w for w, _ in res_lid_ref]
        for (_, p1), (_, p2) in zip(res_lid_cmp, res_lid_ref):
            assert p1 == pytest.approx(p2, abs=1e-12)

        res_mle_ref = generate_k_suggests(g_mle, d_mle, ctx_gap, num_tokens=1, lm_type="MLE", n=3, k_pred=5)
        res_mle_cmp = generate_k_suggests(cg_mle, cd_mle, ctx_gap, num_tokens=1, lm_type="MLE", n=3, k_pred=5)
        assert [w for w, _ in res_mle_cmp] == [w for w, _ in res_mle_ref]
        for (_, p1), (_, p2) in zip(res_mle_cmp, res_mle_ref):
            assert p1 == pytest.approx(p2, abs=1e-12)
    finally:
        for lm_obj in (cg_lid, cd_lid, cg_mle, cd_mle):
            lm_obj.close()


@pytest.mark.asyncio
async def test_model_and_suggestions_service_prefer_sqlite_compact_ngram(tmp_path: Path):
    """
    Verifica che `ModelService._train_and_persist_ngram` usi il checkpoint SQLite compatto
    quando presente (zero upload GridFS) e che `SuggestionsService` lo utilizzi anche per
    documenti MongoDB legacy con `CORPUS_NAMES=None`.
    """
    train_data, padded_vocab = padded_everygram_pipeline(3, [["ΕΔΟΞΕΝ", "ΤΗΙ", "ΒΟΥΛΗΙ"]])
    g_lm = Lidstone(0.001, 3)
    g_lm.fit(train_data, padded_vocab)
    train_data_d, padded_vocab_d = padded_everygram_pipeline(3, [["ΕΔΟΞΕΝ", "ΤΗΙ", "ΒΟΥΛΗΙ"]])
    d_lm = Lidstone(0.001, 3)
    d_lm.fit(train_data_d, padded_vocab_d)

    ckpt_dir = tmp_path / "ngrams"
    ckpt_dir.mkdir()
    db_path = ckpt_dir / "ngram_n3.sqlite"
    export_nltk_models_to_sqlite(g_lm, d_lm, db_path, n=3)

    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))
    mock_fs = MagicMock()
    mock_fs.upload_from_stream = AsyncMock()

    model_service = ModelService(db_collection=mock_collection, gridfs=mock_fs)
    model_dict = {"LM_SCORE": "LIDSTONE", "GAMMA": 0.001, "N": 3, "CORPUS_NAMES": None}

    with patch(
        "backend.api.services.model_service.get_ngram_checkpoint_path",
        return_value=ckpt_dir / "ngram_lidstone_n3_gamma0.001.pkl.zlib",
    ):
        await model_service._train_and_persist_ngram(model_dict)

    assert model_dict["STORAGE_FORMAT"] == "sqlite_compact"
    assert model_dict["GLOBAL_MODEL_FILE_ID"] is None
    assert model_dict["DOMAIN_MODEL_FILE_ID"] is None
    mock_fs.upload_from_stream.assert_not_called()

    # Verifica predizione tramite SuggestionsService anche su documento legacy senza STORAGE_FORMAT
    legacy_doc = {
        "_id": ObjectId(),
        "TYPE": ModelType.NGRAMS,
        "LM_SCORE": "LIDSTONE",
        "GAMMA": 0.001,
        "N": 3,
        "CORPUS_NAMES": None,
        "GLOBAL_MODEL_FILE_ID": "legacy-gid",
        "DOMAIN_MODEL_FILE_ID": "legacy-did",
    }
    mock_collection.find_one = AsyncMock(return_value=legacy_doc)
    sugg_service = SuggestionsService(db_collection=mock_collection, gridfs=mock_fs, feedback_db=MagicMock())
    SuggestionsService._compact_ngram_cache.clear()

    with patch(
        "backend.api.services.suggestions_service.get_compact_ngram_checkpoint_path",
        return_value=db_path,
    ):
        preds = await sugg_service.get_predictions(
            model_id=str(legacy_doc["_id"]),
            context="ΕΔΟΞΕΝ ΤΗΙ [......]",
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
        )
    assert len(preds) >= 1
    mock_fs.open_download_stream_by_name.assert_not_called()

    # Pulizia connessioni SQLite di test
    for g_cmp, d_cmp in SuggestionsService._compact_ngram_cache.values():
        g_cmp.close()
        d_cmp.close()
    SuggestionsService._compact_ngram_cache.clear()


def test_ithaca_micro_batch_parity_and_ttl_eviction():
    """Verifica equivalenza tra micro-batch fisso e full-batch e pulizia TTL di Ithaca."""
    class DummyAlphabet:
        def __init__(self):
            self.pad = "#"
            self.sos = "<"
            self.unk = "^"
            self.space = " "
            self.missing = "-"
            chars = list("αβγδεζηθικλμνξοπρςστυφχψω")
            self.idx2char = np.array([self.pad, self.sos, self.unk, self.space, self.missing] + chars)
            self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

    alphabet = DummyAlphabet()
    max_len = 12
    char_ids = np.full((1, max_len), alphabet.char2idx[alphabet.pad], dtype=np.int32)
    word_ids = np.zeros((1, max_len), dtype=np.int32)
    char_ids[0, 0] = alphabet.char2idx[alphabet.sos]
    char_ids[0, 1] = alphabet.char2idx["α"]
    char_ids[0, 2] = alphabet.char2idx[alphabet.missing]
    char_ids[0, 3] = alphabet.char2idx[alphabet.missing]
    char_ids[0, 4] = alphabet.char2idx["ς"]

    observed_batch_sizes: list[int] = []

    def deterministic_forward(params, text_char, text_word):
        arr = np.asarray(text_char)
        observed_batch_sizes.append(arr.shape[0])
        b, seq_len = arr.shape
        out = np.full((b, seq_len, len(alphabet.idx2char)), -10.0, dtype=np.float32)
        for row_i in range(b):
            # Punteggi deterministici dipendenti dalla riga per verificare isolamento batch
            prev_ch = int(arr[row_i, 2])
            out[row_i, 2, alphabet.char2idx["β"]] = -0.2
            out[row_i, 2, alphabet.char2idx["γ"]] = -0.5
            out[row_i, 3, alphabet.char2idx["ο"]] = -0.1 if prev_ch == alphabet.char2idx["β"] else -0.8
            out[row_i, 3, alphabet.char2idx["α"]] = -0.6
        return out

    res_full = ithaca_predict.decode_gap_hcb_beam_search(
        forward_fn=deterministic_forward,
        params=None,
        char_ids_initial=char_ids,
        word_ids_initial=word_ids,
        mask_indices=[2, 3],
        alphabet=alphabet,
        K=5,
        beam_width=6,
        forward_batch_size=None,
    )
    observed_batch_sizes.clear()
    res_mb = ithaca_predict.decode_gap_hcb_beam_search(
        forward_fn=deterministic_forward,
        params=None,
        char_ids_initial=char_ids,
        word_ids_initial=word_ids,
        mask_indices=[2, 3],
        alphabet=alphabet,
        K=5,
        beam_width=6,
        forward_batch_size=4,
    )

    assert all(bs == 4 for bs in observed_batch_sizes)
    assert [w for w, _ in res_mb] == [w for w, _ in res_full]
    for (_, s1), (_, s2) in zip(res_mb, res_full):
        assert s1 == pytest.approx(s2, abs=1e-6)

    # Verifica eviction TTL su _LOADED_MODELS + __torso_fn__ associati
    ithaca_predict.clear_ithaca_cache()
    ithaca_predict._LOADED_MODELS["/tmp/fake_ithaca.pkl"] = ("m", "p", "a", {}, "f")
    ithaca_predict._RESOLVED_CHECKPOINT_PATHS["CNR-ILC/gs-ithaca-tlg"] = "/tmp/fake_ithaca.pkl"
    ithaca_predict._ITHACA_LAST_USED["/tmp/fake_ithaca.pkl"] = time.monotonic() - 1000.0
    ithaca_predict._LOADED_MODELS["__torso_fn__::CNR-ILC/gs-ithaca-tlg::768"] = "torso"

    assert ithaca_predict.evict_expired_ithaca(ttl_seconds=600.0) is True
    assert len(ithaca_predict._LOADED_MODELS) == 0
    assert len(ithaca_predict._RESOLVED_CHECKPOINT_PATHS) == 0
