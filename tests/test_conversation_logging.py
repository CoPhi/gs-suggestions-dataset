"""
Test unitari e di integrazione per il logging JSONL opt-in delle conversazioni
complete DSPy <-> LLM (`models/rag/conversation_logging.py`).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import dspy
import numpy as np
import pytest
from dspy.utils.dummies import DummyLM

from models.rag import conversation_logging
from models.rag.inference import fill_mask_rag
from models.rag.modules import GreekRAGInfiller
from models.rag.retriever import GreekHybridRetriever, normalize_for_retrieval


def _dummy_embedder(texts: list[str]) -> np.ndarray:
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, text in enumerate(texts):
        norm = normalize_for_retrieval(text)
        for pos in range(max(1, len(norm) - 2)):
            bucket = sum(ord(ch) for ch in norm[pos : pos + 3]) % dim
            out[idx, bucket] += 1.0
        nrm = np.linalg.norm(out[idx])
        out[idx] = out[idx] / nrm if nrm > 0 else np.eye(1, dim, 0, dtype=np.float32)[0]
    return out


@pytest.fixture(autouse=True)
def _clean_conv_logging(monkeypatch):
    monkeypatch.delenv("RAG_CONVERSATION_LOG_ENABLED", raising=False)
    monkeypatch.delenv("RAG_CONVERSATION_LOG_FILE", raising=False)
    conversation_logging.reset_conversation_logging_for_tests()
    yield
    conversation_logging.reset_conversation_logging_for_tests()


def test_conversation_logging_disabled_by_default(tmp_path, monkeypatch):
    log_file = tmp_path / "dspy_conversations.jsonl"
    monkeypatch.setenv("RAG_CONVERSATION_LOG_FILE", str(log_file))

    dummy = DummyLM(
        [
            {
                "philological_reasoning": "Epiteto omerico di Ares.",
                "candidates": [{"supplement": "λητα", "confidence": 0.95, "rationale": "forma omerica"}],
            }
        ]
    )
    preds = fill_mask_rag(
        text="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        lm=dummy,
        retriever=None,
        top_k_passages=0,
        K=5,
        session_id="sess-disabled",
    )

    assert preds[0][0] == "λητα"
    assert not log_file.exists()


def test_full_conversation_trace_with_refinement_and_no_secrets(tmp_path, monkeypatch):
    log_file = tmp_path / "dspy_conversations.jsonl"
    monkeypatch.setenv("RAG_CONVERSATION_LOG_ENABLED", "true")
    monkeypatch.setenv("RAG_CONVERSATION_LOG_FILE", str(log_file))

    corpus = [
        "[Corpus: tlg | Hom. Il. 5.31] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα",
    ]
    retriever = GreekHybridRetriever(corpus=corpus, embedder=_dummy_embedder, k=1)

    # La prima risposta ha lunghezza errata (2 lettere anziché 4) e attiva ConstraintRefinementSignature
    dummy = DummyLM(
        [
            {
                "philological_reasoning": "Primo tentativo troppo corto.",
                "candidates": [{"supplement": "λη", "confidence": 0.6, "rationale": "troppo corto"}],
            },
            {
                "candidates": [{"supplement": "λητα", "confidence": 0.94, "rationale": "esattamente 4 lettere"}],
            },
        ]
    )
    # Inietta un finto api_key in kwargs tramite wrapper per verificare che non finisca mai nel JSONL
    dummy.kwargs = {**(getattr(dummy, "kwargs", {}) or {}), "api_key": "super-secret-token-xyz", "temperature": 0.2}

    preds, _details = fill_mask_rag(
        text="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        checkpoint="Qwen/Qwen2.5-72B-Instruct",
        lm=dummy,
        retriever=retriever,
        top_k_passages=1,
        K=5,
        enable_refinement=True,
        return_details=True,
        human_feedback="Preferisci forma epica",
        rejected_candidates=["πλητα"],
        session_id="sess-hitl-001",
        model_id="model-abc",
    )

    assert preds[0][0] == "λητα"
    assert log_file.exists()
    raw_text = log_file.read_text(encoding="utf-8")
    assert "super-secret-token-xyz" not in raw_text

    records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    assert len(records) == 1
    rec = records[0]

    assert rec["session_id"] == "sess-hitl-001"
    assert rec["model_id"] == "model-abc"
    assert rec["status"] == "ok"
    assert rec["input"]["human_feedback"] == "Preferisci forma epica"
    assert rec["input"]["rejected_candidates"] == ["πλητα"]
    assert len(rec["retrieval"]["passages_sent_to_lm"]) == 1
    assert "τειχεσιβλῆτα" in rec["retrieval"]["passages_sent_to_lm"][0]

    step_names = [s["step_name"] for s in rec["dspy_steps"]]
    assert step_names == ["GreekInfillingSignature", "ConstraintRefinementSignature"]

    assert len(rec["llm_calls"]) == 2
    assert rec["llm_calls"][0]["step_name"] == "GreekInfillingSignature"
    assert rec["llm_calls"][1]["step_name"] == "ConstraintRefinementSignature"
    for call in rec["llm_calls"]:
        assert isinstance(call["messages"], list) and len(call["messages"]) >= 2
        assert call["raw_outputs"]
        assert call["cache_hit"] is False
        assert call["latency_ms"] >= 0
        assert "api_key" not in (call.get("kwargs") or {})

    assert rec["final_output"]["candidates"][0]["supplement"] == "λητα"


def test_deterministic_cache_hit_detection(tmp_path, monkeypatch):
    log_file = tmp_path / "dspy_conversations.jsonl"
    monkeypatch.setenv("RAG_CONVERSATION_LOG_ENABLED", "true")
    monkeypatch.setenv("RAG_CONVERSATION_LOG_FILE", str(log_file))

    recorder = conversation_logging.ConversationTraceRecorder(
        session_id="sess-cache",
        checkpoint="Qwen/Qwen2.5-72B-Instruct",
        requested_backend="hf_api",
    )
    fake_lm = MagicMock()
    fake_lm.model = "huggingface/Qwen/Qwen2.5-72B-Instruct"
    fake_lm.history = []

    with recorder.make_dspy_context(fake_lm, attempt="primary") as _ctx:
        tracker = dspy.settings.usage_tracker
        cb = dspy.settings.callbacks[-1]

        # Chiamata 1: live (usage_tracker.add_usage viene invocato da DSPy prima di on_lm_end)
        cb.on_lm_start("c1", fake_lm, {"messages": [{"role": "user", "content": "q1"}], "kwargs": {}})
        tracker.add_usage(fake_lm.model, {"prompt_tokens": 100, "completion_tokens": 20})
        cb.on_lm_end("c1", ["risposta 1"], None)

        # Chiamata 2: cache_hit (usage_tracker.add_usage NON viene invocato da DSPy)
        cb.on_lm_start("c2", fake_lm, {"messages": [{"role": "user", "content": "q1"}], "kwargs": {}})
        cb.on_lm_end("c2", ["risposta 1"], None)

    record = recorder.finalize_and_write(
        effective_backend="hf_api",
        effective_model=fake_lm.model,
        status="ok",
    )

    assert len(record["llm_calls"]) == 2
    assert record["llm_calls"][0]["cache_hit"] is False
    assert record["llm_calls"][0]["usage"]["prompt_tokens"] == 100
    assert record["llm_calls"][1]["cache_hit"] is True
    assert record["llm_calls"][1]["usage"] == {}


def test_fallback_to_neuro_lexical_and_error_recording(tmp_path, monkeypatch):
    log_file = tmp_path / "dspy_conversations.jsonl"
    monkeypatch.setenv("RAG_CONVERSATION_LOG_ENABLED", "true")
    monkeypatch.setenv("RAG_CONVERSATION_LOG_FILE", str(log_file))

    corpus = ["[Corpus: DCLP | Philodemus] τὸ μὲν εὐπαρακολούθητα ποιεῖν τοῖς ἀκούουσιν"]
    retriever = GreekHybridRetriever(corpus=corpus, embedder=_dummy_embedder, k=1)

    with (
        patch.object(GreekRAGInfiller, "forward", side_effect=RuntimeError("403 inference providers denied")),
        patch("models.rag.inference.get_fallback_dspy_lms", return_value=[]),
        patch("models.rag.inference._get_cached_mlm", side_effect=RuntimeError("skip heavy MLM")),
    ):
        preds = fill_mask_rag(
            text="μὲν εὐπαρακολούθητα π[.]ιειν",
            checkpoint="Qwen/Qwen2.5-72B-Instruct",
            backend="auto",
            retriever=retriever,
            K=5,
            allow_local_fallback=True,
            session_id="sess-fallback",
        )
    assert preds[0][0] == "ο"

    # Verifica anche registrazione errore quando allow_local_fallback=False
    failing_lm = DummyLM([{}])
    with (
        patch.object(GreekRAGInfiller, "forward", side_effect=RuntimeError("timeout remoto")),
        pytest.raises(RuntimeError, match="timeout remoto"),
    ):
        fill_mask_rag(
            text="μὲν εὐπαρακολούθητα π[.]ιειν",
            checkpoint="Qwen/Qwen2.5-72B-Instruct",
            lm=failing_lm,
            retriever=None,
            top_k_passages=0,
            allow_local_fallback=False,
            session_id="sess-error",
        )

    records = conversation_logging.load_conversation_records(log_path=log_file)
    assert len(records) == 2

    fb_rec = records[0]
    assert fb_rec["session_id"] == "sess-fallback"
    assert fb_rec["status"] == "fallback_neuro_lexical"
    assert fb_rec["effective_backend"] == "neuro_lexical"
    assert fb_rec["fallback_events"][0]["kind"] == "fallback_neuro_lexical"

    err_rec = records[1]
    assert err_rec["session_id"] == "sess-error"
    assert err_rec["status"] == "error"
    assert "RuntimeError: timeout remoto" in err_rec["error"]


def test_neuro_lexical_apriori_is_excluded(tmp_path, monkeypatch):
    log_file = tmp_path / "dspy_conversations.jsonl"
    monkeypatch.setenv("RAG_CONVERSATION_LOG_ENABLED", "true")
    monkeypatch.setenv("RAG_CONVERSATION_LOG_FILE", str(log_file))

    corpus = ["[Corpus: DCLP | Philodemus] τὸ μὲν εὐπαρακολούθητα ποιεῖν τοῖς ἀκούουσιν"]
    retriever = GreekHybridRetriever(corpus=corpus, embedder=_dummy_embedder, k=1)

    with patch("models.rag.inference._get_cached_mlm", side_effect=RuntimeError("skip heavy MLM")):
        preds = fill_mask_rag(
            text="μὲν εὐπαρακολούθητα π[.]ιειν",
            checkpoint="CNR-ILC/gs-GreBerta",
            backend="neuro_lexical",
            retriever=retriever,
            K=5,
            session_id="sess-bert-rag",
        )

    assert preds[0][0] == "ο"
    assert not log_file.exists()


def test_cli_loader_and_formatter(tmp_path, monkeypatch, capsys):
    log_file = tmp_path / "dspy_conversations.jsonl"
    monkeypatch.setenv("RAG_CONVERSATION_LOG_ENABLED", "true")
    monkeypatch.setenv("RAG_CONVERSATION_LOG_FILE", str(log_file))

    dummy = DummyLM(
        [
            {
                "philological_reasoning": "Analisi sintattica.",
                "candidates": [{"supplement": "λητα", "confidence": 0.9, "rationale": "valido"}],
            }
        ]
    )
    fill_mask_rag(
        text="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        checkpoint="Qwen/Qwen2.5-72B-Instruct",
        lm=dummy,
        retriever=None,
        top_k_passages=0,
        session_id="sess-cli-alpha",
    )

    loaded = conversation_logging.load_conversation_records(
        log_path=log_file,
        session_id="sess-cli",
        model="Qwen2.5",
        last=5,
    )
    assert len(loaded) == 1

    formatted = conversation_logging.format_conversation_record(loaded[0], show_system=False)
    assert "sess-cli-alpha" in formatted
    assert "GreekInfillingSignature" in formatted
    assert "[SYSTEM] <nascosto con --no-system" in formatted
    assert "λητα" in formatted

    monkeypatch.setattr(
        "sys.argv",
        ["conversation_logging", "--file", str(log_file), "--session-id", "sess-cli-alpha", "--last", "1"],
    )
    conversation_logging.main()
    cli_out = capsys.readouterr().out
    assert "sess-cli-alpha" in cli_out
    assert "CONVERSAZIONE LLM" in cli_out
