"""
Test unitari per il logging osservabile delle chiamate LLM remote
(`models/rag/llm_logging.py`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from models.rag import llm_logging


@pytest.fixture(autouse=True)
def _clean_llm_logging(monkeypatch):
    for name in (
        "RAG_LLM_LOG_DEST",
        "RAG_LLM_LOG_FILE",
        "RAG_LLM_LOG_LEVEL",
        "RAG_LLM_LOG_HEADERS",
        "RAG_LLM_LOG_PROMPTS",
    ):
        monkeypatch.delenv(name, raising=False)
    llm_logging.reset_llm_logging_for_tests()
    yield
    llm_logging.reset_llm_logging_for_tests()


def _success_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="req-123",
        model="Qwen/Qwen2.5-72B-Instruct",
        choices=[SimpleNamespace(finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45, total_tokens=165),
        _hidden_params={"response_cost": 0.001234},
    )


def test_success_call_is_logged_to_file_without_secrets(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_calls.log"
    monkeypatch.setenv("RAG_LLM_LOG_DEST", "file")
    monkeypatch.setenv("RAG_LLM_LOG_FILE", str(log_file))

    logger = llm_logging.configure_llm_logging(force=True)
    callback = llm_logging.HuggingFaceCallLogger(logger)

    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=1.25)

    callback.log_pre_api_call(
        "huggingface/Qwen/Qwen2.5-72B-Instruct",
        [{"role": "user", "content": "Completa la lacuna nel testo greco"}],
        {
            "litellm_call_id": "call-abc",
            "litellm_params": {"custom_llm_provider": "huggingface", "timeout": 60},
            "temperature": 0.2,
            "max_tokens": 256,
        },
    )
    callback.log_success_event(
        {
            "litellm_call_id": "call-abc",
            "response_headers": {
                "x-inference-provider": "nebius",
                "x-ratelimit-remaining-requests": "99",
                "authorization": "Bearer secret-token",
            },
        },
        _success_response(),
        start,
        end,
    )

    content = log_file.read_text(encoding="utf-8")
    assert "HF/LLM request start" in content
    assert "HF/LLM request success" in content
    assert "status=200" in content
    assert "elapsed=1.250s" in content
    assert "finish_reason=stop" in content
    assert "input_tokens=120" in content
    assert "output_tokens=45" in content
    assert "cost_usd=0.001234" in content
    assert "x-inference-provider" in content
    assert "authorization" not in content
    assert "secret-token" not in content


def test_failure_call_logs_http_status_and_rate_limit_headers(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_calls.log"
    monkeypatch.setenv("RAG_LLM_LOG_DEST", "file")
    monkeypatch.setenv("RAG_LLM_LOG_FILE", str(log_file))

    logger = llm_logging.configure_llm_logging(force=True)
    callback = llm_logging.HuggingFaceCallLogger(logger)

    class FakeHuggingFaceError(Exception):
        status_code = 429
        headers = {
            "retry-after": "2",
            "x-ratelimit-remaining-requests": "0",
            "authorization": "Bearer secret-token",
        }
        message = "Rate limit exceeded for provider"

    start = datetime.now(timezone.utc)
    callback.log_failure_event(
        {"litellm_call_id": "call-err", "exception": FakeHuggingFaceError("rate limited")},
        None,
        start,
        start + timedelta(milliseconds=250),
    )

    content = log_file.read_text(encoding="utf-8")
    assert "HF/LLM request failure" in content
    assert "status=429" in content
    assert "elapsed=0.250s" in content
    assert "error_type=FakeHuggingFaceError" in content
    assert "retry-after" in content
    assert "x-ratelimit-remaining-requests" in content
    assert "authorization" not in content
    assert "secret-token" not in content


def test_header_mode_all_includes_custom_headers_but_never_secrets(tmp_path, monkeypatch):
    log_file = tmp_path / "llm_calls.log"
    monkeypatch.setenv("RAG_LLM_LOG_DEST", "file")
    monkeypatch.setenv("RAG_LLM_LOG_FILE", str(log_file))
    monkeypatch.setenv("RAG_LLM_LOG_HEADERS", "all")

    logger = llm_logging.configure_llm_logging(force=True)
    callback = llm_logging.HuggingFaceCallLogger(logger)
    start = datetime.now(timezone.utc)

    callback.log_success_event(
        {
            "response_headers": {
                "x-custom-debug": "visible",
                "set-cookie": "session=abc",
                "authorization": "Bearer secret-token",
            }
        },
        _success_response(),
        start,
        start + timedelta(seconds=0.1),
    )

    content = log_file.read_text(encoding="utf-8")
    assert "x-custom-debug" in content
    assert "set-cookie" not in content
    assert "authorization" not in content
    assert "secret-token" not in content


def test_off_destination_suppresses_output_and_skips_callback(monkeypatch, capsys):
    monkeypatch.setenv("RAG_LLM_LOG_DEST", "off")
    logger = llm_logging.configure_llm_logging(force=True)
    callback = llm_logging.HuggingFaceCallLogger(logger)
    callback.log_pre_api_call("huggingface/Qwen/test", [], {})

    assert capsys.readouterr().out == ""
    assert logger.disabled is True


def test_install_registers_callback_only_once(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "callbacks", list(getattr(litellm, "callbacks", []) or []))

    assert llm_logging.install_llm_call_logging(force=True) is True
    assert llm_logging.install_llm_call_logging() is True

    handlers = [
        callback
        for callback in litellm.callbacks
        if isinstance(callback, llm_logging.HuggingFaceCallLogger)
    ]
    assert len(handlers) == 1
    assert isinstance(logging.getLogger(llm_logging.LOGGER_NAME).handlers[0], logging.StreamHandler)


def test_both_destination_logs_to_stdout_and_file(tmp_path, monkeypatch, capsys):
    log_file = tmp_path / "llm_calls.log"
    monkeypatch.setenv("RAG_LLM_LOG_DEST", "both")
    monkeypatch.setenv("RAG_LLM_LOG_FILE", str(log_file))

    logger = llm_logging.configure_llm_logging(force=True)
    callback = llm_logging.HuggingFaceCallLogger(logger)
    callback.log_pre_api_call("huggingface/Qwen/Qwen2.5-72B-Instruct", [], {})
    llm_logging.log_llm_skipped_event(
        "huggingface/Qwen/Qwen2.5-72B-Instruct",
        "circuit-breaker attivo",
    )
    llm_logging.log_llm_fallback_event(
        "Qwen/Qwen2.5-72B-Instruct",
        "neuro_lexical (CNR-ILC/gs-GreBerta)",
        "permessi HF_TOKEN Inference Providers assenti",
    )

    stdout_out = capsys.readouterr().out
    file_out = log_file.read_text(encoding="utf-8")
    for text in (stdout_out, file_out):
        assert "HF/LLM request start" in text
        assert "HF/LLM request skipped" in text
        assert "HF/LLM fallback activated" in text

