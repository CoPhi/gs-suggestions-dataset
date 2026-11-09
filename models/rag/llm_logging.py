"""
Logging osservabile delle chiamate ai provider LLM remoti (Hugging Face Router,
server OpenAI-compatible come vLLM/TGI) effettuate tramite LiteLLM/DSPy.

Hugging Face non espone uno stato asincrono per le richieste di inferenza: la
chiamata è sincrona e il provider comunica solo l'esito HTTP (es. 200, 401,
403, 429, 503) insieme agli header di risposta (request-id, rate limit,
provider effettivo). Questo modulo intercetta gli eventi LiteLLM e li traduce
in righe di log leggibili, utili per capire se una richiesta è in corso, se è
riuscita e con quali consumi di token, oppure perché è fallita.

Configurazione tramite variabili d'ambiente (tutte opzionali):

- ``RAG_LLM_LOG_DEST``: ``stdout`` (default), ``stderr``, ``file``, ``both``, ``off``.
- ``RAG_LLM_LOG_FILE``: percorso del file (default ``logs/llm_calls.log``).
- ``RAG_LLM_LOG_LEVEL``: livello logging (default ``INFO``).
- ``RAG_LLM_LOG_HEADERS``: ``safe`` (default), ``all``, ``none`` per gli header HTTP.
- ``RAG_LLM_LOG_PROMPTS``: ``true`` per includere un'anteprima dei messaggi.

Le credenziali (``api_key``, header ``Authorization``, cookie e token) non
vengono mai scritte nei log.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGGER_NAME = "gs.llm"
DEFAULT_LOG_FILE = Path("logs/llm_calls.log")

# Header di risposta potenzialmente interessanti per diagnosticare una chiamata
# remota (rate limit, routing del provider, richiesta correlata). In modalità
# "safe" vengono loggati solo questi; in modalità "all" tutti quelli non sensibili.
_INTERESTING_HEADER_PREFIXES = (
    "x-ratelimit",
    "x-request",
    "x-correlation",
    "x-inference",
    "x-hf",
    "x-provider",
    "x-model",
    "retry-after",
    "server-timing",
    "cf-ray",
    "via",
    "x-served-by",
    "x-cache",
)

# Frammenti che indicano header/credenziali da non loggare in nessun caso.
_SENSITIVE_HEADER_FRAGMENTS = (
    "authorization",
    "cookie",
    "token",
    "secret",
    "api-key",
    "apikey",
    "x-amz-security",
)

_HANDLER_MARKER = "_gs_llm_call_logging_handler"
_configured = False
_callback_installed = False
_configure_lock = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "si", "sì")


def _resolve_level() -> int:
    raw = (os.getenv("RAG_LLM_LOG_LEVEL") or "INFO").strip().upper()
    try:
        return int(raw)
    except ValueError:
        return getattr(logging, raw, logging.INFO)


def _resolve_destination() -> str:
    dest = (os.getenv("RAG_LLM_LOG_DEST") or "stdout").strip().lower()
    aliases = {
        "console": "stdout",
        "out": "stdout",
        "none": "off",
        "disabled": "off",
        "disable": "off",
        "false": "off",
        "0": "off",
    }
    return aliases.get(dest, dest)


def _resolve_header_mode() -> str:
    mode = (os.getenv("RAG_LLM_LOG_HEADERS") or "safe").strip().lower()
    return mode if mode in ("safe", "all", "none") else "safe"


def _remove_our_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def _build_handlers(dest: str, level: int) -> list[logging.Handler]:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = []

    if dest in ("stdout", "both"):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(formatter)
        setattr(stdout_handler, _HANDLER_MARKER, True)
        handlers.append(stdout_handler)
    elif dest == "stderr":
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(level)
        stderr_handler.setFormatter(formatter)
        setattr(stderr_handler, _HANDLER_MARKER, True)
        handlers.append(stderr_handler)

    if dest in ("file", "both"):
        log_path = Path(os.getenv("RAG_LLM_LOG_FILE") or DEFAULT_LOG_FILE).expanduser()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Rotazione locale per evitare che un file di log cresca indefinitamente
            # durante sessioni di inferenza prolungate.
            file_handler: logging.Handler = RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, _HANDLER_MARKER, True)
            handlers.append(file_handler)
        except Exception as exc:
            print(
                f"[gs.llm] Impossibile aprire il file di log '{log_path}' ({exc}); "
                "uso lo standard output.",
                file=sys.stderr,
            )
            if not handlers:
                fallback_handler = logging.StreamHandler(sys.stdout)
                fallback_handler.setLevel(level)
                fallback_handler.setFormatter(formatter)
                setattr(fallback_handler, _HANDLER_MARKER, True)
                handlers.append(fallback_handler)

    return handlers


def configure_llm_logging(force: bool = False) -> logging.Logger:
    """
    Configura (una sola volta) il logger dedicato ``gs.llm``.

    Args:
        force: Ricostruisce gli handler anche se la configurazione è già attiva
               (usato nei test o per applicare a caldo nuove variabili d'ambiente).

    Returns:
        Il logger configurato. Se ``RAG_LLM_LOG_DEST=off`` restituisce un logger
        silenzioso, così le chiamate di inferenza non producono output aggiuntivo.
    """
    global _configured

    logger = logging.getLogger(LOGGER_NAME)
    with _configure_lock:
        if _configured and not force:
            return logger

        _remove_our_handlers(logger)
        level = _resolve_level()
        dest = _resolve_destination()

        logger.setLevel(level)
        # Evita la duplicazione sul root logger (es. uvicorn) quando si logga su stdout.
        logger.propagate = False

        if dest == "off":
            logger.disabled = True
            null_handler = logging.NullHandler()
            setattr(null_handler, _HANDLER_MARKER, True)
            logger.addHandler(null_handler)
            _configured = True
            return logger

        logger.disabled = False
        handlers = _build_handlers(dest, level)
        if not handlers:
            logger.disabled = True
            null_handler = logging.NullHandler()
            setattr(null_handler, _HANDLER_MARKER, True)
            logger.addHandler(null_handler)
        else:
            for handler in handlers:
                logger.addHandler(handler)

        _configured = True

    if not logger.disabled:
        logger.info(
            "LLM call logging enabled | dest=%s | level=%s | headers=%s | prompts=%s",
            dest,
            logging.getLevelName(level),
            _resolve_header_mode(),
            _env_bool("RAG_LLM_LOG_PROMPTS", default=False),
        )
    return logger


def _safe_headers(source: Any, mode: str | None = None) -> dict[str, str]:
    """Estrae gli header HTTP da ``httpx.Headers``/dict, escludendo i segreti."""
    if not source or mode == "none":
        return {}

    mode = mode or _resolve_header_mode()
    try:
        items = source.items() if hasattr(source, "items") else dict(source).items()
    except Exception:
        return {}

    selected: dict[str, str] = {}
    for raw_key, raw_value in items:
        key = str(raw_key).lower()
        if any(fragment in key for fragment in _SENSITIVE_HEADER_FRAGMENTS):
            continue
        if mode == "safe" and not key.startswith(_INTERESTING_HEADER_PREFIXES):
            continue
        try:
            value = str(raw_value)
        except Exception:
            continue
        selected[key] = value[:200]
    return selected


def _elapsed_seconds(start_time: Any, end_time: Any) -> float | None:
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        return max(0.0, (end_time - start_time).total_seconds())
    return None


def _truncate(value: Any, limit: int = 800) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[:limit]}…"


def _messages_preview(messages: Any, limit: int = 240) -> str:
    if not messages:
        return ""
    try:
        last = messages[-1]
    except Exception:
        return ""
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
    if not isinstance(content, str):
        content = str(content or "")
    return _truncate(content, limit=limit)


def _infer_provider(model: str | None) -> str:
    if not model or "/" not in model:
        return "unknown"
    provider = model.split("/", 1)[0]
    if provider == "huggingface":
        return "huggingface (HF Router)"
    return provider


def _endpoint(model: str | None, api_base: str | None) -> str:
    if api_base:
        return api_base
    if model and model.startswith("huggingface/"):
        return "https://router.huggingface.co"
    return "default"


def _usage_fields(response_obj: Any) -> dict[str, Any]:
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _finish_reason(response_obj: Any) -> str | None:
    choices = getattr(response_obj, "choices", None)
    if choices is None and isinstance(response_obj, dict):
        choices = response_obj.get("choices")
    if not choices:
        return None
    first = choices[0]
    return getattr(first, "finish_reason", None) if not isinstance(first, dict) else first.get("finish_reason")


class HuggingFaceCallLogger:
    """
    Callback LiteLLM che registra inizio, esito e latenza delle chiamate remote.

    Implementa volutamente solo gli hook sincroni: LiteLLM esegue i callback
    sincroni anche per le richieste asincrone, evitando righe duplicate.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(LOGGER_NAME)

    def log_pre_api_call(self, model: Any, messages: Any, kwargs: dict[str, Any]) -> None:
        try:
            params = kwargs.get("litellm_params") or {}
            fields: list[str] = [
                f"model={_truncate(model or kwargs.get('model'), 120)}",
                f"provider={_truncate(params.get('custom_llm_provider') or _infer_provider(str(model or '')), 80)}",
                f"endpoint={_truncate(_endpoint(str(model or ''), params.get('api_base')), 160)}",
            ]
            request_id = kwargs.get("litellm_call_id") or params.get("litellm_call_id") or kwargs.get("id")
            if request_id:
                fields.append(f"request_id={request_id}")
            if params.get("timeout") is not None:
                fields.append(f"timeout={params['timeout']}")
            if kwargs.get("num_retries") is not None:
                fields.append(f"max_retries={kwargs['num_retries']}")
            if kwargs.get("stream") or params.get("stream"):
                fields.append("stream=true")
            if kwargs.get("temperature") is not None:
                fields.append(f"temperature={kwargs['temperature']}")
            if kwargs.get("max_tokens") is not None:
                fields.append(f"max_tokens={kwargs['max_tokens']}")
            if _env_bool("RAG_LLM_LOG_PROMPTS", default=False):
                preview = _messages_preview(messages)
                if preview:
                    fields.append(f'prompt_preview="{preview}"')
            self.logger.info("HF/LLM request start | %s", " | ".join(fields))
        except Exception:
            # Il logging non deve mai interrompere l'inferenza.
            pass

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            params = kwargs.get("litellm_params") or {}
            hidden = getattr(response_obj, "_hidden_params", None) or {}
            elapsed = _elapsed_seconds(start_time, end_time)
            request_id = (
                getattr(response_obj, "id", None)
                or kwargs.get("litellm_call_id")
                or params.get("litellm_call_id")
            )
            fields: list[str] = ["status=200"]
            if elapsed is not None:
                fields.append(f"elapsed={elapsed:.3f}s")
            if request_id:
                fields.append(f"request_id={request_id}")
            model = getattr(response_obj, "model", None) or kwargs.get("model")
            if model:
                fields.append(f"model={_truncate(model, 120)}")
            finish_reason = _finish_reason(response_obj)
            if finish_reason:
                fields.append(f"finish_reason={finish_reason}")
            usage = _usage_fields(response_obj)
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if usage.get(key) is not None:
                    fields.append(f"{key}={usage[key]}")
            cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
            if cost is not None:
                fields.append(f"cost_usd={cost:.6f}")
            if kwargs.get("cache_hit"):
                fields.append("cache_hit=true")
            headers = _safe_headers(
                kwargs.get("response_headers")
                or (hidden.get("additional_headers") if isinstance(hidden, dict) else getattr(hidden, "additional_headers", None))
                or (hidden.get("headers") if isinstance(hidden, dict) else getattr(hidden, "headers", None))
            )
            if headers:
                fields.append(f"headers={headers}")
            self.logger.info("HF/LLM request success | %s", " | ".join(fields))
        except Exception:
            pass

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            exc = kwargs.get("exception") or response_obj
            elapsed = _elapsed_seconds(start_time, end_time)
            exc_resp = getattr(exc, "response", None)
            status_code = (
                getattr(exc, "status_code", None)
                or getattr(exc, "status", None)
                or getattr(exc_resp, "status_code", None)
            )
            fields: list[str] = []
            if status_code is not None:
                fields.append(f"status={status_code}")
            if elapsed is not None:
                fields.append(f"elapsed={elapsed:.3f}s")
            fields.append(f"error_type={type(exc).__name__}")
            error_message = getattr(exc, "message", None) or exc
            fields.append(f'error="{_truncate(error_message)}"')
            request_id = kwargs.get("litellm_call_id") or (kwargs.get("litellm_params") or {}).get(
                "litellm_call_id"
            )
            if request_id:
                fields.append(f"request_id={request_id}")
            headers = _safe_headers(
                getattr(exc, "headers", None)
                or getattr(exc_resp, "headers", None)
                or kwargs.get("response_headers")
            )
            if headers:
                fields.append(f"headers={headers}")
            self.logger.warning("HF/LLM request failure | %s", " | ".join(fields))
        except Exception:
            pass


def log_llm_skipped_event(model: str, reason: str) -> None:
    """
    Registra sul logger ``gs.llm`` il salto di una chiamata remota (es. quando il
    circuit-breaker di sessione per ``HF_TOKEN`` è già attivo).
    """
    try:
        logger = configure_llm_logging()
        if not logger.disabled:
            logger.warning(
                'HF/LLM request skipped | model=%s | reason="%s"',
                _truncate(model, 120),
                _truncate(reason, 300),
            )
    except Exception:
        pass


def log_llm_fallback_event(primary_model: str, target_model: str, reason: str) -> None:
    """
    Registra sul logger ``gs.llm`` l'attivazione di un fallback (verso Ollama o
    verso il motore neuro-lessicale locale BERT+RAG) a seguito del fallimento del
    provider LLM remoto primario.
    """
    try:
        logger = configure_llm_logging()
        if not logger.disabled:
            logger.warning(
                'HF/LLM fallback activated | primary=%s | target=%s | reason="%s"',
                _truncate(primary_model, 120),
                _truncate(target_model, 120),
                _truncate(reason, 300),
            )
    except Exception:
        pass


def _build_litellm_callback(logger: logging.Logger) -> "HuggingFaceCallLogger":
    """
    Crea l'istanza del callback ereditando anche da ``litellm.CustomLogger``.

    LiteLLM dispaccia gli hook solo per le istanze di ``CustomLogger``: la
    sottoclasse viene creata qui (e non a livello di modulo) per non importare
    LiteLLM quando il logging è disabilitato o si usa solo un motore locale.
    """
    from litellm.integrations.custom_logger import CustomLogger

    class _LiteLLMHuggingFaceCallLogger(HuggingFaceCallLogger, CustomLogger):
        pass

    return _LiteLLMHuggingFaceCallLogger(logger)


def install_llm_call_logging(force: bool = False) -> bool:
    """
    Configura il logger e registra il callback LiteLLM una sola volta.

    Returns:
        True se il callback è stato registrato (o era già presente), False se
        il logging è disabilitato o LiteLLM non è installato.
    """
    global _callback_installed

    if _resolve_destination() == "off" and not force:
        return False

    logger = configure_llm_logging(force=force)
    if logger.disabled:
        return False

    with _configure_lock:
        if _callback_installed and not force:
            return True
        try:
            import litellm
        except Exception:
            logger.debug("LiteLLM non disponibile: callback di logging non registrato")
            return False

        callbacks = list(getattr(litellm, "callbacks", None) or [])
        if not any(isinstance(callback, HuggingFaceCallLogger) for callback in callbacks):
            callbacks.append(_build_litellm_callback(logger))
        # LiteLLM propaga i callback globali anche su success/failure e, per le
        # chiamate sincrone di DSPy, invoca gli hook sincroni una sola volta.
        litellm.callbacks = callbacks
        _callback_installed = True
    return True


def reset_llm_logging_for_tests() -> None:
    """Ripristina lo stato globale del logger (solo per la suite di test)."""
    global _configured, _callback_installed

    logger = logging.getLogger(LOGGER_NAME)
    with _configure_lock:
        _remove_our_handlers(logger)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
        logger.disabled = False
        _configured = False
        _callback_installed = False
