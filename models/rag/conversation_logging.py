"""
Logging JSONL opt-in delle conversazioni complete DSPy <-> LLM per il debug
delle sessioni di integrazione dei filologi.

A differenza di ``models/rag/llm_logging.py`` (orientato allo stato HTTP e agli
header di LiteLLM su stdout), questo modulo cattura a livello di framework DSPy
3.4+ (tramite ``BaseCallback`` e ``UsageTracker`` in ``dspy.context``) l'intera
sequenza di turni di ogni invocazione RAG con LLM:
- input del filologo e vincoli papirologici / Human-in-the-loop;
- passi paralleli (*loci similes*) recuperati dal retriever;
- ogni step DSPy (``GreekInfillingSignature``, eventuale ``ConstraintRefinementSignature``)
  e i relativi scambi completi con l'LLM (``messages`` inviati, risposta grezza,
  output parsato da DSPy, token, costo, latenza, flag ``cache_hit``, errori);
- eventuali eventi di circuit-breaker e fallback verso Ollama o ``neuro_lexical``;
- output finale restituito al filologo.

Configurazione tramite variabili d'ambiente:
- ``RAG_CONVERSATION_LOG_ENABLED``: ``false`` (default) o ``true`` per attivare la scrittura.
- ``RAG_CONVERSATION_LOG_FILE``: percorso del file JSONL (default ``logs/dspy_conversations.jsonl``).

Le credenziali (``api_key``, header ``Authorization``, token e segreti) vengono
rimosse prima della serializzazione.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

import dspy
from dspy.utils.callback import BaseCallback
from dspy.utils.usage_tracker import UsageTracker

DEFAULT_CONVERSATION_LOG_FILE = Path("logs/dspy_conversations.jsonl")
DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20 MiB per file prima della rotazione
DEFAULT_BACKUP_COUNT = 5

_SENSITIVE_KEY_EXACT = frozenset(
    {
        "token",
        "api_key",
        "apikey",
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "password",
        "passwd",
        "secret",
        "x-api-key",
    }
)
_SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "cookie",
    "password",
)
_SENSITIVE_KEY_SUFFIXES = (
    "_token",
    "-token",
    "_apikey",
    "-apikey",
)

_writer_lock = threading.Lock()
_cached_handler_path: Path | None = None
_cached_handler: RotatingFileHandler | None = None
_warned_write_error = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "si", "sì")


def is_conversation_logging_enabled() -> bool:
    """Restituisce True se il logging JSONL delle conversazioni DSPy <-> LLM è abilitato."""
    return _env_bool("RAG_CONVERSATION_LOG_ENABLED", default=False)


def resolve_conversation_log_file() -> Path:
    """Restituisce il percorso del file JSONL configurato tramite ``RAG_CONVERSATION_LOG_FILE``."""
    raw = os.getenv("RAG_CONVERSATION_LOG_FILE") or str(DEFAULT_CONVERSATION_LOG_FILE)
    return Path(raw).expanduser()


def _is_sensitive_key(key: str) -> bool:
    # Esclude solo credenziali e chiavi di autenticazione preservando i campi NLP/token-count
    # legittimi (es. prompt_tokens, completion_tokens, max_tokens, token_str).
    low = key.lower().strip()
    if low in _SENSITIVE_KEY_EXACT or low.endswith(_SENSITIVE_KEY_SUFFIXES):
        return True
    return any(fragment in low for fragment in _SENSITIVE_KEY_SUBSTRINGS)


def _to_json_safe(value: Any) -> Any:
    """Converte ricorsivamente oggetti DSPy, Pydantic o Python in strutture JSON-serializzabili."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _to_json_safe(value.model_dump())
        except Exception:
            pass
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return _to_json_safe(asdict(value))
        except Exception:
            pass
    if isinstance(value, dspy.Prediction):
        return _serialize_prediction(value)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if _is_sensitive_key(key_str):
                continue
            sanitized[key_str] = _to_json_safe(v)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    # Supporta eventuali oggetti Prediction/Example di DSPy che espongono `.toDict()`
    if hasattr(value, "toDict") and callable(value.toDict):
        try:
            return _to_json_safe(value.toDict())
        except Exception:
            pass
    return str(value)


def _sanitize_mapping(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    safe = _to_json_safe(data)
    return safe if isinstance(safe, dict) else {}


def _serialize_prediction(pred: Any) -> dict[str, Any]:
    """Estrae in forma leggibile e serializzabile i campi principali di una ``dspy.Prediction``."""
    if pred is None:
        return {}
    if isinstance(pred, dict):
        return _sanitize_mapping(pred)
    if hasattr(pred, "toDict") and callable(pred.toDict):
        try:
            return _sanitize_mapping(pred.toDict())
        except Exception:
            pass
    store = getattr(pred, "_store", None)
    if isinstance(store, dict):
        return _sanitize_mapping(store)
    return {"repr": str(pred)}


def _serialize_lm_outputs(outputs: Any) -> list[Any]:
    """Normalizza le risposte grezze consegnate a ``BaseCallback.on_lm_end``."""
    if outputs is None:
        return []
    if hasattr(outputs, "message") and hasattr(outputs, "model"):
        # Gestisce dspy.lm15.Response per eventuali chiamate native DSPy 3.4+
        msg = getattr(outputs, "message", None)
        parts = getattr(msg, "parts", None)
        text = "".join(getattr(p, "text", str(p)) for p in (parts or []))
        return [text] if text else [_to_json_safe(outputs)]
    if isinstance(outputs, (list, tuple)):
        return [_to_json_safe(item) for item in outputs]
    return [_to_json_safe(outputs)]


def _get_or_create_handler(log_path: Path) -> RotatingFileHandler:
    """Restituisce un ``RotatingFileHandler`` thread-safe ricreandolo se il percorso cambia."""
    global _cached_handler_path, _cached_handler
    resolved = log_path.resolve()
    if _cached_handler is not None and _cached_handler_path == resolved:
        return _cached_handler

    if _cached_handler is not None:
        try:
            _cached_handler.close()
        except Exception:
            pass
        _cached_handler = None
        _cached_handler_path = None

    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        resolved,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    _cached_handler = handler
    _cached_handler_path = resolved
    return handler


def _write_jsonl_record(record: dict[str, Any], log_path: Path | None = None) -> None:
    """Scrive una singola riga JSONL in modo thread-safe senza mai propagare eccezioni all'inferenza."""
    global _warned_write_error
    target_path = log_path or resolve_conversation_log_file()
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _writer_lock:
        try:
            handler = _get_or_create_handler(target_path)
            # Usiamo direttamente RotatingFileHandler tramite un LogRecord minimale
            # così da delegare rotazione automatica (maxBytes/backupCount) e encoding UTF-8.
            log_record = logging.LogRecord(
                name="gs.dspy.conversation",
                level=logging.INFO,
                pathname=__file__,
                lineno=0,
                msg=line,
                args=(),
                exc_info=None,
            )
            handler.emit(log_record)
            handler.flush()
        except Exception as exc:
            if not _warned_write_error:
                _warned_write_error = True
                print(
                    f"[gs.dspy] Impossibile scrivere il log conversazioni su '{target_path}': {exc}",
                    file=sys.stderr,
                )


def reset_conversation_logging_for_tests() -> None:
    """Chiude e azzera gli handler file in cache (usato esclusivamente nei test)."""
    global _cached_handler_path, _cached_handler, _warned_write_error
    with _writer_lock:
        if _cached_handler is not None:
            try:
                _cached_handler.close()
            except Exception:
                pass
        _cached_handler = None
        _cached_handler_path = None
        _warned_write_error = False


class _TraceUsageTracker(UsageTracker):
    """
    Tracker locale per singola esecuzione usato per distinguere deterministicamente
    un ``cache_hit`` DSPy da una chiamata effettiva al modello.

    In DSPy 3.4 ``dspy.clients.execution.finalize`` invoca ``usage_tracker.add_usage``
    esclusivamente quando ``not result.cache_hit``: confrontando ``live_calls`` tra
    ``on_lm_start`` e ``on_lm_end`` il rilevamento funziona su qualsiasi backend
    (LiteLLM, Ollama, HuggingFaceLocalLM, DummyLM).
    """

    def __init__(self, parent_tracker: UsageTracker | None = None) -> None:
        super().__init__()
        self._parent_tracker = parent_tracker
        self.live_calls: int = 0
        self.last_usage: dict[str, Any] | None = None

    def add_usage(self, lm: str, usage_entry: dict[str, Any]) -> None:
        self.live_calls += 1
        flattened = self._flatten_usage_entry(usage_entry or {})
        self.last_usage = _sanitize_mapping(flattened)
        super().add_usage(lm, usage_entry)
        if self._parent_tracker is not None:
            try:
                self._parent_tracker.add_usage(lm, usage_entry)
            except Exception:
                pass


class DSPyConversationCallback(BaseCallback):
    """
    Callback DSPy dedicato alla singola esecuzione di ``fill_mask_rag``.

    Raccoglie in strutture isolate per richiesta:
    - gli step dei predittori con signature (``GreekInfillingSignature``, ``ConstraintRefinementSignature``);
    - gli scambi completi inviati/ricevuti da ciascuna chiamata LLM (prompt/messages,
      risposta grezza, token, costo, latenza, flag ``cache_hit`` ed eventuali errori di parsing).
    """

    def __init__(
        self,
        *,
        attempt: str = "primary",
        usage_tracker: _TraceUsageTracker | None = None,
        steps_sink: list[dict[str, Any]] | None = None,
        llm_calls_sink: list[dict[str, Any]] | None = None,
    ) -> None:
        self.attempt = attempt
        self.usage_tracker = usage_tracker
        self.steps: list[dict[str, Any]] = steps_sink if steps_sink is not None else []
        self.llm_calls: list[dict[str, Any]] = llm_calls_sink if llm_calls_sink is not None else []
        self._pending_modules: dict[str, dict[str, Any]] = {}
        self._active_step_stack: list[dict[str, Any]] = []
        self._pending_lms: dict[str, dict[str, Any]] = {}

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        # Filtriamo il modulo contenitore esterno (`GreekRAGInfiller`) per tracciare in modo
        # leggibile i singoli predittori dotati di signature (`generate_infill`, `refine_infill`).
        signature = getattr(instance, "signature", None)
        if signature is None:
            return
        step_name = getattr(signature, "__name__", None) or type(instance).__name__
        raw_kwargs = inputs.get("kwargs") if isinstance(inputs, dict) and "kwargs" in inputs else inputs
        step_entry: dict[str, Any] = {
            "step_name": step_name,
            "predictor_class": type(instance).__name__,
            "attempt": self.attempt,
            "inputs": _sanitize_mapping(raw_kwargs if isinstance(raw_kwargs, dict) else {"raw": raw_kwargs}),
            "parsed_output": None,
            "adapter_parse_errors": [],
            "latency_ms": None,
            "error": None,
        }
        self._pending_modules[call_id] = {
            "entry": step_entry,
            "t0": time.perf_counter(),
        }
        self._active_step_stack.append(step_entry)

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: BaseException | None = None,
    ) -> None:
        pending = self._pending_modules.pop(call_id, None)
        if pending is None:
            return
        entry = pending["entry"]
        entry["latency_ms"] = round((time.perf_counter() - pending["t0"]) * 1000.0, 2)
        if exception is not None:
            entry["error"] = f"{type(exception).__name__}: {exception}"
        else:
            entry["parsed_output"] = _serialize_prediction(outputs)

        # Rimuove lo step dallo stack mantenendo l'ordine anche in caso di eccezioni annidate
        for idx in range(len(self._active_step_stack) - 1, -1, -1):
            if self._active_step_stack[idx] is entry:
                self._active_step_stack.pop(idx)
                break

        self.steps.append(entry)

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ) -> None:
        if exception is not None and self._active_step_stack:
            self._active_step_stack[-1]["adapter_parse_errors"].append(
                f"{type(exception).__name__}: {exception}"
            )

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        history = getattr(instance, "history", None)
        start_hist_len = len(history) if isinstance(history, list) else 0
        active_step = self._active_step_stack[-1]["step_name"] if self._active_step_stack else "direct_lm"
        model_name = getattr(instance, "model", None) or getattr(instance, "model_id", None) or "unknown"

        raw_inputs = inputs if isinstance(inputs, dict) else {}
        messages = _to_json_safe(raw_inputs.get("messages")) if raw_inputs.get("messages") is not None else None
        prompt = _to_json_safe(raw_inputs.get("prompt")) if raw_inputs.get("prompt") is not None else None
        lm_kwargs = _sanitize_mapping(raw_inputs.get("kwargs"))

        self._pending_lms[call_id] = {
            "t0": time.perf_counter(),
            "step_name": active_step,
            "model": str(model_name),
            "lm_class": type(instance).__name__,
            "lm_instance": instance,
            "start_hist_len": start_hist_len,
            "start_live_calls": self.usage_tracker.live_calls if self.usage_tracker is not None else None,
            "messages": messages,
            "prompt": prompt,
            "kwargs": lm_kwargs,
        }

    def _match_history_entry(
        self,
        instance: Any,
        start_hist_len: int,
        serialized_outputs: list[Any],
    ) -> dict[str, Any] | None:
        history = getattr(instance, "history", None)
        if not isinstance(history, list) or len(history) <= start_hist_len:
            return None
        # Scansiona al contrario i nuovi record di history: se la stessa istanza LM è condivisa
        # tra più thread (es. confronti paralleli), cerca il match con gli output correnti.
        recent = history[start_hist_len:]
        for entry in reversed(recent):
            if not isinstance(entry, dict):
                continue
            if serialized_outputs and _serialize_lm_outputs(entry.get("outputs")) == serialized_outputs:
                return entry
        last = recent[-1]
        return last if isinstance(last, dict) else None

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: BaseException | None = None,
    ) -> None:
        pending = self._pending_lms.pop(call_id, None)
        if pending is None:
            return

        latency_ms = round((time.perf_counter() - pending["t0"]) * 1000.0, 2)
        serialized_outputs = _serialize_lm_outputs(outputs) if exception is None else []

        cache_hit: bool | None = None
        usage: dict[str, Any] = {}
        if self.usage_tracker is not None and pending["start_live_calls"] is not None:
            if exception is None:
                cache_hit = self.usage_tracker.live_calls == pending["start_live_calls"]
                if not cache_hit and self.usage_tracker.last_usage:
                    usage = dict(self.usage_tracker.last_usage)
            else:
                cache_hit = False

        hist_entry = (
            self._match_history_entry(
                pending["lm_instance"],
                pending["start_hist_len"],
                serialized_outputs,
            )
            if exception is None
            else None
        )
        if hist_entry is not None:
            if not usage and isinstance(hist_entry.get("usage"), dict):
                usage = _sanitize_mapping(hist_entry.get("usage"))
            # Se il prompt non era esplicito in on_lm_start ma è stato ricostruito in history,
            # lo recuperiamo qui per garantire visibilità completa della conversazione.
            if pending["messages"] is None and hist_entry.get("messages") is not None:
                pending["messages"] = _to_json_safe(hist_entry.get("messages"))
            cost_usd = hist_entry.get("cost")
            response_model = hist_entry.get("response_model") or pending["model"]
        else:
            cost_usd = None
            response_model = pending["model"]

        call_record = {
            "call_index": len(self.llm_calls) + 1,
            "step_name": pending["step_name"],
            "attempt": self.attempt,
            "model": pending["model"],
            "response_model": str(response_model) if response_model else pending["model"],
            "lm_class": pending["lm_class"],
            "messages": pending["messages"],
            "prompt": pending["prompt"],
            "kwargs": pending["kwargs"],
            "raw_outputs": serialized_outputs,
            "usage": usage,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "error": f"{type(exception).__name__}: {exception}" if exception is not None else None,
        }
        self.llm_calls.append(call_record)

        # In caso di errore HTTP/LLM, il modulo/predittore non produce output:
        # memorizziamo comunque lo step pendente se non è già stato chiuso da on_module_end.
        # (on_module_end rimane il punto canonico di chiusura dello step).


class ConversationTraceRecorder:
    """
    Raccoglitore di alto livello per una singola invocazione ``fill_mask_rag``.

    Aggrega contesto di sessione, vincoli, passi recuperati, tutti i turni
    DSPy <-> LLM (primari e di fallback) e risultato finale, scrivendo un unico
    record JSONL al termine dell'esecuzione.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        model_id: str | None = None,
        checkpoint: str = "",
        requested_backend: str = "auto",
        input_payload: dict[str, Any] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id.strip() if isinstance(session_id, str) and session_id.strip() else None
        self.model_id = str(model_id) if model_id else None
        self.checkpoint = checkpoint
        self.requested_backend = requested_backend
        self.input_payload = _sanitize_mapping(input_payload or {})
        self.log_path = log_path

        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()

        self.retrieval_payload: dict[str, Any] = {
            "passages_sent_to_lm": [],
            "total_retrieved_count": 0,
        }
        self.dspy_steps: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.fallback_events: list[dict[str, Any]] = []
        self._finalized = False

    @contextlib.contextmanager
    def make_dspy_context(
        self,
        lm: dspy.BaseLM,
        attempt: str = "primary",
    ) -> Iterator[None]:
        """Attiva in ``dspy.context`` un callback e un tracker isolati per questo tentativo."""
        existing_callbacks = list(dspy.settings.get("callbacks", []) or [])
        parent_tracker = getattr(dspy.settings, "usage_tracker", None)
        tracker = _TraceUsageTracker(parent_tracker=parent_tracker)
        callback = DSPyConversationCallback(
            attempt=attempt,
            usage_tracker=tracker,
            steps_sink=self.dspy_steps,
            llm_calls_sink=self.llm_calls,
        )
        with dspy.context(
            lm=lm,
            callbacks=[*existing_callbacks, callback],
            usage_tracker=tracker,
        ):
            yield

    def record_fallback(
        self,
        primary: str,
        target: str,
        reason: str,
        kind: str = "fallback",
    ) -> None:
        self.fallback_events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "primary": primary,
                "target": target,
                "reason": reason,
            }
        )

    def set_retrieval(
        self,
        retrieved_passages: list[str] | None,
        all_retrieved_passages: list[str] | None = None,
    ) -> None:
        sent = list(retrieved_passages or [])
        all_p = list(all_retrieved_passages or sent)
        self.retrieval_payload = {
            "passages_sent_to_lm": _to_json_safe(sent),
            "total_retrieved_count": len(all_p),
        }

    def finalize_and_write(
        self,
        *,
        effective_backend: str,
        effective_model: str,
        status: str = "ok",
        candidates: list[tuple[str, float]] | None = None,
        philological_reasoning: str = "",
        error: str | None = None,
    ) -> dict[str, Any]:
        if self._finalized:
            return {}
        self._finalized = True

        duration_ms = round((time.perf_counter() - self._t0) * 1000.0, 2)
        record: dict[str, Any] = {
            "timestamp": self.started_at,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "requested_backend": self.requested_backend,
            "effective_backend": effective_backend,
            "effective_model": effective_model,
            "status": status,
            "duration_ms": duration_ms,
            "input": self.input_payload,
            "retrieval": self.retrieval_payload,
            "dspy_steps": self.dspy_steps,
            "llm_calls": self.llm_calls,
            "fallback_events": self.fallback_events,
            "final_output": {
                "philological_reasoning": philological_reasoning,
                "candidates": [
                    {"supplement": supp, "score": float(score)}
                    for supp, score in (candidates or [])
                ],
            },
            "error": error,
        }
        _write_jsonl_record(record, log_path=self.log_path)
        return record


def load_conversation_records(
    log_path: Path | None = None,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
    model: str | None = None,
    last: int | None = None,
) -> list[dict[str, Any]]:
    """Carica e filtra le tracce di conversazione da un file JSONL."""
    target = log_path or resolve_conversation_log_file()
    if not target.exists():
        return []

    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue

            if session_id:
                rec_sess = str(rec.get("session_id") or "")
                if rec_sess != session_id and not rec_sess.startswith(session_id):
                    continue
            if trace_id:
                rec_trace = str(rec.get("trace_id") or "")
                if rec_trace != trace_id and not rec_trace.startswith(trace_id):
                    continue
            if model:
                needle = model.lower()
                hay = f"{rec.get('checkpoint', '')} {rec.get('effective_model', '')}".lower()
                if needle not in hay:
                    continue

            records.append(rec)

    if last is not None and last > 0:
        return records[-last:]
    return records


def format_conversation_record(
    record: dict[str, Any],
    *,
    show_system: bool = True,
    raw_json: bool = False,
) -> str:
    """Formatta una traccia JSONL in un blocco testuale leggibile per il debug da terminale."""
    if raw_json:
        return json.dumps(record, ensure_ascii=False, indent=2)

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(
        f"[{record.get('timestamp', '?')}] status={record.get('status', '?')} "
        f"| duration={record.get('duration_ms', '?')}ms"
    )
    lines.append(
        f"session_id={record.get('session_id') or '-'} | trace_id={record.get('trace_id', '-')}"
    )
    lines.append(
        f"model={record.get('checkpoint', '-')} -> effective={record.get('effective_model', '-')} "
        f"({record.get('effective_backend', '-')})"
    )

    inp = record.get("input") or {}
    lines.append("-" * 88)
    lines.append(f"CONTESTO: {inp.get('context_with_gap') or inp.get('raw_text', '')}")
    lines.append(
        f"VINCOLI : chars=[{inp.get('min_chars')}, {inp.get('max_chars')}] "
        f"| prefix='{inp.get('word_prefix', '')}' | suffix='{inp.get('word_suffix', '')}' "
        f"| K={inp.get('K')}"
    )
    if inp.get("human_feedback"):
        lines.append(f"HITL FEEDBACK: {inp.get('human_feedback')}")
    if inp.get("rejected_candidates"):
        lines.append(f"HITL SCARTATI: {', '.join(str(x) for x in inp.get('rejected_candidates') or [])}")

    ret = record.get("retrieval") or {}
    passages = ret.get("passages_sent_to_lm") or []
    lines.append(
        f"RETRIEVAL: {len(passages)} passi inviati all'LLM "
        f"(su {ret.get('total_retrieved_count', len(passages))} totali)"
    )
    for idx, p in enumerate(passages, 1):
        lines.append(f"  [{idx}] {p}")

    fallbacks = record.get("fallback_events") or []
    if fallbacks:
        lines.append("EVENTI FALLBACK / SKIP:")
        for fb in fallbacks:
            lines.append(
                f"  - [{fb.get('kind')}] {fb.get('primary')} -> {fb.get('target')}: {fb.get('reason')}"
            )

    llm_calls = record.get("llm_calls") or []
    lines.append(f"CONVERSAZIONE LLM ({len(llm_calls)} chiamate):")
    for call in llm_calls:
        cache_tag = "CACHE_HIT" if call.get("cache_hit") else "LIVE"
        usage = call.get("usage") or {}
        tok_str = (
            f"tokens(in={usage.get('prompt_tokens')}, out={usage.get('completion_tokens')})"
            if usage
            else "tokens(-)"
        )
        lines.append(
            f"\n  --- Call #{call.get('call_index')} | step={call.get('step_name')} "
            f"| attempt={call.get('attempt')} | {cache_tag} | {call.get('latency_ms')}ms | {tok_str} ---"
        )
        messages = call.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                role = str(msg.get("role", "unknown")).upper() if isinstance(msg, dict) else "MSG"
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                if role == "SYSTEM" and not show_system:
                    lines.append(f"  [{role}] <nascosto con --no-system: {len(str(content))} caratteri>")
                    continue
                lines.append(f"  [{role}]")
                lines.append(str(content))
        elif call.get("prompt"):
            lines.append("  [PROMPT]")
            lines.append(str(call.get("prompt")))

        if call.get("error"):
            lines.append(f"  [ERROR] {call.get('error')}")
        else:
            lines.append("  [ASSISTANT RAW]")
            for out in call.get("raw_outputs") or []:
                lines.append(str(out))

    steps = record.get("dspy_steps") or []
    if steps:
        lines.append("\nSTEP DSPy PARSATI:")
        for st in steps:
            lines.append(
                f"  * {st.get('step_name')} ({st.get('attempt')}, {st.get('latency_ms')}ms)"
            )
            if st.get("adapter_parse_errors"):
                lines.append(f"    parse_errors={st.get('adapter_parse_errors')}")
            if st.get("error"):
                lines.append(f"    error={st.get('error')}")
            elif st.get("parsed_output"):
                lines.append(
                    f"    parsed_output={json.dumps(st.get('parsed_output'), ensure_ascii=False)}"
                )

    final_out = record.get("final_output") or {}
    lines.append("\nOUTPUT FINALE:")
    if final_out.get("philological_reasoning"):
        lines.append(f"  CoT: {final_out.get('philological_reasoning')}")
    cands = final_out.get("candidates") or []
    lines.append(
        "  Candidati: "
        + (", ".join(f"{c['supplement']} ({c['score']:.4f})" for c in cands) if cands else "<nessuno>")
    )
    if record.get("error"):
        lines.append(f"  ERRORE: {record.get('error')}")

    return "\n".join(lines)


def _format_sessions_summary(records: list[dict[str, Any]]) -> str:
    if not records:
        return "Nessuna sessione registrata."
    by_session: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = str(rec.get("session_id") or f"(no-session:{rec.get('trace_id', '')[:8]})")
        by_session.setdefault(key, []).append(rec)

    lines = ["SESSION_ID | TURNI | ULTIMO_TS | MODELLI | CONTESTO"]
    lines.append("-" * 88)
    for sess_id, items in by_session.items():
        last_rec = items[-1]
        models = sorted({str(r.get("checkpoint", "?")) for r in items})
        ctx = str((last_rec.get("input") or {}).get("context_with_gap") or "")
        if len(ctx) > 55:
            ctx = ctx[:52] + "..."
        lines.append(
            f"{sess_id} | {len(items)} | {last_rec.get('timestamp', '?')} | {', '.join(models)} | {ctx}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ispeziona i log JSONL delle conversazioni DSPy <-> LLM per il debug delle sessioni."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help=f"Percorso del file JSONL (default: env RAG_CONVERSATION_LOG_FILE o {DEFAULT_CONVERSATION_LOG_FILE})",
    )
    parser.add_argument(
        "-n",
        "--last",
        type=int,
        default=5,
        help="Mostra gli ultimi N record (0 per mostrare tutti; default: 5)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Filtra per session_id (match esatto o prefisso)",
    )
    parser.add_argument(
        "--trace-id",
        type=str,
        default=None,
        help="Filtra per trace_id (match esatto o prefisso)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filtra per sottostringa del nome modello/checkpoint",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Mostra un riepilogo tabellare delle sessioni registrate",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Omette il testo completo del messaggio SYSTEM (mostrando solo il conteggio caratteri)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Stampa i record filtrati in JSON formattato",
    )
    args = parser.parse_args()

    log_path = Path(args.file).expanduser() if args.file else resolve_conversation_log_file()
    if not log_path.exists():
        print(
            f"Nessun file di log trovato in '{log_path}'. "
            "Imposta RAG_CONVERSATION_LOG_ENABLED=true per abilitare la registrazione."
        )
        return

    records = load_conversation_records(
        log_path=log_path,
        session_id=args.session_id,
        trace_id=args.trace_id,
        model=args.model,
        last=None if args.list_sessions or args.last <= 0 else args.last,
    )
    if args.list_sessions:
        items = records[-args.last :] if args.last and args.last > 0 else records
        print(_format_sessions_summary(items))
        return

    if not records:
        print(f"Nessuna traccia trovata in '{log_path}' con i filtri specificati.")
        return

    for rec in records:
        print(
            format_conversation_record(
                rec,
                show_system=not args.no_system,
                raw_json=args.json,
            )
        )


if __name__ == "__main__":
    main()
