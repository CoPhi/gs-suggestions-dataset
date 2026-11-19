"""
Gestore centralizzato della cache in-memory per i modelli BERT/MLM e rilascio della memoria al kernel.

Unifica in un'unica cache LRU con TTL di inattività le istanze usate da:
1. Predizioni BERT dirette (`SuggestionsService._predict_bert`)
2. Motore neuro-lessicale RAG (`models.rag.inference._get_cached_mlm`)
3. Embedder semantico del retriever ibrido (`models.rag.retriever.GreekBertEmbedder`),
   condividendo `.base_model` senza duplicare i pesi in RAM.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from backend.config.settings import GS_MAX_BERT_MODELS

_LIBC: Any | None = None
_LIBC_INITIALIZED = False


def _get_libc() -> Any | None:
    """Carica `libc.so.6` su Linux una sola volta per invocare `malloc_trim(0)`."""
    global _LIBC, _LIBC_INITIALIZED
    if _LIBC_INITIALIZED:
        return _LIBC
    _LIBC_INITIALIZED = True
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                _LIBC = libc
        except Exception:
            _LIBC = None
    return _LIBC


def trim_memory(clear_jax: bool = False) -> None:
    """
    Forza la raccolta dei riferimenti Python e restituisce le pagine libere delle arene glibc al kernel.
    Esegue opzionalmente il flush delle cache XLA/JIT di JAX (solo se già importato) e della cache CUDA.
    """
    gc.collect()

    if clear_jax and "jax" in sys.modules:
        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass

    if "torch" in sys.modules:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Su Linux glibc trattiene spesso blocchi anonimi nelle arene dopo la distruzione dei tensori:
    # malloc_trim(0) restituisce immediatamente le pagine non usate all'OS riducendo l'RSS visibile.
    libc = _get_libc()
    if libc is not None:
        try:
            libc.malloc_trim(0)
        except Exception:
            pass


def _default_mlm_loader(checkpoint: str, device: str | None = None) -> tuple[Any, Any]:
    """Carica un modello MaskedLM Hugging Face e il suo tokenizer in modalità eval."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN")
    kwargs: dict[str, Any] = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kwargs)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer


class SharedBertCache:
    """
    Cache thread-safe condivisa per modelli `AutoModelForMaskedLM` e relativi tokenizer.
    Supporta serializzazione dei caricamenti per-checkpoint, limite LRU ed eviction per TTL.
    """

    def __init__(self, max_models: int = GS_MAX_BERT_MODELS) -> None:
        self.cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._last_used: dict[str, float] = {}
        self._global_lock = threading.Lock()
        self._ckpt_locks: dict[str, threading.Lock] = {}
        self.max_models = max(1, int(max_models))

    def _get_ckpt_lock(self, checkpoint: str) -> threading.Lock:
        with self._global_lock:
            lock = self._ckpt_locks.get(checkpoint)
            if lock is None:
                lock = threading.Lock()
                self._ckpt_locks[checkpoint] = lock
            return lock

    def get_or_load_mlm(
        self,
        checkpoint: str,
        loader_fn: Callable[[str], tuple[Any, Any]] | None = None,
        max_models: int | None = None,
        device: str | None = None,
    ) -> tuple[Any, Any]:
        """Restituisce `(mlm_model, tokenizer)` dalla cache oppure lo carica una sola volta sotto lock."""
        now = time.monotonic()
        with self._global_lock:
            if checkpoint in self.cache:
                self.cache.move_to_end(checkpoint)
                self._last_used[checkpoint] = now
                return self.cache[checkpoint]

        ckpt_lock = self._get_ckpt_lock(checkpoint)
        with ckpt_lock:
            now = time.monotonic()
            with self._global_lock:
                if checkpoint in self.cache:
                    self.cache.move_to_end(checkpoint)
                    self._last_used[checkpoint] = now
                    return self.cache[checkpoint]

            if loader_fn is not None:
                model, tokenizer = loader_fn(checkpoint)
            else:
                model, tokenizer = _default_mlm_loader(checkpoint, device=device)

            if hasattr(model, "eval"):
                model.eval()

            limit = max(1, int(max_models if max_models is not None else self.max_models))
            evicted = False
            with self._global_lock:
                self.cache[checkpoint] = (model, tokenizer)
                self.cache.move_to_end(checkpoint)
                self._last_used[checkpoint] = time.monotonic()
                while len(self.cache) > limit:
                    evicted_key, _ = self.cache.popitem(last=False)
                    self._last_used.pop(evicted_key, None)
                    evicted = True

            if evicted:
                trim_memory(clear_jax=False)

            return model, tokenizer

    def get_encoder_backbone(
        self,
        checkpoint: str,
        fallback_checkpoint: str | None = None,
        device: str | None = None,
    ) -> tuple[Any, Any]:
        """
        Restituisce `(encoder_backbone, tokenizer)` condividendo i pesi del modello MLM già in cache.
        Per `AutoModelForMaskedLM` (es. RobertaForMaskedLM/BertForMaskedLM) `.base_model` espone
        direttamente il backbone encoder che restituisce `last_hidden_state`.
        """
        try:
            mlm_model, tokenizer = self.get_or_load_mlm(checkpoint, device=device)
        except Exception:
            if not fallback_checkpoint or fallback_checkpoint == checkpoint:
                raise
            mlm_model, tokenizer = self.get_or_load_mlm(fallback_checkpoint, device=device)

        encoder = getattr(mlm_model, "base_model", mlm_model)
        if hasattr(encoder, "eval"):
            encoder.eval()
        return encoder, tokenizer

    def pop(self, checkpoint: str, default: Any = None) -> tuple[Any, Any] | Any:
        """Rimuove esplicitamente un checkpoint dalla cache e rilascia la memoria."""
        with self._global_lock:
            item = self.cache.pop(checkpoint, None)
            self._last_used.pop(checkpoint, None)
        if item is not None:
            del item
            trim_memory(clear_jax=False)
        return default

    def clear(self) -> None:
        """Svuota completamente la cache BERT e rilascia la memoria al sistema."""
        with self._global_lock:
            had_items = bool(self.cache)
            self.cache.clear()
            self._last_used.clear()
        if had_items:
            trim_memory(clear_jax=False)

    def evict_expired(self, ttl_seconds: float) -> list[str]:
        """Scarica i modelli non utilizzati da più di `ttl_seconds` secondi."""
        if ttl_seconds <= 0:
            return []

        now = time.monotonic()
        expired_keys: list[str] = []
        with self._global_lock:
            for key in list(self.cache.keys()):
                last_used = self._last_used.get(key)
                # Se una voce è stata inserita manualmente nei test direttamente su `.cache`,
                # inizializza il timestamp al momento corrente evitando rimozioni premature.
                if last_used is None:
                    self._last_used[key] = now
                    continue
                if (now - last_used) > ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self.cache.pop(key, None)
                self._last_used.pop(key, None)

        if expired_keys:
            trim_memory(clear_jax=False)
        return expired_keys


shared_bert_cache = SharedBertCache(max_models=GS_MAX_BERT_MODELS)
