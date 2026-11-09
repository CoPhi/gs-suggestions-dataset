"""
Gestione dei Language Models (LLM) di Hugging Face per il framework DSPy 3.4+.

Supporta tre modalità di esecuzione per modelli come Qwen (es. Qwen/Qwen2.5-7B-Instruct)
e Llama (es. meta-llama/Llama-3.1-8B-Instruct):
1. 'local': Esecuzione diretta in-process su GPU/CPU tramite Hugging Face Transformers
   (`AutoModelForCausalLM` + `AutoTokenizer`) con cache LRU e supporto opzionale a 4-bit.
2. 'hf_api': Interrogazione remota tramite Hugging Face Serverless Inference API / Endpoints
   usando `HF_TOKEN` e `dspy.LM("huggingface/<repo_id>")`.
3. 'openai_compatible': Interrogazione di un server locale o remoto compatibile OpenAI
   (es. vLLM, Ollama, TGI, SGLang) tramite `dspy.LM("openai/<repo_id>", api_base=...)`.
"""

from __future__ import annotations

import gc
import json
import os
import re
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import dspy
import torch
from dotenv import load_dotenv
from dspy.clients.engines.base import validate_request
from dspy.lm15 import Message, Response, TextPart, Usage, response_to_events
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class TransformersLocalEngine:
    """
    Engine canonico DSPy 3.4 (`dspy.lm15`) per l'esecuzione locale di modelli
    `AutoModelForCausalLM` di Hugging Face (Qwen, Llama, Mistral, ecc.).
    """

    _hf_cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
    MAX_CACHED_MODELS: int = 1

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        default_temperature: float = 0.2,
        default_max_tokens: int = 1536,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @classmethod
    def clear_cache(cls, model_id: str | None = None) -> None:
        """Libera la memoria occupata dai modelli caricati nella cache LRU."""
        if model_id is not None:
            cls._hf_cache.pop(model_id, None)
        else:
            cls._hf_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_model_and_tokenizer(self) -> tuple[Any, Any]:
        if self.model_id in self._hf_cache:
            self._hf_cache.move_to_end(self.model_id)
            return self._hf_cache[self.model_id]

        while len(self._hf_cache) >= self.MAX_CACHED_MODELS:
            self._hf_cache.popitem(last=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        hf_token = os.getenv("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=hf_token,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "token": hf_token,
            "trust_remote_code": True,
        }
        if self.device == "cuda" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            if self.torch_dtype == "auto":
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=load_kwargs.get("torch_dtype", torch.float16),
                )
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if "device_map" not in load_kwargs:
            model = model.to(self.device)
        model.eval()

        self._hf_cache[self.model_id] = (model, tokenizer)
        return model, tokenizer

    def _extract_messages(self, request: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if getattr(request, "system", None) is not None:
            sys_content = (
                request.system
                if isinstance(request.system, str)
                else "".join(getattr(p, "text", "") for p in request.system)
            )
            if sys_content:
                messages.append({"role": "system", "content": sys_content})

        for msg in getattr(request, "messages", []) or []:
            role = getattr(msg, "role", "user")
            text = getattr(msg, "text", None)
            if text is None and hasattr(msg, "parts"):
                text = "".join(getattr(p, "text", "") for p in msg.parts)
            messages.append({"role": str(role), "content": text or ""})
        return messages

    def _generate_from_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Response:
        model, tokenizer = self._get_model_and_tokenizer()
        temp = self.default_temperature if temperature is None else float(temperature)
        max_new_tokens = int(max_tokens or self.default_max_tokens)

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = (
                "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
                + "\n\nASSISTANT:\n"
            )

        model_device = next(model.parameters()).device
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)
        input_len = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": temp > 1e-5,
        }
        if temp > 1e-5:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0][input_len:]
        output_len = int(generated_ids.shape[0])
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return Response(
            id=None,
            model=self.model_id,
            message=Message.assistant([TextPart(response_text)]),
            finish_reason="stop",
            usage=Usage(
                input_tokens=input_len,
                output_tokens=output_len,
                total_tokens=input_len + output_len,
            ),
        )

    def complete(self, request: Any) -> Response:
        validate_request(request)
        messages = self._extract_messages(request)
        cfg = getattr(request, "config", None)
        temperature = getattr(cfg, "temperature", None) if cfg else None
        max_tokens = getattr(cfg, "max_tokens", None) if cfg else None
        return self._generate_from_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete_legacy(self, lm: Any, request: dict[str, Any], **context: Any) -> Any:
        from dspy.clients.call_result import CallResult, combine

        messages = request.get("messages") or []
        temperature = request.get("temperature", self.default_temperature)
        max_tokens = request.get("max_tokens", self.default_max_tokens)
        count = int(request.get("n") or 1)
        results = [
            CallResult.native(
                self._generate_from_messages(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            for _ in range(count)
        ]
        return results[0] if count == 1 else combine(results, model_type="chat")

    def stream(self, request: Any):
        return response_to_events(self.complete(request))

    def close(self) -> None:
        pass


class AsyncTransformersLocalEngine:
    """Controparte asincrona di `TransformersLocalEngine` per DSPy 3.4+."""

    def __init__(self, sync_engine: TransformersLocalEngine) -> None:
        self.sync = sync_engine

    async def complete(self, request: Any) -> Response:
        import asyncio

        return await asyncio.to_thread(self.sync.complete, request)

    async def complete_legacy(self, lm: Any, request: dict[str, Any], **context: Any) -> Any:
        import asyncio

        return await asyncio.to_thread(self.sync.complete_legacy, lm, request, **context)

    async def stream(self, request: Any):
        for event in self.sync.stream(request):
            yield event

    async def aclose(self) -> None:
        pass


class HuggingFaceLocalLM(dspy.LM):
    """
    Sottoclasse di `dspy.LM` che esegue modelli Hugging Face (`Qwen`, `Llama`, ecc.)
    localmente su GPU/CPU tramite `TransformersLocalEngine`.
    """

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1536,
        cache: bool = True,
    ) -> None:
        sync_engine = TransformersLocalEngine(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )
        async_engine = AsyncTransformersLocalEngine(sync_engine)
        super().__init__(
            model=model_id,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            engine=sync_engine,
            async_engine=async_engine,
        )
        self.model_id = model_id


_LM_INSTANCES: OrderedDict[tuple[str, str, str | None, float, int, bool], dspy.BaseLM] = OrderedDict()

# Circuit-breaker di sessione: se `HF_TOKEN` è presente ma privo dei permessi
# "Make calls to Inference Providers" (es. token fine-grained solo per repo Hub),
# evita di instradare nuove richieste `backend='auto'` verso `hf_api`.
_HF_INFERENCE_API_UNAVAILABLE: bool = False


def mark_hf_api_unavailable() -> None:
    """Segna Hugging Face Serverless Inference Providers come non utilizzabile nella sessione corrente."""
    global _HF_INFERENCE_API_UNAVAILABLE
    _HF_INFERENCE_API_UNAVAILABLE = True


def reset_hf_api_availability() -> None:
    """Ripristina lo stato del circuit-breaker per Hugging Face Inference Providers (utile nei test)."""
    global _HF_INFERENCE_API_UNAVAILABLE
    _HF_INFERENCE_API_UNAVAILABLE = False


def is_hf_api_unavailable() -> bool:
    """Restituisce True se `hf_api` è stato disabilitato per assenza di permessi sul token."""
    return _HF_INFERENCE_API_UNAVAILABLE


def is_hf_permission_or_auth_error(exc: BaseException) -> bool:
    """
    Rileva se un'eccezione sollevata da DSPy/LiteLLM dipende da permessi insufficienti
    o credenziali non valide su Hugging Face Inference Providers.
    """
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "sufficient permissions to call inference providers",
            "huggingfaceexception",
            "inference providers",
            "401",
            "403",
            "invalid credentials",
            "unauthorized",
        )
    )


def get_available_system_ram_gb() -> float:
    """
    Legge la RAM fisica effettivamente disponibile (`MemAvailable`) da `/proc/meminfo` su Linux.
    Usato per prevenire l'intervento dell'OOM killer del kernel quando Ollama carica modelli
    pesanti (es. 9.6 GB) interamente su CPU (`num_gpu=0`).
    """
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024**2)
        except Exception:
            pass
    return 8.0


def discover_ollama_models(
    api_base: str | None = None,
    timeout: float = 0.4,
    check_ram_budget: bool = True,
) -> tuple[str | None, list[str]]:
    """
    Interroga il demone locale Ollama (`GET /api/tags`) sugli endpoint candidati
    (`OLLAMA_API_BASE`, `127.0.0.1:11434`, `host.docker.internal:11434`) e restituisce
    `(ollama_host, lista_modelli_compatibili)`.

    Se `check_ram_budget=True` e la GPU ha `< 8 GB` di VRAM (per cui i modelli > 3B
    vengono forzati su CPU con `num_gpu=0`), scarta i modelli la cui dimensione su disco
    eccede la RAM disponibile meno 1.5 GB di margine operativo, prevenendo `oom-kill`.
    """
    candidates: list[str] = []
    for base in (
        api_base,
        os.getenv("OLLAMA_API_BASE"),
        "http://127.0.0.1:11434",
        "http://host.docker.internal:11434",
    ):
        if base:
            clean_base = base.removesuffix("/v1").rstrip("/")
            if clean_base not in candidates:
                candidates.append(clean_base)

    low_vram = True
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        low_vram = vram_gb < 8.0

    avail_ram_gb = get_available_system_ram_gb() if check_ram_budget else 999.0

    for host in candidates:
        try:
            req = urllib.request.Request(f"{host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    continue
                payload = json.loads(resp.read().decode("utf-8"))
                models_list = payload.get("models") or []
                compatible: list[str] = []
                for m in models_list:
                    name = m.get("name") or m.get("model")
                    if not name:
                        continue
                    size_bytes = float(m.get("size") or 0)
                    size_gb = size_bytes / (1024**3)
                    # Su GPU < 8 GB i modelli non compatti girano su CPU (`num_gpu=0`):
                    # richiediamo almeno 1.5 GB di margine sulla RAM disponibile per evitare OOM kill
                    if (
                        check_ram_budget
                        and low_vram
                        and not _is_compact_hf_model(name)
                        and size_gb > 0
                        and (size_gb + 1.5) > avail_ram_gb
                    ):
                        continue
                    compatible.append(str(name))
                return host, compatible
        except Exception:
            continue

    return None, []


def _is_compact_hf_model(model_id: str) -> bool:
    """Rileva modelli compatti (<= 3B parametri) eseguibili nativamente su GPU consumer da 4-8 GB."""
    return bool(re.search(r"(?:^|[-_/])(0\.5|1|1\.5|2|3)b(?:[-_.]|$)", model_id.lower()))


def is_ollama_model_within_ram_budget(
    model_name: str,
    api_base: str | None = None,
    timeout: float = 0.4,
) -> tuple[bool, str]:
    """
    Verifica preventivamente via `GET /api/tags` se un modello Ollama può essere caricato
    in memoria senza innescare l'OOM killer del kernel Linux quando `num_gpu=0` (GPU < 8 GB).
    """
    clean_name = model_name
    for prefix in ("ollama_chat/", "ollama/", "openai/"):
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix) :]

    if _is_compact_hf_model(clean_name):
        return True, ""

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 8.0:
            return True, ""

    avail_ram_gb = get_available_system_ram_gb()
    host = (api_base or os.getenv("OLLAMA_API_BASE") or "http://127.0.0.1:11434").removesuffix("/v1").rstrip("/")
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                for m in payload.get("models") or []:
                    m_name = m.get("name") or m.get("model") or ""
                    if m_name == clean_name or m_name.startswith(f"{clean_name}:"):
                        size_gb = float(m.get("size") or 0) / (1024**3)
                        if size_gb > 0 and (size_gb + 1.5) > avail_ram_gb:
                            return (
                                False,
                                f"il modello Ollama '{clean_name}' ({size_gb:.1f} GiB) richiede più RAM di quella libera ({avail_ram_gb:.1f} GiB)",
                            )
    except Exception:
        pass
    return True, ""


def get_fallback_dspy_lms(
    checkpoint: str,
    backend: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 1536,
) -> list[dspy.BaseLM]:
    """
    Restituisce una coda ordinata di istanze `dspy.BaseLM` di fallback da tentare
    qualora l'LLM primario fallisca a runtime (es. `HF_TOKEN` privo di permessi
    Inference Providers).
    """
    if backend not in ("auto", "hf_api", "openai_compatible"):
        return []

    fallbacks: list[dspy.BaseLM] = []
    ollama_host, ollama_models = discover_ollama_models(check_ram_budget=True)
    if ollama_host and ollama_models:
        clean_primary = checkpoint
        for prefix in ("huggingface/", "openai/", "ollama_chat/", "ollama/"):
            if clean_primary.startswith(prefix):
                clean_primary = clean_primary[len(prefix) :]
        for tag in ollama_models:
            if tag == clean_primary:
                continue
            try:
                fb_lm = build_dspy_lm(
                    checkpoint=f"ollama_chat/{tag}",
                    backend="openai_compatible",
                    api_base=ollama_host,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                fallbacks.append(fb_lm)
            except Exception:
                continue
    return fallbacks


def build_dspy_lm(
    checkpoint: str = "Qwen/Qwen2.5-7B-Instruct",
    backend: Literal["auto", "hf_api", "local", "openai_compatible"] = "auto",
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1536,
    load_in_4bit: bool = False,
    device: str | None = None,
    cache: bool = True,
) -> dspy.BaseLM:
    """
    Costruisce e restituisce un'istanza `dspy.BaseLM` pronta per `dspy.configure(lm=...)`
    o `with dspy.context(lm=...):`.

    Args:
        checkpoint: Nome del repository su Hugging Face (es. 'Qwen/Qwen2.5-7B-Instruct',
                    'Qwen/Qwen2.5-1.5B-Instruct') oppure tag Ollama (es. 'gemma4:e4b', 'ollama/qwen2.5:7b').
        backend: Strategia di inferenza ('auto', 'hf_api', 'local', 'openai_compatible').
        api_base: URL base per server compatibili OpenAI (vLLM, Ollama, TGI).
        api_key: Chiave API (se omessa, legge `HF_TOKEN` o `RAG_API_KEY` da `.env`).
        temperature: Temperatura di campionamento.
        max_tokens: Numero massimo di token generati.
        load_in_4bit: Se True (solo per `backend='local'` su GPU), abilita la quantizzazione 4-bit.
        device: Device esplicito ('cuda', 'cpu') per il caricamento locale.
        cache: Abilita la cache delle risposte di DSPy.
    """
    is_explicit_ollama = checkpoint.startswith(("ollama/", "ollama_chat/"))
    clean_ckpt = checkpoint
    for prefix in ("huggingface/", "openai/", "ollama_chat/", "ollama/"):
        if clean_ckpt.startswith(prefix):
            clean_ckpt = clean_ckpt[len(prefix) :]

    # Riconosce tag nativi Ollama (es. 'gemma4:e4b' o 'ollama/qwen2.5:7b') distinti dai repo HF ('org/repo')
    is_ollama_tag = is_explicit_ollama or (":" in clean_ckpt and "/" not in clean_ckpt)
    resolved_api_base = api_base or os.getenv("RAG_API_BASE")
    if is_ollama_tag and not resolved_api_base:
        resolved_api_base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")

    has_usable_hf_token = bool(api_key or (os.getenv("HF_TOKEN") and not _HF_INFERENCE_API_UNAVAILABLE))

    resolved_backend = backend
    if is_ollama_tag and backend in ("auto", "openai_compatible"):
        resolved_backend = "ollama"
    elif resolved_backend == "auto":
        if resolved_api_base:
            resolved_backend = "openai_compatible"
        elif torch.cuda.is_available():
            # Evita di caricare modelli >= 7B (~15 GB fp16) su GPU con VRAM < 12 GB se è disponibile HF_TOKEN
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if _is_compact_hf_model(clean_ckpt) or vram_gb >= 12.0:
                resolved_backend = "local"
            elif has_usable_hf_token:
                resolved_backend = "hf_api"
            else:
                # Se HF_TOKEN non è abilitato per Inference Providers e la GPU ha < 12 GB,
                # verifica se c'è un modello Ollama compatibile con la RAM disponibile
                ollama_host, ollama_models = discover_ollama_models(check_ram_budget=True)
                if ollama_host and ollama_models:
                    clean_ckpt = ollama_models[0]
                    resolved_api_base = ollama_host
                    resolved_backend = "ollama"
                else:
                    resolved_backend = "hf_api" if (os.getenv("HF_TOKEN") or api_key) else "local"
        elif has_usable_hf_token:
            resolved_backend = "hf_api"
        else:
            ollama_host, ollama_models = discover_ollama_models(check_ram_budget=True)
            if ollama_host and ollama_models:
                clean_ckpt = ollama_models[0]
                resolved_api_base = ollama_host
                resolved_backend = "ollama"
            else:
                resolved_backend = "hf_api" if (os.getenv("HF_TOKEN") or api_key) else "local"

    cache_key = (
        clean_ckpt,
        resolved_backend,
        resolved_api_base,
        float(temperature),
        int(max_tokens),
        bool(load_in_4bit),
    )
    if cache_key in _LM_INSTANCES:
        _LM_INSTANCES.move_to_end(cache_key)
        return _LM_INSTANCES[cache_key]

    if resolved_backend == "local":
        lm: dspy.BaseLM = HuggingFaceLocalLM(
            model_id=clean_ckpt,
            device=device,
            load_in_4bit=load_in_4bit,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )
    elif resolved_backend == "ollama":
        # Usa il provider nativo `ollama_chat/` (senza suffisso /v1) per disabilitare il thinking implicito
        # (`think=False`, che saturerebbe `max_tokens`), limitare `num_ctx=2048` per contenere la KV-cache
        # e forzare `num_gpu=0` su schede < 8 GB per prevenire errori `GGML_ASSERT` nello split CPU/GPU.
        ollama_host = (resolved_api_base or "http://127.0.0.1:11434").removesuffix("/v1").rstrip("/")
        ollama_kwargs: dict[str, Any] = {"think": False, "num_ctx": 2048}
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 8.0 and not _is_compact_hf_model(clean_ckpt):
                ollama_kwargs["num_gpu"] = 0
        lm = dspy.LM(
            model=f"ollama_chat/{clean_ckpt}",
            api_base=ollama_host,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **ollama_kwargs,
        )
    elif resolved_backend == "openai_compatible":
        token = api_key or os.getenv("RAG_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        base_url = resolved_api_base or "http://localhost:8001/v1"
        lm = dspy.LM(
            model=f"openai/{clean_ckpt}",
            api_base=base_url,
            api_key=token,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )
    elif resolved_backend == "hf_api":
        token = api_key or os.getenv("HF_TOKEN")
        lm = dspy.LM(
            model=f"huggingface/{clean_ckpt}",
            api_key=token,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )
    else:
        raise ValueError(
            f"Backend '{backend}' non supportato. Valori ammessi: 'auto', 'hf_api', 'local', 'openai_compatible'."
        )

    _LM_INSTANCES[cache_key] = lm
    if len(_LM_INSTANCES) > 4:
        _LM_INSTANCES.popitem(last=False)
    return lm
