import tomllib
from pathlib import Path
from contextlib import asynccontextmanager
import dspy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import predictions
from backend.api.routes import models
from backend.api.exceptions import (
    ModelNotFoundError,
    ModelAlreadyExistsError,
    InvalidContextError,
    DatasetNotFoundError,
)
from backend.api.services.model_service import ModelService


def get_version() -> str:
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data.get("project", {}).get("version") or data.get("tool", {}).get(
        "poetry", {}
    ).get("version", "unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = ModelService()
        await service.init_models()
    except Exception as e:
        print(f"Error during auto-initialization of models: {e}")
    yield


app = FastAPI(
    title="gs-api",
    version=get_version(),
    description="""API per GreekSchools. Questa API offre l'accesso e la gestione del ciclo di vita di modelli linguistici basati su N-grammi, BERT, Ithaca e DSPy RAG per la generazione di supplementi testuali e il restauro di testi antichi.

### Modelli supportati:
- **Ngrams**: Modelli statistici MLE e Lidstone con K-smoothing.
- **BERT**: Modelli Masked Language Modeling (es. `CNR-ILC/gs-aristoBERTo`, `CNR-ILC/gs-GreBerta`, `CNR-ILC/gs-Logion`).
- **Ithaca**: Modello BigBird character-level fine-tunato sul corpus TLG (es. `CNR-ILC/gs-ithaca-tlg`) con decoding HCB beam search per il ripristino di lacune papirologiche ed epigrafiche.
- **RAG**: Sistema di Retrieval-Augmented Generation orchestrato con DSPy 3.4+ su corpora antichi (TLG & MAAT) e modelli LLM open-weights (es. `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-2-9b-it`, `meta-llama/Llama-3.1-8B-Instruct`).
""",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, include_in_schema=True)
app.include_router(predictions.router, include_in_schema=True)


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ModelAlreadyExistsError)
async def model_already_exists_handler(request: Request, exc: ModelAlreadyExistsError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(InvalidContextError)
async def invalid_context_handler(request: Request, exc: InvalidContextError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(DatasetNotFoundError)
async def dataset_not_found_handler(request: Request, exc: DatasetNotFoundError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(dspy.utils.exceptions.LMError)
async def dspy_lm_error_handler(request: Request, exc: Exception):
    err_str = str(exc)
    if "Inference Providers" in err_str or "HuggingfaceException" in err_str:
        hint = (
            "Il token HF_TOKEN configurato non dispone del permesso 'Make calls to Inference Providers' "
            "su Hugging Face Hub. Abilita tale permesso sul token oppure usa il backend 'auto' con fallback locale."
        )
    elif "Ollama" in err_str or "Connection" in err_str:
        hint = (
            "Il runtime LLM locale (Ollama/vLLM) non risponde o ha esaurito la memoria RAM/VRAM disponibile."
        )
    else:
        hint = (
            "Verifica i permessi di HF_TOKEN ('Make calls to Inference Providers') "
            "o la disponibilità del runtime locale (Ollama/vLLM)."
        )
    return JSONResponse(
        status_code=502,
        content={
            "detail": f"Errore di comunicazione con il backend LLM del sistema RAG: {exc}. {hint}"
        },
    )


