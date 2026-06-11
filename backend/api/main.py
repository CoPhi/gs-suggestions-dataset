import tomllib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import predictions
from backend.api.routes import models
from backend.api.exceptions import (
    ModelNotFoundError,
    ModelAlreadyExistsError,
    InvalidContextError,
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
    description="""API per GreekSchools. Questa API è progettata per offrire l'accesso a modelli linguistici basati su n-grammi e BERT per la generazione di supplementi testuali. """,
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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
