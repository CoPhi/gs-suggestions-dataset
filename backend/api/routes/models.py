from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path
from fastapi.responses import JSONResponse

from backend.api.exceptions import ModelAlreadyExistsError, ModelNotFoundError
from backend.api.models import Model, ModelsResponse
from backend.api.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["Models"])


def get_service() -> ModelService:
    return ModelService()


@router.get(
    "/{id}",
    response_model=Model,
    responses={
        200: {
            "description": "ID del modello",
            "content": {
                "application/json": {
                    "examples": {
                        "NgramModel": {
                            "summary": "Example Ngram Model",
                            "value": {
                                "LM_SCORE": "string",
                                "GAMMA": "float | None",
                                "N": 3,
                                "GLOBAL_MODEL_FILE_ID": "string",
                                "DOMAIN_MODEL_FILE_ID": "string",
                            },
                        },
                        "BERTModel": {
                            "summary": "Example BERT Model",
                            "value": {
                                "CHECKPOINT": "string",
                                "MODEL_FILE_ID": "string",
                            },
                        },
                    }
                }
            },
        },
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_model(
    id: Annotated[str, Path(title="ID", description="ID del modello")],
    service: ModelService = Depends(get_service),
):
    try:
        return JSONResponse(status_code=200, content={"model": await service.get_model(id)})
    except ModelNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.get(
    "",
    response_model=ModelsResponse,
    responses={
        200: {"description": "Lista dei modelli"},
        404: {"description": "Models not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_models(service: ModelService = Depends(get_service)):
    try:
        return JSONResponse(
            status_code=200, content={"models": await service.get_all_models()}
        )
    except ModelNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.post(
    "",
    status_code=201,
    responses={
        201: {
            "description": "Model created successfully",
            "content": {"application/json": {"example": {"ID": "string"}}},
        },
        400: {"description": "Invalid model type"},
        409: {"description": "Model already exists"},
        500: {"description": "Internal server error"},
    },
)
async def create_model(
    model: Annotated[
        Model,
        Body(
            discriminator="TYPE",
            title="Modello",
            description="Modello linguistico da creare",
            openapi_examples={
                "MLE": {
                    "summary": "Esempio creazione modello MLE",
                    "description": "Parametri per la creazione di un modello MLE.",
                    "value": {"LM_SCORE": "MLE", "N": 3, "TYPE": "Ngrams"},
                },
                "Lidstone": {
                    "summary": "Esempio creazione modello Lidstone",
                    "description": "Parametri per la creazione di un modello Lidstone.",
                    "value": {
                        "LM_SCORE": "LIDSTONE",
                        "GAMMA": 0.1,
                        "N": 3,
                        "TYPE": "Ngrams",
                    },
                },
                "BERT": {
                    "summary": "Esempio creazione modello BERT",
                    "description": "Parametri per la creazione di un modello BERT.",
                    "value": {"CHECKPOINT": "CNR-ILC/gs-aristoBERTo", "TYPE": "BERT"},
                },
            },
        ),
    ],
    service: ModelService = Depends(get_service),
):
    try:
        model_id = await service.create_model(model)
        return JSONResponse(status_code=201, content={"ID": model_id})
    except ModelAlreadyExistsError as e:
        return JSONResponse(status_code=409, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.post("/init/", include_in_schema=False)
@router.post(
    "/init",
    status_code=201,
    responses={
        201: {
            "description": "Models created successfully",
            "content": {"application/json": {"example": {"IDs": ["string"]}}},
        },
        409: {"description": "Model already exists"},
        500: {"description": "Internal server error"},
    },
)
async def create_models(service: ModelService = Depends(get_service)):
    try:
        ids = await service.init_models()
        return JSONResponse(status_code=201, content={"IDs": ids})
    except ModelAlreadyExistsError as e:
        return JSONResponse(status_code=409, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.delete(
    "/{id}",
    response_model=Model,
    responses={
        200: {"description": "Model deleted successfully"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    },
)
async def delete_model(
    id: Annotated[str, Path(title="ID", description="ID del modello da eliminare")],
    service: ModelService = Depends(get_service),
):
    try:
        deleted = await service.delete_model(id)
        return JSONResponse(status_code=200, content={"model": deleted})
    except ModelNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
