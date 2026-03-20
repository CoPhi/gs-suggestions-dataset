import re
from fastapi import APIRouter, Query, Path, Body
from fastapi.responses import JSONResponse
from api.schema import serial_model
from api.database import collection, fs
from api.models import (
    NgramModel,
    BERTModel,
    Model,
    ModelsResponse,
)
from bson import ObjectId
from train.training import pipeline_train
from inference.suggests import generate_k_suggests
from predictions.bert import fill_mask
from config.settings import (
    LM_TYPES,
    GAMMA,
    N,
    BERT_CHECKPOINTS,
)
from utils.preprocess import test_case_contains_lacuna
import pickle
import zlib
from uuid import uuid4
from typing import Annotated, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
from fastapi.responses import RedirectResponse


def save_to_gridfs(data, file_id=None):
    pickled_data = pickle.dumps(data)
    compressed_data = zlib.compress(pickled_data)
    filename = file_id or f"{uuid4()}"
    fs.put(compressed_data, filename=filename)
    return filename


router = APIRouter(prefix="/models", tags=["Models"])


# @router.get("/", include_in_schema=False)
# async def root():
#     return RedirectResponse(url="/docs")  # si fa il redirect alla documentazione


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
                                "CORPUS_NAMES": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "nullable": "true",
                                },
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
async def get_model(id: Annotated[str, Path(title="ID", description="ID del modello")]):
    try:
        model = serial_model(id)
        if model is None:
            return JSONResponse(status_code=404, content={"detail": "Model not found"})
        return JSONResponse(status_code=200, content={"model": model})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.get(
    "",
    response_model=ModelsResponse,
    responses={
        200: {
            "description": "Lista dei modelli",
            "content": {
                "application/json": {
                    "example": {
                        "models": [
                            {
                                "LM_SCORE": "string",
                                "GAMMA": "float | None",
                                "N": 3,
                                "CORPUS_NAMES": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "nullable": "true",
                                },
                                "GLOBAL_MODEL_FILE_ID": "string",
                                "DOMAIN_MODEL_FILE_ID": "string",
                            },
                            {
                                "CHECKPOINT": "string",
                                "MODEL_FILE_ID": "string",
                            },
                        ]
                    }
                },
            },
        },
        404: {"description": "Models not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_models():
    try:
        models = collection.find()
        if models:
            return JSONResponse(
                status_code=200,
                content={
                    "models": [serial_model(str(model["_id"])) for model in models]
                },
            )
        else:
            return JSONResponse(
                status_code=404, content={"detail": "Models are not found"}
            )
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
        409: {"description": "Model already exists"},
        404: {"description": "Model not found"},
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
                    "description": "Parametri di esempio per la creazione di un modello linguistico che usa `MLE` come tipo di smoothing. [MLE](https://it.wikipedia.org/wiki/Metodo_della_massima_verosimiglianza)",
                    "value": {
                        "LM_SCORE": "MLE",
                        "N": 3,
                        "TYPE": "Ngrams",
                    },
                },
                "Lidstone": {
                    "summary": "Esempio creazione modello Lidstone",
                    "description": "Parametri di esempio per la creazione di un modello linguistico che usa `LIDSTONE` come tipo di smoothing. [Lidstone](https://en.wikipedia.org/wiki/Additive_smoothing)",
                    "value": {
                        "LM_SCORE": "LIDSTONE",
                        "GAMMA": 0.1,
                        "N": 3,
                        "TYPE": "Ngrams",
                    },
                },
                "BERT": {
                    "summary": "Esempio creazione modello BERT",
                    "description": "Parametri di esempio per la creazione di un modello linguistico BERT.",
                    "value": {
                        "CHECKPOINT": "CNR-ILC/gs-aristoBERTo",
                        "TYPE": "BERT",
                    },
                },
            },
        ),
    ],
):

    if isinstance(model, NgramModel):
        try:
            model_dict = model.model_dump()
            if collection.find_one(model_dict):
                return JSONResponse(
                    status_code=409,
                    content={"detail": "Model already exist in db"},
                )

            global_ngram_model, domain_ngram_model, _ = pipeline_train(
                lm_type=model_dict["LM_SCORE"],
                gamma=model_dict["GAMMA"],
                n=model_dict["N"],
            )
            model_dict["GLOBAL_MODEL_FILE_ID"] = save_to_gridfs(global_ngram_model)
            model_dict["DOMAIN_MODEL_FILE_ID"] = save_to_gridfs(domain_ngram_model)
            model_id = collection.insert_one(model_dict).inserted_id
            return JSONResponse(status_code=201, content={"ID": str(model_id)})
        except ValueError as v:
            return JSONResponse(status_code=404, content={"detail": str(v)})
        except Exception as e:
            return JSONResponse(status_code=500, content=str(e))

    elif isinstance(model, BERTModel):
        model_dict = model.model_dump()
        try:
            if collection.find_one(model_dict):
                return JSONResponse(
                    status_code=409,
                    content={"message": "Model already exist in db"},
                )
            bert_model = AutoModelForMaskedLM.from_pretrained(model_dict["CHECKPOINT"])

            bert_tokenizer = AutoTokenizer.from_pretrained(model_dict["CHECKPOINT"])
            model_dict["MODEL_FILE_ID"] = save_to_gridfs(bert_model)
            model_dict["TOKENIZER_FILE_ID"] = save_to_gridfs(bert_tokenizer)
            model_id = collection.insert_one({**model_dict, "TYPE": "BERT"}).inserted_id
            return JSONResponse(status_code=201, content={"ID": str(model_id)})

        except ValueError as v:
            return JSONResponse(status_code=404, content={"message": str(v)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    else:
        return JSONResponse(status_code=400, content={"detail": "Invalid model type"})


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
async def create_models():
    ids = []
    # Selezione migliori iperparametri tramite BO (WIP)
    for lm_score in LM_TYPES:
        try:
            model_dict = {
                "LM_SCORE": lm_score,  # Si presuppone che sia ottimizzato
                "GAMMA": GAMMA,  # Si presuppone che sia ottimizzato
                "N": N,  # Si presuppone che sia ottimizzato
                "CORPUS_NAMES": None,
                "TYPE": "Ngrams",
            }

            if collection.find_one(model_dict):
                return JSONResponse(
                    status_code=409, content={"message": "Model already exist in db"}
                )

            global_ngram_model, domain_ngram_model, _ = pipeline_train(
                lm_type=model_dict["LM_SCORE"],
                gamma=model_dict["GAMMA"],
                n=model_dict["N"],
            )
            model_dict["GLOBAL_MODEL_FILE_ID"] = save_to_gridfs(global_ngram_model)
            model_dict["DOMAIN_MODEL_FILE_ID"] = save_to_gridfs(domain_ngram_model)
            model_id = collection.insert_one(
                {**model_dict, "TYPE": "Ngrams"}
            ).inserted_id
            ids.append(str(model_id))
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    # BERT models
    for checkpoint in BERT_CHECKPOINTS:
        try:
            model_dict = {
                "CHECKPOINT": checkpoint,
                "TYPE": "BERT",
            }
            bert_model = AutoModelForMaskedLM.from_pretrained(model_dict["CHECKPOINT"])
            bert_tokenizer = AutoTokenizer.from_pretrained(model_dict["CHECKPOINT"])
            model_dict["MODEL_FILE_ID"] = save_to_gridfs(bert_model)
            model_dict["TOKENIZER_FILE_ID"] = save_to_gridfs(bert_tokenizer)
            model_id = collection.insert_one({**model_dict, "TYPE": "BERT"}).inserted_id
            ids.append(str(model_id))
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})
    return JSONResponse(status_code=201, content={"IDs": ids})


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
):
    try:
        model = serial_model(id)
        if not model:
            return JSONResponse(status_code=404, content={"message": "Model not found"})

        # Elimina file GridFS associati
        if "MODEL_FILE_ID" in model:
            file_doc = fs.find_one({"filename": model["MODEL_FILE_ID"]})
            if file_doc:
                fs.delete(file_doc._id)

        if model.get("TYPE") == "BERT" and "TOKENIZER_FILE_ID" in model:
            tokenizer_doc = fs.find_one({"filename": model["TOKENIZER_FILE_ID"]})
            if tokenizer_doc:
                fs.delete(tokenizer_doc._id)

        collection.delete_one({"_id": ObjectId(id)})

        return JSONResponse(status_code=200, content={"model": model})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
