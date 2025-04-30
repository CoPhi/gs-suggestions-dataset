from fastapi import APIRouter, Query, Path, Body, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from api.schema import serial_model
from api.database import collection, fs
from api.models import NgramModel, BERTModel, Model, PredictionCount
from api import LEFT_CONTEXT_PATTERN
from bson import ObjectId
from train.training import pipeline_train, train_lm
from train import load_abs
from inference import generate_k_suggests
from finetuning import hcb_beam_search, convert_lacuna_to_masks
from config.settings import (
    K_PREDICTIONS,
    LM_TYPES,
    GAMMA,
    N,
    MIN_FREQ,
    BERT_CHECKPOINTS,
)
import pickle
import zlib
from uuid import uuid4
from typing import Annotated, Literal, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
from fastapi.responses import RedirectResponse


def save_to_gridfs(data, file_id=None):
    pickled_data = pickle.dumps(data)
    compressed_data = zlib.compress(pickled_data)
    filename = file_id or f"{uuid4()}"
    fs.put(compressed_data, filename=filename)
    return filename


router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@router.get(
    "/model/{id}",
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
                                "MIN_FREQ": "float",
                                "N": 3,
                                "CORPUS_NAMES": "list[string] | None",
                                "MODEL_FILE_ID": "string",
                            },
                        },
                        "BERTModel": {
                            "summary": "Example BERT Model",
                            "value": {
                                "MODEL": "string",
                                "TOKENIZER": "string",
                                "MODEL_FILE_ID": "string",
                                "TOKENIZER_FILE_ID": "string",
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
    "/models",
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
                                "MIN_FREQ": "float",
                                "N": 3,
                                "CORPUS_NAMES": "list[string] | None",
                                "MODEL_FILE_ID": "string",
                            },
                            {
                                "MODEL": "string",
                                "TOKENIZER": "string",
                                "MODEL_FILE_ID": "string",
                                "TOKENIZER_FILE_ID": "string",
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
        raise JSONResponse(status_code=500, content={"detail": str(e)})


@router.post(
    "/model",
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
                    "description": "Parametri di esempio per la creazione di un modello linguistico che usa MLE.",
                    "value": {
                        "LM_SCORE": "MLE",
                        "MIN_FREQ": 3,
                        "N": 3,
                        "TYPE": "Ngrams",
                    },
                },
                "Lidstone": {
                    "summary": "Esempio creazione modello Lidstone",
                    "description": "Parametri di esempio per la creazione di un modello linguistico che usa Lidstone.",
                    "value": {
                        "LM_SCORE": "LIDSTONE",
                        "GAMMA": 0.1,
                        "MIN_FREQ": 3,
                        "N": 3,
                        "TYPE": "Ngrams",
                    },
                },
                "BERT": {
                    "summary": "Esempio creazione modello BERT",
                    "description": "Parametri di esempio per la creazione di un modello linguistico BERT.",
                    "value": {
                        "MODEL": "CNR-ILC/gs-aristoBERTo",
                        "TOKENIZER": "CNR-ILC/gs-aristoBERTo",
                        "TYPE": "BERT",
                    },
                },
            },
        ),
    ],
):

    match model:
        case NgramModel():
            try:
                model_dict = dict(model)

                if collection.find_one(model_dict):
                    return JSONResponse(
                        status_code=409,
                        content={"detail": "Model already exist in db"},
                    )

                abs = load_abs(
                    corpus_set=model_dict["CORPUS_NAMES"]
                )  # Carico i blocchi anonimi di default

                ngram_model = train_lm(
                    train_abs=abs,
                    lm_type=model_dict["LM_SCORE"],
                    min_freq=model_dict["MIN_FREQ"],
                    gamma=model_dict["GAMMA"],
                    n=model_dict["N"],
                )
                model_dict["MODEL_FILE_ID"] = save_to_gridfs(ngram_model)
                model_id = collection.insert_one(model_dict).inserted_id
                return JSONResponse(status_code=201, content={"ID": str(model_id)})
            except ValueError as v:
                return JSONResponse(status_code=404, content={"detail": str(v)})
            except Exception as e:
                raise JSONResponse(status_code=500, content=str(e))

        case BERTModel():
            try:
                model_dict = dict(model)
                if collection.find_one(model_dict):
                    return JSONResponse(
                        status_code=409,
                        content={"message": "Model already exist in db"},
                    )

                bert_model = AutoModelForMaskedLM.from_pretrained(
                    model_dict["MODEL"], token=True
                )
                bert_tokenizer = AutoTokenizer.from_pretrained(model_dict["TOKENIZER"])

                model_dict["MODEL_FILE_ID"] = save_to_gridfs(bert_model)
                model_dict["TOKENIZER_FILE_ID"] = save_to_gridfs(bert_tokenizer)

                model_id = collection.insert_one(model_dict).inserted_id
                return JSONResponse(status_code=201, content={"ID": str(model_id)})

            except ValueError as v:
                return JSONResponse(status_code=404, content={"message": str(v)})
            except Exception as e:
                return JSONResponse(status_code=500, content={"detail": str(e)})


@router.post(
    "/models",
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

    # Ngrams models
    abs = load_abs()  # Carico i blocchi anonimi di default
    # Selezione migliori iperparametri tramite BO (WIP)
    for lm_score in LM_TYPES:
        try:
            model_dict = {
                "LM_SCORE": lm_score,
                "GAMMA": GAMMA,
                "MIN_FREQ": MIN_FREQ,
                "N": N,
                "CORPUS_NAMES": None,
                "TYPE": "Ngrams",
            }

            if collection.find_one(model_dict):
                return JSONResponse(
                    status_code=409, content={"message": "Model already exist in db"}
                )

            ngram_model = train_lm(
                train_abs=abs,
                lm_type=model_dict["LM_SCORE"],
                min_freq=model_dict["MIN_FREQ"],
                gamma=model_dict["GAMMA"],
                n=model_dict["N"],
            )
            model_dict["MODEL_FILE_ID"] = save_to_gridfs(ngram_model)
            model_id = collection.insert_one(model_dict).inserted_id
            ids.append(str(model_id))
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

        # BERT models
        for checkpoint in BERT_CHECKPOINTS:
            try:
                model_dict = {
                    "MODEL": checkpoint,
                    "TOKENIZER": checkpoint,
                    "TYPE": "BERT",
                }
                bert_model = AutoModelForMaskedLM.from_pretrained(
                    model_dict["MODEL"], token=True
                )

                bert_tokenizer = AutoTokenizer.from_pretrained(model_dict["TOKENIZER"])
                model_dict["MODEL_FILE_ID"] = save_to_gridfs(bert_model)
                model_dict["TOKENIZER_FILE_ID"] = save_to_gridfs(bert_tokenizer)

                model_id = collection.insert_one(model_dict).inserted_id
                ids.append(str(model_id))
            except Exception as e:
                return JSONResponse(status_code=500, content={"detail": str(e)})
    return JSONResponse(status_code=201, content={"IDs": ids})


# predictions
@router.get(
    "/predictions",
    responses={
        200: {
            "description": "Predictions",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            ["token", 0.5],
                            ["another_token", 0.3],
                        ]
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_predictions(
    model_id: Annotated[str, Query(title="ID", description="ID del modello")],
    context: Annotated[
        str,
        Query(
            title="Contesto di generazione",
            description="Deve contenere una lacuna da riempire, indicata tramite `[...]`, il numero di puntini indica la grandezza della lacuna nei modelli `BERT`",
            max_length=512,
        ),
    ],
    num_predictions: Annotated[
        PredictionCount,
        Query(
            title="Numero di predizioni da generare",
            description="Numero di generazioni che il modello deve eseguire; valori consentiti: 1, 5, 10, 20",
        ),
    ],
    num_tokens: Annotated[
        Optional[int],
        Query(
            title="Numero di token",
            description="Numero previsto di token da generare, deve essere presente solo se il modello è di tipo `Ngrams`",
            ge=1,
            le=10,
        ),
    ] = None,
):
    print(num_predictions)
    try:
        model = collection.find_one({"_id": ObjectId(model_id)})
        if model is None:
            return JSONResponse(status_code=404, content={"detail": "Model not found"})

        dict_model = dict(model)

        # Se non è presente la lacuna nel contesto, restituisce bad request
        if "[" not in context:
            return JSONResponse(
                status_code=400,
                content={"detail": "Context must contain a gap indicated by `[...]`"},
            )

        if dict_model["TYPE"] == "Ngrams":
            model_filename = dict_model["MODEL_FILE_ID"]
            file_document = fs.find_one(
                {"filename": model_filename}
            )  # Cerca il file per nome
            if not file_document:
                return JSONResponse(
                    status_code=404, content={"message": "Model not found"}
                )

            model_file = fs.get(file_document._id)  # Recupera il file usando il suo _id
            decompressed_model = pickle.loads(zlib.decompress(model_file.read()))

            left_context = LEFT_CONTEXT_PATTERN.sub("[", context).split("[")[0]
            predictions = [
                {
                    "sentence": left_context + suggestion[0],
                    "token_str": suggestion[0],
                    "score": suggestion[1],
                }
                for suggestion in generate_k_suggests(
                    lm=decompressed_model,
                    context=context,
                    num_tokens=num_tokens,
                    lm_type=dict_model["LM_SCORE"],
                    n=dict_model["N"],
                    k_pred=num_predictions.value,
                )
            ]
            return JSONResponse(
                status_code=200,
                content={"predictions": predictions},
            )

        elif dict_model["TYPE"] == "BERT":
            model_filename = dict_model["MODEL_FILE_ID"]
            tokenizer_filename = dict_model["TOKENIZER_FILE_ID"]
            model_document = fs.find_one({"filename": model_filename})
            tokenizer_document = fs.find_one({"filename": tokenizer_filename})
            if not model_document or not tokenizer_document:
                return JSONResponse(
                    status_code=404, content={"message": "Model not found"}
                )
            model_file = fs.get(model_document._id)
            tokenizer_file = fs.get(tokenizer_document._id)
            decompressed_model = pickle.loads(zlib.decompress(model_file.read()))
            decompressed_tokenizer = pickle.loads(
                zlib.decompress(tokenizer_file.read())
            )

            predictions = [
                {"token_str": suggestion[0], "score": suggestion[1]}
                for suggestion in hcb_beam_search(
                    decompressed_model,
                    decompressed_tokenizer,
                    convert_lacuna_to_masks(context),
                )
            ]

            return JSONResponse(
                status_code=200,
                content={"predictions": predictions},
            )

        else:
            return JSONResponse(status_code=404, content={"message": "Model not found"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
