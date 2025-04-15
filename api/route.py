from fastapi import APIRouter, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from api.schema import serial_model
from api.database import collection, fs
from api.models import NgramModel, BERTModel, Model
from api import LEFT_CONTEXT_PATTERN
from bson import ObjectId
from train.training import pipeline_train
from inference import generate_k_suggests
from finetuning import hcb_beam_search, convert_lacuna_to_masks
from config.settings import K_PREDICTIONS, LM_TYPES, GAMMA, TEST_SIZE, N, MIN_FREQ
import pickle
import zlib
from uuid import uuid4
from typing import Annotated, Optional
from transformers import AutoModelForMaskedLM, AutoTokenizer
import re


def save_to_gridfs(data, file_id=None):
    pickled_data = pickle.dumps(data)
    compressed_data = zlib.compress(pickled_data)
    filename = file_id or f"{uuid4()}"
    fs.put(compressed_data, filename=filename)
    return filename


router = APIRouter()


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
                                "K_PRED": "float",
                                "TEST_SIZE": "float",
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
                                "K_PRED": "float",
                                "TEST_SIZE": "float",
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
                        "K_PRED": 10,
                        "TEST_SIZE": 0.05,
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
                        "K_PRED": 10,
                        "GAMMA": 0.1,
                        "TEST_SIZE": 0.05,
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
                if model_dict["K_PRED"] not in K_PREDICTIONS:
                    return JSONResponse(
                        status_code=400, content={"detail": "Invalid K_PRED value"}
                    )

                if collection.find_one(model_dict):
                    return JSONResponse(
                        status_code=409,
                        content={"detail": "Model already exist in db"},
                    )

                ngram_model, _ = pipeline_train(
                    lm_type=model_dict["LM_SCORE"],
                    gamma=model_dict["GAMMA"],
                    min_freq=model_dict["MIN_FREQ"],
                    test_size=model_dict["TEST_SIZE"],
                    n=model_dict["N"],
                    corpus_set=model_dict["CORPUS_NAMES"],
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
    for lm_score, K_pred in zip(LM_TYPES, K_PREDICTIONS):
        try:
            model_dict = {
                "LM_SCORE": lm_score,
                "GAMMA": GAMMA,
                "K_PRED": K_pred,
                "TEST_SIZE": TEST_SIZE,
                "MIN_FREQ": MIN_FREQ,
                "N": N,
                "CORPUS_NAMES": None,
            }

            if collection.find_one(model_dict):
                return JSONResponse(
                    status_code=409, content={"message": "Model already exist in db"}
                )

            ngram_model, _ = pipeline_train(
                lm_type=model_dict["LM_SCORE"],
                gamma=model_dict["GAMMA"],
                min_freq=model_dict["MIN_FREQ"],
                test_size=model_dict["TEST_SIZE"],
                n=model_dict["N"],
                corpus_set=model_dict["CORPUS_NAMES"],
            )

            model_dict["MODEL_FILE_ID"] = save_to_gridfs(ngram_model)
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
            description="Deve contenere una lacuna da riempire, indicata tramite `[...]`",
            max_length=512,
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
                {"sentence": left_context+suggestion , "token_str": suggestion, "score": 0}
                for suggestion in generate_k_suggests(
                    lm=decompressed_model,
                    context=context,
                    num_tokens=num_tokens,
                    lm_type=dict_model["LM_SCORE"],
                    n=dict_model["N"],
                    k_pred=dict_model["K_PRED"],
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
                    convert_lacuna_to_masks(context, decompressed_tokenizer),
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
