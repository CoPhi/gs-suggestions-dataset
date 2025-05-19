from fastapi import APIRouter, Query, Path, Body
from fastapi.responses import JSONResponse
from api.schema import serial_model
from api.database import collection, fs
from api.models import NgramModel, BERTModel, Model, PredictionCount, PredictionsResponse, ModelsResponse
from api import LEFT_CONTEXT_PATTERN
from bson import ObjectId
from train.training import train_lm
from train import load_abs
from inference import generate_k_suggests
from finetuning import fill_mask
from config.settings import (
    LM_TYPES,
    GAMMA,
    N,
    BERT_CHECKPOINTS,
)
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


router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")  # si fa il redirect alla documentazione


@router.get(
    "/models/{id}",
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
    "/models",
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
    "/models",
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

                global_ngram_model, domain_ngram_model = train_lm(
                    train_abs=abs,
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

        case BERTModel():
            try:
                model_dict = dict(model)
                if collection.find_one(model_dict):
                    return JSONResponse(
                        status_code=409,
                        content={"message": "Model already exist in db"},
                    )
                print(model_dict)
                bert_model = AutoModelForMaskedLM.from_pretrained(
                    model_dict["CHECKPOINT"]
                )

                bert_tokenizer = AutoTokenizer.from_pretrained(model_dict["CHECKPOINT"])
                model_dict["MODEL_FILE_ID"] = save_to_gridfs(bert_model)
                model_dict["TOKENIZER_FILE_ID"] = save_to_gridfs(bert_tokenizer)

                model_id = collection.insert_one(
                    {**model_dict, "TYPE": "BERT"}
                ).inserted_id
                return JSONResponse(status_code=201, content={"ID": str(model_id)})

            except ValueError as v:
                return JSONResponse(status_code=404, content={"message": str(v)})
            except Exception as e:
                return JSONResponse(status_code=500, content={"detail": str(e)})


@router.post(
    "/models/init/",
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

            global_ngram_model, domain_ngram_model = train_lm(
                train_abs=abs,
                lm_type=model_dict["LM_SCORE"],
                gamma=model_dict["GAMMA"],
                n=model_dict["N"],
            )
            model_dict["GLOBAL_MODEL_FILE_ID"] = save_to_gridfs(global_ngram_model)
            model_dict["DOMAIN_MODEL_FILE_ID"] = save_to_gridfs(domain_ngram_model)
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


# predictions
@router.get(
    "/predictions",
    response_model=PredictionsResponse, 
    responses={
        200: {
            "description": "Predictions",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": 
                            {
                                "type": "array"
                            }
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
            global_model_filename = dict_model["GLOBAL_MODEL_FILE_ID"]
            domain_model_filename = dict_model["DOMAIN_MODEL_FILE_ID"]
            global_model_file_document = fs.find_one(
                {"filename": global_model_filename}
            )  # Cerca il file per nome
            domain_model_file_document = fs.find_one(
                {"filename": domain_model_filename}
            )  # Cerca il file per nome

            if not (global_model_file_document or domain_model_file_document):
                return JSONResponse(
                    status_code=404, content={"message": "Model not found"}
                )

            global_model_file = fs.get(
                global_model_file_document._id
            )  # Recupera il file usando il suo _id
            domain_model_file = fs.get(domain_model_file_document._id)
            decompressed_global_model = pickle.loads(
                zlib.decompress(global_model_file.read())
            )
            decompressed_domain_model = pickle.loads(
                zlib.decompress(domain_model_file.read())
            )

            left_context = LEFT_CONTEXT_PATTERN.sub("[", context).split("[")[0]
            predictions = [
                {
                    "sentence": left_context + suggestion[0],
                    "token_str": suggestion[0],
                    "score": suggestion[1],
                }
                for suggestion in generate_k_suggests(
                    g_lm=decompressed_global_model,
                    d_lm=decompressed_domain_model,
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
                {
                    "sentence": prediction[0],
                    "token_str": prediction[1],
                    "score": prediction[2],
                }
                for prediction in fill_mask(
                    decompressed_model, decompressed_tokenizer, context, num_predictions
                )
            ]

            if not predictions:
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": str("No predictions generated, check the context")
                    },
                )

            return JSONResponse(
                status_code=200,
                content={"predictions": predictions},
            )

        else:
            return JSONResponse(status_code=404, content={"message": "Model not found"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.delete(
    "/models/{id}",
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

        # Rimuovi il documento dalla collezione
        collection.delete_one({"_id": ObjectId(id)})

        return JSONResponse(
            status_code=200, content={"model": model}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
