import re
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from api.database import collection, fs
from api.models import PredictionCount, PredictionsResponse
from bson import ObjectId
from inference.suggests import generate_k_suggests
from predictions.bert import fill_mask

from utils.preprocess import test_case_contains_lacuna
import pickle
import zlib
from uuid import uuid4
from typing import Annotated, Optional
from fastapi.responses import RedirectResponse

router = APIRouter(tags=["Predictions"])

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")  # si fa il redirect alla documentazione


@router.get(
    "/predictions",
    response_model=PredictionsResponse,
    responses={
        200: {
            "description": "Predictions",
            "content": {
                "application/json": {"example": {"predictions": {"type": "array"}}}
            },
        },
        204: {"description": "No content"},  # quando non ci sono predizioni
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
        if test_case_contains_lacuna(context) is None:
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

            predictions = [
                {
                    "sentence": re.sub(
                        r"\S*\[.*?\]\S*", suggestion[0], context, count=1
                    ),
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

            return JSONResponse(
                status_code=200,
                content={"predictions": predictions},
            )

        else:
            return JSONResponse(status_code=404, content={"message": "Model not found"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
