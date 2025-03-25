from fastapi import APIRouter, HTTPException
from api.schema import serial_model, list_models
from api.database import collection
from api.models import NgramModel
from bson import ObjectId
from models.training import pipeline_train
from models.infer import generate_k_suggests
from config.settings import K_PREDICTIONS, LM_TYPES, GAMMA, TEST_SIZE, N
import pickle
import zlib

router = APIRouter()


# get model/id
@router.get("/model/{id}")
async def get_model(id: str):
    try:
        model = collection.find_one({"_id": ObjectId(id)})
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return serial_model(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# get models/
@router.get("/models")
async def get_models():
    try:
        models = collection.find()
        if models:
            return list_models(models)
        else:
            raise HTTPException(status_code=404, detail="Models not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# post model/
@router.post("/model")
async def post_model(model: NgramModel):
    try:
        model_dict = dict(model)
        if model_dict["K_PRED"] not in K_PREDICTIONS:
            raise HTTPException(status_code=400, detail="Invalid K_PRED")

        ngram_model, _ = pipeline_train(
            lm_type=model_dict["LM_SCORE"],
            gamma=model_dict["GAMMA"],
            test_size=model_dict["TEST_SIZE"],
            n=model_dict["N"],
            corpus_set=model_dict["CORPUS_NAMES"],
        )

        # Compress the pickled model
        pickled_model = pickle.dumps(ngram_model)
        compressed_model = zlib.compress(pickled_model)

        model_dict["MODEL"] = compressed_model
        model_id = collection.insert_one(model_dict).inserted_id
        return {"ID": str(model_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# post models


@router.post("/model")
async def create_models():
    for lm_score, K_pred in zip(LM_TYPES, K_PREDICTIONS):
        try:
            model_dict = {
                "LM_SCORE": lm_score,
                "GAMMA": GAMMA,
                "K_PRED": K_pred,
                "TEST_SIZE": TEST_SIZE,
                "N": N,
                "CORPUS_NAMES": None,
            }

            # Aggiungere logica per vedere se non sono già presenti nel db

            ngram_model, _ = pipeline_train(
                lm_type=model_dict["LM_SCORE"],
                gamma=model_dict["GAMMA"],
                test_size=model_dict["TEST_SIZE"],
                n=model_dict["N"],
                corpus_set=model_dict["CORPUS_NAMES"],
            )

            # Compress the pickled model
            pickled_model = pickle.dumps(ngram_model)
            compressed_model = zlib.compress(pickled_model)

            model_dict["MODEL"] = compressed_model
            model_id = collection.insert_one(model_dict).inserted_id
            return {"ID": str(model_id)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


#predictions
@router.get("/predictions")
async def get_predictions(model_id: str, context: str, num_words: int):
    try:
        model = collection.find_one({"_id": ObjectId(model_id)})
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")

        dict_model = dict(model)
        decompressed_model = pickle.loads(zlib.decompress(dict_model["MODEL"]))

        return generate_k_suggests(
            decompressed_model, context, num_words, dict_model["N"], dict_model["K_PRED"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
