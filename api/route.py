from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from api.schema import serial_model, list_models
from api.database import collection, fs
from api.models import NgramModel
from bson import ObjectId
from train.training import pipeline_train
from inference import generate_k_suggests
from config.settings import K_PREDICTIONS, LM_TYPES, GAMMA, TEST_SIZE, N, MIN_FREQ
import pickle
import zlib
from uuid import uuid4

router = APIRouter()


# get model/id
@router.get("/model/{id}")
async def get_model(id: str):
    try:
        model = collection.find_one({"_id": ObjectId(id)})
        if model is None:
            return JSONResponse(status_code=404, content={"message": "Model not found"})
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
            return JSONResponse(
                status_code=404, content={"message": "Models are not found"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# post model/
@router.post("/model")
async def create_model(model: NgramModel):
    try:
        model_dict = dict(model)
        if model_dict["K_PRED"] not in K_PREDICTIONS:
            return JSONResponse(
                status_code=400, content={"message": "Invalid K_PRED value"}
            )

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

        pickled_model = pickle.dumps(ngram_model)
        compressed_model = zlib.compress(pickled_model)

        # Una volta compresso, si usa GridFS per salvare il modello in MongoDB
        filename = f"{uuid4()}"
        fs.put(compressed_model, filename=filename)

        model_dict["MODEL_FILE_ID"] = filename
        model_id = collection.insert_one(model_dict).inserted_id
        return {"ID": str(model_id)}
    except ValueError as v:
        return JSONResponse(status_code=404, content={"message": str(v)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models")
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
            
            pickled_model = pickle.dumps(ngram_model)
            compressed_model = zlib.compress(pickled_model)
            filename = f"{uuid4()}"
            fs.put(compressed_model, filename=filename)

            model_dict["MODEL_FILE_ID"] = filename
            model_id = collection.insert_one(model_dict).inserted_id
            ids.append(str(model_id))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"IDs": ids}


# predictions
@router.get("/predictions")
async def get_predictions(model_id: str, context: str, num_words: int):
    try:
        model = collection.find_one({"_id": ObjectId(model_id)})
        if model is None:
            return JSONResponse(status_code=404, content={"message": "Model not found"})

        dict_model = dict(model)

        if dict_model["TYPE"] == "Ngrams":
            model_filename = dict_model["MODEL_FILE_ID"] 
            file_document = fs.find_one({"filename": model_filename})  # Cerca il file per nome
            if not file_document:
                return JSONResponse(status_code=404, content={"message": "Model not found"})
        
            model_file = fs.get(file_document._id)  # Recupera il file usando il suo _id
            decompressed_model = pickle.loads(zlib.decompress(model_file.read()))

            return generate_k_suggests(
                decompressed_model,
                context,
                num_words,
                dict_model["LM_SCORE"],
                dict_model["N"],
                dict_model["K_PRED"],
            )
            
        if dict_model["TYPE"] == "BERT":
            pass
        
        else:
            return JSONResponse(status_code=404, content={"message": "Model not found"})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
