import pickle
import zlib
from typing import Any
from uuid import uuid4

from bson import ObjectId

from api.database import collection, fs
from api.exceptions import ModelAlreadyExistsError, ModelNotFoundError
from api.models import BERTModel, NgramModel, Model
from config.settings import BERT_CHECKPOINTS, GAMMA, LM_TYPES, N
from train.training import pipeline_train


class ModelService:
    """Service layer per la gestione del ciclo di vita dei modelli linguistici."""

    def __init__(self, db_collection=collection, gridfs=fs) -> None:
        self._collection = db_collection
        self._fs = gridfs

    async def get_model(self, model_id: str) -> dict:
        """Restituisce un singolo modello serializzato oppure solleva ModelNotFoundError."""
        model = await self._serial_model(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return model

    async def get_all_models(self) -> list[dict]:
        """Restituisce tutti i modelli disponibili"""
        models = [model async for model in self._collection.find()]
        return [await self._serial_model(str(m["_id"])) for m in models]

    async def create_model(self, model: Model) -> str:
        """Crea un modello Ngram o BERT, lo memorizza nel db e restituisce il suo ID."""
        if isinstance(model, NgramModel):
            return await self._create_ngram_model(model)
        if isinstance(model, BERTModel):
            return await self._create_bert_model(model)
        raise ValueError("Unsupported model type")

    async def init_models(self) -> list[str]:
        """Crea l'insieme di modelli di default definiti nella configurazione."""
        ids: list[str] = []
        for lm_score in LM_TYPES:
            ids.append(await self._create_ngram_model_from_params(lm_score, GAMMA, N))
        for checkpoint in BERT_CHECKPOINTS:
            ids.append(await self._create_bert_model_from_checkpoint(checkpoint))
        return ids

    async def delete_model(self, model_id: str) -> dict:
        """Elimina un modello e i file GridFS associati; restituisce il modello eliminato."""
        model = await self._serial_model(model_id)
        if not model:
            raise ModelNotFoundError(f"Model '{model_id}' not found")

        if model["TYPE"] == "Ngrams":
            await self._delete_gridfs_files(model)

        await self._collection.delete_one({"_id": ObjectId(model_id)})
        return model

    # Private helpers

    async def _serial_model(self, id: str) -> dict:
        """
        Serializza un documento del database in un dizionario Python, convertendo l'ObjectId in stringa, e lo cerca nel db.
        """
        document = await self._collection.find_one({"_id": ObjectId(id)})
        if not document:
            return {}

        document["_id"] = str(document["_id"])
        return document

    async def _save_to_gridfs(self, data: Any, file_id: str | None = None) -> str:
        """Serializza, comprime e salva un oggetto su GridFS; restituisce il filename."""
        compressed = zlib.compress(pickle.dumps(data))
        filename = file_id or str(uuid4())
        await self._fs.upload_from_stream(filename, compressed)
        return filename

    async def _check_duplicate(self, model_dict: dict) -> None:
        """Solleva ModelAlreadyExistsError se il modello è già presente in MongoDB."""
        if await self._collection.find_one(model_dict):
            raise ModelAlreadyExistsError("Model already exists in db")

    async def _create_ngram_model(self, model: NgramModel) -> str:
        model_dict = model.model_dump()
        await self._check_duplicate(model_dict)
        return await self._train_and_persist_ngram(model_dict)

    async def _create_ngram_model_from_params(
        self, lm_score: str, gamma: float, n: int
    ) -> str:
        model_dict = {
            "LM_SCORE": lm_score,
            "GAMMA": gamma,
            "N": n,
            "CORPUS_NAMES": None,
            "TYPE": "Ngrams",
        }
        await self._check_duplicate(model_dict)
        return await self._train_and_persist_ngram(model_dict)

    async def _train_and_persist_ngram(self, model_dict: dict) -> str:
        global_model, domain_model, _ = pipeline_train(
            lm_type=model_dict["LM_SCORE"],
            gamma=model_dict["GAMMA"],
            n=model_dict["N"],
        )
        model_dict["GLOBAL_MODEL_FILE_ID"] = await self._save_to_gridfs(global_model)
        model_dict["DOMAIN_MODEL_FILE_ID"] = await self._save_to_gridfs(domain_model)
        model_dict.setdefault("TYPE", "Ngrams")
        result = await self._collection.insert_one(model_dict)
        return str(result.inserted_id)

    async def _create_bert_model(self, model: BERTModel) -> str:
        model_dict = model.model_dump()
        await self._check_duplicate(model_dict)
        result = await self._collection.insert_one(model_dict)
        return str(result.inserted_id)

    async def _create_bert_model_from_checkpoint(self, checkpoint: str) -> str:
        model_dict = {"CHECKPOINT": checkpoint, "TYPE": "BERT"}
        await self._check_duplicate(model_dict)
        result = await self._collection.insert_one(model_dict)
        return str(result.inserted_id)

    async def _delete_gridfs_files(self, model: dict) -> None:
        """Rimuove da GridFS tutti i file associati al modello."""
        file_keys = [
            "GLOBAL_MODEL_FILE_ID",
            "DOMAIN_MODEL_FILE_ID",
        ]
        for key in file_keys:
            try:
                await self._fs.delete_by_name(model[key])
            except Exception:
                pass
