import pickle
import zlib
from typing import Any
from uuid import uuid4

from bson import ObjectId
from transformers import AutoModelForMaskedLM, AutoTokenizer

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

    def get_model(self, model_id: str) -> dict:
        """Restituisce un singolo modello serializzato oppure solleva ModelNotFoundError."""
        model = self._serial_model(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return model

    def get_all_models(self) -> list[dict]:
        """Restituisce tutti i modelli disponibili oppure solleva ModelNotFoundError."""
        models = list(self._collection.find())
        if not models:
            raise ModelNotFoundError("No models found")
        return [self._serial_model(str(m["_id"])) for m in models]

    def create_model(self, model: Model) -> str:
        """Crea un modello Ngram o BERT, lo memorizza nel db e restituisce il suo ID."""
        if isinstance(model, NgramModel):
            return self._create_ngram_model(model)
        if isinstance(model, BERTModel):
            return self._create_bert_model(model)
        raise ValueError("Unsupported model type")

    def init_models(self) -> list[str]:
        """Crea l'insieme di modelli di default definiti nella configurazione."""
        ids: list[str] = []
        for lm_score in LM_TYPES:
            ids.append(self._create_ngram_model_from_params(lm_score, GAMMA, N))
        for checkpoint in BERT_CHECKPOINTS:
            ids.append(self._create_bert_model_from_checkpoint(checkpoint))
        return ids

    def delete_model(self, model_id: str) -> dict:
        """Elimina un modello e i file GridFS associati; restituisce il modello eliminato."""
        model = self._serial_model(model_id)
        if not model:
            raise ModelNotFoundError(f"Model '{model_id}' not found")

        self._delete_gridfs_files(model)
        self._collection.delete_one({"_id": ObjectId(model_id)})
        return model

    # Private helpers

    def _serial_model(self, id: str) -> dict:
        """
        Serializza un documento del database in un dizionario Python, convertendo l'ObjectId in stringa, e lo cerca nel db.
        """
        document = self._collection.find_one({"_id": ObjectId(id)})
        if not document:
            return {}

        document["_id"] = str(document["_id"])
        return document

    def _save_to_gridfs(self, data: Any, file_id: str | None = None) -> str:
        """Serializza, comprime e salva un oggetto su GridFS; restituisce il filename."""
        compressed = zlib.compress(pickle.dumps(data))
        filename = file_id or str(uuid4())
        self._fs.put(compressed, filename=filename)
        return filename

    def _check_duplicate(self, model_dict: dict) -> None:
        """Solleva ModelAlreadyExistsError se il modello è già presente in MongoDB."""
        if self._collection.find_one(model_dict):
            raise ModelAlreadyExistsError("Model already exists in db")

    def _create_ngram_model(self, model: NgramModel) -> str:
        model_dict = model.model_dump()
        self._check_duplicate(model_dict)
        return self._train_and_persist_ngram(model_dict)

    def _create_ngram_model_from_params(
        self, lm_score: str, gamma: float, n: int
    ) -> str:
        model_dict = {
            "LM_SCORE": lm_score,
            "GAMMA": gamma,
            "N": n,
            "CORPUS_NAMES": None,
            "TYPE": "Ngrams",
        }
        self._check_duplicate(model_dict)
        return self._train_and_persist_ngram(model_dict)

    def _train_and_persist_ngram(self, model_dict: dict) -> str:
        global_model, domain_model, _ = pipeline_train(
            lm_type=model_dict["LM_SCORE"],
            gamma=model_dict["GAMMA"],
            n=model_dict["N"],
        )
        model_dict["GLOBAL_MODEL_FILE_ID"] = self._save_to_gridfs(global_model)
        model_dict["DOMAIN_MODEL_FILE_ID"] = self._save_to_gridfs(domain_model)
        model_dict.setdefault("TYPE", "Ngrams")
        return str(self._collection.insert_one(model_dict).inserted_id)

    def _create_bert_model(self, model: BERTModel) -> str:
        model_dict = model.model_dump()
        self._check_duplicate(model_dict)
        return self._create_bert_model_from_checkpoint(model_dict["CHECKPOINT"])

    def _create_bert_model_from_checkpoint(self, checkpoint: str) -> str:
        model_dict = {"CHECKPOINT": checkpoint, "TYPE": "BERT"}
        self._check_duplicate(model_dict)
        bert_model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model_dict["MODEL_FILE_ID"] = self._save_to_gridfs(bert_model)
        model_dict["TOKENIZER_FILE_ID"] = self._save_to_gridfs(bert_tokenizer)
        return str(self._collection.insert_one(model_dict).inserted_id)

    def _delete_gridfs_files(self, model: dict) -> None:
        """Rimuove da GridFS tutti i file associati al modello."""
        file_keys = [
            "MODEL_FILE_ID",
            "TOKENIZER_FILE_ID",
            "GLOBAL_MODEL_FILE_ID",
            "DOMAIN_MODEL_FILE_ID",
        ]
        for key in file_keys:
            if key in model:
                doc = self._fs.find_one({"filename": model[key]})
                if doc:
                    self._fs.delete(doc._id)
