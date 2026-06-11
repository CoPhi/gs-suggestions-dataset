import pickle
import zlib
import asyncio
from functools import partial
from typing import Any
from uuid import uuid4

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from backend.api.database import collection, fs
from backend.api.exceptions import ModelAlreadyExistsError, ModelNotFoundError
from backend.api.models import BERTModel, NgramModel, Model
from backend.config.settings import BERT_CHECKPOINTS, GAMMA, LM_TYPES, N
from models.ngrams.train.training import pipeline_train
from backend.api.services.suggestions_service import SuggestionsService


class ModelService:
    """Service layer per la gestione del ciclo di vita dei modelli linguistici."""

    def __init__(self, db_collection=collection, gridfs=fs) -> None:
        self._collection = db_collection
        self._fs = gridfs

    async def get_model(self, model_id: str) -> dict:
        """Restituisce un singolo modello serializzato oppure solleva ModelNotFoundError."""
        model = await self._serial_model(model_id)
        if not model:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return model

    async def get_all_models(self) -> list[dict]:
        """Restituisce tutti i modelli disponibili."""
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
        """
        Crea l'insieme di modelli di default definiti nella configurazione.

        L'operazione è idempotente: i modelli già presenti vengono silenziosamente
        ignorati e non causano l'interruzione della procedura.
        """
        ids: list[str] = []
        for lm_score in LM_TYPES:
            try:
                ids.append(
                    await self._create_ngram_model_from_params(lm_score, GAMMA, N)
                )
            except ModelAlreadyExistsError:
                pass  # modello già presente, si prosegue
        for checkpoint in BERT_CHECKPOINTS:
            try:
                ids.append(await self._create_bert_model_from_checkpoint(checkpoint))
            except ModelAlreadyExistsError:
                pass  # modello già presente, si prosegue
        return ids

    async def delete_model(self, model_id: str) -> dict:
        """Elimina un modello e i file GridFS associati; restituisce il modello eliminato."""
        model = await self._serial_model(model_id)
        if not model:
            raise ModelNotFoundError(f"Model '{model_id}' not found")

        if model["TYPE"] == "Ngrams":
            await self._delete_gridfs_files(model)
        elif model["TYPE"] == "BERT":
            SuggestionsService._bert_cache.pop(model.get("CHECKPOINT"), None)

        await self._collection.delete_one({"_id": ObjectId(model_id)})
        return model

    # Private helpers

    async def _serial_model(self, id: str) -> dict:
        """
        Serializza un documento del database in un dizionario Python,
        convertendo l'ObjectId in stringa, e lo cerca nel db.
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

    async def _check_duplicate(self, identity_filter: dict) -> None:
        """
        Solleva ModelAlreadyExistsError se esiste già un documento che corrisponde
        al filtro identitario fornito.

        Args:
            identity_filter: dizionario contenente SOLO le chiavi identificative
                             del modello (es. TYPE + CHECKPOINT per BERT,
                             TYPE + LM_SCORE + N per Ngram). Non includere campi
                             che variano tra creazione e persistenza (es. file ID).
        """
        if await self._collection.find_one(identity_filter):
            raise ModelAlreadyExistsError("Model already exists in db")

    async def _insert_one_safe(self, document: dict) -> str:
        """
        Esegue insert_one gestendo DuplicateKeyError come ModelAlreadyExistsError.

        Protegge dalla race condition TOCTOU: anche se due richieste concorrenti
        superano _check_duplicate, solo una avrà successo nell'insert grazie
        all'indice unico MongoDB; l'altra riceverà ModelAlreadyExistsError.
        """
        try:
            result = await self._collection.insert_one(document)
            return str(result.inserted_id)
        except DuplicateKeyError as e:
            raise ModelAlreadyExistsError("Model already exists in db") from e

    async def _create_ngram_model(self, model: NgramModel) -> str:
        model_dict = model.model_dump()

        identity_filter = {
            "TYPE": "Ngrams",
            "LM_SCORE": model_dict["LM_SCORE"],
            "N": model_dict["N"],
        }
        await self._check_duplicate(identity_filter)
        return await self._train_and_persist_ngram(model_dict)

    async def _create_ngram_model_from_params(
        self, lm_score: str, gamma: float, n: int
    ) -> str:
        identity_filter = {"TYPE": "Ngrams", "LM_SCORE": lm_score, "N": n}
        await self._check_duplicate(identity_filter)
        model_dict = {
            "LM_SCORE": lm_score,
            "GAMMA": gamma,
            "N": n,
            "CORPUS_NAMES": None,
            "TYPE": "Ngrams",
        }
        return await self._train_and_persist_ngram(model_dict)

    async def _train_and_persist_ngram(self, model_dict: dict) -> str:
        loop = asyncio.get_running_loop()
        train_func = partial(
            pipeline_train,
            lm_type=model_dict["LM_SCORE"],
            gamma=model_dict["GAMMA"],
            n=model_dict["N"],
        )
        global_model, domain_model, _ = await loop.run_in_executor(None, train_func)

        model_dict["GLOBAL_MODEL_FILE_ID"] = await self._save_to_gridfs(global_model)
        model_dict["DOMAIN_MODEL_FILE_ID"] = await self._save_to_gridfs(domain_model)
        model_dict.setdefault("TYPE", "Ngrams")
        return await self._insert_one_safe(model_dict)

    async def _create_bert_model(self, model: BERTModel) -> str:
        model_dict = model.model_dump()
        identity_filter = {"TYPE": "BERT", "CHECKPOINT": model_dict["CHECKPOINT"]}
        await self._check_duplicate(identity_filter)
        return await self._insert_one_safe(model_dict)

    async def _create_bert_model_from_checkpoint(self, checkpoint: str) -> str:
        identity_filter = {"TYPE": "BERT", "CHECKPOINT": checkpoint}
        await self._check_duplicate(identity_filter)
        model_dict = {"CHECKPOINT": checkpoint, "TYPE": "BERT"}
        return await self._insert_one_safe(model_dict)

    async def _delete_gridfs_files(self, model: dict) -> None:
        """Rimuove da GridFS tutti i file associati al modello."""
        file_keys = [
            "GLOBAL_MODEL_FILE_ID",
            "DOMAIN_MODEL_FILE_ID",
        ]
        for key in file_keys:
            try:
                await self._fs.delete_by_name(model[key])
                SuggestionsService._ngram_cache.pop(model[key], None)
            except Exception:
                pass
