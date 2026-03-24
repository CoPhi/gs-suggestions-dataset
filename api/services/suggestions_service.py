import asyncio
import pickle
import re
import zlib
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from bson import ObjectId
from gridfs.errors import NoFile

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from transformers import AutoModelForMaskedLM, AutoTokenizer

from api.database import collection, fs
from api.exceptions import InvalidContextError, ModelNotFoundError
from api.models import ModelType
from inference.suggests import generate_k_suggests
from predictions.bert import fill_mask
from utils.preprocess import test_case_contains_lacuna

LACUNA_PATTERN = r"\S*\[.*?\]\S*"


def _load_bert_checkpoint(checkpoint: str) -> tuple:
    """Funzione pickle-able per caricare un modello BERT e il suo tokenizer, da eseguire in un thread separato."""
    try:
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return model, tokenizer
    except OSError as e:
        raise ModelNotFoundError(
            f"Impossibile caricare il checkpoint '{checkpoint}': {e}"
        )


class SuggestionsService:
    """Service layer for generating textual suggestions via N-gram or BERT models."""

    def __init__(self, db_collection=collection, gridfs=fs) -> None:
        self._collection = db_collection
        self._fs = gridfs

    async def get_predictions(
        self,
        model_id: str,
        context: str,
        num_tokens: int,
        num_predictions: Any,
    ) -> list[dict]:
        model = await self._fetch_model(model_id)
        self._validate_context(context)
        model_type = model.get("TYPE")

        if model_type == ModelType.NGRAMS:
            return await self._predict_ngrams(
                model, context, num_tokens, num_predictions
            )
        try:
            if model_type == ModelType.BERT:
                return await self._predict_bert(model, context, num_predictions)
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Checkpoint '{model['CHECKPOINT']}' not found on HuggingFace Hub"
            )

        raise ModelNotFoundError(f"Unsupported model type: {model_type!r}")

    # Private helpers

    async def _fetch_model(self, model_id: str) -> dict:
        try:
            model = await self._collection.find_one({"_id": ObjectId(model_id)})
        except Exception as exc:
            raise ModelNotFoundError(f"Invalid model ID: {model_id!r}") from exc
        if model is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return dict(model)

    def _validate_context(self, context: str) -> None:
        if test_case_contains_lacuna(context) is None:
            raise InvalidContextError("Context must contain a gap indicated by `[...]`")

    async def _load_compressed_file(self, filename: str) -> Any:
        """Fetch a GridFS file by name, decompress it, and deserialise with pickle."""
        try:
            stream = await self._fs.open_download_stream_by_name(filename)
        except NoFile:
            raise ModelNotFoundError(f"Model file '{filename}' not found in GridFS")
        raw = await stream.read()
        return pickle.loads(zlib.decompress(raw))

    async def _predict_ngrams(
        self, model: dict, context: str, num_tokens: int, num_predictions: Any
    ) -> list[dict]:
        global_model = await self._load_compressed_file(model["GLOBAL_MODEL_FILE_ID"])
        domain_model = await self._load_compressed_file(model["DOMAIN_MODEL_FILE_ID"])

        suggestions = generate_k_suggests(
            g_lm=global_model,
            d_lm=domain_model,
            context=context,
            num_tokens=num_tokens,
            lm_type=model["LM_SCORE"],
            n=model["N"],
            k_pred=num_predictions.value,
        )
        return [
            {
                "sentence": re.sub(LACUNA_PATTERN, suggestion[0], context, count=1),
                "token_str": suggestion[0],
                "score": suggestion[1],
            }
            for suggestion in suggestions
        ]

    async def _predict_bert(
        self, model: dict, context: str, num_predictions: Any
    ) -> list[dict]:
        """Genera predizioni usando un modello BERT pre-addestrato specificato dal checkpoint."""
        checkpoint = model["CHECKPOINT"]

        await self._validate_hf_checkpoint(checkpoint)

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            bert_model, tokenizer = await loop.run_in_executor(
                executor, _load_bert_checkpoint, checkpoint
            )
        return [
            {"sentence": p[0], "token_str": p[1], "score": p[2]}
            for p in fill_mask(bert_model, tokenizer, context, num_predictions)
        ]

    async def _validate_hf_checkpoint(self, checkpoint: str) -> None:
        """Verifica che il checkpoint esista su HuggingFace Hub e sia un modello fill-mask."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            info = await loop.run_in_executor(executor, HfApi().model_info, checkpoint)
        pipeline_tag = getattr(info, "pipeline_tag", None)
        if pipeline_tag and pipeline_tag != "fill-mask":
            raise ValueError(
                f"Checkpoint '{checkpoint}' ha task '{pipeline_tag}', atteso 'fill-mask'"
            )
