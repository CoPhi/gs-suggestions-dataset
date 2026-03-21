import pickle
import re
import zlib
from typing import Any

from bson import ObjectId

from api.database import collection, fs
from api.exceptions import InvalidContextError, ModelNotFoundError
from api.models import ModelType
from inference.suggests import generate_k_suggests
from predictions.bert import fill_mask
from utils.preprocess import test_case_contains_lacuna

LACUNA_PATTERN = r"\S*\[.*?\]\S*"

class SuggestionsService:
    """Service layer for generating textual suggestions via N-gram or BERT models."""

    def __init__(self, db_collection=collection, gridfs=fs) -> None:
        self._collection = db_collection
        self._fs = gridfs

    def get_predictions(
        self,
        model_id: str,
        context: str,
        num_tokens: int,
        num_predictions: Any,
    ) -> list[dict]:
        """Return a ranked list of predictions for the given lacuna context."""
        model = self._fetch_model(model_id)
        self._validate_context(context)

        model_type = model.get("TYPE")

        if model_type == ModelType.NGRAMS:
            return self._predict_ngrams(model, context, num_tokens, num_predictions)
        if model_type == ModelType.BERT:
            return self._predict_bert(model, context, num_predictions)

        raise ModelNotFoundError(f"Unsupported model type: {model_type!r}")

    # Private helper methods
    def _fetch_model(self, model_id: str) -> dict:
        """Retrieve and return the model document from MongoDB."""
        try:
            model = self._collection.find_one({"_id": ObjectId(model_id)})
        except Exception as exc:
            raise ModelNotFoundError(f"Invalid model ID: {model_id!r}") from exc

        if model is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")

        return dict(model)

    def _validate_context(self, context: str) -> None:
        """Raise InvalidContextError if the context lacks a lacuna marker."""
        if test_case_contains_lacuna(context) is None:
            raise InvalidContextError(
                "Context must contain a gap indicated by `[...]`"
            )

    def _load_compressed_file(self, filename: str) -> Any:
        """Fetch a GridFS file, decompress it, and deserialise with pickle."""
        document = self._fs.find_one({"filename": filename})
        if document is None:
            raise ModelNotFoundError(f"Model file '{filename}' not found in GridFS")
        raw = self._fs.get(document._id).read()
        return pickle.loads(zlib.decompress(raw))

    def _predict_ngrams(
        self,
        model: dict,
        context: str,
        num_tokens: int,
        num_predictions: Any,
    ) -> list[dict]:
        """Generate predictions using the N-gram language model."""
        global_model = self._load_compressed_file(model["GLOBAL_MODEL_FILE_ID"])
        domain_model = self._load_compressed_file(model["DOMAIN_MODEL_FILE_ID"])

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

    def _predict_bert(
        self,
        model: dict,
        context: str,
        num_predictions: Any,
    ) -> list[dict]:
        """Generate predictions using a fine-tuned BERT model."""
        bert_model = self._load_compressed_file(model["MODEL_FILE_ID"])
        tokenizer = self._load_compressed_file(model["TOKENIZER_FILE_ID"])

        return [
            {
                "sentence": prediction[0],
                "token_str": prediction[1],
                "score": prediction[2],
            }
            for prediction in fill_mask(bert_model, tokenizer, context, num_predictions)
        ]
