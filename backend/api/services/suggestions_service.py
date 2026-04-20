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

from backend.api.database import collection, fs
from backend.api.exceptions import InvalidContextError, ModelNotFoundError
from backend.api.models import ModelType
from models.ngrams.inference.suggests import generate_k_suggests
from models.bert.inference.predict import fill_mask
from models.bert.finetuning import get_model_config
from backend.core.preprocess import (
    remove_punctuation,
    process_editorial_marks,
    test_case_contains_lacuna,
    normalize_greek,
)

LACUNA_PATTERN = r"\S*\[.*?\]\S*"
BERT_LACUNA_PATTERN = r"\[.*?\]"
INTRA_WORD_LACUNA_PATTERN = re.compile(r'(\S+)\[\.+\](\S+)')

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

        config = get_model_config(checkpoint)
        remove_punctuation_model = config.get("remove_punct")

        cleaned = process_editorial_marks(context, preserve_lacunae=True)

        normalized = normalize_greek(
            cleaned,
            case_folding=config.get("case_folding", "upper"),
            strip_diacritics_flag=config.get("strip_diacritics", True),
        )
        
        meta = self._prepare_bert_input(normalized)

        suggestions = fill_mask(
            text=(
                remove_punctuation(normalized, preserve_lacunae=True)
                if remove_punctuation_model
                else normalized
            ),
            intra_word=meta["intra_word"],
            prefix=meta["prefix"],
            suffix=meta["suffix"],
            model=bert_model,
            tokenizer=tokenizer,
            K=num_predictions.value,
            normalize_probs=True,
        )

        return [
            {
                "sentence": re.sub(BERT_LACUNA_PATTERN, p[0], context, count=1).lower(),
                "token_str": p[0].lower(),
                "score": float(p[1]),
            }
            for p in suggestions
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
            
    def _prepare_bert_input(self, normalized_text: str) -> dict:
        """
        Rileva se la lacuna è intra-parola (es. "ΚΑΤ[.]ΣΚΕΥΑΖΕΙΝ") 
        o inter-parola (es. "ΑΛΛΑ [..] ΕΧΕΙ").

        Ritorna un dict con:
        - bert_text:   testo con [MASK] al posto della lacuna
        - n_chars:     numero di caratteri nella lacuna
        - intra_word:  bool
        - prefix:      parte sinistra della parola (solo se intra_word)
        - suffix:      parte destra della parola (solo se intra_word)
        """
        intra_match = INTRA_WORD_LACUNA_PATTERN.search(normalized_text)

        if intra_match:
            prefix = intra_match.group(1)          # "ΚΑΤ"
            suffix = intra_match.group(2)          # "ΣΚΕΥΑΖΕΙΝ"
            dots   = re.search(r'\[(\.+)\]', intra_match.group(0)).group(1)
            n_chars = len(dots)                    # 1

            # Per WordPiece: sostituisce l'intera parola con [MASK]
            # Per BPE: idem, fill_mask gestirà internamente k maschere
            bert_text = INTRA_WORD_LACUNA_PATTERN.sub("[MASK]", normalized_text, count=1)

            return {
                "bert_text": bert_text,
                "n_chars": n_chars,
                "intra_word": True,
                "prefix": prefix,
                "suffix": suffix,
            }

        # Lacuna inter-parola
        dots_match = re.search(r'\[(\.+)\]', normalized_text)
        n_chars = len(dots_match.group(1)) if dots_match else 1
        bert_text = re.sub(r'\[\.+\]', "[MASK]", normalized_text, count=1)

        return {
            "bert_text": bert_text,
            "n_chars": n_chars,
            "intra_word": False,
            "prefix": "",
            "suffix": "",
        }
