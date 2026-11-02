import asyncio
import pickle
import re
import zlib
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from collections import OrderedDict
import torch
import gc

from bson import ObjectId
from gridfs.errors import NoFile

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from transformers import AutoModelForMaskedLM, AutoTokenizer

from datetime import datetime, timezone
from backend.api.database import collection, feedback_collection, fs
from backend.api.exceptions import InvalidContextError, ModelNotFoundError
from backend.api.models import FeedbackCreateRequest, ModelType
from models.ngrams.inference.suggests import generate_k_suggests
from models.bert.inference.predict import fill_mask
from backend.core.preprocess import normalize_greek, test_case_contains_lacuna

LACUNA_PATTERN = r"\S*\[.*?\]\S*"
BERT_LACUNA_PATTERN = r"\[.*?\]"
INTRA_WORD_LACUNA_PATTERN = re.compile(r"(\S+)\[\.+\](\S+)")


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

    _bert_cache: OrderedDict[str, tuple] = OrderedDict()
    _ngram_cache: dict[str, Any] = {}
    MAX_BERT_MODELS = 3

    def __init__(
        self,
        db_collection=collection,
        gridfs=fs,
        feedback_db=feedback_collection,
    ) -> None:
        self._collection = db_collection
        self._fs = gridfs
        self._feedback_collection = feedback_db

    async def save_feedback(self, payload: FeedbackCreateRequest) -> dict[str, str]:
        """
        Persiste in MongoDB la validazione o lo scarto di un supplemento da parte del filologo
        (Human-in-the-loop Fase 1) per costruire il dataset curato di esempi positivi/negativi.
        """
        doc = {
            "model_id": payload.model_id,
            "context": payload.context,
            "token_str": payload.token_str,
            "sentence": payload.sentence,
            "is_accepted": payload.is_accepted,
            "notes": payload.notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = await self._feedback_collection.insert_one(doc)
        return {"id": str(result.inserted_id), "status": "saved"}

    async def get_feedback_list(
        self,
        limit: int = 100,
        is_accepted: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Recupera gli ultimi feedback annotati dai filologi, filtrabili per esito (accettato/scartato)."""
        query: dict[str, Any] = {}
        if is_accepted is not None:
            query["is_accepted"] = is_accepted

        cursor = self._feedback_collection.find(query).sort("created_at", -1).limit(limit)
        items: list[dict[str, Any]] = []
        async for doc in cursor:
            serialized = dict(doc)
            if "_id" in serialized:
                serialized["id"] = str(serialized.pop("_id"))
            items.append(serialized)
        return items

    async def get_predictions(
        self,
        model_id: str,
        context: str,
        num_tokens: int,
        num_predictions: Any,
        human_feedback: str | None = None,
        rejected_candidates: list[str] | None = None,
    ) -> list[dict]:
        model = await self._fetch_model(model_id)
        self._validate_context(context)
        model_type = model.get("TYPE")

        predictions: list[dict] | None = None
        if model_type == ModelType.NGRAMS:
            predictions = await self._predict_ngrams(
                model, context, num_tokens, num_predictions
            )
        else:
            try:
                if model_type == ModelType.BERT:
                    predictions = await self._predict_bert(
                        model, context, num_predictions
                    )
                elif model_type == ModelType.ITHACA:
                    predictions = await self._predict_ithaca(
                        model, context, num_predictions
                    )
                elif model_type == ModelType.RAG:
                    predictions = await self._predict_rag(
                        model,
                        context,
                        num_predictions,
                        human_feedback=human_feedback,
                        rejected_candidates=rejected_candidates,
                    )
            except RepositoryNotFoundError:
                raise ModelNotFoundError(
                    f"Checkpoint '{model['CHECKPOINT']}' not found on HuggingFace Hub"
                )

        if predictions is None:
            raise ModelNotFoundError(f"Unsupported model type: {model_type!r}")

        # Filtra deterministicamente eventuali supplementi scartati dal filologo nella sessione
        # corrente (HITL Fase 2) su tutte le famiglie di modelli (N-grammi, BERT, Ithaca, RAG).
        if rejected_candidates:
            rejected_norm = {
                normalize_greek(r.strip(), case_folding="fold", strip_diacritics_flag=True)
                for r in rejected_candidates
                if r and r.strip()
            }
            if rejected_norm:
                predictions = [
                    pred
                    for pred in predictions
                    if normalize_greek(
                        str(pred.get("token_str", "")),
                        case_folding="fold",
                        strip_diacritics_flag=True,
                    )
                    not in rejected_norm
                ]

        return predictions

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
        if filename in self._ngram_cache:
            return self._ngram_cache[filename]

        try:
            stream = await self._fs.open_download_stream_by_name(filename)
        except NoFile:
            raise ModelNotFoundError(f"Model file '{filename}' not found in GridFS")
        raw = await stream.read()
        model = pickle.loads(zlib.decompress(raw))
        self._ngram_cache[filename] = model
        return model

    async def _predict_ngrams(
        self, model: dict, context: str, num_tokens: int, num_predictions: Any
    ) -> list[dict]:
        global_model = await self._load_compressed_file(model["GLOBAL_MODEL_FILE_ID"])
        domain_model = await self._load_compressed_file(model["DOMAIN_MODEL_FILE_ID"])

        loop = asyncio.get_running_loop()
        predict_func = partial(
            generate_k_suggests,
            g_lm=global_model,
            d_lm=domain_model,
            context=context,
            num_tokens=num_tokens,
            lm_type=model["LM_SCORE"],
            n=model["N"],
            k_pred=num_predictions.value,
        )
        suggestions = await loop.run_in_executor(None, predict_func)

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

        if checkpoint not in self._bert_cache:
            await self._validate_hf_checkpoint(checkpoint)
            loop = asyncio.get_running_loop()
            bert_model, tokenizer = await loop.run_in_executor(
                None, _load_bert_checkpoint, checkpoint
            )
            self._bert_cache[checkpoint] = (bert_model, tokenizer)
            self._bert_cache.move_to_end(checkpoint)

            if len(self._bert_cache) > self.MAX_BERT_MODELS:
                self._bert_cache.popitem(last=False)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            bert_model, tokenizer = self._bert_cache[checkpoint]
            self._bert_cache.move_to_end(checkpoint)

        loop = asyncio.get_running_loop()
        predict_func = partial(
            fill_mask,
            text=context,
            model=bert_model,
            tokenizer=tokenizer,
            K=num_predictions.value,
            normalize_probs=True,
            checkpoint=checkpoint,
        )
        suggestions = await loop.run_in_executor(None, predict_func)

        return [
            {
                "sentence": re.sub(BERT_LACUNA_PATTERN, p[0], context, count=1).lower(),
                "token_str": p[0].lower(),
                "score": float(p[1]),
            }
            for p in suggestions
        ]

    async def _predict_ithaca(
        self, model: dict, context: str, num_predictions: Any
    ) -> list[dict]:
        """Genera predizioni usando il modello Ithaca specificato dal checkpoint."""
        checkpoint = model["CHECKPOINT"]
        from models.ithaca.inference.predict import fill_mask_ithaca

        loop = asyncio.get_running_loop()
        predict_func = partial(
            fill_mask_ithaca,
            text=context,
            checkpoint=checkpoint,
            K=num_predictions.value,
        )
        suggestions = await loop.run_in_executor(None, predict_func)

        return [
            {
                "sentence": re.sub(BERT_LACUNA_PATTERN, p[0], context, count=1),
                "token_str": p[0],
                "score": float(p[1]),
            }
            for p in suggestions
        ]

    async def _predict_rag(
        self,
        model: dict,
        context: str,
        num_predictions: Any,
        human_feedback: str | None = None,
        rejected_candidates: list[str] | None = None,
    ) -> list[dict]:
        """Genera predizioni usando il sistema DSPy RAG specificato dal modello."""
        from models.rag.inference import fill_mask_rag

        checkpoint = model["CHECKPOINT"]
        backend = model.get("BACKEND", "auto")
        top_k_passages = int(model.get("TOP_K_PASSAGES", 5))
        compiled_program_path = model.get("COMPILED_PROGRAM_PATH")

        loop = asyncio.get_running_loop()
        # Passa i parametri HITL (human_feedback / rejected_candidates) solo se valorizzati
        # per preservare la compatibilità con eventuali mock di test con firma minimale.
        rag_kwargs: dict[str, Any] = {
            "text": context,
            "checkpoint": checkpoint,
            "backend": backend,
            "top_k_passages": top_k_passages,
            "compiled_program_path": compiled_program_path,
            "K": num_predictions.value,
            "normalize_probs": True,
            "return_details": True,
        }
        if human_feedback:
            rag_kwargs["human_feedback"] = human_feedback
        if rejected_candidates:
            rag_kwargs["rejected_candidates"] = rejected_candidates

        predict_func = partial(fill_mask_rag, **rag_kwargs)
        raw_result = await loop.run_in_executor(None, predict_func)
        # Mantiene retrocompatibilità se `fill_mask_rag` (es. in test mock) restituisce solo la lista di tuple
        if (
            isinstance(raw_result, tuple)
            and len(raw_result) == 2
            and isinstance(raw_result[1], dict)
        ):
            suggestions, details = raw_result
        else:
            suggestions, details = raw_result, {}

        results: list[dict] = []
        for idx, p in enumerate(suggestions):
            item: dict[str, Any] = {
                "sentence": re.sub(BERT_LACUNA_PATTERN, p[0], context, count=1).lower(),
                "token_str": p[0].lower(),
                "score": float(p[1]),
            }
            # Allega ragionamento filologico e loci similes solo al primo candidato per non duplicare il payload JSON
            if idx == 0 and details:
                reasoning = details.get("philological_reasoning")
                passages = details.get("retrieved_passages")
                if reasoning:
                    item["philological_reasoning"] = reasoning
                if passages:
                    item["retrieved_passages"] = list(passages)
            results.append(item)
        return results

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
