"""
Test unitari e di integrazione per il sistema di Text-Infilling DSPy RAG (`models/rag/`).
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import numpy as np
import pytest
from bson import ObjectId
from dspy.utils.dummies import DummyLM

from backend.api.models import ModelType, PredictionCount, RAGModel
from backend.api.services.model_service import ModelService
from backend.api.services.suggestions_service import SuggestionsService
from models.bert.dataset.dev_set import DevCase
from models.rag.evaluation import evaluate_rag_on_cases, format_rag_evaluation_markdown
from models.rag.inference import fill_mask_rag
from models.rag.lm import (
    _is_compact_hf_model,
    build_dspy_lm,
    is_hf_api_unavailable,
    is_hf_permission_or_auth_error,
    reset_hf_api_availability,
)
from models.rag.modules import (
    GreekRAGInfiller,
    extract_lacuna_metadata,
    extract_parallel_attestations,
    sanitize_greek_supplement,
)
from models.rag.optimizer import (
    compile_greek_rag_infiller,
    compute_cer,
    dev_cases_to_dspy_examples,
    greek_infilling_dspy_metric,
)
from models.rag.retriever import (
    GreekHybridRetriever,
    build_rag_corpus_from_data,
    normalize_for_retrieval,
)
from models.rag.signatures import format_gap_constraints


def _dummy_greek_embedder(texts: list[str]) -> np.ndarray:
    """
    Embedder deterministico leggero basato su conteggio di trigrammi di caratteri greci,
    usato nei test unitari per testare `dspy.Embedder` e `dspy.Embeddings` senza scaricare pesi da HF.
    """
    dim = 64
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        norm = normalize_for_retrieval(t)
        for idx in range(max(1, len(norm) - 2)):
            chunk = norm[idx : idx + 3]
            bucket = sum(ord(c) * (j + 1) for j, c in enumerate(chunk)) % dim
            out[i, bucket] += 1.0
        norm_val = np.linalg.norm(out[i])
        if norm_val > 0:
            out[i] /= norm_val
        else:
            out[i, 0] = 1.0
    return out


# ==============================================================================
# 1. TEST VINCOLI PAPIROLOGICI E PREFISSO E/O SUFFISSO
# ==============================================================================


def test_extract_lacuna_metadata_inter_and_intra_word():
    # 1. Lacuna inter-parola
    meta_inter = extract_lacuna_metadata("ἐνταῦθα δὲ [...] τοὺς λόγους")
    assert meta_inter["min_chars"] == 3
    assert meta_inter["max_chars"] == 3
    assert meta_inter["word_prefix"] == ""
    assert meta_inter["word_suffix"] == ""
    assert meta_inter["is_partial_word"] is False

    # 2. Lacuna intra-parola finale (solo prefisso)
    meta_prefix = extract_lacuna_metadata("Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]")
    assert meta_prefix["min_chars"] == 4
    assert meta_prefix["max_chars"] == 4
    assert meta_prefix["word_prefix"] == "τειχεσιβ"
    assert meta_prefix["word_suffix"] == ""
    assert meta_prefix["is_partial_word"] is True

    # 3. Lacuna intra-parola iniziale (solo suffisso)
    meta_suffix = extract_lacuna_metadata("[....]νε τειχεσιβλῆτα")
    assert meta_suffix["min_chars"] == 4
    assert meta_suffix["word_prefix"] == ""
    assert meta_suffix["word_suffix"] == "νε"
    assert meta_suffix["is_partial_word"] is True

    # 4. Lacuna intra-parola centrale (coesistenza di prefisso E suffisso)
    meta_both = extract_lacuna_metadata("βροτολοιγὲ μιαιφόνε τει[....]βλῆτα")
    assert meta_both["min_chars"] == 4
    assert meta_both["max_chars"] == 4
    assert meta_both["word_prefix"] == "τει"
    assert meta_both["word_suffix"] == "βλῆτα"
    assert meta_both["is_partial_word"] is True

    constraints_str = format_gap_constraints(
        min_chars=meta_both["min_chars"],
        max_chars=meta_both["max_chars"],
        word_prefix=meta_both["word_prefix"],
        word_suffix=meta_both["word_suffix"],
    )
    assert "τει<supplemento>βλῆτα" in constraints_str


def test_extract_lacuna_metadata_multiple_lacunae():
    text = "ἐνταῦθα δὲ [...] τοὺς λόγους [....] καὶ τὰς πράξεις ...."
    meta = extract_lacuna_metadata(text)
    # Solo la prima lacuna deve rimanere [...], le altre diventano <UNK>
    assert meta["context_with_gap"].count("[...]") == 1
    assert "[....]" not in meta["context_with_gap"]
    assert "<UNK>" in meta["context_with_gap"]


def test_sanitize_greek_supplement_strips_redundant_affixes():
    # Rifiuta caratteri latini e numeri
    assert sanitize_greek_supplement("hello") == ""
    assert sanitize_greek_supplement("123") == ""

    # Rimuove automaticamente prefisso ripetuto dall'LLM
    assert (
        sanitize_greek_supplement(
            "τειχεσιβλῆτα",
            word_prefix="τειχεσιβ",
            word_suffix="",
            min_chars=4,
            max_chars=4,
        )
        == "λητα"
    )

    # Rimuove sia prefisso sia suffisso ripetuti dall'LLM
    assert (
        sanitize_greek_supplement(
            "τειχεσιβλῆτα",
            word_prefix="τει",
            word_suffix="βλῆτα",
            min_chars=4,
            max_chars=4,
        )
        == "χεσι"
    )


# ==============================================================================
# 2. TEST RETRIEVER IBRIDO (BM25 + DSPy Embeddings) E ANTI-LEAKAGE
# ==============================================================================


def test_hybrid_retriever_and_persistence(tmp_path: Path):
    sample_blocks = [
        {
            "corpus_id": "DCLP",
            "file_id": "175276",
            "title": "P. Bagnall. 13",
            "language": "grc",
            "training_text": "Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα",
        },
        {
            "corpus_id": "tlg",
            "file_id": "tlg0007081.xml",
            "title": "Apophthegmata",
            "language": "grc",
            "training_text": "Ἀρτοξέρξης ὁ Περσῶν βασιλεύς οὐχ ἧττον οἰόμενος βασιλικὸν καὶ φιλάνθρωπον εἶναι",
        },
        {
            "corpus_id": "tlg",
            "file_id": "tlg0007082.xml",
            "title": "Moralia",
            "language": "grc",
            "training_text": "ἐνταῦθα δὲ καὶ τοὺς λόγους αὐτοὺς καθ' αὑτοὺς ὥσπερ δείγματα τῶν βίων",
        },
    ]

    corpus = build_rag_corpus_from_data(blocks_override=sample_blocks, min_chunk_words=3)
    assert len(corpus) == 3

    retriever = GreekHybridRetriever(
        corpus=corpus,
        embedder=_dummy_greek_embedder,
        k=2,
    )

    pred = retriever(
        query="Ἆρες βροτολοιγὲ τειχεσιβ[....]",
        word_prefix="τειχεσιβ",
        k=2,
    )
    assert len(pred.passages) == 2
    assert "τειχεσιβλῆτα" in pred.passages[0]

    # Verifica salvataggio e caricamento da disco
    save_dir = tmp_path / "rag_idx"
    retriever.save(save_dir)
    loaded = GreekHybridRetriever.from_saved(save_dir, embedder=_dummy_greek_embedder)
    pred_loaded = loaded("Ἀρτοξέρξης ὁ Περσῶν [...]", k=1)
    assert len(pred_loaded.passages) == 1
    assert "Ἀρτοξέρξης" in pred_loaded.passages[0]

    # Verifica filtro anti-data-leakage quando la query coincide quasi interamente col passaggio
    leak_pred = loaded(
        "Ἀρτοξέρξης ὁ Περσῶν βασιλεύς οὐχ ἧττον οἰόμενος [...] καὶ φιλάνθρωπον εἶναι",
        k=2,
        exclude_query_overlap=True,
    )
    for passage in leak_pred.passages:
        assert "Ἀρτοξέρξης" not in passage


# ==============================================================================
# 3. TEST MODULO DSPY, SELF-REFINEMENT E FILL_MASK_RAG
# ==============================================================================


def test_greek_rag_infiller_and_fill_mask_rag_with_dummylm():
    corpus = [
        "[Corpus: DCLP | P. Bagnall. 13] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα",
        "[Corpus: tlg | Apophthegmata] ἐνταῦθα δὲ καὶ τοὺς λόγους αὐτοὺς καθ' αὑτοὺς",
    ]
    retriever = GreekHybridRetriever(corpus=corpus, embedder=_dummy_greek_embedder, k=2)

    dummy_lm = DummyLM(
        [
            {
                "philological_reasoning": "Il verso omerico attesta l'epiteto di Ares τειχεσιβλῆτα (4 lettere mancanti: λητα).",
                "candidates": [
                    {"supplement": "λῆτα", "confidence": 0.95},
                    {"supplement": "πλῆτα", "confidence": 0.60},
                ],
            }
        ]
    )

    preds, details = fill_mask_rag(
        text="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        lm=dummy_lm,
        retriever=retriever,
        K=5,
        normalize_probs=True,
        return_details=True,
    )

    assert len(preds) >= 1
    assert preds[0][0] == "λητα"
    assert math.isclose(sum(score for _, score in preds), 1.0, rel_tol=1e-5)
    assert details["word_prefix"] == "τειχεσιβ"
    assert len(details["retrieved_passages"]) == 2


def test_constraint_refinement_triggered_when_initial_length_invalid():
    """
    Verifica che se la prima generazione Chain-of-Thought restituisce solo candidati
    che violano la lunghezza della lacuna (es. 6 lettere invece di 3), il modulo
    attiva automaticamente `refine_infill` (`ConstraintRefinementSignature`).
    """
    dummy_lm = DummyLM(
        [
            # 1a chiamata (generate_infill): restituisce candidato troppo lungo (6 lettere invece di 3)
            {
                "philological_reasoning": "Tentativo iniziale troppo lungo.",
                "candidates": [{"supplement": "λόγους", "confidence": 0.8}],
            },
            # 2a chiamata (refine_infill): corregge a 3 lettere ('καὶ')
            {
                "candidates": [{"supplement": "καὶ", "confidence": 0.92}],
            },
        ]
    )

    infiller = GreekRAGInfiller(retriever=None, enable_refinement=True)
    with dspy.context(lm=dummy_lm):
        pred = infiller(
            context_with_gap="ἐνταῦθα δὲ [...] τοὺς λόγους",
            min_chars=3,
            max_chars=3,
            num_candidates=5,
        )

    valid_supps = [c.supplement for c in pred.candidates]
    assert "καὶ" in valid_supps


# ==============================================================================
# 4. TEST COMPILAZIONE DSPY (BOOTSTRAPFEWSHOT) E VALUTAZIONE
# ==============================================================================


def test_dspy_optimizer_bootstrap_and_evaluation(tmp_path: Path):
    dev_cases = [
        DevCase(
            x="ἐνταῦθα δὲ [...] τοὺς λόγους",
            y=["καὶ"],
            gap_length=3,
            corpus_id="tlg",
            file_id="tlg0007081.xml",
            gap_type="word",
        ),
        DevCase(
            x="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
            y=["λητα"],
            gap_length=4,
            corpus_id="DCLP",
            file_id="175276",
            gap_type="suffix",
        ),
    ]

    examples = dev_cases_to_dspy_examples(dev_cases)
    assert len(examples) == 2

    # DummyLM con risposte valide per il bootstrap e per la successiva valutazione
    scripted_answers = [
        {
            "philological_reasoning": "Connettivo καί di 3 lettere.",
            "candidates": [{"supplement": "καὶ", "confidence": 0.95}],
        },
        {
            "philological_reasoning": "Epiteto omerico τειχεσιβλῆτα.",
            "candidates": [{"supplement": "λῆτα", "confidence": 0.95}],
        },
        {
            "philological_reasoning": "Connettivo καί di 3 lettere.",
            "candidates": [{"supplement": "καὶ", "confidence": 0.95}],
        },
        {
            "philological_reasoning": "Epiteto omerico τειχεσιβλῆτα.",
            "candidates": [{"supplement": "λῆτα", "confidence": 0.95}],
        },
    ]
    dummy_lm = DummyLM(scripted_answers)

    compiled_json = tmp_path / "compiled_rag_test.json"
    compiled_prog = compile_greek_rag_infiller(
        trainset=examples,
        lm=dummy_lm,
        retriever=None,
        optimizer_type="bootstrap",
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        output_path=compiled_json,
    )
    assert compiled_json.exists()
    assert isinstance(compiled_prog, GreekRAGInfiller)

    # Valutazione stratificata
    eval_lm = DummyLM(
        [
            {
                "philological_reasoning": "Connettivo καί.",
                "candidates": [{"supplement": "καὶ", "confidence": 0.95}],
            },
            {
                "philological_reasoning": "Epiteto omerico.",
                "candidates": [{"supplement": "λῆτα", "confidence": 0.95}],
            },
        ]
    )
    metrics = evaluate_rag_on_cases(
        cases=dev_cases,
        lm=eval_lm,
        retriever=None,
        top_k_passages=0,
        compiled_program_path=compiled_json,
    )
    assert "overall" in metrics
    assert metrics["overall"]["top1"] == 100.0
    assert metrics["overall"]["cer"] == 0.0
    md = format_rag_evaluation_markdown(metrics, model_name="Qwen/Qwen2.5-7B-Instruct")
    assert "Qwen/Qwen2.5-7B-Instruct" in md


# ==============================================================================
# 5. TEST INTEGRAZIONE FASTAPI (MODELSERVICE & SUGGESTIONSSERVICE)
# ==============================================================================


@pytest.mark.asyncio
async def test_fastapi_rag_model_creation_and_prediction():
    mock_collection = MagicMock()
    mock_fs = MagicMock()

    inserted_id = ObjectId()
    mock_collection.find_one = AsyncMock(return_value=None)
    mock_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=inserted_id)
    )

    model_service = ModelService(db_collection=mock_collection, gridfs=mock_fs)
    rag_model = RAGModel(
        CHECKPOINT="Qwen/Qwen2.5-7B-Instruct",
        BACKEND="auto",
        TOP_K_PASSAGES=5,
        TYPE="RAG",
    )
    created_id = await model_service.create_model(rag_model)
    assert created_id == str(inserted_id)

    # Test SuggestionsService per un modello di tipo RAG
    stored_doc = {
        "_id": inserted_id,
        "CHECKPOINT": "Qwen/Qwen2.5-7B-Instruct",
        "BACKEND": "auto",
        "TOP_K_PASSAGES": 5,
        "COMPILED_PROGRAM_PATH": None,
        "TYPE": ModelType.RAG,
    }
    mock_collection.find_one = AsyncMock(return_value=stored_doc)
    sugg_service = SuggestionsService(db_collection=mock_collection, gridfs=mock_fs)

    with patch(
        "models.rag.inference.fill_mask_rag",
        return_value=[("και", 0.8), ("ουν", 0.2)],
    ):
        preds = await sugg_service.get_predictions(
            model_id=str(inserted_id),
            context="ἐνταῦθα δὲ [...] τοὺς λόγους",
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
        )

    assert len(preds) == 2
    assert preds[0]["token_str"] == "και"
    assert preds[0]["score"] == 0.8
    assert "και" in preds[0]["sentence"]

    # Verifica che _predict_rag alleghi philological_reasoning e retrieved_passages quando restituiti da fill_mask_rag
    with patch(
        "models.rag.inference.fill_mask_rag",
        return_value=(
            [("λητα", 0.9), ("πλητα", 0.1)],
            {
                "philological_reasoning": "Epiteto omerico di Ares τειχεσιβλῆτα.",
                "retrieved_passages": ["[Corpus: tlg | Ilias] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα"],
            },
        ),
    ):
        rich_preds = await sugg_service.get_predictions(
            model_id=str(inserted_id),
            context="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
        )

    assert len(rich_preds) == 2
    assert rich_preds[0]["token_str"] == "λητα"
    assert rich_preds[0]["philological_reasoning"] == "Epiteto omerico di Ares τειχεσιβλῆτα."
    assert len(rich_preds[0]["retrieved_passages"]) == 1
    assert "philological_reasoning" not in rich_preds[1]


def test_build_dspy_lm_ollama_tag_and_compact_detection():
    assert _is_compact_hf_model("Qwen/Qwen2.5-1.5B-Instruct") is True
    assert _is_compact_hf_model("Qwen/Qwen2.5-3B-Instruct") is True
    assert _is_compact_hf_model("Qwen/Qwen2.5-7B-Instruct") is False

    lm = build_dspy_lm(checkpoint="gemma4:e4b", backend="auto")
    assert isinstance(lm, dspy.LM)
    assert "gemma4:e4b" in lm.model


def test_rag_neuro_lexical_fallback_on_hf_permission_error():
    """
    Verifica che quando l'LLM remoto restituisce l'errore di permessi HF_TOKEN
    ('This authentication method does not have sufficient permissions to call Inference Providers'),
    `fill_mask_rag` attivi il circuit-breaker e degradi automaticamente al motore neuro-lessicale
    locale restituendo candidati validi, `philological_reasoning` e `retrieved_passages` senza 502.
    """
    reset_hf_api_availability()
    assert is_hf_api_unavailable() is False

    corpus = [
        "[Corpus: DCLP | Philodemus] τὸ μὲν εὐπαρακολούθητα ποιεῖν τοῖς ἀκούουσιν",
        "[Corpus: tlg | Moralia] ἐνταῦθα δὲ καὶ τοὺς λόγους αὐτοὺς ποιεῖν",
    ]
    retriever = GreekHybridRetriever(corpus=corpus, embedder=_dummy_greek_embedder, k=2)

    # Verifica `match_affix_words` sull'indice BM25 per `π[.]ιειν`
    affix_matches = retriever.lexical_index.match_affix_words(
        word_prefix="π",
        word_suffix="ιειν",
        min_gap_chars=1,
        max_gap_chars=1,
    )
    assert any(supp == "ο" and word == "ποιειν" for word, supp, _df in affix_matches)

    hf_err = RuntimeError(
        '[Qwen/Qwen2.5-7B-Instruct] litellm.APIError: HuggingfaceException - '
        '{"error":"This authentication method does not have sufficient permissions to call Inference Providers on behalf of user CNR-ILC"}'
    )
    assert is_hf_permission_or_auth_error(hf_err) is True

    with (
        patch.object(GreekRAGInfiller, "forward", side_effect=hf_err),
        patch("models.rag.inference.get_fallback_dspy_lms", return_value=[]),
        patch("models.rag.inference._get_cached_mlm", side_effect=RuntimeError("skip heavy MLM in unit test")),
    ):
        preds, details = fill_mask_rag(
            text="μὲν εὐπαρακολούθητα π[.]ιειν",
            checkpoint="Qwen/Qwen2.5-7B-Instruct",
            backend="auto",
            retriever=retriever,
            K=5,
            return_details=True,
            allow_local_fallback=True,
        )

    assert is_hf_api_unavailable() is True
    assert len(preds) >= 1
    assert preds[0][0] == "ο"
    assert "neuro-lessicale locale" in details["philological_reasoning"]
    assert len(details["retrieved_passages"]) == 2
    reset_hf_api_availability()


# ==============================================================================
# 6. TEST HUMAN-IN-THE-LOOP (FASE 1: FEEDBACK LOG & FASE 2: IN-SESSION REFINEMENT)
# ==============================================================================


def test_hitl_in_session_refinement_filters_rejected_candidates_and_injects_feedback():
    """
    Verifica che `format_gap_constraints` e `fill_mask_rag` (Fase 2 HITL):
    1. Includano le indicazioni del filologo (`human_feedback`) e i candidati scartati
       in `gap_constraints`.
    2. Filtrino deterministicamente qualsiasi candidato coincidente (anche a meno di
       accenti/spiriti) con `rejected_candidates`.
    """
    constraints_str = format_gap_constraints(
        min_chars=4,
        max_chars=4,
        word_prefix="τειχεσιβ",
        human_feedback="Preferisci epiteto omerico epicizzante",
        rejected_candidates=["πλῆτα"],
    )
    assert "Preferisci epiteto omerico epicizzante" in constraints_str
    assert "πλῆτα" in constraints_str

    dummy_lm = DummyLM(
        [
            {
                "philological_reasoning": "Escludendo πλῆτα su indicazione del filologo, l'epiteto omerico è τειχεσιβλῆτα.",
                "candidates": [
                    {"supplement": "πλῆτα", "confidence": 0.99},
                    {"supplement": "λῆτα", "confidence": 0.90},
                ],
            }
        ]
    )

    preds, details = fill_mask_rag(
        text="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        lm=dummy_lm,
        retriever=None,
        top_k_passages=0,
        K=5,
        return_details=True,
        human_feedback="Preferisci epiteto omerico epicizzante",
        rejected_candidates=["πλητα"],
    )

    returned_tokens = [tok for tok, _score in preds]
    assert "πλητα" not in returned_tokens
    assert "λητα" in returned_tokens
    assert "Preferisci epiteto omerico epicizzante" in details["gap_constraints"]


@pytest.mark.asyncio
async def test_hitl_feedback_persistence_and_service_filtering():
    """
    Verifica che `SuggestionsService.save_feedback` (Fase 1 HITL) salvi il documento
    su MongoDB e che `SuggestionsService.get_predictions` (Fase 2 HITL) escluda i
    `rejected_candidates` e inoltri `human_feedback` al predittore RAG.
    """
    from backend.api.models import FeedbackCreateRequest

    mock_collection = MagicMock()
    mock_fs = MagicMock()
    mock_feedback_collection = MagicMock()

    feedback_id = ObjectId()
    mock_feedback_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id=feedback_id)
    )

    sugg_service = SuggestionsService(
        db_collection=mock_collection,
        gridfs=mock_fs,
        feedback_db=mock_feedback_collection,
    )

    fb_req = FeedbackCreateRequest(
        model_id="67f000000000000000000001",
        context="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        token_str="λητα",
        sentence="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλητα",
        is_accepted=True,
        notes="Epiteto omerico attestato in Il. 5.31",
    )
    saved = await sugg_service.save_feedback(fb_req)
    assert saved["id"] == str(feedback_id)
    assert saved["status"] == "saved"
    mock_feedback_collection.insert_one.assert_awaited_once()
    inserted_doc = mock_feedback_collection.insert_one.call_args[0][0]
    assert inserted_doc["token_str"] == "λητα"
    assert inserted_doc["is_accepted"] is True
    assert "created_at" in inserted_doc

    # Verifica che get_predictions filtri rejected_candidates e inoltri human_feedback
    model_id = ObjectId()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": model_id,
            "CHECKPOINT": "Qwen/Qwen2.5-7B-Instruct",
            "BACKEND": "auto",
            "TOP_K_PASSAGES": 5,
            "COMPILED_PROGRAM_PATH": None,
            "TYPE": ModelType.RAG,
        }
    )

    with patch(
        "models.rag.inference.fill_mask_rag",
        return_value=([("πλητα", 0.6), ("λητα", 0.4)], {}),
    ) as mocked_rag:
        filtered_preds = await sugg_service.get_predictions(
            model_id=str(model_id),
            context="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
            human_feedback="Solo forme omeriche",
            rejected_candidates=["πλῆτα"],
        )

    assert len(filtered_preds) == 1
    assert filtered_preds[0]["token_str"] == "λητα"
    assert mocked_rag.call_args.kwargs["human_feedback"] == "Solo forme omeriche"
    assert mocked_rag.call_args.kwargs["rejected_candidates"] == ["πλῆτα"]

