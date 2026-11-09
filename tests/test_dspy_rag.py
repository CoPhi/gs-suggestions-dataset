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
from models.rag.evaluation import (
    evaluate_rag_impact_on_test_abs,
    evaluate_rag_on_cases,
    format_rag_evaluation_markdown,
    format_rag_impact_markdown,
    load_cases_from_test_abs_csv,
)
from models.rag.inference import fill_mask_rag, is_bert_checkpoint
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
    parse_passage_metadata,
    resolve_candidate_attestations,
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
        score=0.985,
        rank=1,
        reconstructed_word="τειχεσιβλητα",
        tlg_attested=True,
        candidate_reasoning="Epiteto formulare dell'Iliade riferito al dio Ares",
        model_label="gs-GreBerta + RAG",
        model_type="BERT",
        checkpoint="CNR-ILC/gs-GreBerta",
        use_rag=True,
        top_k=5,
        alternative_candidates=[
            {"token_str": "βλητα", "score": 0.82, "rank": 2, "tlg_attested": False},
            {"token_str": "πλητα", "score": 0.45, "rank": 3, "tlg_attested": False},
        ],
    )
    saved = await sugg_service.save_feedback(fb_req)
    assert saved["id"] == str(feedback_id)
    assert saved["status"] == "saved"
    mock_feedback_collection.insert_one.assert_awaited_once()
    inserted_doc = mock_feedback_collection.insert_one.call_args[0][0]
    assert inserted_doc["token_str"] == "λητα"
    assert inserted_doc["is_accepted"] is True
    assert inserted_doc["score"] == 0.985
    assert inserted_doc["rank"] == 1
    assert inserted_doc["reconstructed_word"] == "τειχεσιβλητα"
    assert inserted_doc["tlg_attested"] is True
    assert inserted_doc["model_label"] == "gs-GreBerta + RAG"
    assert inserted_doc["model_type"] == "BERT"
    assert inserted_doc["checkpoint"] == "CNR-ILC/gs-GreBerta"
    assert inserted_doc["use_rag"] is True
    assert len(inserted_doc["alternative_candidates"]) == 2
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


# ==============================================================================
# 7. TEST ATTESTAZIONI NEL CORPUS (COT CONSULTABILE) E VALUTAZIONE TEST_ABS.CSV
# ==============================================================================


def test_corpus_attestations_and_rag_impact_on_test_abs():
    """
    Verifica che:
    1. `parse_passage_metadata` e `resolve_candidate_attestations` identifichino
       esattamente dove un suggerimento del retriever è attestato nella collezione
       di documenti (corpus_id, source_id, attested_form, passage, doc_freq).
    2. `fill_mask_rag(..., return_details=True)` restituisca `candidate_attestations`
       e arricchisca `philological_reasoning` (CoT) con la citazione del corpus.
    3. `load_cases_from_test_abs_csv` ed `evaluate_rag_impact_on_test_abs` valutino
       `test_abs.csv` confrontando No-RAG vs With-RAG e generando il report Markdown.
    """
    meta = parse_passage_metadata("[Corpus: DCLP | P. Bagnall. 13] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα")
    assert meta["corpus_id"] == "DCLP"
    assert meta["source_id"] == "P. Bagnall. 13"
    assert "τειχεσιβλῆτα" in meta["passage"]

    corpus = [
        "[Corpus: DCLP | P. Bagnall. 13] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα",
        "[Corpus: PHerc. 1004 | col. 64] οὐ γὰρ ἦν φανερώτερα τὰ κατ' αὐτὰ λείπων ὑπὲρ ὧν ἐποίει τοὺς λόγους",
    ]
    retriever = GreekHybridRetriever(corpus=corpus, embedder=_dummy_greek_embedder, k=2)

    atts = retriever.find_attestations(
        supplement="λητα",
        word_prefix="τειχεσιβ",
        retrieved_passages=[corpus[0]],
    )
    assert len(atts) == 1
    assert atts[0]["corpus_id"] == "DCLP"
    assert atts[0]["source_id"] == "P. Bagnall. 13"
    assert atts[0]["attested_form"] == "τειχεσιβλῆτα"
    assert "**τειχεσιβλῆτα**" in atts[0]["highlighted_passage"]
    assert atts[0]["doc_freq"] >= 1

    dummy_lm = DummyLM(
        [
            {
                "philological_reasoning": "Analisi contestuale del participio presente dai documenti recuperati.",
                "candidates": [
                    {
                        "supplement": "ι",
                        "confidence": 0.90,
                        "rationale": "Integra il participio presente attivo λείπων coerente col verbo ἐποίει.",
                    }
                ],
            }
        ]
    )
    preds, details = fill_mask_rag(
        text="τὰ κατ' αὐτὰ λε[.]πων ὑπὲρ",
        lm=dummy_lm,
        retriever=retriever,
        top_k_passages=1,
        K=5,
        return_details=True,
    )
    assert preds[0][0] == "ι"
    assert "candidate_attestations" in details
    assert "ι" in details["candidate_attestations"]
    assert details["candidate_attestations"]["ι"][0]["corpus_id"] == "PHerc. 1004"
    assert "**λείπων**" in details["candidate_attestations"]["ι"][0]["highlighted_passage"]
    assert "PHerc. 1004 | col. 64" in details["philological_reasoning"]
    assert "**λείπων**" in details["philological_reasoning"]
    # Verifica CoT di 2° livello (rationale specifico per suggerimento + attestazione evidenziata)
    assert "candidate_rationales" in details
    assert "ι" in details["candidate_rationales"]
    assert "participio presente attivo" in details["candidate_rationales"]["ι"]
    assert "**λείπων**" in details["candidate_rationales"]["ι"]
    # Verifica restituzione sia dei top_k passaggi primari sia di tutti i loci similes trovati
    assert len(details["retrieved_passages"]) == 1
    assert len(details["all_retrieved_passages"]) == 2

    # Verifica il caricamento da test_abs.csv e il confronto No-RAG vs With-RAG
    csv_cases = load_cases_from_test_abs_csv("test_abs.csv", max_cases=3)
    assert len(csv_cases) == 3

    # Test di evaluate_rag_impact_on_test_abs su un caso controllato dove RAG recupera l'attestazione
    test_case = [
        DevCase(
            x="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
            y=["λητα"],
            gap_length=4,
            corpus_id="PHerc. 1004",
            file_id="R 1 | col. 64",
            gap_type="suffix",
        )
    ]
    ablation_lm = DummyLM(
        [
            # 1a chiamata (No-RAG): l'LLM senza documenti sbaglia supplemento
            {
                "philological_reasoning": "Ipotesi generica senza contesto documentale.",
                "candidates": [{"supplement": "ωτα", "confidence": 0.70}],
            },
            {
                "candidates": [{"supplement": "ωταc", "confidence": 0.70}],
            },
            # 2a chiamata (With-RAG): con il documento recuperato propone il supplemento corretto
            {
                "philological_reasoning": "Attestato in P. Bagnall. 13 come τειχεσιβλῆτα.",
                "candidates": [{"supplement": "λῆτα", "confidence": 0.95}],
            },
        ]
    )
    impact = evaluate_rag_impact_on_test_abs(
        cases=test_case,
        lm=ablation_lm,
        retriever=retriever,
        top_k_passages=2,
    )
    assert impact["no_rag"]["overall"]["top1"] == 0.0
    assert impact["with_rag"]["overall"]["top1"] == 100.0
    assert impact["delta"]["overall"]["top1"] == 100.0
    assert impact["corpus_attestation_analysis"]["gold_attested_count"] == 1
    md_report = format_rag_impact_markdown(impact, model_name="Qwen/Qwen2.5-7B-Instruct")
    assert "With-RAG" in md_report
    assert "DCLP | P. Bagnall. 13" in md_report


# ==============================================================================
# 8. TEST POST-PROCESSING ATTESTAZIONE LESSICALE TLG (BOUNDARY RECONSTRUCTION)
# ==============================================================================


@pytest.mark.asyncio
async def test_tlg_boundary_reconstruction_and_postprocessing():
    """
    Verifica il post-processing di attestazione lessicale sul TLG (`tlg.wl.gz`):
    1. `get_normalized_vocab` carica il lessico pre-normalizzato uniformando `ς` -> `σ`
       ed escludendo consonanti isolate da elisione (`δ`, `τ`).
    2. `check_tlg_attestation` e `postprocess_tlg_attestations` ricostruiscono la forma
       con prefisso e/o suffisso senza alterare score o ranking dei suggerimenti.
    3. `SuggestionsService.get_predictions` valorizza `tlg_attested` e `reconstructed_word`
       sia per i modelli MLM (BERT) dopo HCB Beam Search sia per DSPy RAG.
    """
    from models.bert.inference.vocab import (
        check_tlg_attestation,
        get_normalized_vocab,
        postprocess_tlg_attestations,
    )

    vocab = get_normalized_vocab()
    assert len(vocab) > 500_000
    # Verifica normalizzazione sigma finale (ς -> σ) e presenza di forme omeriche/classiche del TLG
    assert "τειχεσιβλητα" in vocab
    assert "λογοσ" in vocab
    assert "ανθρωπων" in vocab
    # Verifica che consonanti singole spurie da elisione (δ', τ') non siano attestate come parole intere
    assert "δ" not in vocab
    assert "τ" not in vocab
    assert "ο" in vocab

    # 1. Boundary reconstruction con prefisso ("τειχεσιβ" + "λητα" -> "τειχεσιβλητα")
    attested_pref, recon_pref = check_tlg_attestation(
        supplement="λῆτα",
        context_with_gap="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        vocab=vocab,
    )
    assert attested_pref is True
    assert recon_pref == "τειχεσιβλῆτα"

    # 2. Boundary reconstruction con prefisso E suffisso ("τει" + "χεσι" + "βλῆτα" -> "τειχεσιβλῆτα")
    attested_both, recon_both = check_tlg_attestation(
        supplement="χεσι",
        context_with_gap="βροτολοιγὲ μιαιφόνε τει[....]βλῆτα",
        vocab=vocab,
    )
    assert attested_both is True
    assert recon_both == "τειχεσιβλῆτα"

    # 3. Forma inesistente nel TLG ("τειχεσιβ" + "ζζζζ" -> False)
    attested_fake, recon_fake = check_tlg_attestation(
        supplement="ζζζζ",
        context_with_gap="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        vocab=vocab,
    )
    assert attested_fake is False
    assert recon_fake == "τειχεσιβζζζζ"

    # 4. Invarianza di ranking e score durante postprocess_tlg_attestations
    raw_cands = [("ζζζζ", 0.75), ("λητα", 0.25)]
    annotated = postprocess_tlg_attestations(
        candidates=raw_cands,
        context_with_gap="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        vocab=vocab,
    )
    assert [a["token_str"] for a in annotated] == ["ζζζζ", "λητα"]
    assert [a["score"] for a in annotated] == [0.75, 0.25]
    assert annotated[0]["tlg_attested"] is False
    assert annotated[1]["tlg_attested"] is True
    assert annotated[1]["reconstructed_word"] == "τειχεσιβλητα"

    # 5. Integrazione in SuggestionsService per modelli BERT e RAG
    mock_collection = MagicMock()
    sugg_service = SuggestionsService(
        db_collection=mock_collection,
        gridfs=MagicMock(),
        feedback_db=MagicMock(),
    )
    bert_id = ObjectId()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": bert_id,
            "CHECKPOINT": "CNR-ILC/gs-GreBerta",
            "TYPE": ModelType.BERT,
        }
    )
    sugg_service._bert_cache["CNR-ILC/gs-GreBerta"] = (MagicMock(), MagicMock())
    with patch(
        "backend.api.services.suggestions_service.fill_mask",
        return_value=[("ζζζζ", 0.70), ("λητα", 0.30)],
    ):
        bert_preds = await sugg_service.get_predictions(
            model_id=str(bert_id),
            context="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
        )
    assert len(bert_preds) == 2
    assert bert_preds[0]["token_str"] == "ζζζζ"
    assert bert_preds[0]["tlg_attested"] is False
    assert bert_preds[1]["token_str"] == "λητα"
    assert bert_preds[1]["tlg_attested"] is True
    assert bert_preds[1]["reconstructed_word"] == "τειχεσιβλητα"


@pytest.mark.asyncio
async def test_explicit_bert_neuro_lexical_rag_a_priori():
    """
    Verifica che il modello neuro-lessicale (BERT + RAG) possa essere scelto a priori
    (sia tramite checkpoint BERT o backend='neuro_lexical', sia tramite use_rag=True
    su un modello BERT) senza tentare la connessione a LLM esterni e senza etichette di fallback.
    """
    assert is_bert_checkpoint("CNR-ILC/gs-GreBerta") is True
    assert is_bert_checkpoint("CNR-ILC/gs-aristoBERTo") is True
    assert is_bert_checkpoint("CNR-ILC/gs-Logion") is True
    assert is_bert_checkpoint("Qwen/Qwen2.5-72B-Instruct") is False
    assert is_bert_checkpoint("meta-llama/Llama-3.3-70B-Instruct") is False

    corpus = [
        "[Corpus: tlg | Hom. Il. 5.31] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα",
    ]
    retriever = GreekHybridRetriever(corpus=corpus, embedder=_dummy_greek_embedder, k=1)

    with (
        patch("models.rag.inference.build_dspy_lm") as mock_build_lm,
        patch(
            "models.rag.inference._get_cached_mlm",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "models.rag.inference.fill_mask",
            return_value=[("λητα", -0.2), ("πολι", -1.4)],
        ),
    ):
        results, details = fill_mask_rag(
            text="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
            checkpoint="CNR-ILC/gs-GreBerta",
            backend="neuro_lexical",
            retriever=retriever,
            K=5,
            return_details=True,
        )
        # Non deve mai invocare build_dspy_lm quando il modello scelto è BERT / neuro_lexical
        mock_build_lm.assert_not_called()

    assert len(results) >= 1
    assert results[0][0] == "λητα"
    assert "neuro-lessicale locale" in details["philological_reasoning"]
    assert "fallback automatico" not in details["philological_reasoning"]

    # Verifica anche l'accostamento on-the-fly BERT + RAG tramite use_rag=True in SuggestionsService
    mock_collection = MagicMock()
    sugg_service = SuggestionsService(
        db_collection=mock_collection,
        gridfs=MagicMock(),
        feedback_db=MagicMock(),
    )
    bert_id = ObjectId()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": bert_id,
            "CHECKPOINT": "CNR-ILC/gs-aristoBERTo",
            "TYPE": ModelType.BERT,
        }
    )
    with patch(
        "models.rag.inference.fill_mask_rag",
        return_value=(
            [("λητα", 0.92)],
            {
                "philological_reasoning": "Sintesi filologica eseguita in modalità neuro-lessicale locale (Retrieval Ibrido BM25+gs-aristoBERTo + HCB Beam Search)",
                "retrieved_passages": ["[Corpus: tlg | Hom. Il. 5.31] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα"],
                "all_retrieved_passages": ["[Corpus: tlg | Hom. Il. 5.31] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα"],
                "candidate_attestations": {},
                "candidate_rationales": {"λητα": "Completa la forma τειχεσιβλητα."},
                "tlg_attestations": {"λητα": {"tlg_attested": True, "reconstructed_word": "τειχεσιβλητα"}},
            },
        ),
    ) as mock_rag:
        rag_bert_preds = await sugg_service.get_predictions(
            model_id=str(bert_id),
            context="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
            use_rag=True,
        )
        mock_rag.assert_called_once()
        assert mock_rag.call_args.kwargs["checkpoint"] == "CNR-ILC/gs-aristoBERTo"
        assert mock_rag.call_args.kwargs["backend"] == "neuro_lexical"

    assert len(rag_bert_preds) == 1
    assert rag_bert_preds[0]["token_str"] == "λητα"
    assert "gs-aristoBERTo" in rag_bert_preds[0]["philological_reasoning"]


@pytest.mark.asyncio
async def test_batch_parallel_predictions_with_barrier():
    """
    Verifica che `get_batch_predictions` esegua in parallelo le chiamate ai modelli consultati
    sincronizzandole tramite barriera (`asyncio.Barrier`) e restituendo i metadati del modello
    di provenienza per ciascuna suggestion box.
    """
    bert_id = ObjectId()
    rag_id = ObjectId()
    models_db = {
        bert_id: {
            "_id": bert_id,
            "CHECKPOINT": "CNR-ILC/gs-GreBerta",
            "TYPE": ModelType.BERT,
        },
        rag_id: {
            "_id": rag_id,
            "CHECKPOINT": "Qwen/Qwen2.5-72B-Instruct",
            "BACKEND": "auto",
            "TOP_K_PASSAGES": 5,
            "TYPE": ModelType.RAG,
        },
    }

    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(
        side_effect=lambda q: models_db.get(q.get("_id"))
    )
    sugg_service = SuggestionsService(
        db_collection=mock_collection,
        gridfs=MagicMock(),
        feedback_db=MagicMock(),
    )
    sugg_service._bert_cache["CNR-ILC/gs-GreBerta"] = (MagicMock(), MagicMock())

    with (
        patch(
            "backend.api.services.suggestions_service.fill_mask",
            return_value=[("λητα", 0.85)],
        ),
        patch(
            "models.rag.inference.fill_mask_rag",
            return_value=([("βλητα", 0.91)], {"philological_reasoning": "Analisi Qwen"}),
        ),
    ):
        batch_res = await sugg_service.get_batch_predictions(
            model_ids=[str(bert_id), str(rag_id)],
            context="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσι[....]",
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
            use_rag=False,
        )

    assert len(batch_res) == 2
    assert batch_res[0]["model_id"] == str(bert_id)
    assert batch_res[0]["model_label"] == "gs-GreBerta"
    assert batch_res[0]["model_type"] == "BERT"
    assert batch_res[0]["predictions"][0]["token_str"] == "λητα"

    assert batch_res[1]["model_id"] == str(rag_id)
    assert batch_res[1]["model_label"] == "Qwen2.5-72B-Instruct"
    assert batch_res[1]["model_type"] == "RAG"
    assert batch_res[1]["predictions"][0]["token_str"] == "βλητα"


@pytest.mark.asyncio
async def test_sublinear_dot_validation_rejection():
    """
    Verifica che stringhe contenenti punti sublineari (U+0323) vengano rifiutate
    sollevando `InvalidContextError` con messaggio 'la stringa contiene carattere sconosciuti'.
    """
    from backend.api.exceptions import InvalidContextError

    sugg_service = SuggestionsService(
        db_collection=MagicMock(),
        gridfs=MagicMock(),
        feedback_db=MagicMock(),
    )
    # Contesto con punto sublineare U+0323 (es. α̣)
    bad_context = "τὰ κατ' α\u0323ὐτὰ [....] ὑπὲρ"
    with pytest.raises(InvalidContextError, match="la stringa contiene carattere sconosciuti"):
        await sugg_service.get_predictions(
            model_id=str(ObjectId()),
            context=bad_context,
            num_tokens=1,
            num_predictions=PredictionCount.FIVE,
        )






