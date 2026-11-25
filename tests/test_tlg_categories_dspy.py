"""
Test unitari e di integrazione per il supporto alle categorie testuali TLG (`<cla>`)
nel modulo DSPy RAG, nell'ottimizzatore e nelle route FastAPI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest
from bson import ObjectId
from dspy.utils.dummies import DummyLM
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.models import ModelType, PredictionCount
from backend.api.routes.predictions import get_service
from models.bert.dataset.dev_set import DevCase
from models.rag.categories import (
    DEFAULT_TEXT_CATEGORY_PROMPT,
    REASONING_CATEGORIES,
    TLG_CATEGORIES,
    TLG_CATEGORY_BY_CODE,
    extract_cla_categories_from_canon,
    format_text_category_for_prompt,
    get_reasoning_category_for_cla,
    get_reasoning_profile,
    list_tlg_categories,
    load_tlg_work_categories_from_canon,
    normalize_text_categories,
    normalize_text_category,
)
from models.rag.conversation_logging import (
    format_conversation_record,
    load_conversation_records,
)
from models.rag.inference import fill_mask_rag
from models.rag.modules import GreekRAGInfiller
from models.rag.optimizer import (
    build_mixed_optimization_cases,
    build_synthetic_tlg_category_cases,
    compile_greek_rag_infiller,
    dev_cases_to_dspy_examples,
    greek_infilling_dspy_metric,
    greek_infilling_gepa_metric,
    score_philological_reasoning,
)
from models.rag.signatures import (
    ConstraintRefinementSignature,
    GreekCandidate,
    GreekInfillingSignature,
)

EXTERNAL_CANON_XML = Path(
    "/home/gabriele/Documenti/projects/greekschools/datasets/tlg-bulk/tlg/canon.xml"
)


# ==============================================================================
# 1. TEST CATALOGO 75 CATEGORIE <cla> E NORMALIZZAZIONE
# ==============================================================================


def test_tlg_categories_catalog_completeness_and_canon_sync():
    assert len(TLG_CATEGORIES) == 75
    codes = [c.code for c in TLG_CATEGORIES]
    assert len(set(codes)) == 75
    for cat in TLG_CATEGORIES:
        assert cat.code == cat.code.strip()
        assert cat.label.strip()
        assert cat.description.strip()

    listed = list_tlg_categories()
    assert len(listed) == 75
    assert listed[0].keys() == {
        "code",
        "label",
        "description",
        "reasoning_category",
        "reasoning_focus",
    }
    for item in listed:
        assert item["reasoning_category"] in REASONING_CATEGORIES
        assert item["reasoning_focus"].strip()
        assert get_reasoning_category_for_cla(item["code"]) == item["reasoning_category"]

    # Verifica la sincronizzazione esatta con canon.xml quando presente su disco
    if EXTERNAL_CANON_XML.exists():
        extracted = extract_cla_categories_from_canon(EXTERNAL_CANON_XML)
        assert set(extracted) == set(codes)


def test_normalize_and_format_text_categories():
    # Normalizzazione tollerante di maiuscole, asterischi e punti finali
    assert normalize_text_category("Phil.") == "phil."
    assert normalize_text_category("*Hist.") == "hist."
    assert normalize_text_category("epic") == "epic."
    assert normalize_text_category("Narr. Fict") == "narr.fict."
    assert normalize_text_category("nat hist") == "nat.hist."

    # Supporto multi-valore con deduplicazione preservando l'ordine
    assert normalize_text_categories(["Phil.", "*rhet", "PHIL", "epic."]) == [
        "phil.",
        "rhet.",
        "epic.",
    ]
    assert normalize_text_categories("comic., iamb; *COMIC.") == ["comic.", "iamb."]

    # Valori assenti/neutri -> lista vuota e prompt neutro
    assert normalize_text_categories(None) == []
    assert normalize_text_categories([]) == []
    assert normalize_text_categories("") == []
    assert format_text_category_for_prompt(None) == DEFAULT_TEXT_CATEGORY_PROMPT
    assert format_text_category_for_prompt([]) == DEFAULT_TEXT_CATEGORY_PROMPT

    formatted = format_text_category_for_prompt(["phil.", "rhet."])
    assert "Filosofia (phil.)" in formatted
    assert "Retorica (rhet.)" in formatted
    assert "Categoria di ragionamento: Filosofia e argomentazione" in formatted

    epic_profile = get_reasoning_profile(["epic."])
    assert epic_profile["reasoning_categories"] == ["poesia_metrica"]
    assert "metr" in epic_profile["cues"]

    with pytest.raises(ValueError, match="non riconosciuta"):
        normalize_text_category("genere_inesistente")


def test_extract_and_map_from_synthetic_canon_xml(tmp_path: Path):
    canon_file = tmp_path / "canon.xml"
    canon_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<tlgcanon>
  <auth id="0012">
    <work id="0012 001"><cla value="epic.">Epic.</cla></work>
  </auth>
  <auth id="0007">
    <work id="0007 081">
      <cla value="phil.">*Phil.</cla>
      <cla value="rhet.">Rhet.</cla>
    </work>
  </auth>
</tlgcanon>
""",
        encoding="utf-8",
    )
    cats = extract_cla_categories_from_canon(canon_file)
    assert cats == ["epic.", "phil.", "rhet."]

    wmap = load_tlg_work_categories_from_canon(canon_file)
    assert wmap["0012 001"] == ["epic."]
    assert wmap["tlg0012001.xml"] == ["epic."]
    assert wmap["tlg0007081.xml"] == ["phil.", "rhet."]


# ==============================================================================
# 2. TEST SIGNATURE, MODULO DSPY, INFERENZA E CONVERSATION LOGGING
# ==============================================================================


def test_dspy_signatures_expose_text_category_input_field():
    assert "text_category" in GreekInfillingSignature.input_fields
    assert "text_category" in ConstraintRefinementSignature.input_fields


def test_greek_rag_infiller_passes_text_category_to_generate_and_refine():
    dummy_lm = DummyLM(
        [
            # 1a chiamata (generate_infill): candidato di lunghezza errata -> attiva refine_infill
            {
                "philological_reasoning": "Analisi filosofica iniziale.",
                "candidates": [{"supplement": "λόγους", "confidence": 0.7}],
            },
            # 2a chiamata (refine_infill): candidato di 3 lettere valido
            {
                "candidates": [{"supplement": "καὶ", "confidence": 0.9}],
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
            text_category=["phil.", "rhet."],
        )

    assert pred.text_categories == ["phil.", "rhet."]
    assert "Filosofia (phil.)" in pred.text_category
    assert "Retorica (rhet.)" in pred.text_category
    assert any(c.supplement == "καὶ" for c in pred.candidates)

    # Entrambe le chiamate DSPy (generate_infill e refine_infill) devono aver ricevuto la categoria
    assert len(dummy_lm.history) == 2
    for entry in dummy_lm.history:
        joined = json.dumps(entry["messages"], ensure_ascii=False)
        assert "Filosofia (phil.)" in joined
        assert "Retorica (rhet.)" in joined

    # Verifica il fallback neutro senza chiamate LLM extra quando omessa
    neutral_lm = DummyLM(
        [
            {
                "philological_reasoning": "Analisi neutra.",
                "candidates": [{"supplement": "καὶ", "confidence": 0.9}],
            }
        ]
    )
    with dspy.context(lm=neutral_lm):
        pred_neutral = infiller(
            context_with_gap="ἐνταῦθα δὲ [...] τοὺς λόγους",
            min_chars=3,
            max_chars=3,
            num_candidates=5,
        )
    assert pred_neutral.text_categories == []
    assert pred_neutral.text_category == DEFAULT_TEXT_CATEGORY_PROMPT
    assert len(neutral_lm.history) == 1
    assert DEFAULT_TEXT_CATEGORY_PROMPT in json.dumps(
        neutral_lm.history[0]["messages"], ensure_ascii=False
    )


def test_fill_mask_rag_propagates_and_logs_text_categories(tmp_path: Path, monkeypatch):
    log_file = tmp_path / "dspy_conv.jsonl"
    monkeypatch.setenv("RAG_CONVERSATION_LOG_ENABLED", "true")
    monkeypatch.setenv("RAG_CONVERSATION_LOG_FILE", str(log_file))

    dummy_lm = DummyLM(
        [
            {
                "philological_reasoning": "Epiteto epico omerico.",
                "candidates": [{"supplement": "λητα", "confidence": 0.95}],
            }
        ]
    )
    preds, details = fill_mask_rag(
        text="Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβ[....]",
        lm=dummy_lm,
        use_default_retriever=False,
        top_k_passages=0,
        K=3,
        return_details=True,
        text_categories=["*Epic"],
    )
    assert preds
    assert details["text_categories"] == ["epic."]
    assert "Epica (epic.)" in details["text_category"]

    records = load_conversation_records(log_file)
    assert len(records) == 1
    assert records[0]["input"]["text_categories"] == ["epic."]
    formatted = format_conversation_record(records[0])
    assert "CATEGORIA:" in formatted
    assert "Epica (epic.)" in formatted

    with pytest.raises(ValueError, match="non riconosciuta"):
        fill_mask_rag(
            text="ἐνταῦθα δὲ [...] τοὺς λόγους",
            lm=dummy_lm,
            use_default_retriever=False,
            top_k_passages=0,
            text_categories=["non_valido"],
        )


# ==============================================================================
# 3. TEST OPTIMIZER MISTO STRATIFICATO (P.HERC. + TLG CANON.XML)
# ==============================================================================


def test_optimizer_mixed_trainset_and_bootstrap_compile(tmp_path: Path):
    canon_file = tmp_path / "canon.xml"
    canon_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<tlgcanon>
  <auth id="0012">
    <work id="0012 001"><cla value="epic.">Epic.</cla></work>
  </auth>
  <auth id="0006">
    <work id="0006 001"><cla value="trag.">Trag.</cla></work>
  </auth>
</tlgcanon>
""",
        encoding="utf-8",
    )
    tlg_blocks = [
        {
            "corpus_id": "tlg",
            "file_id": "tlg0012001.xml",
            "language": "grc",
            "training_text": "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος οὐλομένην ἣ μυρί Ἀχαιοῖς ἄλγεα ἔθηκεν",
        },
        {
            "corpus_id": "tlg",
            "file_id": "tlg0006001.xml",
            "language": "grc",
            "training_text": "εἴθε ὤφελε Ἀργοῦς μὴ διαπτάσθαι σκάφος Κόλχων ἐς αἶαν κυανέας Συμπληγάδας",
        },
    ]
    herc_cases = [
        DevCase(
            x="ἐνταῦθα δὲ [...] τοὺς λόγους",
            y=["καὶ"],
            gap_length=3,
            corpus_id="PHerc.",
            file_id="1004",
            gap_type="word",
        ),
        DevCase(
            x="περὶ δὲ τῶν [...] φιλοσόφων",
            y=["ἀρχαίων"],
            gap_length=7,
            corpus_id="PHerc.",
            file_id="1004",
            gap_type="word",
        ),
    ]

    synthetic_tlg = build_synthetic_tlg_category_cases(
        max_cases=2,
        canon_xml_path=canon_file,
        blocks_override=tlg_blocks,
        seed=7,
    )
    assert len(synthetic_tlg) == 2
    assert {c.text_categories[0] for c in synthetic_tlg if c.text_categories} == {
        "epic.",
        "trag.",
    }

    mixed = build_mixed_optimization_cases(
        herc_cases=herc_cases,
        max_total_cases=4,
        tlg_ratio=0.5,
        herc_default_categories=("phil.",),
        neutral_ratio=0.25,
        canon_xml_path=canon_file,
        tlg_blocks_override=tlg_blocks,
        seed=7,
    )
    assert len(mixed) == 4
    # Almeno 1 caso neutro esplicito (`text_categories == []`)
    assert sum(1 for c in mixed if c.text_categories == []) == 1

    examples = dev_cases_to_dspy_examples(mixed)
    assert len(examples) == 4
    for ex in examples:
        assert "text_category" in ex.inputs()

    # Verifica la compilazione end-to-end con DummyLM e salvataggio/caricamento JSON
    simple_examples = dev_cases_to_dspy_examples(
        [
            DevCase(
                x="ἐνταῦθα δὲ [...] τοὺς λόγους",
                y=["καὶ"],
                gap_length=3,
                corpus_id="PHerc.",
                file_id="1004",
                text_categories=["phil."],
            ),
            DevCase(
                x="μῆνιν ἄειδε [...] Πηληϊάδεω Ἀχιλῆος",
                y=["θεὰ"],
                gap_length=3,
                corpus_id="tlg",
                file_id="tlg0012001.xml",
                text_categories=["epic."],
            ),
        ]
    )
    dummy_lm = DummyLM(
        [
            {
                "philological_reasoning": "Connettivo καί in prosa filosofica.",
                "candidates": [{"supplement": "καὶ", "confidence": 0.95}],
            },
            {
                "philological_reasoning": "Invocazione epica alla dea.",
                "candidates": [{"supplement": "θεα", "confidence": 0.95}],
            },
        ]
    )
    out_json = tmp_path / "compiled_with_categories.json"
    compiled = compile_greek_rag_infiller(
        trainset=simple_examples,
        lm=dummy_lm,
        retriever=None,
        optimizer_type="bootstrap",
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        output_path=out_json,
    )
    assert out_json.exists()
    assert compiled.default_exclude_query_overlap is True


def test_reasoning_aware_metrics_and_zero_shot_instruction_optimizers(tmp_path: Path):
    example_epic = dspy.Example(
        context_with_gap="μῆνιν ἄειδε [...] Πηληϊάδεω Ἀχιλῆος",
        text_category="epic.",
        min_chars=3,
        max_chars=3,
        gold_supplement="θεα",
    ).with_inputs("context_with_gap", "text_category", "min_chars", "max_chars")

    aligned_pred = dspy.Prediction(
        philological_reasoning=(
            "Il contesto epico in esametri omerici richiede l'invocazione alla dea; "
            "per vincoli metrici, morfologia e sintassi del verso l'accusativo o vocativo "
            "di 3 lettere dai loci similes è θεά."
        ),
        candidates=[
            GreekCandidate(
                supplement="θεά",
                confidence=0.95,
                rationale="Vocativo coerente con l'imperativo epico ἄειδε e il metro esametrico.",
            )
        ],
    )
    weak_pred = dspy.Prediction(
        philological_reasoning="Parola corta.",
        candidates=[GreekCandidate(supplement="θεά", confidence=0.95, rationale="")],
    )

    s_aligned, notes_aligned = score_philological_reasoning(
        aligned_pred.philological_reasoning, ["epic."], aligned_pred.candidates
    )
    s_weak, notes_weak = score_philological_reasoning(
        weak_pred.philological_reasoning, ["epic."], weak_pred.candidates
    )
    assert s_aligned > s_weak
    assert not notes_aligned
    assert any("Poesia e metrica" in n for n in notes_weak)

    assert greek_infilling_dspy_metric(example_epic, aligned_pred) > greek_infilling_dspy_metric(
        example_epic, weak_pred
    )

    gepa_fb = greek_infilling_gepa_metric(example_epic, weak_pred)
    assert isinstance(gepa_fb, dspy.Prediction)
    assert 0.0 < float(gepa_fb.score) < 1.0
    assert "Poesia e metrica" in gepa_fb.feedback

    # Verifica che il default di compile_greek_rag_infiller sia MIPROv2 zero-shot (0 demos)
    dummy_lm = DummyLM([{"philological_reasoning": "ok", "candidates": []}])
    fake_compiled = GreekRAGInfiller(retriever=None, default_exclude_query_overlap=True)
    for p in fake_compiled.predictors():
        p.demos = ["dummy_demo_to_clear"]

    with patch("dspy.MIPROv2") as mock_mipro_cls:
        mock_mipro = MagicMock()
        mock_mipro.compile.return_value = fake_compiled
        mock_mipro_cls.return_value = mock_mipro

        out_mipro = tmp_path / "compiled_mipro_zeroshot.json"
        res_mipro = compile_greek_rag_infiller(
            trainset=[example_epic],
            lm=dummy_lm,
            output_path=out_mipro,
        )
        assert out_mipro.exists()
        assert mock_mipro_cls.call_args.kwargs["max_bootstrapped_demos"] == 0
        assert mock_mipro_cls.call_args.kwargs["max_labeled_demos"] == 0
        assert mock_mipro.compile.call_args.kwargs["fewshot_aware_proposer"] is False
        assert all(getattr(p, "demos", []) == [] for p in res_mipro.predictors())

    # Verifica i branch COPRO e GEPA (entrambi zero-shot, 0 demos)
    with patch("dspy.COPRO") as mock_copro_cls:
        mock_copro = MagicMock()
        mock_copro.compile.return_value = GreekRAGInfiller(retriever=None)
        mock_copro_cls.return_value = mock_copro

        res_copro = compile_greek_rag_infiller(
            trainset=[example_epic],
            lm=dummy_lm,
            optimizer_type="copro",
        )
        assert mock_copro_cls.call_args.kwargs["metric"] is greek_infilling_dspy_metric
        assert all(getattr(p, "demos", []) == [] for p in res_copro.predictors())

    with patch("dspy.GEPA") as mock_gepa_cls:
        mock_gepa = MagicMock()
        mock_gepa.compile.return_value = GreekRAGInfiller(retriever=None)
        mock_gepa_cls.return_value = mock_gepa

        res_gepa = compile_greek_rag_infiller(
            trainset=[example_epic],
            lm=dummy_lm,
            optimizer_type="gepa",
        )
        assert mock_gepa_cls.call_args.kwargs["metric"] is greek_infilling_gepa_metric
        assert all(getattr(p, "demos", []) == [] for p in res_gepa.predictors())


# ==============================================================================
# 4. TEST FASTAPI ROUTES (/predictions/categories, /predictions, /batch, /feedback)
# ==============================================================================


def test_fastapi_categories_and_predictions_validation():
    mock_service = MagicMock()
    mock_service.get_predictions = AsyncMock(
        return_value=[
            {
                "sentence": "ἐνταῦθα δὲ και τοὺς λόγους",
                "token_str": "και",
                "score": 0.9,
                "tlg_attested": True,
                "reconstructed_word": "και",
            }
        ]
    )
    mock_service.get_batch_predictions = AsyncMock(
        return_value=[
            {
                "model_id": str(ObjectId()),
                "model_label": "Qwen",
                "model_type": "RAG",
                "checkpoint": "Qwen/Qwen2.5-72B-Instruct",
                "use_rag": True,
                "predictions": [],
                "error": None,
            }
        ]
    )
    mock_service.save_feedback = AsyncMock(return_value={"id": "fb1", "status": "saved"})

    app.dependency_overrides[get_service] = lambda: mock_service
    try:
        client = TestClient(app)

        # 1. Endpoint catalogo categorie
        resp_cat = client.get("/predictions/categories")
        assert resp_cat.status_code == 200
        body_cat = resp_cat.json()
        assert body_cat["total"] == 75
        codes = {item["code"] for item in body_cat["categories"]}
        assert "phil." in codes
        assert "epic." in codes
        assert len(codes) == len(TLG_CATEGORY_BY_CODE)

        # 2. GET /predictions inoltra text_categories
        resp_ok = client.get(
            "/predictions",
            params=[
                ("model_id", str(ObjectId())),
                ("context", "ἐνταῦθα δὲ [...] τοὺς λόγους"),
                ("num_predictions", 5),
                ("text_categories", "Phil"),
                ("text_categories", "*Rhet."),
            ],
        )
        assert resp_ok.status_code == 200
        assert mock_service.get_predictions.call_args.kwargs["text_categories"] == [
            "Phil",
            "*Rhet.",
        ]

        # 3. GET /predictions/batch inoltra text_categories
        resp_batch = client.get(
            "/predictions/batch",
            params=[
                ("model_ids", str(ObjectId())),
                ("context", "ἐνταῦθα δὲ [...] τοὺς λόγους"),
                ("num_predictions", 5),
                ("text_categories", "epic."),
            ],
        )
        assert resp_batch.status_code == 200
        assert mock_service.get_batch_predictions.call_args.kwargs["text_categories"] == ["epic."]

        # 4. POST /feedback accetta text_categories
        resp_fb = client.post(
            "/feedback",
            json={
                "model_id": str(ObjectId()),
                "context": "ἐνταῦθα δὲ [...] τοὺς λόγους",
                "token_str": "και",
                "is_accepted": True,
                "text_categories": ["phil."],
            },
        )
        assert resp_fb.status_code == 201
        assert mock_service.save_feedback.call_args.args[0].text_categories == ["phil."]
    finally:
        app.dependency_overrides.clear()


def test_fastapi_predictions_rejects_invalid_text_category_with_400():
    from backend.api.services.suggestions_service import SuggestionsService

    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(
        return_value={
            "_id": ObjectId(),
            "CHECKPOINT": "Qwen/Qwen2.5-72B-Instruct",
            "BACKEND": "auto",
            "TOP_K_PASSAGES": 5,
            "TYPE": ModelType.RAG,
        }
    )
    real_service = SuggestionsService(db_collection=mock_collection, gridfs=MagicMock())
    app.dependency_overrides[get_service] = lambda: real_service
    try:
        client = TestClient(app)

        resp = client.get(
            "/predictions",
            params=[
                ("model_id", str(ObjectId())),
                ("context", "ἐνταῦθα δὲ [...] τοὺς λόγους"),
                ("num_predictions", 5),
                ("text_categories", "invalid_genre"),
            ],
        )
        assert resp.status_code == 400
        assert "non riconosciuta" in resp.json()["detail"]

        resp_batch = client.get(
            "/predictions/batch",
            params=[
                ("model_ids", str(ObjectId())),
                ("context", "ἐνταῦθα δὲ [...] τοὺς λόγους"),
                ("num_predictions", 5),
                ("text_categories", "invalid_genre"),
            ],
        )
        assert resp_batch.status_code == 400
        assert "non riconosciuta" in resp_batch.json()["detail"]
    finally:
        app.dependency_overrides.clear()
