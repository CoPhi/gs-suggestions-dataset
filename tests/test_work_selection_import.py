"""
Test unitari per l'importer generico `scripts/tei/import_work_selection.py` (`grc-lab/work-selection`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core.cleaner import load_abs
from scripts.tei.import_work_selection import (
    format_author_name,
    import_work_selections,
)


def test_format_author_name_normalization():
    assert format_author_name("HOMERUS") == "Homerus"
    assert format_author_name("philo judaeus") == "Philo Judaeus"
    assert format_author_name("<PYTHAGORAS>") == "Pythagoras"
    assert (
        format_author_name("ARISTOTELES et CORPUS ARISTOTELICUM")
        == "Aristoteles et Corpus Aristotelicum"
    )
    assert format_author_name("Plutarchus Biogr. et Phil.") == "Plutarchus"
    assert format_author_name("Apollonius Rhodius Epic.") == "Apollonius Rhodius"
    assert (
        format_author_name("Aristoteles Phil. et Corpus Aristotelicum Aristotle")
        == "Aristoteles et Corpus Aristotelicum"
    )


def test_import_work_selections_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tlg_dir = tmp_path / "tlg"
    existing_dir = tmp_path / "data"
    out_dir = existing_dir / "rag_corpora"
    tlg_dir.mkdir()
    existing_dir.mkdir()

    # authtab.xml per risolvere l'autore dell'opera filosofica già presente in `data/`
    (tlg_dir / "authtab.xml").write_text(
        """<authtab corpus="tlg">
<author name="Plato Phil." n="0059"/>
<author name="Homerus Epic." n="0012"/>
</authtab>""",
        encoding="utf-8",
    )

    # Opera 1 (0012 001): nuova poesia con Marginalia e più versi <l>
    (tlg_dir / "tlg0012001.xml").write_text(
        """<TEI xmlns="http://www.tei-c.org/ns/1.0">
<teiHeader><fileDesc>
  <titleStmt><title>Ilias</title><author>Homerus Epic.</author></titleStmt>
  <publicationStmt><idno type="filename">tlg0012001.xml</idno></publicationStmt>
</fileDesc></teiHeader>
<text><body><div type="Book" n="1">
  <l n="1">Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος</l>
  <l n="2"><seg rend="Marginalia">&gt;</seg> οὐλομένην ἣ μυρί' Ἀχαιοῖς ἄλγε' ἔθηκε</l>
  <l n="3"><seg rend="Marginalia">⸖</seg> πολλὰς δ' ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν</l>
</div></body></text>
</TEI>""",
        encoding="utf-8",
    )

    # Opera 2 (0059 001): già presente in data/tlg_0.json (overlap da non duplicare)
    (tlg_dir / "tlg0059001.xml").write_text(
        """<TEI xmlns="http://www.tei-c.org/ns/1.0">
<teiHeader><fileDesc>
  <titleStmt><title>Euthyphro</title><author>Plato Phil.</author></titleStmt>
  <publicationStmt><idno type="filename">tlg0059001.xml</idno></publicationStmt>
</fileDesc></teiHeader>
<text><body><p>Τί νεώτερον ὦ Σώκρατες γέγονεν ἐνθάδε.</p></body></text>
</TEI>""",
        encoding="utf-8",
    )

    existing_blocks = [
        {
            "corpus_id": "tlg",
            "file_id": "tlg0059001.xml",
            "block_index": 1,
            "id": "tlg/tlg0059001.xml/1",
            "title": "Euthyphro",
            "material": "unknown",
            "language": "grc",
            "training_text": "Τί νεώτερον ὦ Σώκρατες γέγονεν ἐνθάδε.",
            "test_cases": [],
        }
    ]
    (existing_dir / "tlg_0.json").write_text(
        json.dumps(existing_blocks, ensure_ascii=False), encoding="utf-8"
    )

    # Manifest grc-lab/work-selection: 1 nuova, 1 overlap, 1 X01 assente su disco
    selection_path = tmp_path / "selection.json"
    selection_payload = {
        "format": "grc-lab/work-selection",
        "version": 4,
        "name": "Test Selection",
        "works": [
            {
                "author_id": "0012",
                "work_id": "0012 001",
                "author": "HOMERUS",
                "work": "Ilias",
                "cla": "epic.",
                "gruppi": "Poesia",
                "secoli": "VIII a.C.",
                "status": "int",
            },
            {
                "author_id": "0059",
                "work_id": "0059 001",
                "author": "PLATO",
                "work": "Euthyphro",
                "cla": "phil.",
                "gruppi": "Filosofia",
                "secoli": "V-IV a.C.",
                "status": "int",
            },
            {
                "author_id": "0012",
                "work_id": "0012 X01",
                "author": "HOMERUS",
                "work": "Epigramma dedicatorium",
                "cla": "epigr.",
                "gruppi": "Poesia",
                "secoli": "VIII a.C.",
                "status": "int",
            },
        ],
    }
    selection_path.write_text(json.dumps(selection_payload), encoding="utf-8")

    report = import_work_selections(
        selection_paths=[selection_path],
        tlg_dir=tlg_dir,
        output_dir=out_dir,
        existing_tlg_dir=existing_dir,
        target_poetry_words=90,
    )

    assert report["total_selection_works"] == 3
    assert report["matched_on_disk"] == 2
    assert report["missing_non_numeric_count"] == 1
    assert report["missing_numeric_count"] == 0
    assert report["skipped_existing_overlap_count"] == 1
    assert report["converted_works_count"] == 1
    assert report["generated_shards"] == ["tlg_rag_0.json"]

    # Verifica che nello shard RAG ci sia solo la nuova opera, con versi uniti e senza Marginalia
    rag_shard = json.loads((out_dir / "tlg_rag_0.json").read_text(encoding="utf-8"))
    assert len(rag_shard) == 1
    assert rag_shard[0]["file_id"] == "tlg0012001.xml"
    assert rag_shard[0]["author"] == "Homerus"
    assert ">" not in rag_shard[0]["training_text"]
    assert "⸖" not in rag_shard[0]["training_text"]
    assert "Μῆνιν ἄειδε" in rag_shard[0]["training_text"]
    assert "προΐαψεν" in rag_shard[0]["training_text"]

    # Verifica tlg_author_map.json per entrambe le opere (nuova + esistente)
    author_map = json.loads((out_dir / "tlg_author_map.json").read_text(encoding="utf-8"))
    assert author_map["tlg0012001.xml"] == "Homerus"
    assert author_map["tlg0059001.xml"] == "Plato"

    # Verifica che load_abs() su DATA_PATH continui a vedere SOLO `data/tlg_0.json` e non `rag_corpora/`
    monkeypatch.setattr("backend.core.cleaner.DATA_PATH", existing_dir)
    loaded = load_abs(corpus_set=["tlg"])
    assert [b["file_id"] for b in loaded] == ["tlg0059001.xml"]
