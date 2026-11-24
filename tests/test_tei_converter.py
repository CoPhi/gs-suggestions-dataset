import pytest
import lxml.etree as ET
from scripts.tei.converter import (
    author,
    idno,
    title,
    material,
    language,
    parse_element_text,
    convert_tei_to_json
)
import tempfile
import os

@pytest.fixture
def sample_tei_xml():
    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="grc">
        <teiHeader>
            <fileDesc>
                <titleStmt>
                    <title>Test Title</title>
                </titleStmt>
                <publicationStmt>
                    <idno type="filename">test_file.xml</idno>
                </publicationStmt>
                <sourceDesc>
                    <supportDesc>
                        <support>
                            <material>papyrus</material>
                        </support>
                    </supportDesc>
                </sourceDesc>
            </fileDesc>
            <profileDesc>
                <langUsage>
                    <language ident="grc">Greek</language>
                </langUsage>
            </profileDesc>
        </teiHeader>
        <text>
            <body>
                <div type="edition">
                    <p>Some <gap quantity="3"/> text <gap/> with <span>nested</span> elements.</p>
                </div>
            </body>
        </text>
    </TEI>
    """
    return ET.fromstring(xml)

def test_idno_extraction(sample_tei_xml):
    assert idno(sample_tei_xml, "some_path.xml") == "test_file.xml"

def test_idno_fallback():
    xml = '<TEI xmlns="http://www.tei-c.org/ns/1.0"></TEI>'
    doc = ET.fromstring(xml)
    assert idno(doc, "fallback.xml") == "fallback.xml"

def test_title_extraction(sample_tei_xml):
    assert title(sample_tei_xml) == "Test Title"

def test_material_extraction(sample_tei_xml):
    assert material(sample_tei_xml) == "papyrus"

def test_language_extraction(sample_tei_xml):
    assert language(sample_tei_xml) == "grc"

@pytest.mark.parametrize(
    "xml_string, expected_text",
    [
        ("<p xmlns='http://www.tei-c.org/ns/1.0'>Hello there</p>", "Hello there"),
        ("<p xmlns='http://www.tei-c.org/ns/1.0'>Hello <gap quantity='4'/> there</p>", "Hello .... there"),
        ("<p xmlns='http://www.tei-c.org/ns/1.0'>Unknown gap <gap/> here</p>", "Unknown gap <gap/> here"),
        ("<ab xmlns='http://www.tei-c.org/ns/1.0'>Text <supplied reason='lost'>supplied</supplied> end</ab>", "Text [supplied] end"),
    ]
)
def test_parse_element_text(xml_string, expected_text):
    element = ET.fromstring(xml_string)
    text = parse_element_text(element)
    assert text == expected_text

def test_convert_tei_to_json(sample_tei_xml):
    # We need a physical file to test convert_tei_to_json properly
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="wb") as f:
        f.write(ET.tostring(sample_tei_xml, encoding="utf-8"))
        temp_path = f.name
        
    try:
        results = convert_tei_to_json(temp_path)
        
        assert len(results) == 1
        obj = results[0]
        
        # Verify JSON schema matches requirements
        assert obj["corpus_id"] == os.path.basename(temp_path).split('.')[0]
        assert obj["file_id"] == "test_file.xml"
        assert obj["title"] == "Test Title"
        assert obj["material"] == "papyrus"
        assert obj["language"] == "grc"
        
        # text should have dots instead of quantity 3, and <gap/> for the empty one
        # Because we strip whitespace inside cleaning
        assert obj["training_text"] == "Some ... text <gap/> with nested elements"
        assert obj["test_cases"] == []
    finally:
        os.remove(temp_path)


def test_author_strip_marginalia_and_poetry_grouping(tmp_path):
    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="grc">
        <teiHeader>
            <fileDesc>
                <titleStmt>
                    <title>Ilias</title>
                    <author>Homerus Epic.</author>
                </titleStmt>
                <publicationStmt>
                    <idno type="filename">tlg0012001.xml</idno>
                </publicationStmt>
            </fileDesc>
        </teiHeader>
        <text>
            <body>
                <div type="Book" n="1">
                    <l n="t"><label type="head">ΙΛΙΑΔΟΣ Α</label></l>
                    <l n="1">Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος</l>
                    <l n="2"><seg rend="Marginalia">&gt;</seg> οὐλομένην ἣ μυρί' Ἀχαιοῖς ἄλγε' ἔθηκε</l>
                    <l n="3"><seg rend="Marginalia">※</seg> πολλὰς δ' ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν</l>
                </div>
                <div type="Book" n="2">
                    <l n="1">Ἄλλοι μέν ῥα θεοί τε καὶ ἀνέρες ἱπποκορυσταὶ</l>
                    <p>Προσθήκη πεζοῦ λόγου ἐξ ἑτέρου τμήματος.</p>
                    <l n="2">εὗδον παννύχιοι Δία δ' οὐκ ἔχε νήδυμος ὕπνος</l>
                </div>
            </body>
        </text>
    </TEI>
    """
    xml_path = tmp_path / "tlg0012001.xml"
    xml_path.write_text(xml, encoding="utf-8")

    doc = ET.fromstring(xml.encode("utf-8"))
    assert author(doc) == "Homerus Epic."

    # Default retrocompatibile: un blocco per verso e marginalia preservati
    default_blocks = convert_tei_to_json(str(xml_path))
    assert len(default_blocks) == 6
    assert ">" in default_blocks[1]["training_text"]

    # Modalità RAG: rimozione Marginalia e aggregazione dei <l> entro ciascun <div>
    grouped_blocks = convert_tei_to_json(
        str(xml_path),
        strip_marginalia=True,
        group_poetry_lines=True,
        target_poetry_words=100,
        author_override="Homerus",
        metadata_extra={"work_id": "0012 001", "title": "SHOULD_NOT_OVERRIDE"},
    )
    # Book 1 (3 versi uniti) | Book 2 verso 1 | <p> | Book 2 verso 2 => 4 blocchi
    assert len(grouped_blocks) == 4
    assert grouped_blocks[0]["author"] == "Homerus"
    assert grouped_blocks[0]["title"] == "Ilias"
    assert grouped_blocks[0]["work_id"] == "0012 001"
    assert ">" not in grouped_blocks[0]["training_text"]
    assert "※" not in grouped_blocks[0]["training_text"]
    assert "Μῆνιν ἄειδε" in grouped_blocks[0]["training_text"]
    assert "προΐαψεν" in grouped_blocks[0]["training_text"]
    assert "Ἄλλοι μέν" in grouped_blocks[1]["training_text"]
    assert "Προσθήκη πεζοῦ" in grouped_blocks[2]["training_text"]
    assert "εὗδον παννύχιοι" in grouped_blocks[3]["training_text"]
