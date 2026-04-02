import pytest
import lxml.etree as ET
from scripts.tei_converter import (
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
        ("<ab xmlns='http://www.tei-c.org/ns/1.0'>Text <supplied>supplied</supplied> end</ab>", "Text supplied end"),
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
        assert obj["training_text"] == "Some ... text <gap/> with nested elements."
        assert obj["test_cases"] == []
    finally:
        os.remove(temp_path)
