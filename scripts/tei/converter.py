import os
import re
import lxml.etree as ET
from packages.maat.maat.converter import Converter
from packages.maat.maat.create import create_training_text
from scripts import RE_WHITESPACE, RE_TEST_CASE

our_namespaces = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

SHARED_CONVERTER = Converter()


def idno(doc, file_path):
    idno_elem = doc.find(".//tei:idno[@type='filename']", namespaces=our_namespaces)
    if idno_elem is None:
        idno_elem = doc.find(".//tei:idno", namespaces=our_namespaces)
    if idno_elem is not None:
        return idno_elem.text
    return os.path.basename(file_path)


def title(doc):
    title_elem = doc.find(".//tei:title", namespaces=our_namespaces)
    if title_elem is not None:
        return "".join(title_elem.itertext()).strip()
    return "unknown"


def material(doc):
    material_elem = doc.find(".//tei:material", namespaces=our_namespaces)
    if material_elem is not None:
        return "".join(material_elem.itertext()).strip()
    return "unknown"


def language(doc):
    lang_elem = doc.find(".//tei:language", namespaces=our_namespaces)
    if lang_elem is not None and lang_elem.get("ident"):
        return lang_elem.get("ident")

    # Try finding lang on TEI or text
    tei_root = doc.getroot()
    xml_lang = tei_root.get("{http://www.w3.org/XML/1998/namespace}lang")
    if xml_lang:
        return xml_lang

    return "grc"


def parse_element_text(element):
    """
    Delegates to maat Converter to process the element text,
    ensuring we strictly follow its logic.
    """
    converted_ab = SHARED_CONVERTER(element)
    training_ab = create_training_text(converted_ab)

    # Extract inner XML from the <ab> wrapper preserving tags like <gap/>
    parts = [training_ab.text or ""]
    parts.extend(
        ET.tostring(child, encoding="unicode", with_tail=True) for child in training_ab
    )
    return "".join(parts)


def convert_tei_to_json(file_path):
    try:
        doc = ET.parse(file_path, ET.XMLParser(recover=True, remove_blank_text=True))
    except (ET.XMLSyntaxError, ET.ParseError, FileNotFoundError) as e:
        print(f"Error parsing {file_path}. Error: {e}")
        return []

    _file_id = idno(doc, file_path)
    _title = title(doc)
    _material = material(doc).lower()
    _lang = language(doc)

    corpus_identifier = os.path.basename(file_path).split(".")[0]
    if re.match(r"^tlg\d+", corpus_identifier):
        corpus_identifier = "tlg"

    body = doc.find(".//tei:body", namespaces=our_namespaces)
    if body is None:
        return []

    # i tag label devono essere assenti
    ET.strip_elements(body, f"{{{our_namespaces['tei']}}}label", with_tail=False)

    # Try to find paragraph-like structural elements
    blocks = body.xpath(".//tei:p | .//tei:ab | .//tei:l", namespaces=our_namespaces)

    if not blocks:
        # Fallback to direct structural divisions
        blocks = body.xpath(".//tei:div", namespaces=our_namespaces)

    if not blocks:
        # Final fallback: whole body
        blocks = [body]

    results = []
    global_block_index = 1

    for block in blocks:
        raw_text = parse_element_text(block)

        # Clean up text
        clean_text = RE_WHITESPACE.sub(" ", raw_text).strip()
        if not clean_text or len(clean_text) < 5:
            continue

        sentences = [clean_text]

        for sentence in sentences:
            if not sentence or len(sentence) < 5:
                continue

            test_cases = []
            matches = RE_TEST_CASE.finditer(sentence)
            for i, match in enumerate(matches):
                start, end = match.start(), match.end()
                expected = match.group(1)

                # formazione test_case
                pre_masked = sentence[:start].replace("[", "").replace("]", "")
                post_masked = sentence[end:].replace("[", "").replace("]", "")
                mask = "." * len(expected)
                masked_text = f"{pre_masked}[{mask}]{post_masked}"

                test_cases.append(
                    {
                        "case_index": i,
                        "id": f"{corpus_identifier}/{_file_id}/{global_block_index}/{i}",
                        "test_case": masked_text,
                    }
                )

            d = {
                "corpus_id": corpus_identifier,
                "file_id": _file_id,
                "block_index": global_block_index,
                "id": f"{corpus_identifier}/{_file_id}/{global_block_index}",
                "title": _title,
                "material": _material,
                "language": _lang,
                "training_text": sentence,
                "test_cases": test_cases,
            }
            results.append(d)
            global_block_index += 1

    return results


if __name__ == "__main__":
    import sys
    import json

    for arg in sys.argv[1:]:
        for item in convert_tei_to_json(arg):
            print(json.dumps(item, ensure_ascii=False))
