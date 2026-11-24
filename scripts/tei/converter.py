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


def author(doc) -> str:
    """Estrae il nome dell'autore dal header TEI (`titleStmt/author` o `author`)."""
    author_elem = doc.find(".//tei:titleStmt/tei:author", namespaces=our_namespaces)
    if author_elem is None:
        author_elem = doc.find(".//tei:author", namespaces=our_namespaces)
    if author_elem is not None:
        return RE_WHITESPACE.sub(" ", "".join(author_elem.itertext())).strip()
    return ""


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


def _strip_marginalia_elements(body) -> None:
    """
    Rimuove i soli `<tei:seg rend="Marginalia">` (segni critici di margine come `>`, `⸖`, `※`)
    preservando il testo `.tail` successivo e lasciando intatti eventuali altri tag `<seg>`.
    """
    for seg in body.xpath(".//tei:seg[@rend='Marginalia']", namespaces=our_namespaces):
        parent = seg.getparent()
        if parent is None:
            continue
        tail = seg.tail or ""
        prev = seg.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
        parent.remove(seg)


def _resolve_structural_container(block, body):
    """
    Individua l'antenato `<tei:div>` più vicino del blocco (o il padre diretto) così da
    non fondere versi appartenenti a libri, carmi o frammenti differenti.
    """
    div_tag = f"{{{our_namespaces['tei']}}}div"
    curr = block.getparent()
    while curr is not None and curr != body:
        if curr.tag == div_tag:
            return curr
        curr = curr.getparent()
    parent = block.getparent()
    return parent if parent is not None else body


def _extract_clean_sentences(
    body,
    *,
    group_poetry_lines: bool = False,
    target_poetry_words: int = 90,
) -> list[str]:
    """
    Estrae i blocchi testuali dal corpo TEI; se `group_poetry_lines=True`, aggrega i versi
    `<tei:l>` contigui dentro lo stesso `<div>` fino a ~`target_poetry_words` parole.
    """
    blocks = body.xpath(".//tei:p | .//tei:ab | .//tei:l", namespaces=our_namespaces)
    if not blocks:
        blocks = body.xpath(".//tei:div", namespaces=our_namespaces)
    if not blocks:
        blocks = [body]

    if not group_poetry_lines:
        sentences: list[str] = []
        for block in blocks:
            raw_text = parse_element_text(block)
            clean_text = RE_WHITESPACE.sub(" ", raw_text).strip()
            if clean_text and len(clean_text) >= 5:
                sentences.append(clean_text)
        return sentences

    l_tag = f"{{{our_namespaces['tei']}}}l"
    sentences = []
    poetry_buf: list[str] = []
    poetry_words = 0
    poetry_container = None

    def _flush_poetry() -> None:
        nonlocal poetry_buf, poetry_words, poetry_container
        if poetry_buf:
            joined = " ".join(poetry_buf).strip()
            if len(joined) >= 5:
                sentences.append(joined)
        poetry_buf = []
        poetry_words = 0
        poetry_container = None

    for block in blocks:
        raw_text = parse_element_text(block)
        clean_text = RE_WHITESPACE.sub(" ", raw_text).strip()
        if not clean_text:
            continue

        if block.tag == l_tag:
            container = _resolve_structural_container(block, body)
            if poetry_container is not None and container != poetry_container:
                _flush_poetry()
            poetry_container = container
            poetry_buf.append(clean_text)
            poetry_words += len(clean_text.split())
            if poetry_words >= target_poetry_words:
                _flush_poetry()
        else:
            _flush_poetry()
            if len(clean_text) >= 5:
                sentences.append(clean_text)

    _flush_poetry()
    return sentences


def convert_tei_to_json(
    file_path,
    *,
    strip_marginalia: bool = False,
    group_poetry_lines: bool = False,
    target_poetry_words: int = 90,
    author_override: str | None = None,
    metadata_extra: dict | None = None,
):
    try:
        doc = ET.parse(file_path, ET.XMLParser(recover=True, remove_blank_text=True))
    except (ET.XMLSyntaxError, ET.ParseError, FileNotFoundError) as e:
        print(f"Error parsing {file_path}. Error: {e}")
        return []

    _file_id = idno(doc, file_path)
    _title = title(doc)
    _author = (author_override if author_override is not None else author(doc)).strip()
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
    if strip_marginalia:
        _strip_marginalia_elements(body)

    sentences = _extract_clean_sentences(
        body,
        group_poetry_lines=group_poetry_lines,
        target_poetry_words=target_poetry_words,
    )

    core_keys = {
        "corpus_id",
        "file_id",
        "block_index",
        "id",
        "title",
        "material",
        "language",
        "training_text",
        "test_cases",
        "author",
    }

    results = []
    global_block_index = 1

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
        if _author:
            d["author"] = _author
        if metadata_extra:
            for k, v in metadata_extra.items():
                if v is not None and k not in core_keys:
                    d[k] = v

        results.append(d)
        global_block_index += 1

    return results


if __name__ == "__main__":
    import sys
    import json

    for arg in sys.argv[1:]:
        for item in convert_tei_to_json(arg):
            print(json.dumps(item, ensure_ascii=False))
