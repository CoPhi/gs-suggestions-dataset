import os
import re
import lxml.etree as ET

our_namespaces = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

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
        
    return "grc"  # default

def parse_element_text(element):
    """
    Recursively extract text from the element.
    Handle <gap> tag correctly.
    """
    if element.tag == "{http://www.tei-c.org/ns/1.0}gap":
        quantity = element.get("quantity")
        if quantity and quantity.isdigit():
            text = "." * int(quantity)
        else:
            text = "<gap/>"
        
        tail = element.tail or ""
        return text + tail

    # For other elements, join text, children's text, and tail
    text = element.text or ""
    for child in element:
        text += parse_element_text(child)
    text += element.tail or ""
    return text

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

    corpus_identifier = os.path.basename(file_path).split('.')[0]
    if re.match(r"^tlg\d+", corpus_identifier):
        corpus_identifier = "tlg"

    body = doc.find(".//tei:body", namespaces=our_namespaces)
    if body is None:
        return []

    # Try to find paragraph-like structural elements
    blocks = body.xpath(".//tei:p | .//tei:ab | .//tei:l", namespaces=our_namespaces)
    
    if not blocks:
        # Fallback to direct structural divisions 
        blocks = body.xpath(".//tei:div", namespaces=our_namespaces)
        
    if not blocks:
        # Final fallback: whole body
        blocks = [body]

    results = []
    
    for idx, block in enumerate(blocks):
        raw_text = parse_element_text(block)
        
        # Clean up text
        clean_text = re.sub(r'\s+', ' ', raw_text).strip()
        if not clean_text or len(clean_text) < 5:
            continue
            
        d = {
            "corpus_id": corpus_identifier,
            "file_id": _file_id,
            "block_index": idx + 1,
            "id": f"{corpus_identifier}/{_file_id}/{idx + 1}",
            "title": _title,
            "material": _material,
            "language": _lang,
            "training_text": clean_text,
            "test_cases": []
        }
        results.append(d)

    return results

if __name__ == "__main__":
    import sys
    import json
    for arg in sys.argv[1:]:
        for item in convert_tei_to_json(arg):
            print(json.dumps(item, ensure_ascii=False))
