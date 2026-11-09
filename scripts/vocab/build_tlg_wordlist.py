"""
Script per la generazione offline del lessico pre-normalizzato dell'intero TLG (`tlg.wl.gz`).

Estrae tutte le forme lessicali attestate nei file TEI XML di `tlg-bulk/tlg` (convertiti
da Diogenes a partire dal CD-ROM TLG_E), applicando:
1. Rimozione di elementi paratestuali (`<label>`, `<seg>`, `<note>`, `<head>`).
2. Esclusione dei token adiacenti a lacune (`[...]`, `[---]`, `...`) per evitare che radici
   tronche entrino nel vocabolario come parole intere.
3. Ricongiunzione delle parole spezzate da trattino di a-capo (`-\n`).
4. Normalizzazione canonica (`normalize_greek(..., case_folding='fold', strip_diacritics_flag=True)`),
   uniformando il sigma finale (`ς` -> `σ`) e rimuovendo spiriti, accenti e punteggiatura.
5. Esclusione delle consonanti singole isolate (1 carattere derivante da elisione come `δ'`, `τ'`),
   conservando come parole di 1 lettera solo le autentiche forme autonome greche (`ο`, `η`, `α`, `ω`).
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from pathlib import Path

import lxml.etree as ET
from tqdm import tqdm

# Aggiunge la root del progetto al PYTHONPATH per importare backend.core
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.preprocess import normalize_greek, remove_punctuation

TEI_NS = "http://www.tei-c.org/ns/1.0"
STRIP_TAGS = (
    f"{{{TEI_NS}}}label",
    f"{{{TEI_NS}}}seg",
    f"{{{TEI_NS}}}note",
    f"{{{TEI_NS}}}head",
    f"{{{TEI_NS}}}bibl",
    f"{{{TEI_NS}}}figure",
)

DEFAULT_TLG_BULK_DIR = PROJECT_ROOT.parent / "datasets" / "tlg-bulk" / "tlg"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "backend" / "core" / "resources" / "tlg.wl.gz"

# Uniche parole greche autonome di 1 sola lettera (articoli/pronomi/congiunzioni/interiezioni: ὁ/ὅ, ἡ/ἤ/ἦ, ἃ, ὦ)
VALID_SINGLE_CHAR_WORDS = frozenset({"ο", "η", "α", "ω"})

_TLG_FILE_REGEX = re.compile(r"^tlg\d+\.xml$", re.IGNORECASE)
_LACUNA_SPAN_REGEX = re.compile(r"\[[\.\-\s…×]+\]|\.{2,}|…")
_LINE_HYPHEN_REGEX = re.compile(r"-\s+")
_APOSTROPHE_CHARS = "'’ʼʽ῾‘‛`"
_TOKEN_SPLIT_REGEX = re.compile(r"[\s,.;:··!?\(\)\[\]\{\}⟨⟩⸤⸥«»\"“”„—–―<>\/\\|+*&^%$#@=~`]+")
_PURE_GREEK_NORM_REGEX = re.compile(r"^[\u0370-\u03ff\u1f00-\u1fff]+$")


def extract_surface_tokens_from_xml(xml_path: Path) -> set[str]:
    """
    Estrae l'insieme dei token greci grezzi dal corpo (`<tei:body>`) di un file TEI XML del TLG,
    ricongiungendo gli a-capo sillabici e scartando i frammenti adiacenti a lacuna.
    """
    try:
        doc = ET.parse(str(xml_path), ET.XMLParser(recover=True, remove_blank_text=False))
    except Exception:
        return set()

    body = doc.find(f".//{{{TEI_NS}}}body")
    if body is None:
        return set()

    # Rimuove speaker label (es. ΣΩ., ΕΥΘ.), marginalia e note mantenendo eventuale tail testuale
    ET.strip_elements(body, *STRIP_TAGS, with_tail=False)

    raw_text = "".join(body.itertext())
    if not raw_text:
        return set()

    # Marca le lacune con U+FFFD prima di rimuovere le parentesi editoriali, così da scartare
    # eventuali prefissi/suffissi tronchi adiacenti alla lacuna.
    marked = _LACUNA_SPAN_REGEX.sub(" \ufffd ", raw_text)

    # Ricongiunge le parole spezzate a fine riga (es. "ἐμμεμέ-\n     νηκεν" -> "ἐμμεμένηκεν")
    rejoined = _LINE_HYPHEN_REGEX.sub("", marked)

    tokens: set[str] = set()
    for raw_tok in _TOKEN_SPLIT_REGEX.split(rejoined):
        if not raw_tok or "\ufffd" in raw_tok:
            continue
        # Rimuove eventuali apostrofi di elisione ad inizio/fine parola (es. δ', ἀλλ', ’βούλευσε)
        cleaned_tok = raw_tok.strip(_APOSTROPHE_CHARS + "-")
        if not cleaned_tok or any(ch in cleaned_tok for ch in _APOSTROPHE_CHARS):
            # Se contiene un apostrofo interno (es. crasi/elisione incollata senza spazio), divide le parti
            for sub_tok in re.split(f"[{re.escape(_APOSTROPHE_CHARS)}]+", cleaned_tok):
                sub_tok = sub_tok.strip("-")
                if sub_tok:
                    tokens.add(sub_tok)
        else:
            tokens.add(cleaned_tok)

    return tokens


def normalize_lexicon_token(token: str) -> str:
    """
    Applica la normalizzazione canonica usata per il lookup TLG:
    - `case_folding="fold"` (converte maiuscole in minuscole e uniforma `ς` -> `σ`)
    - `strip_diacritics_flag=True` (rimuove spiriti, accenti, dieresi e iota sottoscritto)
    - rimozione punteggiatura e spazi invisibili
    - filtro su alfabeto greco puro e sulle consonanti singole da elisione
    """
    if not token:
        return ""

    norm = normalize_greek(token, case_folding="fold", strip_diacritics_flag=True)
    norm = remove_punctuation(norm)
    norm = re.sub(r"[\s\u200B-\u200D\uFEFF]", "", norm).strip()

    if not norm:
        return ""
    if len(norm) == 1 and norm not in VALID_SINGLE_CHAR_WORDS:
        return ""
    if not _PURE_GREEK_NORM_REGEX.match(norm):
        return ""

    return norm


def build_tlg_wordlist(
    tlg_dir: Path = DEFAULT_TLG_BULK_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> set[str]:
    """
    Scansiona l'intero corpus `tlg-bulk/tlg`, normalizza le forme uniche e salva
    l'artefatto compresso `tlg.wl.gz`.
    """
    raw_surface_forms: set[str] = set()

    if tlg_dir.exists() and tlg_dir.is_dir():
        xml_files = sorted(
            p
            for p in tlg_dir.iterdir()
            if p.is_file() and _TLG_FILE_REGEX.match(p.name)
        )
        print(f"Estrazione forme lessicali da {len(xml_files)} file XML in '{tlg_dir}'...")
        for xml_path in tqdm(xml_files, desc="Scanning TLG XML", unit="file"):
            raw_surface_forms.update(extract_surface_tokens_from_xml(xml_path))
    else:
        print(f"ATTENZIONE: Directory '{tlg_dir}' non trovata.")

    print(f"Forme superficiali uniche estratte: {len(raw_surface_forms):,}")
    print("Normalizzazione canonica (casefold + strip diacritics)...")

    normalized_vocab: set[str] = set()
    for raw_form in tqdm(raw_surface_forms, desc="Normalizing", unit="word"):
        norm = normalize_lexicon_token(raw_form)
        if norm:
            normalized_vocab.add(norm)

    sorted_vocab = sorted(normalized_vocab)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=9) as gz_out:
        for word in sorted_vocab:
            gz_out.write(f"{word}\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(
        f"Wordlist TLG salvata in '{output_path}': "
        f"{len(sorted_vocab):,} forme uniche normalizzate ({size_mb:.2f} MB)."
    )
    return normalized_vocab


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera il lessico pre-normalizzato compresso del TLG (tlg.wl.gz) a partire da tlg-bulk."
    )
    parser.add_argument(
        "--tlg-dir",
        type=Path,
        default=DEFAULT_TLG_BULK_DIR,
        help=f"Percorso alla cartella tlg-bulk/tlg contenente i file TEI XML (default: {DEFAULT_TLG_BULK_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Percorso di output per il file compresso .wl.gz (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()
    build_tlg_wordlist(
        tlg_dir=args.tlg_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
