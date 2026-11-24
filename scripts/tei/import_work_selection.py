"""
Script CLI generico per importare selezioni di opere TLG nel formato `grc-lab/work-selection`
all'interno della collezione dedicata al RAG (`data/rag_corpora/`), lasciando intatti i file
`data/tlg_0.json` e `data/tlg_1.json` usati dalle pipeline di addestramento.

Funzionalità principali:
1. Legge uno o più manifest `grc-lab/work-selection` (es. `data/selections/*.json`).
2. Risolve ogni `work_id` (`"0012 001"`) nel corrispondente TEI XML (`tlg0012001.xml`) in `tlg-bulk/tlg`,
   distinguendo nel report le opere con `work_id` non numerico (`X01`/`x01`) assenti su disco.
3. Deduplica rispetto alle opere già presenti in `data/tlg_*.json`.
4. Converte i TEI XML rimuovendo `<seg rend="Marginalia">`, aggregando i versi poetici `<l>`
   contigui (entro i confini di ciascun `<div>`) e normalizzando il nome autore.
5. Esporta `tlg_author_map.json` (coprendo sia le opere già in `data/tlg_*.json` via `authtab.xml`
   sia le nuove opere importate) e `selection_import_report.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import lxml.etree as ET
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import DATA_PATH, RAG_EXTRA_DATA_PATH, RAG_SELECTIONS_DIR
from scripts.tei.converter import author as extract_tei_author
from scripts.tei.converter import convert_tei_to_json
from scripts.tei.pipeline import CorpusWriter

DEFAULT_TLG_BULK_DIR = PROJECT_ROOT.parent / "datasets" / "tlg-bulk" / "tlg"

# Abbreviazioni di genere tipiche di `authtab.xml` e dei header Diogenes da rimuovere dal nome autore
_TLG_GENRE_TOKENS = frozenset(
    {
        "alchem.",
        "apol.",
        "astrol.",
        "astron.",
        "biogr.",
        "bucol.",
        "choliamb.",
        "com.",
        "comic.",
        "comm.",
        "coq.",
        "dialog.",
        "doxogr.",
        "eccl.",
        "eleg.",
        "encom.",
        "epic.",
        "epigr.",
        "epist.",
        "exeget.",
        "fab.",
        "geogr.",
        "gnom.",
        "gramm.",
        "hagiogr.",
        "hexametr.",
        "hist.",
        "homilet.",
        "hymn.",
        "hypoth.",
        "iamb.",
        "imperator",
        "invectiv.",
        "lexicogr.",
        "lyr.",
        "math.",
        "med.",
        "mim.",
        "mus.",
        "myth.",
        "nat.",
        "nat.hist.",
        "onir.",
        "orac.",
        "orat.",
        "paroem.",
        "perieg.",
        "phil.",
        "physiognom.",
        "poem.",
        "poet.",
        "polyhist.",
        "rhet.",
        "satura",
        "satyr.",
        "schol.",
        "scr.",
        "scriptor",
        "soph.",
        "tact.",
        "theol.",
        "trag.",
    }
)
_LOWERCASE_CONNECTORS = frozenset({"et", "vel", "sive", "seu", "aut", "de", "ex", "in", "ab", "ad"})
_TLG_FILENAME_AUTHOR_RE = re.compile(r"^tlg(\d{4})", re.IGNORECASE)


def _smart_title_token(token: str, is_first: bool) -> str:
    """Normalizza il casing di un singolo token autore preservando eventuali forme miste (es. `Ps.-`)."""
    if not token:
        return ""
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return token

    lowered_alpha = "".join(ch.lower() for ch in letters)
    if not is_first and lowered_alpha in _LOWERCASE_CONNECTORS and all(
        ch.isalpha() for ch in token
    ):
        return lowered_alpha

    has_upper = any(ch.isupper() for ch in letters)
    has_lower = any(ch.islower() for ch in letters)
    # Se è già misto (es. `Mc...`, `Mac...`, `Ps.-Galenus`), lo lasciamo inalterato
    if has_upper and has_lower:
        return token

    # Normalizza ogni segmento separato da trattino rendendo maiuscola la prima lettera alfabetica
    parts = token.split("-")
    normalized_parts: list[str] = []
    for part in parts:
        out_chars: list[str] = []
        seen_alpha = False
        for ch in part:
            if ch.isalpha():
                if not seen_alpha:
                    out_chars.append(ch.upper())
                    seen_alpha = True
                else:
                    out_chars.append(ch.lower())
            else:
                out_chars.append(ch)
        normalized_parts.append("".join(out_chars))
    return "-".join(normalized_parts)


def format_author_name(raw_author: str) -> str:
    """
    Pulisce e normalizza i nomi autore provenienti da `grc-lab/work-selection` o da `authtab.xml`:
    - rimuove eventuali parentesi angolari esterne (`<PYTHAGORAS>` -> `Pythagoras`);
    - elimina le etichette abbreviate di genere TLG (`Biogr. et Phil.`, `Epic.`, `Scr. Eccl.`);
    - normalizza il casing mantenendo minuscole le congiunzioni latine (`et`, `vel`, ...).
    """
    if not raw_author:
        return ""

    text = re.sub(r"\s+", " ", raw_author).strip()
    if text.startswith("<") and text.endswith(">") and len(text) > 2:
        text = text[1:-1].strip()

    # Rimuove eventuali spazi/punteggiature residue ai bordi e separa i token preservando la struttura
    raw_tokens = text.split()
    kept_tokens: list[str] = []
    idx = 0
    while idx < len(raw_tokens):
        tok = raw_tokens[idx]
        tok_clean = tok.strip(",;/")
        tok_lower = tok_clean.lower()

        # Salta etichette di genere TLG e congiunzioni rimaste orfane tra/dopo di esse
        if tok_lower in _TLG_GENRE_TOKENS:
            while kept_tokens and kept_tokens[-1].strip(",;/").lower() in _LOWERCASE_CONNECTORS:
                kept_tokens.pop()
            # In alcuni record di `authtab.xml` dopo la qualifica di genere compare un duplicato inglese
            # finale (es. `Aristoteles Phil. et Corpus Aristotelicum Aristotle`), quindi proseguiamo solo
            # se il token successivo fa parte di un composto tipo `et Corpus ...`.
            idx += 1
            continue

        if tok_lower in _LOWERCASE_CONNECTORS:
            # Verifica se la congiunzione collega solo un tag di genere successivo
            next_tok = raw_tokens[idx + 1].strip(",;/").lower() if idx + 1 < len(raw_tokens) else ""
            if not next_tok or next_tok in _TLG_GENRE_TOKENS:
                idx += 1
                continue

        kept_tokens.append(tok_clean)
        idx += 1

    # Rimuove eventuale duplicato inglese singolo in coda presente in alcune voci `authtab.xml`
    # (es. `Homerus` / `Aristoteles et Corpus Aristotelicum Aristotle`)
    if len(kept_tokens) >= 2 and kept_tokens[-1].lower() in {"aristotle", "homer", "plato", "hesiod"}:
        kept_tokens.pop()

    if not kept_tokens:
        kept_tokens = [t.strip(",;/") for t in raw_tokens if t.strip(",;/")]

    formatted = [
        _smart_title_token(tok, is_first=(i == 0))
        for i, tok in enumerate(kept_tokens)
        if tok
    ]
    return " ".join(formatted).strip()


def parse_authtab(authtab_path: Path) -> dict[str, str]:
    """Estrae la mappa `{author_id_4_cifre: nome_autore_pulito}` da `authtab.xml`."""
    if not authtab_path.exists():
        return {}

    try:
        doc = ET.parse(str(authtab_path), ET.XMLParser(recover=True))
    except Exception:
        return {}

    authors: dict[str, str] = {}
    for elem in doc.findall(".//author"):
        aid = (elem.get("n") or "").strip()
        raw_name = (elem.get("name") or "").strip()
        if aid and raw_name:
            clean = format_author_name(raw_name)
            if clean:
                authors[aid] = clean
    return authors


def load_work_selections(selection_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Carica e unisce uno o più manifest `grc-lab/work-selection` deduplicando per `work_id`.
    Restituisce `(opere_uniche, metadati_manifest)`.
    """
    merged_works: list[dict[str, Any]] = []
    seen_work_ids: set[str] = set()
    manifests_meta: list[dict[str, Any]] = []

    for sel_path in selection_paths:
        with open(sel_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        fmt = payload.get("format")
        if fmt != "grc-lab/work-selection" or not isinstance(payload.get("works"), list):
            raise ValueError(
                f"Il file '{sel_path}' non rispetta il formato atteso 'grc-lab/work-selection'."
            )

        sel_name = str(payload.get("name") or sel_path.stem).strip()
        manifests_meta.append(
            {
                "path": str(sel_path),
                "name": sel_name,
                "version": payload.get("version"),
                "works_count": len(payload["works"]),
            }
        )

        for work in payload["works"]:
            work_id = str(work.get("work_id") or "").strip()
            if not work_id or work_id in seen_work_ids:
                continue
            seen_work_ids.add(work_id)
            enriched = dict(work)
            enriched["_selection_name"] = sel_name
            merged_works.append(enriched)

    return merged_works, manifests_meta


def collect_existing_tlg_file_ids(data_dir: Path) -> set[str]:
    """Raccoglie tutti i `file_id` già presenti nei file `tlg_*.json` nella directory base `data/`."""
    existing: set[str] = set()
    if not data_dir.exists():
        return existing

    for json_path in sorted(data_dir.glob("tlg_*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, list):
                for item in items:
                    fid = item.get("file_id")
                    if fid:
                        existing.add(str(fid))
        except Exception as exc:
            print(f"Attenzione: impossibile leggere '{json_path}': {exc}")
    return existing


def _work_id_to_filename(work_id: str) -> tuple[str, str, str]:
    """Converte `work_id` (es. `'0012 001'`) in `(author_id, work_num, 'tlg0012001.xml')`."""
    parts = work_id.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Formato work_id non valido: '{work_id}'")
    aid, wid = parts
    return aid, wid, f"tlg{aid}{wid}.xml"


def import_work_selections(
    selection_paths: Sequence[Path],
    tlg_dir: Path = DEFAULT_TLG_BULK_DIR,
    output_dir: Path = RAG_EXTRA_DATA_PATH,
    existing_tlg_dir: Path = DATA_PATH,
    target_poetry_words: int = 90,
    include_existing: bool = False,
    file_prefix: str = "tlg_rag",
    max_size_mb: int = 50,
) -> dict[str, Any]:
    """
    Converte le opere di una o più selezioni `grc-lab/work-selection` dai file TEI XML di `tlg_dir`
    verso shard JSON dedicati al RAG in `output_dir`, generando `tlg_author_map.json` e
    `selection_import_report.json`.
    """
    if not selection_paths:
        raise ValueError("Nessun file di selezione fornito.")
    if not tlg_dir.exists() or not tlg_dir.is_dir():
        raise FileNotFoundError(f"Directory TLG non trovata: '{tlg_dir}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Rimuove eventuali shard precedenti dello stesso prefisso per rendere l'import idempotente
    for old_shard in output_dir.glob(f"{file_prefix}_*.json"):
        old_shard.unlink()

    works, manifests_meta = load_work_selections(selection_paths)
    existing_file_ids = collect_existing_tlg_file_ids(existing_tlg_dir)
    authtab_map = parse_authtab(tlg_dir / "authtab.xml")
    disk_files = set(os.listdir(tlg_dir))

    writer = CorpusWriter(
        corpus_id="tlg",
        max_size_mb=max_size_mb,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )

    author_map: dict[str, str] = {}
    missing_non_numeric: list[dict[str, str]] = []
    missing_numeric: list[dict[str, str]] = []
    skipped_existing_overlap: list[str] = []
    empty_after_parse: list[str] = []
    converted_works = 0
    total_blocks_written = 0

    for work in tqdm(works, desc="Importing TLG selection", unit="work"):
        work_id = str(work.get("work_id") or "").strip()
        aid, wid, fname = _work_id_to_filename(work_id)
        clean_author = format_author_name(str(work.get("author") or "")) or authtab_map.get(aid, "")

        if fname not in disk_files:
            entry = {
                "work_id": work_id,
                "filename": fname,
                "author": clean_author or str(work.get("author") or ""),
                "work": str(work.get("work") or ""),
            }
            if not wid.isdigit():
                missing_non_numeric.append(entry)
            else:
                missing_numeric.append(entry)
            continue

        if clean_author:
            author_map[fname] = clean_author

        if fname in existing_file_ids and not include_existing:
            skipped_existing_overlap.append(fname)
            continue

        xml_path = tlg_dir / fname
        blocks = convert_tei_to_json(
            str(xml_path),
            strip_marginalia=True,
            group_poetry_lines=True,
            target_poetry_words=target_poetry_words,
            author_override=clean_author or None,
            metadata_extra={
                "work_id": work_id,
                "author_id": aid,
                "gruppi": work.get("gruppi"),
                "cla": work.get("cla"),
                "secoli": work.get("secoli"),
                "status": work.get("status"),
                "selection_name": work.get("_selection_name"),
            },
        )
        if not blocks:
            empty_after_parse.append(fname)
            continue

        if not clean_author and blocks[0].get("author"):
            author_map[fname] = format_author_name(str(blocks[0]["author"]))

        for blk in blocks:
            writer.add(blk)
        converted_works += 1
        total_blocks_written += len(blocks)

    writer.flush()

    # Popola la mappa autore anche per tutte le opere già presenti in `data/tlg_*.json`
    # così che ogni passaggio TLG del RAG mostri `[Corpus: tlg | Autore — Opera]`.
    for fid in sorted(existing_file_ids):
        if fid in author_map:
            continue
        match = _TLG_FILENAME_AUTHOR_RE.match(fid)
        resolved = ""
        if match:
            resolved = authtab_map.get(match.group(1), "")
        if not resolved and fid in disk_files:
            try:
                doc = ET.parse(str(tlg_dir / fid), ET.XMLParser(recover=True))
                resolved = format_author_name(extract_tei_author(doc))
            except Exception:
                resolved = ""
        if resolved:
            author_map[fid] = resolved

    author_map_path = output_dir / "tlg_author_map.json"
    author_map_path.write_text(
        json.dumps(author_map, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    generated_shards = sorted(p.name for p in output_dir.glob(f"{file_prefix}_*.json"))
    report: dict[str, Any] = {
        "manifests": manifests_meta,
        "tlg_dir": str(tlg_dir),
        "output_dir": str(output_dir),
        "total_selection_works": len(works),
        "matched_on_disk": len(works) - len(missing_non_numeric) - len(missing_numeric),
        "missing_non_numeric_count": len(missing_non_numeric),
        "missing_numeric_count": len(missing_numeric),
        "existing_tlg_works_count": len(existing_file_ids),
        "skipped_existing_overlap_count": len(skipped_existing_overlap),
        "converted_works_count": converted_works,
        "empty_after_parse_count": len(empty_after_parse),
        "total_blocks_written": total_blocks_written,
        "author_map_entries": len(author_map),
        "generated_shards": generated_shards,
        "missing_non_numeric_sample": missing_non_numeric[:20],
        "missing_numeric": missing_numeric,
        "empty_after_parse": empty_after_parse,
    }
    report_path = output_dir / "selection_import_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importa una o più selezioni `grc-lab/work-selection` dai TEI XML TLG per il sistema RAG."
    )
    parser.add_argument(
        "--selection",
        nargs="*",
        default=None,
        help=f"Uno o più file JSON grc-lab/work-selection (default: tutti i *.json in '{RAG_SELECTIONS_DIR}').",
    )
    parser.add_argument(
        "--tlg-dir",
        type=str,
        default=str(DEFAULT_TLG_BULK_DIR),
        help=f"Directory contenente i file XML tlg*.xml e authtab.xml (default: {DEFAULT_TLG_BULK_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAG_EXTRA_DATA_PATH),
        help=f"Directory di destinazione per shard RAG e metadati (default: {RAG_EXTRA_DATA_PATH}).",
    )
    parser.add_argument(
        "--existing-tlg-dir",
        type=str,
        default=str(DATA_PATH),
        help=f"Directory base contenente i tlg_*.json esistenti per deduplicazione (default: {DATA_PATH}).",
    )
    parser.add_argument(
        "--target-poetry-words",
        type=int,
        default=90,
        help="Numero target di parole per l'aggregazione di versi <l> contigui (default: 90).",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Se attivo, riconverte anche le opere già presenti in data/tlg_*.json.",
    )
    args = parser.parse_args()

    if args.selection:
        selection_paths = [Path(p) for p in args.selection]
    else:
        selection_paths = sorted(RAG_SELECTIONS_DIR.glob("*.json"))

    if not selection_paths:
        raise FileNotFoundError(
            f"Nessun file di selezione trovato in '{RAG_SELECTIONS_DIR}'. Specificare --selection."
        )

    report = import_work_selections(
        selection_paths=selection_paths,
        tlg_dir=Path(args.tlg_dir),
        output_dir=Path(args.output_dir),
        existing_tlg_dir=Path(args.existing_tlg_dir),
        target_poetry_words=args.target_poetry_words,
        include_existing=args.include_existing,
    )

    print(
        "[Work-Selection Importer] Completato:\n"
        f"  - Opere totali nelle selezioni: {report['total_selection_works']}\n"
        f"  - Trovate in tlg-bulk:          {report['matched_on_disk']}\n"
        f"  - Saltate (X01/x01 non su disco): {report['missing_non_numeric_count']}\n"
        f"  - Mancanti numeriche impreviste:  {report['missing_numeric_count']}\n"
        f"  - Già presenti in data/tlg_*:     {report['skipped_existing_overlap_count']}\n"
        f"  - Nuove opere convertite:         {report['converted_works_count']} "
        f"({report['total_blocks_written']} blocchi in {len(report['generated_shards'])} shard)\n"
        f"  - Mappa autori popolata:          {report['author_map_entries']} opere\n"
        f"  - Report salvato in:              {Path(args.output_dir) / 'selection_import_report.json'}"
    )


if __name__ == "__main__":
    main()
