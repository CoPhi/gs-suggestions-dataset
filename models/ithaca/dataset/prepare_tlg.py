"""
Pipeline di normalizzazione e preparazione del dataset TLG per Ithaca.

Converte il testo del TLG nel formato epigrafico atteso da Ithaca:
1. All-caps (maiuscolo) e rimozione completa dei segni diacritici (accenti, spiriti, iota sottoscritto).
2. Segmentazione in chunk di lunghezza <= 768 caratteri (finestra di contesto di BigBird/Ithaca).
3. Generazione controllata delle lacune nel formato di Ithaca ([----]) stratificata per policy:
   - 'default': lacuna casuale di 1-6 caratteri all'interno di una parola.
   - 'word': intera parola mascherata.
   - 'suffix': terminazione flessiva / desinenza finale della parola (1-6 caratteri).
4. Esportazione in formato JSONL (train, validation, test) compatibile sia con il training
   di Ithaca che con la suite di valutazione di GreekSchools.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Literal

from datasets import load_dataset

from backend.core.preprocess import normalize_greek

# Caratteri ammessi nell'alfabeto epigrafico standard di Ithaca (maiuscolo + numeri + spazi + punteggiatura base)
ITHACA_ALLOWED_CHARS = set(
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    "0123456789"
    " .,;:·-"
)


def clean_to_ithaca_alphabet(text: str) -> str:
    """
    Rimuove caratteri incompatibili con l'alfabeto di Ithaca,
    mantenendo lettere greche maiuscole, spazi e marcatori.
    """
    # Conserviamo i caratteri speciali per le lacune come '[', ']', '-', '.'
    allowed = ITHACA_ALLOWED_CHARS.union({"[", "]", "-", "."})
    cleaned = "".join(c if c in allowed else " " for c in text)
    # Normalizza spazi multipli in spazio singolo
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """
    Divide un testo lungo in chunk di dimensione massima `max_chars`,
    spezzando preferibilmente sui confini di parola o punteggiatura.
    """
    words = text.split(" ")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for word in words:
        if not word:
            continue
        word_len = len(word)
        if current_len + word_len + 1 > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = word_len
        else:
            current_chunk.append(word)
            current_len += word_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if len(c) >= 30]


def inject_lacuna(
    text: str,
    policy: Literal["default", "word", "suffix"] = "default",
    min_gap: int = 1,
    max_gap: int = 6,
    rng: random.Random | None = None,
) -> dict | None:
    """
    Inserisce una lacuna sintetica all'interno del testo secondo la policy specificata:
    - default: sottostringa casuale di 1-6 caratteri in una parola.
    - word: parola intera (tra min_gap e max_gap caratteri).
    - suffix: terminazione flessiva / desinenza finale della parola (1-6 caratteri).

    Restituisce un dizionario con testo mascherato (Ithaca [----] e Leiden [....]) e gold label.
    """
    if rng is None:
        rng = random.Random()

    # Trova tutte le parole composte da sole lettere greche maiuscole
    matches = list(re.finditer(r"[Α-Ω]{2,}", text))
    if not matches:
        return None

    # Filtra parole adatte alla policy
    valid_matches = []
    for m in matches:
        w_len = len(m.group())
        if policy == "word":
            if min_gap <= w_len <= max_gap:
                valid_matches.append(m)
        elif policy == "suffix":
            if w_len > min_gap:
                valid_matches.append(m)
        else:
            if w_len >= min_gap:
                valid_matches.append(m)

    if not valid_matches:
        return None

    match = rng.choice(valid_matches)
    word = match.group()
    w_len = len(word)

    if policy == "word":
        gap_len = w_len
        start_in_word = 0
    elif policy == "suffix":
        gap_len = rng.randint(min_gap, min(max_gap, w_len - 1))
        start_in_word = w_len - gap_len
    else:  # default
        gap_len = rng.randint(min_gap, min(max_gap, w_len))
        start_in_word = rng.randint(0, w_len - gap_len)

    target_fragment = word[start_in_word : start_in_word + gap_len]

    # Formattazione Ithaca: trattini racchiusi da quadre [----]
    placeholder_ithaca = f"[{'-' * gap_len}]"
    masked_word_ithaca = (
        word[:start_in_word] + placeholder_ithaca + word[start_in_word + gap_len :]
    )
    text_ithaca = text[: match.start()] + masked_word_ithaca + text[match.end() :]

    # Formattazione Leiden / GreekSchools: puntini racchiusi da quadre [....]
    placeholder_leiden = f"[{'.' * gap_len}]"
    masked_word_leiden = (
        word[:start_in_word] + placeholder_leiden + word[start_in_word + gap_len :]
    )
    text_leiden = text[: match.start()] + masked_word_leiden + text[match.end() :]

    return {
        "text_original": text,
        "text_ithaca": text_ithaca,
        "text_leiden": text_leiden,
        "gold_target": target_fragment,
        "gap_length": gap_len,
        "policy": policy,
        "start_char": match.start() + start_in_word,
    }


def prepare_tlg_dataset(
    dataset_name: str = "CNR-ILC/gs-dataset-tlg-uncased",
    output_dir: str = "data/ithaca",
    max_samples: int | None = None,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """
    Esegue l'intera pipeline di caricamento, normalizzazione epigrafica,
    chunking e generazione delle lacune multi-policy per Ithaca.
    """
    rng = random.Random(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Caricamento dataset TLG da HuggingFace: '{dataset_name}'...")
    ds = load_dataset(dataset_name)
    train_split = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]

    raw_texts = train_split["text"]
    if max_samples and len(raw_texts) > max_samples:
        raw_texts = raw_texts[:max_samples]

    print(f"Normalizzazione di {len(raw_texts)} testi in greco maiuscolo epigrafico...")
    all_chunks: list[str] = []
    for text in raw_texts:
        if not text or len(text.strip()) < 20:
            continue
        # 1. Normalizzazione maiuscola e rimozione diacritici
        norm_text = normalize_greek(text, case_folding="upper", strip_diacritics_flag=True)
        # 2. Pulizia secondo l'alfabeto di Ithaca
        clean_text = clean_to_ithaca_alphabet(norm_text)
        # 3. Chunking a <= 700 caratteri
        chunks = chunk_text(clean_text, max_chars=700)
        all_chunks.extend(chunks)

    rng.shuffle(all_chunks)
    print(f"Generati {len(all_chunks)} segmenti testuali validi.")

    # Creazione dei casi con lacune sintetiche stratificate
    cases: list[dict] = []
    policies = ["default", "word", "suffix"]

    for idx, chunk in enumerate(all_chunks):
        # Ruota le policy per un bilanciamento uniforme
        selected_policy = policies[idx % len(policies)]
        case = inject_lacuna(chunk, policy=selected_policy, rng=rng)
        if case:
            case["id"] = f"tlg_ithaca_{len(cases):06d}"
            cases.append(case)

    print(f"Generati {len(cases)} casi di test/train completi con lacune sintetiche.")

    # Split deterministico
    n_total = len(cases)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_val - n_test

    train_cases = cases[:n_train]
    val_cases = cases[n_train : n_train + n_val]
    test_cases = cases[n_train + n_val :]

    splits = {
        "train": (train_cases, out_path / "tlg_train.jsonl"),
        "val": (val_cases, out_path / "tlg_val.jsonl"),
        "test": (test_cases, out_path / "tlg_test.jsonl"),
    }

    counts = {}
    for split_name, (data_list, file_path) in splits.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in data_list)
        counts[split_name] = len(data_list)
        print(f"Salvato split '{split_name}': {len(data_list)} casi in {file_path}")

    # Salva anche un file di riepilogo metadati
    summary = {
        "dataset_name": dataset_name,
        "total_cases": n_total,
        "splits": counts,
        "policies": policies,
        "format": "Ithaca [----] and Leiden [....]",
    }
    with open(out_path / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepara il dataset TLG nel formato epigrafico per il fine-tuning di Ithaca"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CNR-ILC/gs-dataset-tlg-uncased",
        help="Dataset HuggingFace o percorso locale dei testi TLG",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ithaca",
        help="Cartella di destinazione dei file JSONL",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Numero massimo di testi sorgente da processare",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Frazione per il validation set (default: 0.1)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Frazione per il test set (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed per la riproducibilità",
    )
    args = parser.parse_args()

    prepare_tlg_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
