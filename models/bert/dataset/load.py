"""
Pipeline unificata per la costruzione, preparazione e pubblicazione dei dataset
su Hugging Face Hub per il progetto GreekSchools.

Supporta la generazione e il push dei seguenti dataset:
  - 'train': Corpus di pre-addestramento MAAT (CNR-ILC/gs-dataset-train)
  - 'eval':  Benchmark di valutazione generale con gold labels (CNR-ILC/gs-dataset-eval)
  - 'herc':  Benchmark di valutazione specifico su Papiri di Ercolano (CNR-ILC/gs-dataset-herc)
  - 'tlg':   Corpus di pre-addestramento TLG cased e uncased (CNR-ILC/gs-dataset-tlg-*)
  - 'all':   Tutti i dataset sopra elencati

Esempi di utilizzo:
  # Verifica in locale (dry-run) del dataset Ercolano
  python -m models.bert.dataset.load --target herc

  # Pubblicazione su Hugging Face Hub del dataset Ercolano
  python -m models.bert.dataset.load --target herc --push

  # Generazione con range personalizzato di lacune (1-6 caratteri)
  python -m models.bert.dataset.load --target herc --min-gap 1 --max-gap 6 --push
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

from datasets import Dataset, DatasetDict, DatasetInfo
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

from backend.config.settings import DATA_PATH
from backend.core import UNK_TOKEN
from backend.core.cleaner import (
    load_abs,
    load_specific_domain_abs,
    load_test_set,
    split_abs_herc_dev,
)
from backend.core.preprocess import (
    normalize_greek,
    remove_punctuation,
)
from models.bert.dataset import (
    _CORPUS_DESCRIPTION,
    _EVAL_DESCRIPTION,
    _HERC_DESCRIPTION,
    _TLG_DESCRIPTION,
    CORPUS_CHECKPOINT,
    EVAL_CHECKPOINT,
    HERC_CHECKPOINT,
    TLG_CHECKPOINT,
)
from models.bert.dataset.dev_set import DevCase, build_dev_set
from models.bert.dataset.train_set import build_train_set
from models.bert.finetuning import (
    BERT_MAX_SEQ_LENGTH,
    CHUNK_SIZE,
    MIN_SENT_TOKEN_TRESHOLD,
    get_model_config,
)

# ==============================================================================
# 1. AUTENTICAZIONE E HUB UTILITIES
# ==============================================================================


def login_hf() -> None:
    """Esegue il login su Hugging Face Hub leggendo HF_TOKEN dal file .env."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("[Hub] Autenticazione Hugging Face completata con successo.")
    else:
        print(
            "[Hub] ATTENZIONE: Nessun HF_TOKEN trovato nel file .env. "
            "Assicurati di essere autenticato via `huggingface-cli login`."
        )


def push_to_hub(
    dataset: DatasetDict,
    checkpoint: str,
    message: str,
    description: str = "",
) -> None:
    """Pubblica *dataset* su HF *checkpoint* iniettando i metadati di provenienza nella card."""
    if description:
        info = DatasetInfo(description=description)
        for split in dataset:
            dataset[split].info.description = info.description
    print(f"[Hub] Caricamento su repository '{checkpoint}' in corso...")
    dataset.push_to_hub(checkpoint, commit_message=message)
    print(f"[Hub] Dataset pubblicato con successo: https://huggingface.co/datasets/{checkpoint}")


# ==============================================================================
# 2. CARICAMENTO E CONVERSIONE FORMATI
# ==============================================================================


def load_train_and_dev_set(test_size: float = 0.1) -> tuple[list, list]:
    """
    Carica i blocchi anonimi da data/ e li suddivide in train e dev set.
    Il dev set contiene esclusivamente blocchi P.Herc.

    Returns:
        (train_abs, dev_abs)
    """
    return split_abs_herc_dev(load_abs(), test_size)


def load_or_generate_test_abs() -> list[dict]:
    """
    Carica i blocchi anonimi del test set da data/test_abs.json.
    Se il file non esiste ma è presente test_abs.csv, lo genera automaticamente.
    """
    test_json_path = DATA_PATH / "test_abs.json"
    if not test_json_path.exists():
        csv_path = Path("test_abs.csv")
        if csv_path.exists():
            try:
                from scripts.valutation_abs_loader import dump_test_cases_into_json_abs
                print(f"[DataLoader] Generazione automatica di '{test_json_path}' da '{csv_path}'...")
                dump_test_cases_into_json_abs(str(test_json_path))
            except Exception as e:
                print(f"[DataLoader] Impossibile convertire {csv_path}: {e}")
    if test_json_path.exists():
        return load_test_set()
    print(f"[DataLoader] Nessun file di test set trovato in {test_json_path}.")
    return []


def dev_set_to_hf_dataset(dev_set: list[DevCase]) -> Dataset:
    """
    Converte una lista di DevCase in un Dataset Hugging Face strutturato.
    Le etichette (campo `y`) sono memorizzate come sequenze di stringhe.
    """
    records = [
        {
            "x": case.x,
            "y": case.y,
            "gap_length": case.gap_length,
            "corpus_id": case.corpus_id,
            "file_id": case.file_id,
            "gap_type": getattr(case, "gap_type", "default"),
        }
        for case in dev_set
    ]
    return Dataset.from_list(records)


# ==============================================================================
# 3. BUILDER SPECIFICI PER DATASET
# ==============================================================================


def build_corpus_train_dataset(test_size: float = 0.2) -> tuple[DatasetDict, str, str]:
    """
    Costruisce il training corpus grezzo (train/dev) da tutti i blocchi MAAT.
    """
    print("\n--- Costruzione Dataset: Pre-addestramento MAAT (Corpus Train) ---")
    train_abs, dev_abs = load_train_and_dev_set(test_size=test_size)
    print(f"Blocchi anonimi allocati: Train={len(train_abs)}, Dev={len(dev_abs)}")

    dataset = DatasetDict(
        {
            "train": build_train_set(train_abs),
            "dev": build_train_set(dev_abs),
        }
    )
    return dataset, CORPUS_CHECKPOINT, _CORPUS_DESCRIPTION


def build_general_eval_dataset(
    test_size: float = 0.2,
    min_gap: int = 1,
    max_gap: int | None = 6,
) -> tuple[DatasetDict, str, str]:
    """
    Costruisce il dataset di valutazione generale con gold labels reali:
    - Split 'dev': derivato dai blocchi P.Herc. del MAAT corpus
    - Split 'test': derivato dal test set controllato (test_abs.json)
    """
    print("\n--- Costruzione Dataset: Valutazione Generale (Eval Set) ---")
    _, dev_abs = load_train_and_dev_set(test_size=test_size)
    test_abs = load_or_generate_test_abs()

    dev_cases = build_dev_set(dev_abs, min_gap_length=min_gap, max_gap_length=max_gap)
    test_cases = build_dev_set(test_abs, min_gap_length=min_gap, max_gap_length=max_gap) if test_abs else []

    splits: dict[str, Dataset] = {"dev": dev_set_to_hf_dataset(dev_cases)}
    if test_cases:
        splits["test"] = dev_set_to_hf_dataset(test_cases)

    return DatasetDict(splits), EVAL_CHECKPOINT, _EVAL_DESCRIPTION


def build_herc_eval_dataset(
    min_gap: int = 1,
    max_gap: int | None = 6,
) -> tuple[DatasetDict, str, str]:
    """
    Costruisce il benchmark di valutazione dedicato ai Papiri di Ercolano (gs-dataset-herc):
    - Split 'dev': casi reali estratti da tutti i blocchi MAAT con "P.Herc." nel titolo
    - Split 'test': casi controllati e verificati da esperti filologi (da test_abs.json / test_abs.csv)
    """
    print("\n--- Costruzione Dataset: Benchmark Papiri di Ercolano (gs-dataset-herc) ---")
    print("1. Caricamento blocchi anonimi da data/...")
    all_abs = load_abs()

    print("2. Filtraggio blocchi con sottostringa 'P.Herc.' nel campo 'title'...")
    herc_abs = load_specific_domain_abs(all_abs, domain_title="P.Herc.")
    print(f"   Trovati {len(herc_abs)} blocchi anonimi appartenenti a P.Herc.")

    print(f"3. Estrazione DevCase per split 'dev' (gap_length: {min_gap}-{max_gap or 'inf'})...")
    dev_cases = build_dev_set(herc_abs, min_gap_length=min_gap, max_gap_length=max_gap)
    print(f"   Generati {len(dev_cases)} casi di sviluppo da MAAT P.Herc.")

    print("4. Caricamento casi controllati per split 'test' (test_abs.json)...")
    test_abs = load_or_generate_test_abs()
    test_cases = build_dev_set(test_abs, min_gap_length=min_gap, max_gap_length=max_gap) if test_abs else []
    print(f"   Generati {len(test_cases)} casi di test controllati.")

    splits: dict[str, Dataset] = {"dev": dev_set_to_hf_dataset(dev_cases)}
    if test_cases:
        splits["test"] = dev_set_to_hf_dataset(test_cases)

    return DatasetDict(splits), HERC_CHECKPOINT, _HERC_DESCRIPTION


def build_tlg_datasets() -> list[tuple[DatasetDict, str, str]]:
    """
    Costruisce i training set dal corpus esclusivo TLG nelle varianti uncased e cased.
    """
    print("\n--- Costruzione Dataset: Pre-addestramento TLG (Thesaurus Linguae Graecae) ---")
    tlg_abs = [ab for ab in load_abs(corpus_set=["tlg"]) if ab.get("corpus_id") == "tlg"]
    print(f"Blocchi anonimi TLG estratti: {len(tlg_abs)}")

    ds_uncased = DatasetDict({"train": build_train_set(tlg_abs, case_folding="none")})
    ds_cased = DatasetDict({"train": build_train_set(tlg_abs, case_folding="upper")})

    return [
        (ds_uncased, f"{TLG_CHECKPOINT}-uncased", _TLG_DESCRIPTION),
        (ds_cased, f"{TLG_CHECKPOINT}-cased", _TLG_DESCRIPTION),
    ]


# ==============================================================================
# 4. CHUNKING E PREPARAZIONE MODEL-SPECIFIC (PER INFERENCE / FINE-TUNING)
# ==============================================================================


def chunk_sentences(sentences: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Divide le frasi in blocchi di esattamente *chunk_size* word token."""
    return [
        " ".join(chunk)
        for sentence in sentences
        for i in range(0, len(sentence.split()), chunk_size)
        if len(chunk := sentence.split()[i : i + chunk_size]) == chunk_size
    ]


def chunk_for_bert(
    sentences: list[str],
    tokenizer,
    max_length: int = BERT_MAX_SEQ_LENGTH,
) -> list[str]:
    """Aggrega frasi rispettando il limite di sub-word token di BERT."""
    chunks: list[str] = []
    current_tokens: list[str] = []
    current_length = 0

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)
        n = len(sent_tokens)

        if n > max_length:
            if current_tokens:
                chunks.append(tokenizer.convert_tokens_to_string(current_tokens))
                current_tokens, current_length = [], 0
            chunks.append(tokenizer.convert_tokens_to_string(sent_tokens[:max_length]))
            continue

        if current_length + n > max_length:
            chunks.append(tokenizer.convert_tokens_to_string(current_tokens))
            current_tokens, current_length = sent_tokens, n
        else:
            current_tokens.extend(sent_tokens)
            current_length += n

    if current_tokens:
        chunks.append(tokenizer.convert_tokens_to_string(current_tokens))

    return chunks


def _quality_filter_subword(
    example: dict,
    tokenizer,
    unk_ratio_threshold: float = 0.1,
) -> bool:
    """Filtra su sub-word token reali del tokenizer specifico."""
    tokens = tokenizer.tokenize(example["text"])
    if len(tokens) < MIN_SENT_TOKEN_TRESHOLD:
        return False
    return tokens.count(tokenizer.unk_token) / max(len(tokens), 1) < unk_ratio_threshold


def _normalize_example(example: dict, config: dict, unk_token: str) -> dict:
    text = normalize_greek(
        example["text"],
        case_folding=config.get("case_folding", "upper"),
        strip_diacritics_flag=config.get("strip_diacritics", True),
    )
    if config.get("remove_punct", False):
        text = remove_punctuation(text)

    cf = config.get("case_folding", "upper")
    if cf == "lower":
        agnostic_unk = UNK_TOKEN.lower()
    elif cf == "upper":
        agnostic_unk = UNK_TOKEN.upper()
    elif cf == "fold":
        agnostic_unk = UNK_TOKEN.casefold()
    else:
        agnostic_unk = UNK_TOKEN

    text = text.replace(agnostic_unk, unk_token)
    return {"text": text}


def _tokenize_example(example: dict, tokenizer) -> dict:
    return tokenizer(example["text"], truncation=False, padding=False)


def prepare_dataset_for_model(
    raw_dataset: Dataset,
    checkpoint: str,
    num_proc: int = 4,
) -> Dataset:
    """
    Pipeline model-specific: normalizza → filtra → tokenizza.
    """
    config = get_model_config(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = int(1e9)

    return (
        raw_dataset.map(
            partial(_normalize_example, config=config, unk_token=tokenizer.unk_token),
            desc=f"Normalizing [{checkpoint}]",
            num_proc=num_proc,
        )
        .filter(
            partial(_quality_filter_subword, tokenizer=tokenizer),
            desc=f"Filtering [{checkpoint}]",
            num_proc=num_proc,
        )
        .map(
            partial(_tokenize_example, tokenizer=tokenizer),
            batched=True,
            desc=f"Tokenizing [{checkpoint}]",
            num_proc=num_proc,
        )
    )


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================


def _print_dataset_summary(dataset: DatasetDict, name: str) -> None:
    """Stampa un riepilogo leggibile del dataset generato."""
    print(f"\n[Riepilogo Dataset: {name}]")
    for split_name, ds in dataset.items():
        print(f"  - Split '{split_name}': {len(ds)} record, colonne: {ds.column_names}")
        if len(ds) > 0:
            first_row = ds[0]
            preview = {
                k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v)
                for k, v in first_row.items()
            }
            print(f"    Esempio record [0]: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gestore unificato per la composizione e pubblicazione dei dataset su Hugging Face Hub."
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="herc",
        choices=["herc", "eval", "train", "tlg", "all"],
        help="Dataset target da comporre/pubblicare (default: herc)",
    )
    parser.add_argument(
        "--push",
        "--push-to-hub",
        action="store_true",
        help="Se specificato, carica il dataset su Hugging Face Hub (altrimenti esegue solo un dry-run locale)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Override opzionale dell'ID del repository su Hugging Face Hub",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=1,
        help="Lunghezza minima della lacuna in caratteri per i dataset di valutazione (default: 1)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=6,
        help="Lunghezza massima della lacuna in caratteri per i dataset di valutazione (default: 6)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Dimensione della quota di test/dev per lo split del corpus di train (default: 0.2)",
    )

    args = parser.parse_args()

    if args.push:
        login_hf()

    items_to_process: list[tuple[DatasetDict, str, str, str]] = []

    targets = ["herc", "eval", "train", "tlg"] if args.target == "all" else [args.target]

    for tgt in targets:
        if tgt == "herc":
            ds, default_repo, desc = build_herc_eval_dataset(
                min_gap=args.min_gap,
                max_gap=args.max_gap,
            )
            repo = args.repo_id or default_repo
            items_to_process.append((ds, repo, "Upload Herculaneum evaluation benchmark (gs-dataset-herc)", desc))

        elif tgt == "eval":
            ds, default_repo, desc = build_general_eval_dataset(
                test_size=args.test_size,
                min_gap=args.min_gap,
                max_gap=args.max_gap,
            )
            repo = args.repo_id or default_repo
            items_to_process.append((ds, repo, "Add evaluation dataset with gold labels", desc))

        elif tgt == "train":
            ds, default_repo, desc = build_corpus_train_dataset(test_size=args.test_size)
            repo = args.repo_id or default_repo
            items_to_process.append((ds, repo, "Add raw training corpus (train+dev split)", desc))

        elif tgt == "tlg":
            tlg_list = build_tlg_datasets()
            for ds, default_repo, desc in tlg_list:
                repo = args.repo_id or default_repo
                items_to_process.append((ds, repo, f"Add TLG raw training corpus [{repo}]", desc))

    for ds, repo, msg, desc in items_to_process:
        _print_dataset_summary(ds, repo)

        if args.push:
            push_to_hub(
                dataset=ds,
                checkpoint=repo,
                message=msg,
                description=desc,
            )
        else:
            print(
                f"\n[DRY RUN completato per {repo}] "
                f"Specifica l'argomento `--push` per effettuare il caricamento effettivo su Hugging Face Hub."
            )


if __name__ == "__main__":
    main()
