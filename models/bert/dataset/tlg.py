"""
Script per creare un dataset di training esclusivamente dai testi TLG (Thesaurus Linguae Graecae).
"""

from __future__ import annotations

import argparse
import os

from datasets import DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

from backend.core.cleaner import load_abs
from models.bert.dataset.load import push_to_hub
from models.bert.dataset.train_set import build_train_set

TLG_CHECKPOINT = "CNR-ILC/gs-dataset-tlg"
TLG_DESCRIPTION = """\
Corpus di greco antico esclusivo del TLG (Thesaurus Linguae Graecae) per il 
pre-addestramento (MLM) di modelli BERT.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea un dataset di training usando solo i testi TLG."
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Carica su HF Hub")
    args = parser.parse_args()

    if args.push_to_hub:
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token)
        else:
            print(
                "ATTENZIONE: Nessun HF_TOKEN trovato nel file .env, impossibile fare il login all'Hub."
            )

    tlg_abs = [
        ab for ab in load_abs(corpus_set=["tlg"]) if ab.get("corpus_id") == "tlg"
    ]
    if not tlg_abs:
        return

    train_dataset_none = DatasetDict(
        {
            "train": build_train_set(tlg_abs, case_folding="none"),
        }
    )

    train_dataset_upper = DatasetDict(
        {
            "train": build_train_set(tlg_abs, case_folding="upper"),
        }
    )

    if args.push_to_hub:
        push_to_hub(
            dataset=train_dataset_none,
            checkpoint=f"{TLG_CHECKPOINT}-uncased",
            message="Add TLG-only raw training corpus (case_folding=none)",
            description=TLG_DESCRIPTION,
        )
        push_to_hub(
            dataset=train_dataset_upper,
            checkpoint=f"{TLG_CHECKPOINT}-cased",
            message="Add TLG-only raw training corpus (case_folding=upper)",
            description=TLG_DESCRIPTION,
        )


if __name__ == "__main__":
    main()
