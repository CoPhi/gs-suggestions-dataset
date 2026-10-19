"""
Script per pubblicare il modello Ithaca fine-tunato su Hugging Face Hub.

Crea o aggiorna il repository su Hugging Face:
1. Carica il checkpoint con i pesi JAX/Flax (checkpoint_tlg.pkl).
2. Carica la configurazione del modello (config.json).
3. Genera e carica una Model Card dettagliata (README.md) con metadati per Digital Classics,
   tabelle comparative di valutazione (se disponibili) e snippet di codice Python per l'inferenza.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()


def build_model_card(
    repo_id: str,
    base_model: str = "deepmind/ithaca",
    dataset_name: str = "CNR-ILC/gs-dataset-tlg-uncased",
    eval_json_path: str | None = None,
    hyperparameters: dict | None = None,
) -> str:
    """Genera il contenuto Markdown della Model Card con metadati YAML, iperparametri e benchmark."""
    table_md = ""
    eval_data: dict = {}
    if eval_json_path and os.path.exists(eval_json_path):
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            table_md = eval_data.get("markdown_table", "")
        except Exception:
            pass

    hp = hyperparameters or eval_data.get("hyperparameters") or {}

    card = f"""---
language:
- grc
license: apache-2.0
tags:
- ancient-greek
- text-restoration
- papyrology
- epigraphy
- jax
- flax
- ithaca
- greekschools
pipeline_tag: fill-mask
---

# {repo_id}: Fine-Tuned Ithaca Model for Ancient Greek Text Restoration

Questo modello è una versione fine-tunata di **Ithaca** (sviluppato da Google DeepMind, Assael et al., *Nature* 2022) addestrato specificamente sul corpus **TLG (Thesaurus Linguae Graecae)** per il progetto **GreekSchools**.

Mentre il modello Ithaca originale è stato addestrato esclusivamente su iscrizioni epigrafiche (corpus PHI), questo checkpoint è stato adattato al registro linguistico, sintattico e lessicale della **letteratura greca e dei trattati filosofici** (tipici dei papiri ercolanesi).

## Caratteristiche Principali
- **Architettura Character-Level**: modella il testo a livello di singoli caratteri anziché subword (BPE/WordPiece), eliminando il problema del disallineamento morfologico nelle terminazioni flessive (desinenze).
- **Transformer BigBird**: attenzione sparsa con supporto a finestre di contesto lunghe (fino a 768 caratteri).
- **Freezing Selettivo**: durante il fine-tuning i rami di attribuzione geografica e temporale (e i blocchi iniziali dell'encoder selezionati tramite NSGA-II) sono stati congelati, focalizzando i gradienti sul restauro testuale (*masked character prediction*).
- **Decodifica con HCB Beam Search**: impiega la correzione pseudo-likelihood HCB (*Hammersley-Clifford-Besag Infilling*) con strategia non-sequenziale Best-to-Worst, garantendo coerenza statistica e allineamento metodologico con i modelli BERT del framework.

## Dettagli di Addestramento
- **Base Model:** `{base_model}`
- **Corpus di Fine-Tuning:** `{dataset_name}`
- **Framework:** JAX / Flax Linen / Optax
- **Ottimizzazione Iperparametri:** Multi-Objective Genetic Algorithm **NSGA-II** (Optuna)
- **Convenzione lacune:** Formato Leiden standard `[...]` o trattini `----` (testo normalizzato privo di accenti e spiriti).

"""
    if hp:
        card += "## Iperparametri Ottimali (Frontiera di Pareto NSGA-II)\n\n"
        card += "| Iperparametro | Valore |\n|:---|:---:|\n"
        for k, v in sorted(hp.items()):
            card += f"| `{k}` | `{v}` |\n"
        card += "\n"

    if table_md:
        card += f"""## Risultati di Valutazione Comparativa (Pre-FT vs Post-FT)

La seguente tabella riporta i risultati della valutazione sul test set TLG, suddiviso per policy di lacuna (*default*, *word*, *suffix* per le terminazioni flessive):

{table_md}

"""

    card += f"""## Utilizzo tramite l'API di GreekSchools

Una volta scaricato o registrato nell'istanza dell'API, il modello può essere interrogato direttamente specificando il checkpoint:

```python
from models.ithaca.inference.predict import fill_mask_ithaca

# Contesto con lacuna in formato Leiden [...] o Ithaca [---]
context = "ΚΑΙ ΤΩΝ ΦΙΛΟΣΟΦ[..] ΕΝ ΤΗΙ ΠΟΛΕΙ"

suggestions = fill_mask_ithaca(
    text=context,
    checkpoint="{repo_id}",
    K=5
)

for cand, score in suggestions:
    print(f"Suggerimento: {{cand}} (Confidenza: {{score:.4f}})")
```

## Citazione

Se utilizzi questo modello nei tuoi studi di papirologia o filologia classica, cita il paper originale di Ithaca e il progetto GreekSchools:

```bibtex
@article{{assael2022restoring,
  title={{Restoring and attributing ancient texts using deep neural networks}},
  author={{Assael, Yannis and Sommerschield, Thea and Shillingford, Brendan and Bordbar, Mahyar and Pavlopoulos, John and Chatzipanagiotou, Marita and Androutsopoulos, Ion and Prag, Jonathan and de Freitas, Nando}},
  journal={{Nature}},
  volume={{603}},
  number={{7900}},
  pages={{280--283}},
  year={{2022}},
  publisher={{Nature Publishing Group}}
}}
```
"""
    return card


def publish_ithaca_to_hub(
    repo_id: str = "CNR-ILC/gs-ithaca-tlg",
    checkpoint_path: str = "checkpoints/ithaca/checkpoint_tlg.pkl",
    config_path: str = "checkpoints/ithaca/config.json",
    eval_json_path: str = "eval/results/eval_results_ithaca_comparison.json",
    hyperparameters: dict | None = None,
    private: bool = False,
) -> bool:
    """
    Carica su Hugging Face Hub il checkpoint fine-tunato di Ithaca (`checkpoint_tlg.pkl`),
    la configurazione (`config.json`) e la Model Card aggiornata (`README.md`).
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "[HuggingFace Hub] AVVISO: Variabile d'ambiente HF_TOKEN non impostata. "
            "Salto il caricamento automatico su Hugging Face Hub."
        )
        return False

    api = HfApi(token=token)

    print(f"\nVerifica/Creazione repository Hugging Face Hub: '{repo_id}'...")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        private=private,
        exist_ok=True,
    )

    if os.path.exists(checkpoint_path):
        print(f"Caricamento {checkpoint_path} -> checkpoint_tlg.pkl...")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="checkpoint_tlg.pkl",
            repo_id=repo_id,
            repo_type="model",
        )
    else:
        print(f"AVVISO: File checkpoint {checkpoint_path} non trovato!")

    if os.path.exists(config_path):
        print(f"Caricamento {config_path} -> config.json...")
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
        )

    print("Generazione e caricamento Model Card (README.md)...")
    readme_content = build_model_card(
        repo_id=repo_id,
        eval_json_path=eval_json_path,
        hyperparameters=hyperparameters,
    )
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nModello Ithaca pubblicato con successo su Hugging Face Hub!")
    print(f"URL: https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pubblica il modello Ithaca fine-tunato su Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="CNR-ILC/gs-ithaca-tlg",
        help="Repository ID di Hugging Face (es. 'CNR-ILC/gs-ithaca-tlg' o 'username/gs-ithaca-tlg')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/ithaca/checkpoint_tlg.pkl",
        help="Percorso al file pickle del checkpoint fine-tunato",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="checkpoints/ithaca/config.json",
        help="Percorso al file config.json",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        default="eval/results/eval_results_ithaca_comparison.json",
        help="File JSON contenente le metriche comparative",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Se impostato, crea il repository come privato",
    )
    args = parser.parse_args()

    success = publish_ithaca_to_hub(
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        eval_json_path=args.eval_json,
        private=args.private,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

