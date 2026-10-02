"""
Gestione dei checkpoint di Ithaca: download, ispezione, freezing delle teste
di attribuzione (data e luogo) e salvataggio dei pesi fine-tunati.
"""

from __future__ import annotations

import os
import pickle
import urllib.request
from pathlib import Path
from typing import Any

import optax

CHECKPOINT_URL = (
    "https://storage.googleapis.com/ithaca-resources/models/checkpoint_v1.pkl"
)
DEFAULT_BASE_CHECKPOINT = "checkpoints/ithaca/checkpoint_v1.pkl"


def ensure_checkpoint_exists(
    checkpoint_path: str = DEFAULT_BASE_CHECKPOINT,
) -> str:
    """
    Verifica che il checkpoint esista; in caso contrario, lo scarica da Google Cloud Storage.
    """
    path = Path(checkpoint_path)
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Download checkpoint ufficiale Ithaca da {CHECKPOINT_URL}...")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = block_num * block_size / total_size * 100
            print(f"\rDownload in corso: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(CHECKPOINT_URL, str(path), reporthook=_progress)
    print(f"\nDownload completato: salvato in {path}")
    return str(path)


def load_ithaca_checkpoint(path: str) -> dict[str, Any]:
    """
    Carica il checkpoint serializzato (.pkl) di Ithaca e ne estrae componenti chiave:
    - params: dizionario dei pesi Flax Linen
    - alphabet: mapping caratteri e vocabolario parole
    - config: iperparametri del modello BigBird
    - region_map: metadati geografici
    """
    if not os.path.exists(path):
        ensure_checkpoint_exists(path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Formato non valido per il checkpoint '{path}'")

    return data


def create_optimizer_with_freezing(
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    weight_decay: float = 1e-4,
    freeze_attribution_heads: bool = True,
) -> optax.GradientTransformation:
    """
    Crea un ottimizzatore Optax (AdamW con cosine decay e warmup).
    Se freeze_attribution_heads=True, azzera completamente i gradienti per i layer
    di classificazione geografica (output_subregions) e datazione (output_date),
    concentrando l'intero gradiente sul Transformer torso e sulla testa di restauro (output_char).
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-7,
    )
    adamw = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    if not freeze_attribution_heads:
        return adamw

    # Funzione di partizionamento dei parametri: 'frozen' vs 'trainable'
    def param_partition_fn(param_path, _):
        path_str = "/".join(str(p) for p in param_path).lower()
        # Blocchiamo esplicitamente le teste di attribuzione data e regione
        if any(head in path_str for head in ["output_date", "output_subregions", "date_mlp", "region_mlp"]):
            return "frozen"
        return "trainable"

    return optax.multi_transform(
        {"trainable": adamw, "frozen": optax.set_to_zero()},
        param_partition_fn,
    )


def save_finetuned_checkpoint(
    output_path: str,
    params: Any,
    alphabet: Any,
    config: dict,
    metadata: dict | None = None,
) -> None:
    """
    Salva il modello fine-tunato preservando la compatibilità con il formato di Ithaca
    e allegando i metadati dell'esperimento (epoche, dataset, loss finale).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "params": params,
        "alphabet": alphabet,
        "config": config,
        "metadata": metadata or {},
    }

    with open(out_file, "wb") as f:
        pickle.dump(payload, f)

    print(f"Checkpoint fine-tunato salvato con successo in: {out_file}")
