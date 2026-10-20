"""
Gestione dei checkpoint di Ithaca: download, ispezione, freezing delle teste
di attribuzione (data e luogo) e salvataggio dei pesi fine-tunati.
"""

from __future__ import annotations

import os
import sys
import pickle
import urllib.request
from pathlib import Path
from typing import Any

# Assicura inclusione del motore Ithaca se presente in packages/ithaca_engine
ithaca_engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/ithaca_engine")
)
if os.path.isdir(ithaca_engine_path) and ithaca_engine_path not in sys.path:
    sys.path.insert(0, ithaca_engine_path)

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


DEFAULT_ITHACA_CONFIG = {
    "vocab_char_size": 164,
    "vocab_word_size": 100004,
    "output_subregions": 85,
    "output_date": 160,
    "output_date_dist": True,
    "output_return_emb": False,
    "use_output_mlp": True,
    "num_heads": 8,
    "num_layers": 6,
    "word_char_emb_dim": 192,
    "emb_dim": 512,
    "qkv_dim": 512,
    "mlp_dim": 2048,
    "max_len": 768,
    "causal_mask": False,
    "feature_combine_type": "concat",
    "posemb_combine_type": "add",
    "region_date_pooling": "first",
    "learn_pos_emb": True,
    "use_bfloat16": False,
    "dropout_rate": 0.1,
    "attention_dropout_rate": 0.1,
    "activation_fn": "gelu",
    "model_type": "bigbird",
}


def load_ithaca_checkpoint(path: str) -> dict[str, Any]:
    """
    Carica il checkpoint serializzato (.pkl) di Ithaca e ne estrae componenti chiave:
    - params: dizionario dei pesi Flax Linen
    - alphabet: mapping caratteri e vocabolario parole (GreekAlphabet)
    - config: iperparametri del modello BigBird (da 'model_config', 'config' o default)
    - region_map: metadati geografici
    """
    if not os.path.exists(path):
        ensure_checkpoint_exists(path)

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Formato non valido per il checkpoint '{path}'")

    # 1. Configurazione modello: cerca 'model_config' (ufficiale DeepMind) o 'config'
    config_raw = data.get("model_config") or data.get("config")
    if config_raw is not None:
        config = dict(config_raw)
    else:
        config = DEFAULT_ITHACA_CONFIG.copy()

    # 2. Normalizzazione parametri Flax Linen (garantisce struttura 'params')
    raw_params = data.get("params", data)
    if isinstance(raw_params, dict) and "params" in raw_params and isinstance(raw_params["params"], dict):
        params = raw_params
    elif isinstance(raw_params, dict):
        params = {"params": raw_params}
    else:
        params = raw_params

    # 3. Istanza alfabeto greco
    alphabet_data = data.get("alphabet")
    try:
        from ithaca.util.alphabet import GreekAlphabet
    except ImportError:
        import sys
        sys.path.insert(0, "packages/ithaca_engine")
        try:
            from ithaca.util.alphabet import GreekAlphabet
        except ImportError:
            import numpy as np

            class GreekAlphabet:
                """Fallback autonomo di GreekAlphabet conforme alle specifiche di DeepMind Ithaca."""
                def __init__(self):
                    self.pad = '#'
                    self.sos = '<'
                    self.unk = '^'
                    self.space = ' '
                    self.missing = '-'
                    greek_chars = list('αβγδεζηθικλμνξοπρςστυφχψωϙϛ')
                    numerals = list('0')
                    punctuation = list('.')
                    self.idx2char = np.array(
                        [self.pad, self.sos, self.unk, self.space, self.missing] +
                        greek_chars + numerals + punctuation
                    )
                    self.char2idx = {c: i for i, c in enumerate(self.idx2char)}
                    self.idx2word = np.array([self.pad, self.sos, self.unk])
                    self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    alphabet = GreekAlphabet()
    if isinstance(alphabet_data, dict):
        if "idx2word" in alphabet_data:
            alphabet.idx2word = alphabet_data["idx2word"]
        if "word2idx" in alphabet_data:
            alphabet.word2idx = alphabet_data["word2idx"]
    elif alphabet_data is not None and hasattr(alphabet_data, "idx2word"):
        alphabet = alphabet_data

    return {
        "params": params,
        "alphabet": alphabet,
        "config": config,
        "model_config": config,
        "region_map": data.get("region_map", {}),
        "metadata": data.get("metadata", {}),
    }


def create_optimizer_with_freezing(
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    weight_decay: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    num_layers_to_freeze: int = 0,
    freeze_attribution_heads: bool = True,
    params: Any | None = None,
) -> optax.GradientTransformation:
    """
    Crea un ottimizzatore Optax (AdamW con decay cosine o linear e warmup).
    - Se freeze_attribution_heads=True, azzera i gradienti per le teste di attribuzione
      geografica (output_subregions) e temporale (output_date) + NSP (mlpblock_1).
    - Se num_layers_to_freeze > 0, congela anche gli embedding iniziali e i primi N
      blocchi dell'encoder BigBird (encoderblock_0 .. encoderblock_{N-1}).
    """
    import jax
    import optax

    safe_total_steps = max(int(total_steps), 2)
    safe_warmup_steps = min(max(int(warmup_steps), 0), safe_total_steps - 1)

    if lr_scheduler_type == "linear":
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=max(safe_warmup_steps, 1),
        )
        decay_fn = optax.linear_schedule(
            init_value=learning_rate,
            end_value=1e-7,
            transition_steps=max(safe_total_steps - safe_warmup_steps, 1),
        )
        if safe_warmup_steps > 0:
            schedule = optax.join_schedules(
                schedules=[warmup_fn, decay_fn],
                boundaries=[safe_warmup_steps],
            )
        else:
            schedule = decay_fn
    else:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=safe_warmup_steps,
            decay_steps=safe_total_steps,
            end_value=1e-7,
        )

    adamw = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    if not freeze_attribution_heads and num_layers_to_freeze <= 0:
        return adamw

    frozen_encoder_blocks = {
        f"encoderblock_{i}" for i in range(max(0, int(num_layers_to_freeze)))
    }

    def _get_labels(p):
        def _label_leaf(path, _):
            path_parts = [str(getattr(k, "key", k)).lower() for k in path]
            path_str = "/".join(path_parts)

            # 1. Congelamento teste di attribuzione (data, regione e NSP)
            if freeze_attribution_heads and any(
                head in path_str
                for head in [
                    "output_date",
                    "output_subregions",
                    "date_mlp",
                    "region_mlp",
                    "mlpblock_1",
                    "mlpblock_2",
                    "mlpblock_3",
                    "dense_1",
                    "dense_2",
                    "dense_3",
                ]
            ):
                return "frozen"

            # 2. Congelamento selettivo dei primi N blocchi BigBird e degli embedding iniziali
            if num_layers_to_freeze > 0:
                if any(
                    emb in path_str
                    for emb in ["char_embeddings", "word_embeddings", "posembed_input"]
                ):
                    return "frozen"
                if any(blk in path_parts for blk in frozen_encoder_blocks):
                    return "frozen"

            return "trainable"

        return jax.tree_util.tree_map_with_path(_label_leaf, p)

    param_labels = _get_labels(params) if params is not None else _get_labels

    return optax.multi_transform(
        {"trainable": adamw, "frozen": optax.set_to_zero()},
        param_labels,
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
        "model_config": config,
        "metadata": metadata or {},
    }

    with open(out_file, "wb") as f:
        pickle.dump(payload, f)

    print(f"Checkpoint fine-tunato salvato con successo in: {out_file}")
