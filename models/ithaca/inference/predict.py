"""
Inference engine per Ithaca per il supporto alle predizioni e all'API di GreekSchools.

Fornisce la funzione `fill_mask_ithaca`:
1. Traduce la convenzione di lacuna Leiden [...] nella convenzione di Ithaca [---].
2. Normalizza il testo in maiuscolo epigrafico senza diacritici.
3. Carica il checkpoint (locale o da Hugging Face Hub) con caching in memoria.
4. Esegue l'inferenza probabilistica sui caratteri mancanti con beam search.
5. Restituisce una lista di suggerimenti ordinati per score decrescente.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# Assicura inclusione del motore Ithaca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
ithaca_engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/ithaca_engine")
)
if os.path.isdir(ithaca_engine_path) and ithaca_engine_path not in sys.path:
    sys.path.insert(0, ithaca_engine_path)

import jax
import jax.numpy as jnp
import numpy as np

from backend.core.preprocess import normalize_greek

# Cache per i modelli caricati: {checkpoint_path: (model, params, alphabet, config, jitted_forward)}
_LOADED_MODELS: dict[str, Any] = {}


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """Risolve il percorso: se è un repo Hugging Face, scarica il pickle da Hub."""
    if os.path.exists(checkpoint):
        return checkpoint

    # Se sembra un identificativo di repo Hugging Face (es. 'CNR-ILC/gs-ithaca-tlg')
    if "/" in checkpoint and not checkpoint.startswith("."):
        try:
            from huggingface_hub import hf_hub_download

            print(f"Download checkpoint Ithaca da Hugging Face Hub: '{checkpoint}'...")
            downloaded_path = hf_hub_download(
                repo_id=checkpoint,
                filename="checkpoint_tlg.pkl",
            )
            return downloaded_path
        except Exception as e:
            # Fallback a checkpoint_v1.pkl su Hub
            try:
                from huggingface_hub import hf_hub_download

                return hf_hub_download(repo_id=checkpoint, filename="checkpoint_v1.pkl")
            except Exception:
                raise FileNotFoundError(
                    f"Impossibile trovare o scaricare il checkpoint Ithaca '{checkpoint}': {e}"
                )

    # Fallback predefinito se locale
    default_local = Path("checkpoints/ithaca/checkpoint_tlg.pkl")
    if default_local.exists():
        return str(default_local)

    base_local = Path("checkpoints/ithaca/checkpoint_v1.pkl")
    if base_local.exists():
        return str(base_local)

    raise FileNotFoundError(
        f"Checkpoint Ithaca non trovato né localmente né su Hugging Face: {checkpoint}"
    )


def get_or_load_ithaca(checkpoint: str) -> tuple:
    """Carica e compila il modello Ithaca con memorizzazione nella cache."""
    resolved = _resolve_checkpoint_path(checkpoint)

    if resolved in _LOADED_MODELS:
        return _LOADED_MODELS[resolved]

    with open(resolved, "rb") as f:
        ckpt_data = pickle.load(f)

    params = ckpt_data["params"]
    alphabet = ckpt_data["alphabet"]
    config = ckpt_data["config"]

    try:
        from ithaca.models.model import Model
    except ImportError:
        sys.path.insert(0, "packages/ithaca_engine")
        from ithaca.models.model import Model

    model = Model(**config)

    @jax.jit
    def forward_fn(p, text_char, text_word):
        outputs = model.apply({"params": p}, text_char=text_char, text_word=text_word)
        return jax.nn.softmax(outputs["char"], axis=-1)

    _LOADED_MODELS[resolved] = (model, params, alphabet, config, forward_fn)
    return _LOADED_MODELS[resolved]


def decode_gap_beam_search(
    char_probs: np.ndarray,
    mask_indices: list[int],
    alphabet: Any,
    K: int = 20,
    beam_width: int = 50,
) -> list[tuple[str, float]]:
    """
    Esegue un beam search lungo i caratteri mascherati nella lacuna.
    Restituisce le prime K sequenze di caratteri più probabili con le rispettive confidenze.
    """
    idx2char = alphabet.idx2char

    # Inizializzazione beam: lista di tuple (stringa, log_prob)
    beam: list[tuple[str, float]] = [("", 0.0)]

    for pos in mask_indices:
        probs = char_probs[pos]  # [vocab_size]
        new_beam = []

        # Esploriamo solo i caratteri greci validi (saltando pad '#' e unk '^')
        top_char_indices = np.argsort(probs)[::-1][:beam_width]

        for cand_str, cand_log_p in beam:
            for c_idx in top_char_indices:
                char_str = str(idx2char[c_idx])
                if char_str in ("#", "^", "<", ">", "[", "]", "-"):
                    continue  # saltiamo token di controllo
                p_val = max(float(probs[c_idx]), 1e-12)
                new_beam.append((cand_str + char_str, cand_log_p + np.log(p_val)))

        # Teniamo i migliori candidati intermedi
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]

    # Normalizzazione esponenziale dei punteggi dei Top-K
    top_k_candidates = beam[:K]
    if not top_k_candidates:
        return []

    max_log_p = top_k_candidates[0][1]
    raw_scores = [np.exp(log_p - max_log_p) for _, log_p in top_k_candidates]
    sum_scores = sum(raw_scores) + 1e-12

    return [
        (cand_str, float(raw_scores[i] / sum_scores))
        for i, (cand_str, _) in enumerate(top_k_candidates)
    ]


def fill_mask_ithaca(
    text: str,
    checkpoint: str = "checkpoints/ithaca/checkpoint_tlg.pkl",
    K: int = 20,
    beam_size: int = 50,
) -> list[tuple[str, float]]:
    """
    Funzione di interfaccia principale compatibile con l'API di GreekSchools:
    Accetta un contesto con lacuna (in formato Leiden [...] o Ithaca [---]),
    esegue la predizione con Ithaca e restituisce una lista ordinata di (suggerimento, score).
    """
    # 1. Rileva e converte la lacuna in formato Ithaca [---]
    match = re.search(r"\[(\.+|\-+)\]", text)
    if not match:
        raise ValueError("Il testo non contiene alcuna lacuna valida nel formato [...] o [---]")

    gap_str = match.group(1)
    gap_len = len(gap_str)
    ithaca_placeholder = f"[{'-' * gap_len}]"

    # Sostituiamo la prima lacuna con la convenzione a trattini
    text_ithaca = text[: match.start()] + ithaca_placeholder + text[match.end() :]

    # 2. Normalizzazione testo (maiuscolo, senza diacritici)
    norm_text = normalize_greek(text_ithaca, case_folding="upper", strip_diacritics_flag=True)

    # 3. Caricamento modello e vocabolario
    _, params, alphabet, config, forward_fn = get_or_load_ithaca(checkpoint)
    max_len = config.get("max_len", 1024)

    # 4. Tokenizzazione input per JAX
    char2idx = getattr(alphabet, "char2idx", None) or {
        c: i for i, c in enumerate(alphabet.idx2char)
    }
    word2idx = getattr(alphabet, "word2idx", None) or {
        w: i for i, w in enumerate(alphabet.idx2word)
    }

    pad_char_id = char2idx.get(alphabet.pad, 0)
    unk_char_id = char2idx.get(alphabet.unk, 1)
    pad_word_id = word2idx.get(alphabet.pad, 0)
    unk_word_id = word2idx.get(alphabet.unk, 1)

    chars = list(norm_text)[:max_len]
    char_ids = np.full((1, max_len), pad_char_id, dtype=np.int32)
    word_ids = np.full((1, max_len), pad_word_id, dtype=np.int32)

    mask_indices = []
    in_gap = False
    for i, c in enumerate(chars):
        char_ids[0, i] = char2idx.get(c, unk_char_id)
        if c == "[":
            in_gap = True
        elif c == "]":
            in_gap = False
        elif in_gap and c == "-":
            mask_indices.append(i)

    words = norm_text.split(" ")
    curr_char = 0
    for w in words:
        w_id = word2idx.get(w, unk_word_id)
        w_len = len(w)
        for offset in range(w_len):
            if curr_char + offset < max_len:
                word_ids[0, curr_char + offset] = w_id
        curr_char += w_len + 1

    # 5. Forward pass JAX
    char_probs = forward_fn(params, jnp.array(char_ids), jnp.array(word_ids))
    char_probs_np = np.array(char_probs[0])  # [max_len, vocab_size]

    # 6. Decodifica beam search
    if not mask_indices:
        return []

    suggestions = decode_gap_beam_search(
        char_probs=char_probs_np,
        mask_indices=mask_indices,
        alphabet=alphabet,
        K=K,
        beam_width=beam_size,
    )

    return suggestions
