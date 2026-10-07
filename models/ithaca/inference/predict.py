"""
Inference engine per Ithaca per il supporto alle predizioni e all'API di GreekSchools.

Fornisce la funzione `fill_mask_ithaca`:
1. Traduce la convenzione di lacuna Leiden [...] nella sequenza di trattini attesa da Ithaca.
2. Normalizza il testo in minuscolo senza diacritici (conforme all'alfabeto di BigBird/Ithaca).
3. Aggiunge il prefisso Start-of-Sequence ('<') e calcola gli embedding di parola allineati via regex.
4. Esegue la decodifica iterativa condizionata usando HCB Beam Search (Hammersley-Clifford-Besag
   Infilling di `packages/hcb_infilling`) con strategia non-sequenziale Best-to-Worst.
5. Restituisce una lista di suggerimenti ordinati per score probabilistico decrescente.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# Assicura inclusione del progetto e del motore Ithaca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
ithaca_engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/ithaca_engine")
)
if os.path.isdir(ithaca_engine_path) and ithaca_engine_path not in sys.path:
    sys.path.insert(0, ithaca_engine_path)

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

    import jax
    import jax.numpy as jnp
    from models.ithaca.finetuning.checkpoint import load_ithaca_checkpoint

    ckpt_data = load_ithaca_checkpoint(resolved)
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
        variables = p if (isinstance(p, dict) and "params" in p) else {"params": p}
        outputs = model.apply(
            variables,
            text_char=text_char,
            text_word=text_word,
            is_training=False,
        )
        if isinstance(outputs, (tuple, list)):
            if len(outputs) > 0 and isinstance(outputs[0], (tuple, list)):
                logits_char = outputs[0][2]
            else:
                logits_char = outputs[2]
        elif isinstance(outputs, dict):
            logits_char = outputs.get("char") or outputs.get("logits_mask") or outputs.get("mask")
        else:
            logits_char = outputs

        return jax.nn.log_softmax(logits_char, axis=-1)

    _LOADED_MODELS[resolved] = (model, params, alphabet, config, forward_fn)
    return _LOADED_MODELS[resolved]


def decode_gap_hcb_beam_search(
    forward_fn: Any,
    params: Any,
    char_ids_initial: np.ndarray,
    word_ids_initial: np.ndarray,
    mask_indices: list[int],
    alphabet: Any,
    K: int = 20,
    beam_width: int = 20,
    strategy: str = "hcb_best_to_worst",
    temperature: float = 1.0,
) -> list[tuple[str, float]]:
    """
    Esegue la decodifica probabilistica per il restauro della lacuna usando HCB
    (Hammersley-Clifford-Besag Infilling con correzione del pivot, mutuato da packages/hcb_infilling)
    e ricerca iterativa Best-to-Worst.

    Ad ogni timestep:
    1. Esegue il forward pass sui candidati attuali per condizionare le predizioni.
    2. Seleziona la posizione a maggiore certezza del modello (Best-to-Worst).
    3. Aggiorna i log-punteggi sottraendo il pivot HCB del token maschera.
    4. Pota e mantiene i migliori beam_width rami per il timestep successivo.
    """
    import jax.numpy as jnp

    idx2char = alphabet.idx2char
    char2idx = getattr(alphabet, "char2idx", None) or {
        c: i for i, c in enumerate(idx2char)
    }

    # ID del token maschera ('-') impiegato come pivot nella correzione HCB
    mask_token_id = char2idx.get(alphabet.missing, 4)

    # Indici dei caratteri alfabetici greci validi (escludiamo token di controllo/punteggiatura tecnica)
    valid_char_indices = np.array(
        [
            i
            for i, c in enumerate(idx2char)
            if str(c) not in ("#", "^", "<", ">", "[", "]", "-")
        ],
        dtype=np.int32,
    )

    if len(mask_indices) == 0:
        return []

    # Popolazione iniziale dei candidati: forma (1, max_len)
    candidates = np.array(char_ids_initial, copy=True)
    candidate_log_likelihoods = np.zeros(1, dtype=np.float32)
    remaining_mask_indices = list(mask_indices)

    for _ in range(len(mask_indices)):
        current_b = len(candidates)
        word_batch = np.repeat(word_ids_initial, repeats=current_b, axis=0)

        # 1. Forward pass JAX condizionato sui caratteri parzialmente ripristinati:
        # Forma: [current_b, max_len, vocab_size] (log-probabilità)
        log_probs = np.array(
            forward_fn(params, jnp.array(candidates), jnp.array(word_batch))
        )
        if temperature > 0 and temperature != 1.0:
            log_probs = log_probs / temperature

        # 2. Scelta della posizione da ripristinare (Best-to-Worst vs Left-to-Right):
        if "best_to_worst" in strategy and len(remaining_mask_indices) > 1:
            # Seleziona la posizione con massima certezza (massima log-prob tra i caratteri validi)
            best_pos = max(
                remaining_mask_indices,
                key=lambda pos: float(np.max(log_probs[:, pos, valid_char_indices])),
            )
        else:
            best_pos = remaining_mask_indices[0]

        remaining_mask_indices.remove(best_pos)

        # 3. Log-probabilità per la posizione scelta: [current_b, vocab_size]
        pos_log_probs = log_probs[:, best_pos, :]
        valid_log_probs = pos_log_probs[:, valid_char_indices]  # [current_b, num_valid]

        # 4. Aggiornamento punteggio HCB (Besag pseudo-likelihood update)
        if "hcb" in strategy:
            # Sottrarre il pivot della maschera previene la sovra-stima di correlazione circolare dell'MLM
            pivot_vals = pos_log_probs[:, mask_token_id]  # [current_b]
            score_deltas = valid_log_probs - pivot_vals[:, None]
        else:
            # Aggiornamento pseudo-likelihood convenzionale
            score_deltas = valid_log_probs

        # Punteggio cumulato per ciascuna biforcazione di candidato x carattere valido
        branch_scores = candidate_log_likelihoods[:, None] + score_deltas  # [current_b, num_valid]

        # 5. Potatura: selezione dei migliori beam_width rami globali
        flat_scores = branch_scores.ravel()
        num_branches = len(flat_scores)
        top_k_indices = np.argsort(flat_scores)[::-1][: min(beam_width, num_branches)]

        cand_indices = top_k_indices // len(valid_char_indices)
        char_sub_indices = top_k_indices % len(valid_char_indices)
        selected_char_ids = valid_char_indices[char_sub_indices]

        # Creazione della nuova popolazione di candidati con la posizione riempita
        next_candidates = candidates[cand_indices].copy()
        next_candidates[:, best_pos] = selected_char_ids
        next_scores = flat_scores[top_k_indices]

        candidates = next_candidates
        candidate_log_likelihoods = next_scores

    # 6. Estrazione e normalizzazione softmax dei candidati completi
    results: list[tuple[str, float]] = []
    seen = set()

    for i in range(len(candidates)):
        cand_str = "".join(str(idx2char[candidates[i, pos]]) for pos in mask_indices)
        if cand_str not in seen:
            seen.add(cand_str)
            results.append((cand_str, float(candidate_log_likelihoods[i])))

    if not results:
        return []

    top_results = results[:K]
    max_log_p = top_results[0][1]
    raw_weights = [np.exp(s - max_log_p) for _, s in top_results]
    sum_w = sum(raw_weights) + 1e-12

    return [(s, float(w / sum_w)) for (s, _), w in zip(top_results, raw_weights)]


def fill_mask_ithaca(
    text: str,
    checkpoint: str = "checkpoints/ithaca/checkpoint_tlg.pkl",
    K: int = 20,
    beam_size: int = 20,
    strategy: Literal[
        "hcb_best_to_worst",
        "hcb_left_to_right",
        "standard_best_to_worst",
        "standard_left_to_right",
    ] = "hcb_best_to_worst",
    temperature: float = 1.0,
) -> list[tuple[str, float]]:
    """
    Funzione di interfaccia principale compatibile con l'API di GreekSchools:
    Accetta un contesto con lacuna (in formato Leiden [...] o [---]),
    esegue il restauro condizionato tramite HCB beam search e restituisce la lista
    ordinata di (suggerimento, score).
    """
    # 1. Rileva e calcola la lunghezza della lacuna (supporta sia Leiden [...] che [---] o trattini puri)
    match = re.search(r"\[(\.+|\-+)\]", text)
    if not match:
        match = re.search(r"(\-{1,})", text)
    if not match:
        raise ValueError("Il testo non contiene alcuna lacuna valida nel formato [...] o [---]")

    is_input_upper = any(c.isupper() for c in text)
    gap_len = len(match.group(1))

    # Sostituiamo la lacuna con trattini puri per la tokenizzazione interna di Ithaca
    text_with_hyphens = text[: match.start()] + ("-" * gap_len) + text[match.end() :]

    # 2. Normalizzazione testo (minuscolo, senza diacritici, conforme a GreekAlphabet)
    clean_text = normalize_greek(
        text_with_hyphens, case_folding="lower", strip_diacritics_flag=True
    )

    # 3. Caricamento modello e vocabolario
    _, params, alphabet, config, forward_fn = get_or_load_ithaca(checkpoint)
    max_len = int(config.get("max_len", 768))

    char2idx = getattr(alphabet, "char2idx", None) or {
        c: i for i, c in enumerate(alphabet.idx2char)
    }
    word2idx = getattr(alphabet, "word2idx", None) or {
        w: i for i, w in enumerate(alphabet.idx2word)
    }

    pad_char_id = char2idx.get(alphabet.pad, 0)
    unk_char_id = char2idx.get(alphabet.unk, 1)
    unk_word_id = word2idx.get(alphabet.unk, 1)

    # 4. Prefisso SOS ('<') fondamentale per la coerenza dei positional embeddings
    text_with_sos = str(alphabet.sos) + clean_text
    text_padded = text_with_sos + str(alphabet.pad) * max(0, max_len - len(text_with_sos))
    text_padded = text_padded[:max_len]

    char_ids = np.full((1, max_len), pad_char_id, dtype=np.int32)
    word_ids = np.full((1, max_len), unk_word_id, dtype=np.int32)
    mask_indices: list[int] = []

    for i, c in enumerate(text_padded):
        char_ids[0, i] = char2idx.get(c, unk_char_id)
        if c == alphabet.missing:
            mask_indices.append(i)

    # Tokenizzazione parole allineata con regex sulle posizioni assolute dei caratteri
    for m in re.finditer(r"\w+", text_padded):
        w_str = m.group()
        if w_str in word2idx:
            word_ids[0, m.start() : m.end()] = word2idx[w_str]

    if not mask_indices:
        return []

    # 5. Esecuzione HCB Beam Search
    suggestions = decode_gap_hcb_beam_search(
        forward_fn=forward_fn,
        params=params,
        char_ids_initial=char_ids,
        word_ids_initial=word_ids,
        mask_indices=mask_indices,
        alphabet=alphabet,
        K=K,
        beam_width=beam_size,
        strategy=strategy,
        temperature=temperature,
    )

    # 6. Preserva il casing coerente con la richiesta: se l'input era maiuscolo, restituisce maiuscolo
    if is_input_upper:
        return [(cand.upper(), score) for cand, score in suggestions]
    return suggestions
