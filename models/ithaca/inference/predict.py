"""
Inference engine per Ithaca per il supporto alle predizioni e all'API di GreekSchools.

Fornisce la funzione `fill_mask_ithaca`:
1. Traduce la convenzione di lacuna Leiden [...] nella sequenza di trattini attesa da Ithaca.
2. Normalizza il testo in minuscolo senza diacritici (conforme all'alfabeto di BigBird/Ithaca).
3. Aggiunge il prefisso Start-of-Sequence ('<') e calcola gli embedding di parola allineati via regex.
4. Esegue la decodifica iterativa condizionata usando HCB Beam Search vincolato alle sole
   lettere greche standard e agli spazi interni ben formati.
5. Applica il postprocessing difensivo (minuscolo costante, normalizzazione contestuale di
   'σ'/'ς' e aggregazione probabilistica dei duplicati via `logaddexp`) e restituisce
   una lista di suggerimenti ordinati per score decrescente.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
import threading
import time
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

from backend.config.settings import GS_ITHACA_MICRO_BATCH
from backend.core.model_cache import trim_memory
from backend.core.preprocess import normalize_greek

# Cache per i modelli caricati: {checkpoint_path: (model, params, alphabet, config, jitted_forward)}
_LOADED_MODELS: dict[str, Any] = {}
_RESOLVED_CHECKPOINT_PATHS: dict[str, str] = {}
_ITHACA_LAST_USED: dict[str, float] = {}
_ITHACA_LOCK = threading.Lock()

# 24 lettere greche standard minuscole (+ sigma finale 'ς'): esclude cifre ('0'),
# punteggiatura ('.'), token di controllo e caratteri epigrafici arcaici ('ϙ', 'ϛ').
STANDARD_GREEK_LOWER_CHARS: frozenset[str] = frozenset("αβγδεζηθικλμνξοπρςστυφχψω")


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
    """Carica e compila il modello Ithaca con cache thread-safe per evitare download/load duplicati."""
    now = time.monotonic()
    with _ITHACA_LOCK:
        cached_resolved = _RESOLVED_CHECKPOINT_PATHS.get(checkpoint)
        if cached_resolved and cached_resolved in _LOADED_MODELS:
            _ITHACA_LAST_USED[cached_resolved] = now
            return _LOADED_MODELS[cached_resolved]
        if checkpoint in _LOADED_MODELS:
            _ITHACA_LAST_USED[checkpoint] = now
            return _LOADED_MODELS[checkpoint]

        resolved = _resolve_checkpoint_path(checkpoint)
        if os.path.exists(resolved):
            _RESOLVED_CHECKPOINT_PATHS[checkpoint] = resolved
        if resolved in _LOADED_MODELS:
            _ITHACA_LAST_USED[resolved] = time.monotonic()
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

        loaded = (model, params, alphabet, config, forward_fn)
        _LOADED_MODELS[resolved] = loaded
        _ITHACA_LAST_USED[resolved] = time.monotonic()
        return loaded


def _normalize_contextual_sigma(candidate: str, next_char_after_gap: str = "") -> str:
    """Normalizza ogni sigma ('σ' mediano, 'ς' finale) in base al carattere successivo."""
    if "σ" not in candidate and "ς" not in candidate:
        return candidate
    out: list[str] = []
    n = len(candidate)
    next_external = next_char_after_gap[:1]
    for i, ch in enumerate(candidate):
        if ch in ("σ", "ς"):
            next_ch = candidate[i + 1] if i + 1 < n else next_external
            out.append("σ" if next_ch in STANDARD_GREEK_LOWER_CHARS else "ς")
        else:
            out.append(ch)
    return "".join(out)


def _is_valid_ithaca_candidate(candidate: str) -> bool:
    """Verifica che il candidato contenga solo lettere greche standard e spazi interni ben formati."""
    if not candidate:
        return False
    if candidate[0] == " " or candidate[-1] == " " or "  " in candidate:
        return False
    has_letter = False
    for ch in candidate:
        if ch in STANDARD_GREEK_LOWER_CHARS:
            has_letter = True
        elif ch != " ":
            return False
    return has_letter


def postprocess_ithaca_candidates(
    raw_candidates: list[tuple[str, float]],
    next_char_after_gap: str = "",
    K: int = 20,
) -> list[tuple[str, float]]:
    """
    Postprocessing difensivo dei candidati grezzi di Ithaca:
    - forza il minuscolo senza diacritici
    - normalizza ortograficamente 'σ'/'ς' rispetto al contesto
    - scarta cifre, punteggiatura, caratteri epigrafici arcaici e spazi malformati
    - fonde i duplicati equivalenti sommando le probabilità (logaddexp) prima della softmax.
    """
    next_after_norm = (
        normalize_greek(
            next_char_after_gap[:1],
            case_folding="lower",
            strip_diacritics_flag=True,
        )
        if next_char_after_gap
        else ""
    )

    merged_scores: dict[str, float] = {}
    for cand_str, log_score in raw_candidates:
        if not isinstance(cand_str, str) or not cand_str:
            continue
        score_val = float(log_score)
        if not np.isfinite(score_val):
            continue

        cand_norm = normalize_greek(
            cand_str, case_folding="lower", strip_diacritics_flag=True
        )
        cand_clean = _normalize_contextual_sigma(
            cand_norm, next_char_after_gap=next_after_norm
        )
        if not _is_valid_ithaca_candidate(cand_clean):
            continue

        if cand_clean in merged_scores:
            # Somma la massa probabilistica di rami equivalenti dopo normalizzazione σ/ς
            merged_scores[cand_clean] = float(
                np.logaddexp(merged_scores[cand_clean], score_val)
            )
        else:
            merged_scores[cand_clean] = score_val

    if not merged_scores:
        return []

    sorted_results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = sorted_results[:K]
    max_log_p = top_results[0][1]
    raw_weights = [float(np.exp(s - max_log_p)) for _, s in top_results]
    sum_w = sum(raw_weights) + 1e-12

    return [(s, float(w / sum_w)) for (s, _), w in zip(top_results, raw_weights)]


def _position_max_log_prob(
    pos: int,
    log_probs: np.ndarray,
    current_candidates: np.ndarray,
    letter_only_indices: np.ndarray,
    space_token_id: int | None,
    first_mask_pos: int,
    last_mask_pos: int,
) -> float:
    """Restituisce la massima log-probabilità valida per una posizione mascherata."""
    best_val = float(np.max(log_probs[:, pos, letter_only_indices]))
    if space_token_id is not None and first_mask_pos < pos < last_mask_pos:
        valid_space_rows = (
            (current_candidates[:, pos - 1] != space_token_id)
            & (current_candidates[:, pos + 1] != space_token_id)
        )
        if np.any(valid_space_rows):
            best_space = float(np.max(log_probs[valid_space_rows, pos, space_token_id]))
            if best_space > best_val:
                best_val = best_space
    return best_val


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
    next_char_after_gap: str | None = None,
    forward_batch_size: int | None = None,
) -> list[tuple[str, float]]:
    """
    Esegue la decodifica probabilistica per il restauro della lacuna usando HCB
    (Hammersley-Clifford-Besag Infilling con correzione del pivot, mutuato da packages/hcb_infilling)
    e ricerca iterativa Best-to-Worst vincolata alle sole lettere greche standard
    e agli spazi strettamente interni ben formati.

    Ad ogni timestep:
    1. Esegue il forward pass sui candidati attuali per condizionare le predizioni.
    2. Seleziona la posizione a maggiore certezza del modello (Best-to-Worst).
    3. Aggiorna i log-punteggi sottraendo il pivot HCB del token maschera.
    4. Pota e mantiene i migliori beam_width rami validi per il timestep successivo.
    """
    import jax.numpy as jnp

    ordered_mask_indices = sorted(mask_indices)
    if not ordered_mask_indices:
        return []

    first_mask_pos = ordered_mask_indices[0]
    last_mask_pos = ordered_mask_indices[-1]

    idx2char = alphabet.idx2char
    char2idx = getattr(alphabet, "char2idx", None) or {
        c: i for i, c in enumerate(idx2char)
    }

    # ID del token maschera ('-') impiegato come pivot nella correzione HCB
    mask_token_id = char2idx.get(alphabet.missing, 4)

    space_char = str(getattr(alphabet, "space", " "))
    space_token_id = char2idx.get(space_char)

    # Vincola il beam alle 24 lettere greche standard (escludendo '0', '.', 'ϙ', 'ϛ' e controlli)
    letter_only_indices = np.array(
        [
            i
            for i, c in enumerate(idx2char)
            if str(c) in STANDARD_GREEK_LOWER_CHARS
        ],
        dtype=np.int32,
    )
    if len(letter_only_indices) == 0:
        return []

    valid_char_indices = np.array(
        [
            i
            for i, c in enumerate(idx2char)
            if str(c) in STANDARD_GREEK_LOWER_CHARS or str(c) == space_char
        ],
        dtype=np.int32,
    )

    space_col_idx: int | None = None
    if space_token_id is not None:
        space_matches = np.flatnonzero(valid_char_indices == space_token_id)
        if len(space_matches) > 0:
            space_col_idx = int(space_matches[0])

    # Popolazione iniziale dei candidati: forma (1, max_len)
    candidates = np.array(char_ids_initial, copy=True)
    candidate_log_likelihoods = np.zeros(1, dtype=np.float32)
    remaining_mask_indices = list(ordered_mask_indices)

    mb = int(forward_batch_size) if (forward_batch_size is not None and forward_batch_size > 0) else None
    word_batch_mb = (
        jnp.array(np.repeat(word_ids_initial, repeats=mb, axis=0))
        if mb is not None
        else None
    )

    for _ in range(len(ordered_mask_indices)):
        current_b = len(candidates)

        # 1. Forward pass JAX condizionato sui caratteri parzialmente ripristinati:
        # Forma: [current_b, max_len, vocab_size] (log-probabilità)
        if mb is not None and word_batch_mb is not None:
            # Micro-batch fisso `(mb, max_len)` per riutilizzare un singolo grafo compilato XLA
            chunk_logs: list[np.ndarray] = []
            for start_idx in range(0, current_b, mb):
                chunk_chars = candidates[start_idx : start_idx + mb]
                actual_len = len(chunk_chars)
                if actual_len < mb:
                    pad_reps = mb - actual_len
                    chunk_chars = np.concatenate(
                        [chunk_chars, np.repeat(chunk_chars[-1:], repeats=pad_reps, axis=0)],
                        axis=0,
                    )
                chunk_out = np.asarray(
                    forward_fn(params, jnp.array(chunk_chars), word_batch_mb)
                )
                chunk_logs.append(chunk_out[:actual_len])
            log_probs = np.concatenate(chunk_logs, axis=0)
        else:
            word_batch = np.repeat(word_ids_initial, repeats=current_b, axis=0)
            log_probs = np.array(
                forward_fn(params, jnp.array(candidates), jnp.array(word_batch))
            )
        if temperature > 0 and temperature != 1.0:
            log_probs = log_probs / temperature

        # 2. Scelta della posizione da ripristinare (Best-to-Worst vs Left-to-Right):
        if "best_to_worst" in strategy and len(remaining_mask_indices) > 1:
            best_pos = max(
                remaining_mask_indices,
                key=lambda pos: _position_max_log_prob(
                    pos=pos,
                    log_probs=log_probs,
                    current_candidates=candidates,
                    letter_only_indices=letter_only_indices,
                    space_token_id=space_token_id,
                    first_mask_pos=first_mask_pos,
                    last_mask_pos=last_mask_pos,
                ),
            )
        else:
            best_pos = remaining_mask_indices[0]

        remaining_mask_indices.remove(best_pos)

        # Lo spazio è ammesso solo in posizioni strettamente interne alla lacuna (mai sui bordi)
        allow_space_at_pos = (
            space_token_id is not None
            and space_col_idx is not None
            and first_mask_pos < best_pos < last_mask_pos
        )
        step_valid_indices = valid_char_indices if allow_space_at_pos else letter_only_indices

        # 3. Log-probabilità per la posizione scelta: [current_b, vocab_size]
        pos_log_probs = log_probs[:, best_pos, :]
        valid_log_probs = pos_log_probs[:, step_valid_indices]

        # 4. Aggiornamento punteggio HCB (Besag pseudo-likelihood update)
        if "hcb" in strategy:
            # Sottrarre il pivot della maschera previene la sovra-stima di correlazione circolare dell'MLM
            pivot_vals = pos_log_probs[:, mask_token_id]
            score_deltas = valid_log_probs - pivot_vals[:, None]
        else:
            # Aggiornamento pseudo-likelihood convenzionale
            score_deltas = valid_log_probs

        # Punteggio cumulato per ciascuna biforcazione di candidato x carattere valido
        branch_scores = candidate_log_likelihoods[:, None] + score_deltas

        if allow_space_at_pos:
            # Impedisce spazi consecutivi dentro la lacuna mascherando i rami con spazio adiacente
            invalid_space_branches = (
                (candidates[:, best_pos - 1] == space_token_id)
                | (candidates[:, best_pos + 1] == space_token_id)
            )
            branch_scores[invalid_space_branches, space_col_idx] = -np.inf

        # 5. Potatura sui soli rami finiti per evitare di espandere rami vietati
        flat_scores = branch_scores.ravel()
        finite_indices = np.flatnonzero(np.isfinite(flat_scores))
        if len(finite_indices) == 0:
            return []

        ordered_finite = finite_indices[np.argsort(flat_scores[finite_indices])[::-1]]
        top_k_indices = ordered_finite[: min(beam_width, len(ordered_finite))]

        num_step_valid = len(step_valid_indices)
        cand_indices = top_k_indices // num_step_valid
        char_sub_indices = top_k_indices % num_step_valid
        selected_char_ids = step_valid_indices[char_sub_indices]

        # Creazione della nuova popolazione di candidati con la posizione riempita
        next_candidates = candidates[cand_indices].copy()
        next_candidates[:, best_pos] = selected_char_ids
        next_scores = flat_scores[top_k_indices]

        candidates = next_candidates
        candidate_log_likelihoods = next_scores

    if next_char_after_gap is None:
        after_pos = last_mask_pos + 1
        if after_pos < char_ids_initial.shape[1]:
            next_id = int(char_ids_initial[0, after_pos])
            resolved_next_char = (
                str(idx2char[next_id]) if 0 <= next_id < len(idx2char) else ""
            )
        else:
            resolved_next_char = ""
    else:
        resolved_next_char = next_char_after_gap

    # 6. Estrazione dei candidati grezzi e postprocessing finale (sigma contestuale + merge + softmax)
    raw_results = [
        (
            "".join(str(idx2char[candidates[i, pos]]) for pos in ordered_mask_indices),
            float(candidate_log_likelihoods[i]),
        )
        for i in range(len(candidates))
    ]

    return postprocess_ithaca_candidates(
        raw_results,
        next_char_after_gap=resolved_next_char,
        K=K,
    )


def clear_ithaca_cache() -> None:
    """Svuota la cache dei modelli Ithaca caricati, libera le compilazioni JIT di JAX e restituisce RAM al kernel."""
    with _ITHACA_LOCK:
        _LOADED_MODELS.clear()
        _RESOLVED_CHECKPOINT_PATHS.clear()
        _ITHACA_LAST_USED.clear()
    trim_memory(clear_jax=True)


def evict_expired_ithaca(ttl_seconds: float) -> bool:
    """Scarica i modelli Ithaca inutilizzati da più di `ttl_seconds` secondi e svuota la cache XLA."""
    if ttl_seconds <= 0:
        return False

    now = time.monotonic()
    removed = False
    with _ITHACA_LOCK:
        expired_models = [
            key
            for key in list(_LOADED_MODELS.keys())
            if not key.startswith("__torso_fn__::")
            and (now - _ITHACA_LAST_USED.get(key, now)) > ttl_seconds
        ]
        if expired_models:
            removed = True
            expired_aliases = {
                alias
                for alias, res in list(_RESOLVED_CHECKPOINT_PATHS.items())
                if res in expired_models
            }
            for key in expired_models:
                _LOADED_MODELS.pop(key, None)
                _ITHACA_LAST_USED.pop(key, None)
            for alias in expired_aliases:
                _RESOLVED_CHECKPOINT_PATHS.pop(alias, None)
            # Rimuove eventuali torso_fn associati o rimasti orfani
            active_models = [
                k for k in _LOADED_MODELS.keys() if not k.startswith("__torso_fn__::")
            ]
            for key in list(_LOADED_MODELS.keys()):
                if key.startswith("__torso_fn__::"):
                    if not active_models or any(
                        m in key for m in (*expired_models, *expired_aliases)
                    ):
                        _LOADED_MODELS.pop(key, None)

    if removed:
        trim_memory(clear_jax=True)
    return removed


def fill_mask_ithaca(
    text: str,
    checkpoint: str = "checkpoints/ithaca/checkpoint_tlg.pkl",
    K: int = 20,
    beam_size: int | None = None,
    strategy: Literal[
        "hcb_best_to_worst",
        "hcb_left_to_right",
        "standard_best_to_worst",
        "standard_left_to_right",
    ] = "hcb_best_to_worst",
    temperature: float = 1.0,
    params_override: Any | None = None,
    alphabet_override: Any | None = None,
    config_override: dict | None = None,
    forward_fn_override: Any | None = None,
) -> list[tuple[str, float]]:
    """
    Funzione di interfaccia principale compatibile con l'API di GreekSchools:
    Accetta un contesto con lacuna (in formato Leiden [...] o [---]),
    esegue il restauro condizionato tramite HCB beam search vincolato e restituisce
    la lista ordinata di (suggerimento, score) sempre in minuscolo.
    """
    # 1. Rileva e calcola la lunghezza della lacuna (supporta sia Leiden [...] che [---] o trattini puri)
    match = re.search(r"\[(\.+|\-+)\]", text)
    if not match:
        match = re.search(r"(\-{1,})", text)
    if not match:
        raise ValueError("Il testo non contiene alcuna lacuna valida nel formato [...] o [---]")

    gap_len = len(match.group(1))
    # Aggiunge un piccolo margine sul beam quando non specificato per compensare fusioni σ/ς
    effective_beam_size = max(20, K + 4) if beam_size is None else beam_size

    # Caricamento modello e vocabolario (con supporto a override in-memory per trial Optuna)
    if (
        params_override is not None
        and alphabet_override is not None
        and config_override is not None
        and forward_fn_override is not None
    ):
        params = params_override
        alphabet = alphabet_override
        config = config_override
        forward_fn = forward_fn_override
    else:
        _, loaded_params, loaded_alphabet, loaded_config, loaded_forward_fn = get_or_load_ithaca(checkpoint)
        params = params_override if params_override is not None else loaded_params
        alphabet = alphabet_override if alphabet_override is not None else loaded_alphabet
        config = config_override if config_override is not None else loaded_config
        forward_fn = forward_fn_override if forward_fn_override is not None else loaded_forward_fn

    max_len = int(config.get("max_len", 768))

    unk_char = str(alphabet.unk)

    prefix = text[: match.start()]
    suffix = text[match.end() :]

    # Uniforma tutte le altre lacune secondarie presenti nel contesto a carattere UNK di Ithaca ('^')
    # evitando che '<' di <UNK> venga scambiato per SOS o che i punti vengano trattati come punteggiatura
    def _clean_secondary_gaps(s: str) -> str:
        s = re.sub(r"<[^>]+>", unk_char, s)
        s = re.sub(r"\[(\.+|\-+)\]", lambda m: unk_char * len(m.group(1)), s)
        s = re.sub(r"(?<!\.)\.{2,}(?!\.)", lambda m: unk_char * len(m.group(0)), s)
        s = re.sub(r"\-+", lambda m: unk_char * len(m.group(0)), s)
        return s

    prefix_clean = _clean_secondary_gaps(prefix)
    suffix_clean = _clean_secondary_gaps(suffix)

    # Sostituiamo la sola lacuna target con trattini puri per la tokenizzazione interna di Ithaca
    text_with_hyphens = prefix_clean + ("-" * gap_len) + suffix_clean

    # 2. Normalizzazione testo (minuscolo, senza diacritici, conforme a GreekAlphabet)
    clean_text = normalize_greek(
        text_with_hyphens, case_folding="lower", strip_diacritics_flag=True
    )

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
    result = decode_gap_hcb_beam_search(
        forward_fn=forward_fn,
        params=params,
        char_ids_initial=char_ids,
        word_ids_initial=word_ids,
        mask_indices=mask_indices,
        alphabet=alphabet,
        K=K,
        beam_width=effective_beam_size,
        strategy=strategy,
        temperature=temperature,
        forward_batch_size=GS_ITHACA_MICRO_BATCH if forward_fn_override is None else None,
    )
    with _ITHACA_LOCK:
        resolved_key = _RESOLVED_CHECKPOINT_PATHS.get(checkpoint)
        if resolved_key and resolved_key in _LOADED_MODELS:
            _ITHACA_LAST_USED[resolved_key] = time.monotonic()
    return result


def get_ithaca_contextual_embeddings(
    text_with_gap: str,
    candidates: list[str],
    checkpoint: str = "checkpoints/ithaca/checkpoint_tlg.pkl",
    params_override: Any | None = None,
    alphabet_override: Any | None = None,
    config_override: dict | None = None,
    torso_fn_override: Any | None = None,
) -> list["torch.Tensor"]:
    """
    Estrae gli embedding contestuali 1D (dim=512, mediati sulle posizioni dei caratteri
    della lacuna) dal torso del modello BigBird di Ithaca (`output_return_emb=True`)
    per ciascun candidato inserito nel contesto.

    Restituisce una lista di `torch.Tensor` di forma `[emb_dim]` compatibile con
    `evaluate_contextual_similarity` e `check_gold_in_dense_cluster`.
    """
    import torch

    if not candidates:
        return []

    match = re.search(r"\[(\.+|\-+)\]", text_with_gap)
    if not match:
        match = re.search(r"(\-{1,})", text_with_gap)
    if not match:
        raise ValueError("Il testo non contiene alcuna lacuna valida nel formato [...] o [---]")

    if (
        params_override is not None
        and alphabet_override is not None
        and config_override is not None
        and torso_fn_override is not None
    ):
        params = params_override
        alphabet = alphabet_override
        config = config_override
        torso_fn = torso_fn_override
    else:
        _, loaded_params, loaded_alphabet, loaded_config, _ = get_or_load_ithaca(checkpoint)
        params = params_override if params_override is not None else loaded_params
        alphabet = alphabet_override if alphabet_override is not None else loaded_alphabet
        config = config_override if config_override is not None else loaded_config
        torso_fn = torso_fn_override

    if torso_fn is None:
        cache_key = f"__torso_fn__::{checkpoint}::{config.get('max_len', 768)}"
        if cache_key in _LOADED_MODELS:
            torso_fn = _LOADED_MODELS[cache_key]
        else:
            import jax

            try:
                from ithaca.models.model import Model
            except ImportError:
                sys.path.insert(0, "packages/ithaca_engine")
                from ithaca.models.model import Model

            emb_config = dict(config)
            emb_config["output_return_emb"] = True
            emb_model = Model(**emb_config)

            @jax.jit
            def _extract_torso(p, text_char, text_word):
                variables = p if (isinstance(p, dict) and "params" in p) else {"params": p}
                _, torso_out = emb_model.apply(
                    variables,
                    text_char=text_char,
                    text_word=text_word,
                    is_training=False,
                )
                return torso_out

            torso_fn = _extract_torso
            _LOADED_MODELS[cache_key] = torso_fn

    max_len = int(config.get("max_len", 768))
    unk_char = str(alphabet.unk)

    prefix = text_with_gap[: match.start()]
    suffix = text_with_gap[match.end() :]

    def _clean_secondary_gaps(s: str) -> str:
        s = re.sub(r"<[^>]+>", unk_char, s)
        s = re.sub(r"\[(\.+|\-+)\]", lambda m: unk_char * len(m.group(1)), s)
        s = re.sub(r"(?<!\.)\.{2,}(?!\.)", lambda m: unk_char * len(m.group(0)), s)
        s = re.sub(r"\-+", lambda m: unk_char * len(m.group(0)), s)
        return s

    prefix_norm = normalize_greek(
        _clean_secondary_gaps(prefix), case_folding="lower", strip_diacritics_flag=True
    )
    suffix_norm = normalize_greek(
        _clean_secondary_gaps(suffix), case_folding="lower", strip_diacritics_flag=True
    )

    char2idx = getattr(alphabet, "char2idx", None) or {
        c: i for i, c in enumerate(alphabet.idx2char)
    }
    word2idx = getattr(alphabet, "word2idx", None) or {
        w: i for i, w in enumerate(alphabet.idx2word)
    }
    pad_char_id = char2idx.get(alphabet.pad, 0)
    unk_char_id = char2idx.get(alphabet.unk, 1)
    unk_word_id = word2idx.get(alphabet.unk, 1)

    num_cands = len(candidates)
    batch_char = np.full((num_cands, max_len), pad_char_id, dtype=np.int32)
    batch_word = np.full((num_cands, max_len), unk_word_id, dtype=np.int32)
    span_ranges: list[tuple[int, int]] = []

    for b_idx, cand in enumerate(candidates):
        cand_norm = normalize_greek(
            str(cand), case_folding="lower", strip_diacritics_flag=True
        ).strip()
        if not cand_norm:
            cand_norm = unk_char

        full_clean = prefix_norm + cand_norm + suffix_norm
        text_with_sos = str(alphabet.sos) + full_clean
        text_padded = (
            text_with_sos + str(alphabet.pad) * max(0, max_len - len(text_with_sos))
        )[:max_len]

        # Posizioni del candidato (offset +1 per SOS)
        span_start = min(1 + len(prefix_norm), max_len - 1)
        span_end = min(span_start + max(len(cand_norm), 1), max_len)
        span_ranges.append((span_start, span_end))

        for i, c in enumerate(text_padded):
            batch_char[b_idx, i] = char2idx.get(c, unk_char_id)

        for m in re.finditer(r"\w+", text_padded):
            w_str = m.group()
            if w_str in word2idx:
                batch_word[b_idx, m.start() : m.end()] = word2idx[w_str]

    import jax.numpy as jnp

    torso_embeddings = np.array(
        torso_fn(params, jnp.array(batch_char), jnp.array(batch_word)),
        dtype=np.float32,
    )

    pooled_tensors: list[torch.Tensor] = []
    for b_idx, (s_start, s_end) in enumerate(span_ranges):
        if s_end > s_start:
            span_vec = torso_embeddings[b_idx, s_start:s_end, :].mean(axis=0)
        else:
            span_vec = torso_embeddings[b_idx, s_start, :]
        pooled_tensors.append(torch.from_numpy(span_vec))

    return pooled_tensors

