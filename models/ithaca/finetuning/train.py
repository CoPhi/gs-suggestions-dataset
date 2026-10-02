"""
Script di Fine-Tuning per Ithaca sul corpus TLG in JAX/Flax.

Funzionalità:
1. Carica il checkpoint pre-addestrato di Ithaca (checkpoint_v1.pkl).
2. Congela le teste di attribuzione geografica e temporale (Optax multi_transform).
3. Allena il Transformer BigBird e la testa di restauro (character-level masked language modeling)
   esclusivamente sulle lacune sintetiche del TLG.
4. Valuta periodicamente la loss di restauro e la mask_accuracy sul set di validazione.
5. Salva il miglior checkpoint fine-tunato (checkpoint_tlg.pkl) per l'inferenza e l'API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Assicura che la root del progetto sia accessibile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Fallback per includere packages/ithaca_engine se clonato localmente
ithaca_engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../packages/ithaca_engine")
)
if os.path.isdir(ithaca_engine_path) and ithaca_engine_path not in sys.path:
    sys.path.insert(0, ithaca_engine_path)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.ithaca.finetuning.checkpoint import (
    create_optimizer_with_freezing,
    ensure_checkpoint_exists,
    load_ithaca_checkpoint,
    save_finetuned_checkpoint,
)


def encode_sequence(
    text_ithaca: str,
    alphabet: Any,
    max_len: int = 768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Codifica una stringa formattata Ithaca (contenente es. '[----]')
    negli array numpy attesi dal modello:
    - text_char: ID dei caratteri (pad='#' per riempimento)
    - text_word: ID delle parole dal vocabolario ausiliario
    - mask_pos: maschera binaria (1 nelle posizioni da predire, 0 altrove)
    """
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

    chars = list(text_ithaca)[:max_len]

    char_ids = np.full(max_len, pad_char_id, dtype=np.int32)
    word_ids = np.full(max_len, pad_word_id, dtype=np.int32)
    mask_pos = np.zeros(max_len, dtype=np.float32)

    # Identifica le posizioni tra '[' e ']' che contengono '-'
    in_gap = False
    for i, c in enumerate(chars):
        char_ids[i] = char2idx.get(c, unk_char_id)
        if c == "[":
            in_gap = True
        elif c == "]":
            in_gap = False
        elif in_gap and c == "-":
            mask_pos[i] = 1.0

    # Tokenizzazione parole a livello di base
    words = text_ithaca.split(" ")
    curr_char_idx = 0
    for w in words:
        w_id = word2idx.get(w, unk_word_id)
        w_len = len(w)
        for offset in range(w_len):
            if curr_char_idx + offset < max_len:
                word_ids[curr_char_idx + offset] = w_id
        curr_char_idx += w_len + 1

    return char_ids, word_ids, mask_pos


def load_jsonl_dataset(
    file_path: str,
    alphabet: Any,
    max_len: int = 768,
    limit: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Carica e codifica gli esempi dal file JSONL in array pronti per i batch."""
    dataset = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset non trovato: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            row = json.loads(line.strip())
            text_ithaca = row["text_ithaca"]
            gold_target = row["gold_target"]

            char_ids, word_ids, mask_pos = encode_sequence(
                text_ithaca, alphabet, max_len=max_len
            )

            # Salta eventuali campioni senza caratteri mascherati nella finestra
            if np.sum(mask_pos) == 0:
                continue

            # Crea il vettore target_chars rimpiazzando i trattini con i caratteri gold
            char2idx = getattr(alphabet, "char2idx", None) or {
                c: i for i, c in enumerate(alphabet.idx2char)
            }
            target_ids = np.array(char_ids, copy=True)
            gold_chars = list(gold_target)
            gold_ptr = 0

            for i in range(max_len):
                if mask_pos[i] == 1.0 and gold_ptr < len(gold_chars):
                    target_ids[i] = char2idx.get(gold_chars[gold_ptr], 1)
                    gold_ptr += 1

            dataset.append(
                {
                    "text_char": char_ids,
                    "text_word": word_ids,
                    "mask_pos": mask_pos,
                    "target_char": target_ids,
                }
            )

    return dataset


def create_batch_generator(
    dataset: list[dict[str, np.ndarray]],
    batch_size: int = 16,
    shuffle: bool = True,
):
    """Generatore di batch iterabile per l'addestramento."""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(dataset), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < batch_size:
            continue  # scartiamo batch incompleto per mantenere forma statica (JIT-friendly)

        batch_char = np.stack([dataset[i]["text_char"] for i in batch_idx])
        batch_word = np.stack([dataset[i]["text_word"] for i in batch_idx])
        batch_mask = np.stack([dataset[i]["mask_pos"] for i in batch_idx])
        batch_target = np.stack([dataset[i]["target_char"] for i in batch_idx])

        yield {
            "text_char": jnp.array(batch_char),
            "text_word": jnp.array(batch_word),
            "mask_pos": jnp.array(batch_mask),
            "target_char": jnp.array(batch_target),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning di Ithaca su dataset TLG con congelamento teste di attribuzione"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/ithaca/tlg_train.jsonl",
        help="Percorso al dataset di training (JSONL)",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/ithaca/tlg_val.jsonl",
        help="Percorso al dataset di validazione (JSONL)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/ithaca/checkpoint_v1.pkl",
        help="Percorso al checkpoint base di Ithaca",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/ithaca",
        help="Cartella dove salvare il checkpoint fine-tunato",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Numero di epoche di fine-tuning (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Dimensione del batch (default: 8 per GPU standard)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate massimo (default: 2e-5)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=300,
        help="Step di warmup per il learning rate",
    )
    parser.add_argument(
        "--no_freeze",
        action="store_true",
        help="Disabilita il freezing delle teste di attribuzione (non consigliato su TLG)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la riproducibilità e l'inizializzazione del dropout RNG",
    )
    args = parser.parse_args()

    # Disabilita preallocazione della VRAM in JAX
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

    ensure_checkpoint_exists(args.checkpoint_path)
    print(f"Caricamento checkpoint base da '{args.checkpoint_path}'...")
    ckpt_data = load_ithaca_checkpoint(args.checkpoint_path)
    params = ckpt_data["params"]
    alphabet = ckpt_data["alphabet"]
    config = ckpt_data["config"]

    # Importazione dell'architettura del modello
    try:
        from ithaca.models.model import Model
    except ImportError:
        # Fallback locale se packages/ithaca_engine è presente
        sys.path.insert(0, "packages/ithaca_engine")
        from ithaca.models.model import Model

    model = Model(**config)

    max_len = int(config.get("max_len", 768))
    print(f"Caricamento e codifica dataset di training e validazione (max_len={max_len})...")
    train_data = load_jsonl_dataset(args.train_path, alphabet, max_len=max_len)
    val_data = load_jsonl_dataset(args.val_path, alphabet, max_len=max_len)
    print(f"Casi di training: {len(train_data)} | Casi di validazione: {len(val_data)}")

    steps_per_epoch = len(train_data) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    freeze_heads = not args.no_freeze
    print(
        f"Configurazione ottimizzatore Optax (AdamW, LR={args.lr}, warmup={args.warmup_steps}, "
        f"freeze_attribution={freeze_heads})..."
    )
    optimizer = create_optimizer_with_freezing(
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        freeze_attribution_heads=freeze_heads,
        params=params,
    )
    opt_state = optimizer.init(params)
    rng = jax.random.PRNGKey(args.seed)

    # Definizione Loss Function (solo sui token mascherati)
    def loss_fn(p, batch, dropout_rng=None):
        variables = p if (isinstance(p, dict) and "params" in p) else {"params": p}
        is_training = dropout_rng is not None
        rngs = {"dropout": dropout_rng} if is_training else {}
        outputs = model.apply(
            variables,
            text_char=batch["text_char"],
            text_word=batch["text_word"],
            is_training=is_training,
            rngs=rngs,
        )
        logits_char = outputs["char"]  # [batch, max_len, vocab_char_size]

        # One-hot encoding del target
        vocab_size = logits_char.shape[-1]
        one_hot_targets = jax.nn.one_hot(batch["target_char"], vocab_size)

        # Cross-entropy
        log_probs = jax.nn.log_softmax(logits_char, axis=-1)
        per_token_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)

        # Mascheramento: calcoliamo la loss solo sulle posizioni mascherate
        mask = batch["mask_pos"]
        total_masked = jnp.maximum(jnp.sum(mask), 1.0)
        masked_loss = jnp.sum(per_token_loss * mask) / total_masked

        # Calcolo accuratezza caratteri mascherati
        pred_chars = jnp.argmax(logits_char, axis=-1)
        correct_chars = (pred_chars == batch["target_char"]) * mask
        accuracy = jnp.sum(correct_chars) / total_masked

        return masked_loss, accuracy

    @jax.jit
    def train_step(p, opt_s, batch, rng_key):
        rng_key, step_rng = jax.random.split(rng_key)
        (loss_val, acc_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            p, batch, step_rng
        )
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_params = optax.apply_updates(p, updates)
        return new_params, new_opt_s, loss_val, acc_val, rng_key

    @jax.jit
    def eval_step(p, batch):
        return loss_fn(p, batch, dropout_rng=None)

    print("\n--- INIZIO FINE-TUNING ITHACA (TLG) ---")
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "checkpoint_tlg.pkl"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_gen = create_batch_generator(
            train_data, batch_size=args.batch_size, shuffle=True
        )

        train_losses, train_accs = [], []
        for step, batch in enumerate(train_gen, 1):
            params, opt_state, loss_v, acc_v, rng = train_step(
                params, opt_state, batch, rng
            )
            train_losses.append(float(loss_v))
            train_accs.append(float(acc_v))

            if step % 20 == 0 or step == steps_per_epoch:
                print(
                    f"\rEpoca {epoch}/{args.epochs} | Step {step}/{steps_per_epoch} | "
                    f"Loss: {np.mean(train_losses[-20:]):.4f} | "
                    f"Char Acc: {np.mean(train_accs[-20:]) * 100:.2f}%",
                    end="",
                    flush=True,
                )

        # Valutazione a fine epoca
        val_gen = create_batch_generator(
            val_data, batch_size=args.batch_size, shuffle=False
        )
        val_losses, val_accs = [], []
        for v_batch in val_gen:
            v_loss, v_acc = eval_step(params, v_batch)
            val_losses.append(float(v_loss))
            val_accs.append(float(v_acc))

        epoch_time = time.time() - epoch_start
        mean_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        mean_val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        print(
            f"\n--> Fine Epoca {epoch}: "
            f"Val Loss: {mean_val_loss:.4f} | Val Char Acc: {mean_val_acc * 100:.2f}% | "
            f"Tempo: {epoch_time:.1f}s"
        )

        # Salvataggio se best checkpoint
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print(f"Nuovo miglior punteggio validazione ({best_val_loss:.4f}). Salvataggio...")
            save_finetuned_checkpoint(
                output_path=str(best_ckpt_path),
                params=params,
                alphabet=alphabet,
                config=config,
                metadata={
                    "epochs": epoch,
                    "val_loss": best_val_loss,
                    "val_char_acc": mean_val_acc,
                    "base_model": "deepmind/ithaca",
                    "dataset": args.train_path,
                    "frozen_heads": freeze_heads,
                },
            )

    # Salvataggio configurazione JSON per riuso immediato
    config_export_path = output_dir / "config.json"
    with open(config_export_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n--- FINE-TUNING COMPLETATO ---")
    print(f"Checkpoint migliore salvato in: {best_ckpt_path}")
    print(f"Configurazione esportata in: {config_export_path}")


if __name__ == "__main__":
    main()
