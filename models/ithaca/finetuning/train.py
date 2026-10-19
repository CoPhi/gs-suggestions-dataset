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
import re
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
    Codifica una stringa formattata Ithaca (contenente es. trattini '----' per le posizioni mascherate)
    negli array numpy attesi dal modello:
    - text_char: ID dei caratteri (con SOS '<' come prefisso e pad='#' per riempimento)
    - text_word: ID delle parole dal vocabolario ausiliario allineati ai caratteri tramite regex
    - mask_pos: maschera binaria (1.0 nelle posizioni da predire '-', 0.0 altrove)
    """
    char2idx = getattr(alphabet, "char2idx", None) or {
        c: i for i, c in enumerate(alphabet.idx2char)
    }
    word2idx = getattr(alphabet, "word2idx", None) or {
        w: i for i, w in enumerate(alphabet.idx2word)
    }

    pad_char_id = char2idx.get(alphabet.pad, 0)
    unk_char_id = char2idx.get(alphabet.unk, 1)
    unk_word_id = word2idx.get(alphabet.unk, 1)

    # 1. Pulizia testo: rimozione di eventuali parentesi quadre editoriali, conversione in minuscolo
    clean_text = text_ithaca.replace("[", "").replace("]", "").replace(".", "-").lower()

    # 2. Prefisso SOS ('<') fondamentale per i positional embeddings di BigBird
    text_with_sos = str(alphabet.sos) + clean_text
    text_padded = text_with_sos + str(alphabet.pad) * max(0, max_len - len(text_with_sos))
    text_padded = text_padded[:max_len]

    # 3. Array dei caratteri e maschera binaria
    char_ids = np.full(max_len, pad_char_id, dtype=np.int32)
    mask_pos = np.zeros(max_len, dtype=np.float32)

    for i, c in enumerate(text_padded):
        char_ids[i] = char2idx.get(c, unk_char_id)
        if c == alphabet.missing:
            mask_pos[i] = 1.0

    # 4. Array delle parole: regex per estrarre parole senza attaccare punteggiatura
    word_ids = np.full(max_len, unk_word_id, dtype=np.int32)
    for m in re.finditer(r"\w+", text_padded):
        w_str = m.group()
        if w_str in word2idx:
            word_ids[m.start() : m.end()] = word2idx[w_str]

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

    char2idx = getattr(alphabet, "char2idx", None) or {
        c: i for i, c in enumerate(alphabet.idx2char)
    }
    unk_char_id = char2idx.get(alphabet.unk, 1)

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            row = json.loads(line.strip())
            text_ithaca = row.get("text_ithaca") or row.get("text_leiden")
            gold_target = str(row["gold_target"]).lower()

            char_ids, word_ids, mask_pos = encode_sequence(
                text_ithaca, alphabet, max_len=max_len
            )

            # Salta eventuali campioni senza caratteri mascherati nella finestra
            if np.sum(mask_pos) == 0:
                continue

            # Crea il vettore target_chars rimpiazzando i trattini con i caratteri gold
            target_ids = np.array(char_ids, copy=True)
            gold_chars = list(gold_target)
            gold_ptr = 0

            for i in range(max_len):
                if mask_pos[i] == 1.0 and gold_ptr < len(gold_chars):
                    target_ids[i] = char2idx.get(gold_chars[gold_ptr], unk_char_id)
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


def _init_wandb_ithaca(
    checkpoint_path: str,
    lr: float,
    batch_size: int,
    epochs: int,
    num_layers_to_freeze: int,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    dropout_rate: float,
    strategy: str,
    temperature: float,
    beam_size: int,
    run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_job_type: str | None = None,
    extra_config: dict | None = None,
) -> str:
    """Inizializza una run W&B sul progetto 'gs-suggestions' per Ithaca."""
    try:
        import wandb
        from models.bert.finetuning import WANDB_PROJECT
    except ImportError:
        return run_name or "ithaca_run"

    if not run_name:
        run_name = (
            f"gs-ithaca-tlg_lr{lr:.1e}_bs{batch_size}_ep{epochs}_frz{num_layers_to_freeze}"
        )

    config_dict = {
        "checkpoint": "CNR-ILC/gs-ithaca-tlg",
        "base_model": checkpoint_path,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_layers_to_freeze": num_layers_to_freeze,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "dropout_rate": dropout_rate,
        "strategy": strategy,
        "temperature": temperature,
        "beam_size": beam_size,
    }
    if extra_config:
        config_dict.update(extra_config)

    tags = (
        wandb_tags
        if wandb_tags is not None
        else ["gs-ithaca-tlg", "ithaca", "finetuning", "jax"]
    )

    if wandb.run is None:
        init_kwargs: dict[str, Any] = {
            "project": WANDB_PROJECT,
            "name": run_name,
            "config": config_dict,
            "tags": tags,
            "resume": "allow",
        }
        if wandb_group:
            init_kwargs["group"] = wandb_group
        if wandb_job_type:
            init_kwargs["job_type"] = wandb_job_type
        try:
            wandb.init(**init_kwargs)
        except Exception as e:
            print(f"[Avviso W&B] Impossibile inizializzare wandb: {e}")
    else:
        wandb.run.name = run_name
        wandb.config.update(config_dict, allow_val_change=True)

    return run_name


def load_eval_cases_jsonl(
    file_path: str, max_cases: int | None = 300
) -> list[dict]:
    """Carica i casi di validazione/test dal file JSONL stratificando sulle policy disponibili."""
    if not os.path.exists(file_path):
        return []
    all_cases: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_cases.append(json.loads(line))
            if max_cases and len(all_cases) >= max_cases:
                break
    return all_cases


def pipeline_ithaca_finetuning(
    checkpoint_path: str = "checkpoints/ithaca/checkpoint_v1.pkl",
    output_dir: str = "checkpoints/ithaca",
    train_path: str = "data/ithaca/tlg_train.jsonl",
    val_path: str = "data/ithaca/tlg_val.jsonl",
    test_path: str = "data/ithaca/tlg_test.jsonl",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    warmup_steps: int | None = None,
    warmup_ratio: float = 0.1,
    weight_decay: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    num_layers_to_freeze: int = 0,
    dropout_rate: float = 0.1,
    freeze_attribution_heads: bool = True,
    strategy: str = "hcb_best_to_worst",
    beam_size: int = 20,
    temperature: float = 1.0,
    max_eval_cases: int = 300,
    evaluate_hcb: bool = True,
    evaluate_on_test: bool = False,
    pre_ft_cached_metrics: dict[str, dict[str, float]] | None = None,
    eval_cases_override: list[dict] | None = None,
    train_data_override: list[dict[str, np.ndarray]] | None = None,
    val_data_override: list[dict[str, np.ndarray]] | None = None,
    save_checkpoint: bool = True,
    log_to_wandb: bool = True,
    run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_job_type: str | None = None,
    extra_config: dict | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Pipeline programmatica completa di fine-tuning e valutazione per Ithaca (JAX/Flax).
    Supporta:
    - Freezing selettivo delle teste di attribuzione e dei primi N blocchi BigBird.
    - Ottimizzazione degli iperparametri di training (lr, wd, warmup_ratio, scheduler, dropout)
      e di decodifica HCB (strategy, temperature, beam_size).
    - Valutazione HCB Beam Search con metriche lessicali (Top-K EM, CER, MRR) e semantiche
      (Cluster Inclusion Rate, Cosine Similarity).
    - Logging completo su Weights & Biases, incluse le tabelle `confronto_pre_post_ft_{policy}`
      e `predictions_table`.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

    ensure_checkpoint_exists(checkpoint_path)
    ckpt_data = load_ithaca_checkpoint(checkpoint_path)
    base_params = ckpt_data["params"]
    params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), base_params)
    alphabet = ckpt_data["alphabet"]
    config = dict(ckpt_data["config"])

    # Applica il dropout_rate scelto (es. da Optuna)
    config["dropout_rate"] = float(dropout_rate)
    config["attention_dropout_rate"] = float(dropout_rate)

    try:
        from ithaca.models.model import Model
    except ImportError:
        sys.path.insert(0, "packages/ithaca_engine")
        from ithaca.models.model import Model

    model = Model(**config)
    max_len = int(config.get("max_len", 768))

    if train_data_override is not None:
        train_data = train_data_override
    else:
        print(f"Caricamento e codifica dataset di training da '{train_path}' (max_len={max_len})...")
        train_data = load_jsonl_dataset(train_path, alphabet, max_len=max_len)

    if val_data_override is not None:
        val_data = val_data_override
    else:
        print(f"Caricamento e codifica dataset di validazione da '{val_path}' (max_len={max_len})...")
        val_data = load_jsonl_dataset(val_path, alphabet, max_len=max_len)

    print(f"Casi di training: {len(train_data)} | Casi di validazione: {len(val_data)}")

    steps_per_epoch = max(1, len(train_data) // batch_size)
    total_steps = max(2, steps_per_epoch * epochs)
    computed_warmup_steps = (
        warmup_steps if warmup_steps is not None else int(warmup_ratio * total_steps)
    )

    if log_to_wandb:
        _init_wandb_ithaca(
            checkpoint_path=checkpoint_path,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            num_layers_to_freeze=num_layers_to_freeze,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            dropout_rate=dropout_rate,
            strategy=strategy,
            temperature=temperature,
            beam_size=beam_size,
            run_name=run_name,
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
            wandb_job_type=wandb_job_type,
            extra_config=extra_config,
        )

    print(
        f"Configurazione ottimizzatore Optax (AdamW, LR={lr:.2e}, wd={weight_decay}, "
        f"sched={lr_scheduler_type}, warmup={computed_warmup_steps}, "
        f"freeze_layers={num_layers_to_freeze}, dropout={dropout_rate}, "
        f"freeze_attribution={freeze_attribution_heads})..."
    )
    optimizer = create_optimizer_with_freezing(
        learning_rate=lr,
        warmup_steps=computed_warmup_steps,
        total_steps=total_steps,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        num_layers_to_freeze=num_layers_to_freeze,
        freeze_attribution_heads=freeze_attribution_heads,
        params=params,
    )
    opt_state = optimizer.init(params)
    rng = jax.random.PRNGKey(seed)

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
        if isinstance(outputs, (tuple, list)):
            if len(outputs) > 0 and isinstance(outputs[0], (tuple, list)):
                logits_char = outputs[0][2]
            else:
                logits_char = outputs[2]
        elif isinstance(outputs, dict):
            logits_char = (
                outputs.get("char") or outputs.get("logits_mask") or outputs.get("mask")
            )
        else:
            logits_char = outputs

        vocab_size = logits_char.shape[-1]
        one_hot_targets = jax.nn.one_hot(batch["target_char"], vocab_size)
        log_probs = jax.nn.log_softmax(logits_char, axis=-1)
        per_token_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)

        mask = batch["mask_pos"]
        total_masked = jnp.maximum(jnp.sum(mask), 1.0)
        masked_loss = jnp.sum(per_token_loss * mask) / total_masked

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
    best_val_acc = 0.0
    best_params = params

    out_dir_path = Path(output_dir)
    if save_checkpoint:
        out_dir_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir_path / "checkpoint_tlg.pkl"

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_gen = create_batch_generator(
            train_data, batch_size=batch_size, shuffle=True
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
                    f"\rEpoca {epoch}/{epochs} | Step {step}/{steps_per_epoch} | "
                    f"Loss: {np.mean(train_losses[-20:]):.4f} | "
                    f"Char Acc: {np.mean(train_accs[-20:]) * 100:.2f}%",
                    end="",
                    flush=True,
                )

        val_gen = create_batch_generator(
            val_data, batch_size=batch_size, shuffle=False
        )
        val_losses, val_accs = [], []
        for v_batch in val_gen:
            v_loss, v_acc = eval_step(params, v_batch)
            val_losses.append(float(v_loss))
            val_accs.append(float(v_acc))

        epoch_time = time.time() - epoch_start
        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        mean_train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        mean_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        mean_val_acc = float(np.mean(val_accs)) if val_accs else 0.0

        print(
            f"\n--> Fine Epoca {epoch}: "
            f"Val Loss: {mean_val_loss:.4f} | Val Char Acc: {mean_val_acc * 100:.2f}% | "
            f"Tempo: {epoch_time:.1f}s"
        )

        if log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/loss": mean_train_loss,
                            "train/char_acc": mean_train_acc * 100.0,
                            "val/loss": mean_val_loss,
                            "val/char_acc": mean_val_acc * 100.0,
                        }
                    )
            except Exception:
                pass

        if mean_val_loss <= best_val_loss:
            best_val_loss = mean_val_loss
            best_val_acc = mean_val_acc
            best_params = params
            if save_checkpoint:
                print(
                    f"Nuovo miglior punteggio validazione ({best_val_loss:.4f}). Salvataggio..."
                )
                save_finetuned_checkpoint(
                    output_path=str(best_ckpt_path),
                    params=best_params,
                    alphabet=alphabet,
                    config=config,
                    metadata={
                        "epochs": epoch,
                        "val_loss": best_val_loss,
                        "val_char_acc": best_val_acc,
                        "base_model": "deepmind/ithaca",
                        "dataset": train_path,
                        "frozen_heads": freeze_attribution_heads,
                        "num_layers_to_freeze": num_layers_to_freeze,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "warmup_ratio": warmup_ratio,
                        "lr_scheduler_type": lr_scheduler_type,
                        "dropout_rate": dropout_rate,
                        "strategy": strategy,
                        "temperature": temperature,
                        "beam_size": beam_size,
                    },
                )

    if save_checkpoint:
        config_export_path = out_dir_path / "config.json"
        export_cfg = dict(config)
        export_cfg["hcb_decoding"] = {
            "strategy": strategy,
            "temperature": temperature,
            "beam_size": beam_size,
        }
        with open(config_export_path, "w", encoding="utf-8") as f:
            json.dump(export_cfg, f, indent=2)
        print(f"Checkpoint migliore salvato in: {best_ckpt_path}")
        print(f"Configurazione esportata in: {config_export_path}")

    # Valutazione HCB Beam Search e generazione tabelle Pre-FT vs Post-FT
    pre_metrics: dict[str, dict[str, float]] = {}
    post_metrics: dict[str, dict[str, float]] = {}
    eval_overall: dict[str, float] = {
        "val_loss": best_val_loss,
        "val_char_acc": best_val_acc * 100.0,
    }

    if evaluate_hcb or evaluate_on_test:
        from models.ithaca.evaluation.compare import (
            evaluate_checkpoint_on_cases,
            log_ithaca_comparison_to_wandb,
            render_comparison_table,
        )

        target_eval_path = test_path if evaluate_on_test else val_path
        eval_cases = (
            eval_cases_override
            if eval_cases_override is not None
            else load_eval_cases_jsonl(target_eval_path, max_cases=max_eval_cases)
        )

        if eval_cases:
            split_prefix = "test" if evaluate_on_test else "eval"
            if pre_ft_cached_metrics is not None:
                pre_metrics = pre_ft_cached_metrics
            else:
                print(f"\nValutazione baseline Pre-FT su {len(eval_cases)} casi ({split_prefix})...")
                pre_metrics = evaluate_checkpoint_on_cases(
                    checkpoint_path=checkpoint_path,
                    cases=eval_cases,
                    strategy=strategy,
                    beam_size=beam_size,
                    temperature=1.0,
                    params_override=base_params,
                    alphabet_override=alphabet,
                    config_override=config,
                    compute_semantic_metrics=True,
                )

            print(f"\nValutazione modello Post-FT su {len(eval_cases)} casi ({split_prefix})...")
            post_metrics, post_preds = evaluate_checkpoint_on_cases(
                checkpoint_path=str(best_ckpt_path) if save_checkpoint else checkpoint_path,
                cases=eval_cases,
                strategy=strategy,
                beam_size=beam_size,
                temperature=temperature,
                params_override=best_params,
                alphabet_override=alphabet,
                config_override=config,
                compute_semantic_metrics=True,
                return_predictions=True,
            )

            eval_overall.update(post_metrics.get("overall", {}))

            table_md = render_comparison_table(pre_metrics, post_metrics)
            print("\n" + "=" * 75)
            print(f"       CONFRONTO PRE-FT VS POST-FT ({split_prefix.upper()} SET)")
            print("=" * 75)
            print(table_md)
            print("=" * 75 + "\n")

            if log_to_wandb:
                try:
                    import wandb

                    if wandb.run is not None:
                        # Log anche con prefisso eval_* diretto per uniformità con Optuna / BERT
                        overall_m = post_metrics.get("overall", {})
                        wandb.log(
                            {
                                f"eval/{k}": float(v)
                                for k, v in overall_m.items()
                                if isinstance(v, (int, float))
                            }
                        )
                        log_ithaca_comparison_to_wandb(
                            pre_metrics_by_policy=pre_metrics,
                            post_metrics_by_policy=post_metrics,
                            predictions_records=post_preds,
                            split_prefix=split_prefix,
                        )
                except Exception as e:
                    print(f"[Avviso W&B] Errore durante il log delle tabelle comparative: {e}")

    if log_to_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    return {
        "best_params": best_params,
        "alphabet": alphabet,
        "config": config,
        "eval_metrics": eval_overall,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
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
        "--test_path",
        type=str,
        default="data/ithaca/tlg_test.jsonl",
        help="Percorso al dataset di test (JSONL)",
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
        default=None,
        help="Step espliciti di warmup (se omesso, usa warmup_ratio)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Frazione di step totali dedicata al warmup (default: 0.1)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay per AdamW (default: 1e-4)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Tipo di scheduler del learning rate (default: cosine)",
    )
    parser.add_argument(
        "--num_layers_to_freeze",
        type=int,
        default=0,
        help="Numero di blocchi encoder BigBird iniziali da congelare (0..5, default: 0)",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Dropout rate per i blocchi del Transformer (default: 0.1)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="hcb_best_to_worst",
        choices=[
            "hcb_best_to_worst",
            "hcb_left_to_right",
            "standard_best_to_worst",
            "standard_left_to_right",
        ],
        help="Strategia di decodifica HCB Beam Search (default: hcb_best_to_worst)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=20,
        help="Dimensione del beam per la valutazione (default: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura di decodifica HCB (default: 1.0)",
    )
    parser.add_argument(
        "--max_eval_cases",
        type=int,
        default=300,
        help="Numero massimo di casi per la valutazione HCB (default: 300)",
    )
    parser.add_argument(
        "--evaluate_on_test",
        action="store_true",
        help="Esegue la valutazione comparativa Pre-FT vs Post-FT sul test set a fine training",
    )
    parser.add_argument(
        "--no_eval_hcb",
        action="store_true",
        help="Salta la valutazione HCB Beam Search finale",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disabilita il logging su Weights & Biases",
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

    pipeline_ithaca_finetuning(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        num_layers_to_freeze=args.num_layers_to_freeze,
        dropout_rate=args.dropout_rate,
        freeze_attribution_heads=not args.no_freeze,
        strategy=args.strategy,
        beam_size=args.beam_size,
        temperature=args.temperature,
        max_eval_cases=args.max_eval_cases,
        evaluate_hcb=not args.no_eval_hcb,
        evaluate_on_test=args.evaluate_on_test,
        log_to_wandb=not args.no_wandb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

