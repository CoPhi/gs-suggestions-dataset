"""
Script per l'esportazione offline dei checkpoint pre-addestrati dei modelli N-grammi.

Addestra i modelli statistici (Lidstone, MLE) partendo dai dati in DATA_PATH e
li serializza compressi con zlib in models/ngrams/checkpoints/. Questi checkpoint
vengono inclusi direttamente nell'immagine Docker di gs-api permettendo
un'inizializzazione istantanea senza dipendere dalla cartella data/ a runtime.
"""
from __future__ import annotations

import argparse
import pickle
import time
import zlib
from pathlib import Path

from backend.config.settings import (
    DATA_PATH,
    GAMMA,
    LM_TYPES,
    N,
    NGRAM_CHECKPOINTS_DIR,
    get_ngram_checkpoint_path,
)
from backend.core.cleaner import load_abs
from models.ngrams.train.training import pipeline_train


def export_checkpoints(
    output_dir: Path = NGRAM_CHECKPOINTS_DIR,
    lm_types: list[str] | None = None,
    gamma: float = GAMMA,
    n: int = N,
    budget: int | None = None,
    reuse_data: bool = True,
) -> list[Path]:
    """
    Addestra ed esporta i checkpoint N-gram nella directory specificata.

    Args:
        output_dir: Cartella di destinazione per i checkpoint.
        lm_types: Lista di modelli da esportare (default: LM_TYPES, es. ["LIDSTONE", "MLE"]).
        gamma: Parametro gamma per lo smoothing Lidstone.
        n: Ordine degli n-grammi.
        budget: Percentuale opzionale di dati da utilizzare.
        reuse_data: Se True, carica i dati anonimi una sola volta per tutti i modelli.

    Returns:
        Lista dei percorsi dei file generati.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = lm_types or LM_TYPES

    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.json")):
        raise FileNotFoundError(
            f"Dataset non trovato in '{DATA_PATH}'. Assicurati che i file .json siano presenti."
        )

    preloaded_abs = None
    if reuse_data:
        print(f"Pre-caricamento dataset da '{DATA_PATH}' (budget={budget}%)...")
        t_load = time.time()
        preloaded_abs = load_abs(corpus_set=None, budget=budget)
        print(
            f"Caricati {len(preloaded_abs):,} blocchi in {time.time() - t_load:.2f}s."
        )

    exported_files: list[Path] = []

    for lm_type in targets:
        lm_upper = lm_type.upper()
        current_gamma = gamma if lm_upper != "MLE" else None
        target_path = get_ngram_checkpoint_path(
            lm_type=lm_upper, n=n, gamma=current_gamma, base_dir=output_dir
        )

        print(
            f"\n--- Addestramento {lm_upper} (N={n}, gamma={current_gamma}) ---"
        )
        t0 = time.time()
        global_model, domain_model, _ = pipeline_train(
            lm_type=lm_upper,
            gamma=current_gamma,
            n=n,
            budget=budget,
            train_abs=preloaded_abs,
        )
        t_train = time.time() - t0
        print(f"Modello {lm_upper} addestrato in {t_train:.2f}s.")

        payload = {
            "lm_type": lm_upper,
            "gamma": current_gamma,
            "n": n,
            "global_model": global_model,
            "domain_model": domain_model,
        }

        print(f"Serializzazione e compressione zlib in corso per '{target_path.name}'...")
        serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(serialized)

        target_path.write_bytes(compressed)
        size_mb = len(compressed) / (1024 * 1024)
        print(
            f"Checkpoint salvato con successo: {target_path} ({size_mb:.2f} MB)"
        )
        exported_files.append(target_path)

    return exported_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Esporta offline i checkpoint dei modelli linguistici N-gram per gs-api."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NGRAM_CHECKPOINTS_DIR,
        help=f"Directory di output per i checkpoint (default: {NGRAM_CHECKPOINTS_DIR})",
    )
    parser.add_argument(
        "--lm-types",
        nargs="+",
        default=LM_TYPES,
        help=f"Modelli da addestrare ed esportare (default: {' '.join(LM_TYPES)})",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help=f"Parametro gamma per Lidstone (default: {GAMMA})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help=f"Ordine n-gram (default: {N})",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Budget percentuale del dataset (default: 100% / None)",
    )
    parser.add_argument(
        "--no-reuse-data",
        action="store_true",
        help="Ricarica i dati per ogni modello invece di condividere la memoria",
    )

    args = parser.parse_args()

    print("==========================================================")
    print(" Esportazione Checkpoint N-gram per Docker gs-api")
    print(f" Output dir : {args.output_dir}")
    print(f" Modelli    : {args.lm_types}")
    print(f" N          : {args.n}")
    print(f" Gamma      : {args.gamma}")
    print(f" Budget     : {args.budget}%" if args.budget else " Budget     : 100% (completo)")
    print("==========================================================")

    start = time.time()
    files = export_checkpoints(
        output_dir=args.output_dir,
        lm_types=args.lm_types,
        gamma=args.gamma,
        n=args.n,
        budget=args.budget,
        reuse_data=not args.no_reuse_data,
    )
    total_time = time.time() - start

    print("\n==========================================================")
    print(f"Completato in {total_time:.2f}s! File generati:")
    for f in files:
        print(f" - {f} ({f.stat().st_size / (1024 * 1024):.2f} MB)")
    print("==========================================================")


if __name__ == "__main__":
    main()
