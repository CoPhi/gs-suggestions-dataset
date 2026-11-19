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
    get_compact_ngram_checkpoint_path,
    get_ngram_checkpoint_path,
)
from backend.core.cleaner import load_abs
from models.ngrams.storage import export_nltk_models_to_sqlite
from models.ngrams.train.training import pipeline_train


def convert_pkl_to_sqlite(pkl_path: Path, output_path: Path | None = None) -> Path:
    """Converte un checkpoint `.pkl.zlib` NLTK esistente nel formato compatto SQLite condiviso."""
    t0 = time.time()
    print(f"Caricamento checkpoint pickle '{pkl_path}' per conversione SQLite...")
    payload = pickle.loads(zlib.decompress(pkl_path.read_bytes()))
    n = int(payload["n"])
    target = output_path or get_compact_ngram_checkpoint_path(n=n, base_dir=pkl_path.parent)
    export_nltk_models_to_sqlite(
        global_model=payload["global_model"],
        domain_model=payload["domain_model"],
        output_path=target,
        n=n,
    )
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"Checkpoint SQLite compatto salvato in '{target}' ({size_mb:.2f} MB) in {time.time() - t0:.2f}s.")
    return target


def export_checkpoints(
    output_dir: Path = NGRAM_CHECKPOINTS_DIR,
    lm_types: list[str] | None = None,
    gamma: float = GAMMA,
    n: int = N,
    budget: int | None = None,
    reuse_data: bool = True,
    output_format: str = "sqlite",
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
        output_format: `'sqlite'` (compatto condiviso, default) oppure `'pkl'` (legacy NLTK).

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
    sqlite_exported = False

    for lm_type in targets:
        lm_upper = lm_type.upper()
        if output_format == "sqlite" and sqlite_exported:
            continue

        current_gamma = gamma if lm_upper != "MLE" else None
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

        if output_format == "sqlite":
            target_path = get_compact_ngram_checkpoint_path(n=n, base_dir=output_dir)
            print(f"Esportazione SQLite compatta condivisa in '{target_path.name}'...")
            export_nltk_models_to_sqlite(
                global_model=global_model,
                domain_model=domain_model,
                output_path=target_path,
                n=n,
            )
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"Checkpoint SQLite salvato con successo: {target_path} ({size_mb:.2f} MB)")
            exported_files.append(target_path)
            sqlite_exported = True
        else:
            target_path = get_ngram_checkpoint_path(
                lm_type=lm_upper, n=n, gamma=current_gamma, base_dir=output_dir
            )
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
    parser.add_argument(
        "--format",
        choices=["sqlite", "pkl"],
        default="sqlite",
        help="Formato di esportazione: 'sqlite' compatto condiviso (default) oppure 'pkl' legacy.",
    )
    parser.add_argument(
        "--from-pkl",
        type=Path,
        default=None,
        help="Converte direttamente un checkpoint .pkl.zlib esistente in .sqlite senza riaddestrare.",
    )

    args = parser.parse_args()

    if args.from_pkl is not None:
        out_file = convert_pkl_to_sqlite(
            pkl_path=args.from_pkl,
            output_path=get_compact_ngram_checkpoint_path(n=args.n, base_dir=args.output_dir),
        )
        print(f"Conversione completata: {out_file}")
        return

    print("==========================================================")
    print(" Esportazione Checkpoint N-gram per Docker gs-api")
    print(f" Output dir : {args.output_dir}")
    print(f" Formato    : {args.format}")
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
        output_format=args.format,
    )
    total_time = time.time() - start

    print("\n==========================================================")
    print(f"Completato in {total_time:.2f}s! File generati:")
    for f in files:
        print(f" - {f} ({f.stat().st_size / (1024 * 1024):.2f} MB)")
    print("==========================================================")


if __name__ == "__main__":
    main()
