from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from backend.core.cleaner import get_sentences, load_abs
from backend.core.preprocess import get_tokens_from_clean_text
from models.bert.finetuning import get_model_config

# Tipi e configurazioni

ModelType = Literal["bert", "ngram"]


class ModelConfig(TypedDict):
    """Schema tipizzato per la configurazione di un modello."""

    label: str
    checkpoint: str | None
    model_type: ModelType


@dataclass(frozen=True)
class _TokenizerConfig:
    """Configurazione immutabile per il tokenizer N-gram."""

    strip_diacritics: bool = False
    remove_punct: bool = False
    case_folding: str | None = None


_NGRAM_CONFIG = _TokenizerConfig(
    strip_diacritics=True,
    remove_punct=True,
    case_folding="upper",
)

# Helpers privati


@lru_cache(maxsize=None)
def _load_tokenizer(checkpoint: str) -> AutoTokenizer:
    """Carica e memorizza in cache il tokenizer BERT per un dato checkpoint."""
    return AutoTokenizer.from_pretrained(checkpoint)


def _annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.1f}",
    offset: float = 0.2,
    fontsize: int = 9,
) -> None:
    """Annota ogni barra con il proprio valore numerico."""
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _render_chart(
    stats_list: list[dict],
    title: str,
    figsize: tuple[int, int],
    save_path: str | Path | None,
) -> None:
    """Renderizza e (opzionalmente) salva il grafico a barre doppie."""
    labels = [s["label"] for s in stats_list]
    means = [s["mean"] for s in stats_list]
    variances = [s["variance"] for s in stats_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    bars_mean = ax1.bar(
        x - width / 2, means, width, label="Media", color="#4f98a3", alpha=0.85
    )
    bars_var = ax2.bar(
        x + width / 2, variances, width, label="Varianza", color="#e8af34", alpha=0.85
    )

    ax1.set_xlabel("Modello")
    ax1.set_ylabel("Media token", color="#4f98a3")
    ax2.set_ylabel("Varianza token", color="#e8af34")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor="#4f98a3")
    ax2.tick_params(axis="y", labelcolor="#e8af34")

    _annotate_bars(ax1, bars_mean)
    _annotate_bars(ax2, bars_var)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left")

    ax1.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")

    plt.show()


# API pubblica


def compute_corpus_token_stats(
    corpus_paths: list[str] | None = None,
    checkpoint: str | None = None,
    model_type: ModelType = "bert",
    *,
    preloaded_data: list | None = None,
) -> dict:
    """
    Calcola statistiche descrittive sui token per un insieme di corpus.

    Args:
        corpus_paths:   Lista di percorsi ai file di testo da analizzare.
                        Ignorato se ``preloaded_data`` è fornito.
        checkpoint:     Checkpoint HuggingFace (obbligatorio per BERT).
        model_type:     Tipo di modello: ``"bert"`` o ``"ngram"``.
        preloaded_data: Lista di blocchi anonimi già caricati da ``load_abs()``.
                        Se fornito, evita un ulteriore accesso al disco.

    Returns:
        Dizionario con ``mean``, ``variance``, ``std`` e ``count``.

    Raises:
        ValueError: Se ``model_type="bert"`` ma ``checkpoint`` non è fornito,
                    o se ``model_type`` non è riconosciuto.

    Examples:
        >>> compute_corpus_token_stats(["tlg", "DCLP"], None, "ngram")
        >>> compute_corpus_token_stats(["tlg", "DCLP"], "CNR-ILC/gs-AristoBERTo", "bert")
    """
    if model_type == "bert":
        if not checkpoint:
            raise ValueError("Il checkpoint è obbligatorio per i modelli BERT.")
        config = get_model_config(checkpoint)
        tokenizer = _load_tokenizer(checkpoint)
        tokenize_fn = tokenizer.tokenize

    elif model_type == "ngram":
        config = _NGRAM_CONFIG
        tokenize_fn = get_tokens_from_clean_text

    else:
        raise ValueError(
            f"model_type '{model_type}' non supportato. Usa 'bert' o 'ngram'."
        )

    # Usa i dati pre-caricati se disponibili, altrimenti legge dal disco
    data = (
        preloaded_data
        if preloaded_data is not None
        else load_abs(corpus_set=corpus_paths)
    )

    sentences = get_sentences(
        abs=data,
        strip_diacritics=(
            config.strip_diacritics
            if isinstance(config, _TokenizerConfig)
            else config.get("strip_diacritics", False)
        ),
        remove_punct=(
            config.remove_punct
            if isinstance(config, _TokenizerConfig)
            else config.get("remove_punct", False)
        ),
        case_folding=(
            config.case_folding
            if isinstance(config, _TokenizerConfig)
            else config.get("case_folding", None)
        ),
    )

    # Conteggio dei token
    if model_type == "ngram":
        token_counts = [len(s) for s in sentences if s]
    else:
        token_counts = [len(tokenize_fn(" ".join(s))) for s in sentences if s]

    if not token_counts:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0, "count": 0}

    arr = np.array(token_counts)

    return {
        "mean": float(arr.mean()),
        "variance": float(arr.var()),
        "std": float(arr.std()),
        "count": int(np.add.reduce(arr)),  # si sommano tutti i token di tutte le frasi
    }


def collect_stats(
    models: list[ModelConfig],
    corpus_paths: list[str] | None,
) -> list[dict]:
    """
    Raccoglie le statistiche di token per ogni modello senza toccare matplotlib.

    Il corpus viene caricato dal disco una sola volta e condiviso tra tutti i
    modelli. Il preprocessing delle frasi (specifico per ogni modello) e il
    calcolo delle statistiche vengono parallelizzati tramite un thread pool.

    Args:
        models:       Lista di configurazioni modello (vedi ``ModelConfig``).
        corpus_paths: Lista di corpus su cui calcolare le statistiche.

    Returns:
        Lista di dizionari ``{label, mean, variance, std, count}``.
    """
    data = load_abs(corpus_set=corpus_paths)

    def _compute(m: ModelConfig) -> dict:
        return {
            "label": m["label"],
            **compute_corpus_token_stats(
                checkpoint=m.get("checkpoint"),
                model_type=m["model_type"],
                preloaded_data=data,
            ),
        }

    with ThreadPoolExecutor() as executor:
        return list(executor.map(_compute, models))


def plot_token_stats(
    models: list[ModelConfig],
    corpus_paths: list[str] | None,
    title: str = "Token stats – blocchi anonimi (training set)",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | Path | None = "token_stats.png",
) -> None:
    """
    Plotta media e varianza del numero di token per ogni modello.

    Args:
        models:       Lista di configurazioni modello (vedi ``ModelConfig``).
        corpus_paths: Lista di corpus da analizzare (es. ``["tlg", "DCLP"]``).
        title:        Titolo del grafico.
        figsize:      Dimensione della figura matplotlib.
        save_path:    Percorso dove salvare il PNG (``None`` per non salvare).

    Examples:
        >>> plot_token_stats(
        ...     models=[
        ...         {"label": "AristoBERTo", "checkpoint": "CNR-ILC/gs-AristoBERTo", "model_type": "bert"},
        ...         {"label": "N-gram", "checkpoint": None, "model_type": "ngram"},
        ...     ],
        ...     corpus_paths=["tlg", "DCLP"],
        ... )
    """
    stats_list = collect_stats(models, corpus_paths)
    _render_chart(stats_list, title, figsize, save_path)


# Entrypoint CLI

_DEFAULT_MODELS: list[ModelConfig] = [
    {
        "label": "AristoBERTo",
        "checkpoint": "CNR-ILC/gs-aristoBERTo",
        "model_type": "bert",
    },
    {
        "label": "GreBERTa",
        "checkpoint": "CNR-ILC/gs-GreBerta",
        "model_type": "bert",
    },
    {
        "label": "Logion",
        "checkpoint": "CNR-ILC/gs-Logion",
        "model_type": "bert",
    },
    {
        "label": "N-gram",
        "checkpoint": None,
        "model_type": "ngram",
    },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mostra le statistiche descrittive relative ai token del train set per modello."
    )
    parser.add_argument(
        "--corpus", nargs="+", default=None, help="Corpus da analizzare."
    )
    parser.add_argument(
        "--save", default="token_stats.png", help="Percorso file PNG di output."
    )
    args = parser.parse_args()

    plot_token_stats(_DEFAULT_MODELS, corpus_paths=args.corpus, save_path=args.save)
