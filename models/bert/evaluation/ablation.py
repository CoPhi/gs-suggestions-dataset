"""
Studio di Ablation della Finestra di Contesto (Context Window Ablation Study).

Valuta la variazione delle prestazioni dei modelli BERT (Pre-FT vs Post-FT)
al variare della finestra di parole greche simmetriche o asimmetriche attorno
alla lacuna [.....].

Permette di misurare la resilienza dei modelli in assenza di macro-contesto
(condizione frequente nei piccoli frammenti papiracei reali).
"""

import os
import sys
import gc
import re
import json
import argparse
from pathlib import Path
from typing import Any

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Assicura che la root del progetto sia in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from models.bert.dataset.dev_set import DevCase
from models.bert.finetuning import (
    ModelRegistry,
    WANDB_API_KEY,
    WANDB_PROJECT,
    wandb_login,
)
from models.bert.finetuning.pipeline import (
    evaluate_metrics_on_test_set,
)
from scripts.evaluate_comparison import load_cases_by_policy

# Mappatura modelli canonici Pre-FT e Post-FT
DEFAULT_MODEL_PAIRS = [
    ("CNR-ILC/gs-aristoBERTo", "Jacobo/aristoBERTo"),
    ("CNR-ILC/gs-GreBerta", "bowphs/GreBerta"),
    ("CNR-ILC/gs-Logion", "cabrooks/LOGION-50k_wordpiece"),
]

MODEL_COLORS = {
    "aristoBERTo": "#2ca02c",  # Verde
    "GreBerta": "#1f77b4",     # Blu
    "Logion": "#ff7f0e",       # Arancione
}


def truncate_context_by_words(
    text: str,
    window_size: int | None = None,
    window_left: int | None = None,
    window_right: int | None = None,
) -> str:
    """
    Tronca il testo lasciando al massimo N parole greche a sinistra e a destra
    della parola che racchiude il segnaposto di lacuna `[.....]`.

    - Se sia `window_size` che `(window_left, window_right)` sono < 0 o None,
      il testo viene restituito integrale (Full Context).
    - Se `window_size == 0`, viene restituita solo la parola contenente la lacuna
      (zero parole di contesto esterno).
    """
    if window_size is not None and window_size >= 0:
        wl = window_size
        wr = window_size
    else:
        wl = window_left if (window_left is not None and window_left >= 0) else -1
        wr = window_right if (window_right is not None and window_right >= 0) else -1

    if wl < 0 and wr < 0:
        return text

    match = re.search(r"\[\.+\]", text)
    if not match:
        return text

    start_gap = match.start()
    end_gap = match.end()

    # Espandi a sinistra fino a whitespace o inizio stringa
    word_start = start_gap
    while word_start > 0 and not text[word_start - 1].isspace():
        word_start -= 1

    # Espandi a destra fino a whitespace o fine stringa
    word_end = end_gap
    while word_end < len(text) and not text[word_end].isspace():
        word_end += 1

    masked_word = text[word_start:word_end]
    left_part = text[:word_start].strip()
    right_part = text[word_end:].strip()

    left_words = left_part.split() if left_part else []
    right_words = right_part.split() if right_part else []

    if wl >= 0:
        left_words = left_words[-wl:] if wl > 0 else []
    if wr >= 0:
        right_words = right_words[:wr] if wr > 0 else []

    parts = []
    if left_words:
        parts.append(" ".join(left_words))
    parts.append(masked_word)
    if right_words:
        parts.append(" ".join(right_words))

    return " ".join(parts).strip()


def truncate_cases_by_words(cases: list[DevCase], window_size: int) -> list[DevCase]:
    """Crea una copia dei casi di test con il testo 'x' troncato alla finestra richiesta."""
    truncated = []
    for c in cases:
        truncated.append(
            DevCase(
                x=truncate_context_by_words(c.x, window_size=window_size),
                y=c.y,
                gap_length=c.gap_length,
                corpus_id=c.corpus_id,
                file_id=c.file_id,
                gap_type=c.gap_type,
            )
        )
    return truncated


def window_label(w: int) -> str:
    """Formatta l'etichetta visuale per la dimensione della finestra."""
    return "Full" if w < 0 else f"W={w}"


def plot_context_ablation_curves(
    results: dict[str, dict[str, dict[str, float]]],
    output_path: str = "eval/results/context_ablation_plot.png",
    windows: list[int] | None = None,
) -> None:
    """
    Genera grafici a linee delle prestazioni (EM@1, EM@5, BERTScore F1@1)
    al variare della dimensione della finestra per ciascun modello.
    """
    if not results:
        print("[Avviso Plot] Nessun risultato disponibile per generare i grafici.")
        return

    # Estrai le finestre ordinate se non fornite
    if windows is None:
        first_model = next(iter(results.keys()))
        win_keys = list(results[first_model].keys())
        # Ordina: numeri crescenti e poi 'Full'
        windows = sorted(
            [int(k.replace("W=", "")) for k in win_keys if k != "Full"]
        ) + ([-1] if "Full" in win_keys else [])

    x_labels = [window_label(w) for w in windows]
    x_indices = list(range(len(x_labels)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    metrics_to_plot = [
        ("top1", "Exact Match @1 (%)", axes[0]),
        ("top5", "Exact Match @5 (%)", axes[1]),
        ("bertscore_f1_top1", "BERTScore F1 @1 (%)", axes[2]),
    ]

    for model_name, win_dict in results.items():
        is_post_ft = "gs-" in model_name or "CNR-ILC" in model_name
        # Identifica la famiglia per il colore
        family = "GreBerta"
        for fam in ["aristoBERTo", "GreBerta", "Logion"]:
            if fam.lower() in model_name.lower():
                family = fam
                break

        color = MODEL_COLORS.get(family, "#333333")
        linestyle = "-" if is_post_ft else "--"
        marker = "o" if is_post_ft else "s"
        linewidth = 2.2 if is_post_ft else 1.5
        alpha = 1.0 if is_post_ft else 0.8
        label_name = f"{family} ({'Post-FT' if is_post_ft else 'Pre-FT'})"

        for metric_key, metric_title, ax in metrics_to_plot:
            y_vals = []
            for w in windows:
                lbl = window_label(w)
                m = win_dict.get(lbl, {})
                y_vals.append(m.get(metric_key, 0.0))

            ax.plot(
                x_indices,
                y_vals,
                label=label_name,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=linewidth,
                alpha=alpha,
                markersize=6,
            )

    for metric_key, metric_title, ax in metrics_to_plot:
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Finestra di Contesto (Parole a SX / DX)", fontsize=10)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Legenda unica
    handles, labels = axes[0].get_legend_handles_labels()
    # Rimuovi duplicati mantenendo l'ordine
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=min(6, len(by_label)),
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Grafico Salvato] Curve di ablation esportate con successo su: {output_path}")


def generate_markdown_report(
    results: dict[str, dict[str, dict[str, float]]],
    dataset_name: str,
    split: str,
    max_cases: int,
    windows: list[int],
    output_path: str = "eval/results/context_ablation_report.md",
) -> str:
    """Genera un report Markdown strutturato con tabelle comparative per finestra."""
    x_labels = [window_label(w) for w in windows]

    md = []
    md.append("# Studio di Ablation della Finestra di Contesto (GreekSchools)")
    md.append("")
    md.append(f"- **Dataset di Valutazione**: `{dataset_name}` (`{split}` split)")
    md.append(f"- **Numero Casi di Test**: {max_cases}")
    md.append(f"- **Unità di Misura Finestra**: Parole greche simmetriche attorno a `[.....]`")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Exact Match @1 (%) al Variare della Finestra")
    md.append("")

    header_cols = ["Modello", "Stato"] + x_labels + ["Delta (Full → W=min)"]
    md.append("| " + " | ".join(header_cols) + " |")
    md.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for model_name, win_dict in results.items():
        is_post_ft = "gs-" in model_name or "CNR-ILC" in model_name
        state = "Post-FT" if is_post_ft else "Pre-FT"
        short_name = model_name.split("/")[-1].replace("gs-", "")

        row = [short_name, state]
        vals = []
        for w in windows:
            val = win_dict.get(window_label(w), {}).get("top1", 0.0)
            row.append(f"{val:.2f}%")
            vals.append(val)

        # Delta drop dal full context alla finestra minima
        full_val = win_dict.get("Full", {}).get("top1", vals[-1])
        min_val = vals[0]
        delta_drop = min_val - full_val
        row.append(f"{delta_drop:+.2f}%")

        md.append("| " + " | ".join(row) + " |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. BERTScore F1 @1 (%) al Variare della Finestra")
    md.append("")

    md.append("| " + " | ".join(header_cols) + " |")
    md.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for model_name, win_dict in results.items():
        is_post_ft = "gs-" in model_name or "CNR-ILC" in model_name
        state = "Post-FT" if is_post_ft else "Pre-FT"
        short_name = model_name.split("/")[-1].replace("gs-", "")

        row = [short_name, state]
        vals = []
        for w in windows:
            val = win_dict.get(window_label(w), {}).get("bertscore_f1_top1", 0.0)
            row.append(f"{val:.2f}%")
            vals.append(val)

        full_val = win_dict.get("Full", {}).get("bertscore_f1_top1", vals[-1])
        min_val = vals[0]
        delta_drop = min_val - full_val
        row.append(f"{delta_drop:+.2f}%")

        md.append("| " + " | ".join(row) + " |")

    md_content = "\n".join(md)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Report Salvato] Report Markdown esportato su: {output_path}")
    return md_content


def run_ablation_study(
    models: list[str],
    windows: list[int],
    dataset_name: str,
    split: str = "test",
    policy: str = "default",
    max_cases: int = 50,
    K: int = 10,
    beam_size: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: str = "eval/results/context_ablation_results.json",
    output_plot: str = "eval/results/context_ablation_plot.png",
    output_md: str = "eval/results/context_ablation_report.md",
    log_to_wandb: bool = False,
    wandb_project: str = WANDB_PROJECT,
    wandb_group: str = "context_ablation",
) -> dict:
    """Esegue l'intero pipeline di ablation della finestra di contesto sui modelli specificati."""
    print("=" * 80)
    print("      STUDIO DI ABLATION: FINESTRA DI CONTESTO (WORD-BASED)")
    print("=" * 80)
    print(f"Modelli da valutare:   {models}")
    print(f"Finestre testate (W):  {[window_label(w) for w in windows]}")
    print(f"Dataset:               {dataset_name} [split: {split}, policy: {policy}]")
    print(f"Numero casi di test:   {max_cases}")
    print(f"Dispositivo:           {device}")
    print("=" * 80 + "\n")

    # 1. Caricamento una tantum dei casi di test originali
    policy_cases_dict = load_cases_by_policy(
        dataset_name=dataset_name,
        split=split,
        max_cases=max_cases,
        policy=policy,
    )
    raw_cases = policy_cases_dict.get(policy) or next(iter(policy_cases_dict.values()))
    print(f"Casi di test inizializzati con successo: {len(raw_cases)}")

    ablation_results: dict[str, dict[str, dict[str, float]]] = {}

    # 2. Iterazione sui modelli
    for model_idx, model_name in enumerate(models, start=1):
        print("\n" + "#" * 80)
        print(f"[{model_idx}/{len(models)}] Caricamento Modello: {model_name}")
        print("#" * 80)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()

        ablation_results[model_name] = {}

        for w in windows:
            w_lbl = window_label(w)
            print(f"\n---> Valutazione {model_name} su Finestra {w_lbl}...")

            # Creazione dataset tagliato a W parole
            truncated_cases = truncate_cases_by_words(raw_cases, window_size=w)

            metrics = evaluate_metrics_on_test_set(
                split_name=f"ablation_{w_lbl}",
                cases=truncated_cases,
                model=model,
                tokenizer=tokenizer,
                checkpoint=model_name,
                max_cases=max_cases,
                K=K,
                beam_size=beam_size,
            )

            ablation_results[model_name][w_lbl] = metrics

        # Pulizia VRAM
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 3. Salvataggio JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Risultati Salvati] File JSON esportato su: {output_json}")

    # 4. Generazione grafici
    plot_context_ablation_curves(ablation_results, output_path=output_plot, windows=windows)

    # 5. Generazione report Markdown
    generate_markdown_report(
        results=ablation_results,
        dataset_name=dataset_name,
        split=split,
        max_cases=max_cases,
        windows=windows,
        output_path=output_md,
    )

    # 6. Tracciamento opzionale su Weights & Biases
    if log_to_wandb:
        try:
            import wandb

            if WANDB_API_KEY:
                wandb_login()

            ds_short = dataset_name.split("/")[-1]
            wandb.init(
                project=wandb_project,
                name=f"context_ablation_{ds_short}",
                group=wandb_group,
                tags=["context_ablation", ds_short, "hcb"],
                job_type="ablation_study",
                config={
                    "models": models,
                    "windows": windows,
                    "eval_dataset": dataset_name,
                    "split": split,
                    "policy": policy,
                    "max_cases": max_cases,
                    "K": K,
                    "beam_size": beam_size,
                },
                reinit=True,
            )

            # Tabella riassuntiva W&B
            table_cols = ["Modello", "Stato", "Finestra", "EM@1 (%)", "EM@5 (%)", "BERTScore F1 (%)"]
            wandb_tbl = wandb.Table(columns=table_cols)

            for m_name, win_dict in ablation_results.items():
                is_post_ft = "gs-" in m_name or "CNR-ILC" in m_name
                state = "Post-FT" if is_post_ft else "Pre-FT"
                for w in windows:
                    lbl = window_label(w)
                    m = win_dict.get(lbl, {})
                    wandb_tbl.add_data(
                        m_name,
                        state,
                        lbl,
                        m.get("top1", 0.0),
                        m.get("top5", 0.0),
                        m.get("bertscore_f1_top1", 0.0),
                    )

            wandb.log({
                "context_ablation_table": wandb_tbl,
                "context_ablation_curves": wandb.Image(output_plot),
            })
            wandb.finish()
            print("[W&B] Metriche e grafici pubblicati con successo su Weights & Biases.")
        except Exception as e:
            print(f"[Avviso W&B] Impossibile sincronizzare i risultati con WandB: {e}")

    return ablation_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Studio di ablation della finestra di contesto su modelli BERT."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Modelli da valutare: 'all' (tutti i 6 modelli pre e post-FT), "
            "'ft' (solo fine-tuned), 'base' (solo pre-addestrati), "
            "o lista separata da virgole di checkpoint HuggingFace."
        ),
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="2,5,10,25,50,100,-1",
        help="Elenco separato da virgole delle finestre di parole per lato (-1 = full context).",
    )
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default="CNR-ILC/gs-dataset-tlg-uncased",
        help="Dataset di valutazione da utilizzare.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split del dataset di valutazione (default: 'test').",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="default",
        choices=["default", "word", "suffix"],
        help="Policy di generazione delle lacune sintetiche.",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=50,
        help="Numero massimo di casi di test da valutare (default: 50).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Numero di suggerimenti Top-K generati da HCB (default: 10).",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="Ampiezza del beam search in fill_mask (default: 10).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="eval/results/context_ablation_results.json",
        help="Percorso di output per il file JSON.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="eval/results/context_ablation_plot.png",
        help="Percorso di output per il grafico comparativo PNG.",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="eval/results/context_ablation_report.md",
        help="Percorso di output per il report Markdown riassuntivo.",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Abilita la registrazione dei risultati su Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=WANDB_PROJECT,
        help="Nome del progetto Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="context_ablation",
        help="Gruppo per raggruppare le run di ablation su W&B.",
    )
    return parser.parse_args()


def resolve_models_list(arg_models: str) -> list[str]:
    """Risolve l'argomento stringa dei modelli in una lista di checkpoint."""
    if arg_models == "all":
        # Tutti e 6: Pre-FT e Post-FT
        selected = []
        for ft_ckpt, base_ckpt in DEFAULT_MODEL_PAIRS:
            selected.extend([base_ckpt, ft_ckpt])
        return selected
    elif arg_models in ("ft", "post-ft"):
        return [ft for ft, _ in DEFAULT_MODEL_PAIRS]
    elif arg_models in ("base", "pre-ft"):
        return [base for _, base in DEFAULT_MODEL_PAIRS]
    else:
        return [m.strip() for m in arg_models.split(",") if m.strip()]


def main():
    args = parse_args()
    model_list = resolve_models_list(args.models)
    windows_list = [int(w.strip()) for w in args.windows.split(",") if w.strip()]

    run_ablation_study(
        models=model_list,
        windows=windows_list,
        dataset_name=args.eval_dataset_name,
        split=args.split,
        policy=args.policy,
        max_cases=args.max_cases,
        K=args.K,
        beam_size=args.beam_size,
        output_json=args.output_json,
        output_plot=args.output_plot,
        output_md=args.output_md,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
    )


if __name__ == "__main__":
    main()
