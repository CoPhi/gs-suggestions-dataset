import os
import sys
import gc
import json
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

# Assicura che la root del progetto sia nel sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bert.finetuning import ModelRegistry, HF_TOKEN, WANDB_API_KEY, wandb_login
from models.bert.finetuning.pipeline import (
    _load_eval_split,
    evaluate_metrics_on_test_set,
    generate_synthetic_cases,
)
from models.bert.evaluation.metrics import reset_scorer_cache

load_dotenv()


def load_cases(dataset_name: str, split: str = "test", max_cases: int = 300):
    """
    Carica i casi di valutazione dal dataset HuggingFace specificato.
    Se contiene già campi 'x' e 'y', usa _load_eval_split.
    Se è un dataset testuale grezzo (campo 'text'), genera casi sintetici.
    """
    print(f"Caricamento dataset di valutazione da '{dataset_name}' (split: {split})...")
    ds = load_dataset(dataset_name)

    target_split = split
    if target_split not in ds:
        # Fallback su uno split disponibile
        available = list(ds.keys())
        target_split = available[0]
        print(f"[Avviso] Split '{split}' non trovato in {dataset_name}. Utilizzo '{target_split}'.")

    split_data = ds[target_split]

    # Verifica se è un dataset di valutazione strutturato (x, y, gap_length)
    if "x" in split_data.column_names and "y" in split_data.column_names:
        cases = _load_eval_split(ds, target_split)
    else:
        print("Il dataset contiene testo non mascherato. Generazione di casi sintetici...")
        cases = generate_synthetic_cases(split_data, n=max_cases, max_gap=6, policy="default")

    if max_cases and len(cases) > max_cases:
        cases = cases[:max_cases]

    print(f"Caricati {len(cases)} casi di test per la valutazione.")
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Confronto comparativo Pre-FT (Baseline) vs Post-FT (Fine-tuned)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="CNR-ILC/gs-GreBerta",
        help="Checkpoint o percorso locale del modello Fine-Tuned (Post-FT)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Checkpoint del modello base (Pre-FT). Se omesso, viene dedotto dal ModelRegistry",
    )
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default="CNR-ILC/gs-dataset-eval",
        help="Dataset Hugging Face da utilizzare per la valutazione",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split da valutare ('test' o 'dev')",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=300,
        help="Numero massimo di casi da valutare",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Percorso opzionale per salvare i risultati in formato JSON",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Percorso opzionale per salvare la tabella comparativa in CSV",
    )
    args = parser.parse_args()

    if HF_TOKEN:
        login(token=HF_TOKEN)

    checkpoint = args.checkpoint
    base_model = args.base_model
    if not base_model:
        base_model = ModelRegistry().base_model_map.get(checkpoint, checkpoint)

    print("\n" + "=" * 80)
    print("               CONFRONTO METRICHE: PRE-FT VS POST-FT")
    print("=" * 80)
    print(f"Modello Pre-FT (Base):        {base_model}")
    print(f"Modello Post-FT (Fine-tuned): {checkpoint}")
    print(f"Dataset di valutazione:       {args.eval_dataset_name} [split: {args.split}]")
    print(f"Numero casi di test:          {args.max_cases}")
    print("=" * 80 + "\n")

    # 1. Caricamento dei casi di test
    cases = load_cases(args.eval_dataset_name, split=args.split, max_cases=args.max_cases)
    if not cases:
        print("Errore: nessun caso di test estratto.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilizzato per l'inferenza: {device.upper()}\n")

    # Pulizia scorer BERTScore cache
    reset_scorer_cache()

    # 2. Valutazione Modello Pre-FT
    print(f"--> [1/2] Valutazione baseline Pre-FT ({base_model})...")
    tokenizer_pre = AutoTokenizer.from_pretrained(base_model)
    model_pre = AutoModelForMaskedLM.from_pretrained(base_model).to(device)
    model_pre.eval()

    pre_ft_metrics = evaluate_metrics_on_test_set(
        split_name="pre_ft",
        cases=cases,
        model=model_pre,
        tokenizer=tokenizer_pre,
        checkpoint=checkpoint,
        max_cases=args.max_cases,
    )

    del model_pre
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 3. Valutazione Modello Post-FT
    print(f"\n--> [2/2] Valutazione modello Post-FT ({checkpoint})...")
    tokenizer_post = AutoTokenizer.from_pretrained(checkpoint)
    model_post = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model_post.eval()

    post_ft_metrics = evaluate_metrics_on_test_set(
        split_name="post_ft",
        cases=cases,
        model=model_post,
        tokenizer=tokenizer_post,
        checkpoint=checkpoint,
        max_cases=args.max_cases,
    )

    del model_post
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 4. Tabella di confronto
    comparison_keys = [
        ("top1", "Exact Match @1 (%)"),
        ("top5", "Exact Match @5 (%)"),
        ("top10", "Exact Match @10 (%)"),
        ("top20", "Exact Match @20 (%)"),
        ("bertscore_f1_top1", "BERTScore F1 @1 (%)"),
        ("bertscore_f1_top5", "BERTScore F1 @5 (%)"),
        ("bertscore_f1_top10", "BERTScore F1 @10 (%)"),
        ("bertscore_f1_top20", "BERTScore F1 @20 (%)"),
        ("cos_sim_top1_max", "CosSim Max @1 (%)"),
        ("cos_sim_top5_max", "CosSim Max @5 (%)"),
        ("cos_sim_top10_max", "CosSim Max @10 (%)"),
        ("cos_sim_top20_max", "CosSim Max @20 (%)"),
        ("cluster_inclusion_rate", "Cluster Inclusion Rate (%)"),
        ("mean_inclusion_margin", "Mean Inclusion Margin"),
        ("mean_gold_centroid_cosine_sim", "Gold Centroid CosSim (%)"),
    ]

    print("\n" + "=" * 80)
    print(f"       RISULTATI CONFRONTO: PRE-FT VS POST-FT (DATASET: {args.eval_dataset_name})")
    print("=" * 80)
    print(f"{'Metrica':<30} | {'Pre-FT':<12} | {'Post-FT':<12} | {'Delta':<12}")
    print("-" * 80)

    rows = []
    for key, label in comparison_keys:
        val_pre = float(pre_ft_metrics.get(key, 0.0))
        val_post = float(post_ft_metrics.get(key, 0.0))
        delta = val_post - val_pre
        
        is_pct = "(%)" in label
        delta_str = f"{delta:+.2f}%" if is_pct else f"{delta:+.4f}"
        pre_str = f"{val_pre:.2f}%" if is_pct else f"{val_pre:.4f}"
        post_str = f"{val_post:.2f}%" if is_pct else f"{val_post:.4f}"

        print(f"{label:<30} | {pre_str:>10} | {post_str:>10} | {delta_str:>10}")
        rows.append({
            "metric": label,
            "metric_key": key,
            "pre_ft": val_pre,
            "post_ft": val_post,
            "delta": delta,
        })

    print("=" * 80 + "\n")

    # 5. Salvataggio su file
    if args.output_json:
        out_data = {
            "checkpoint": checkpoint,
            "base_model": base_model,
            "eval_dataset": args.eval_dataset_name,
            "split": args.split,
            "n_cases": len(cases),
            "pre_ft_metrics": pre_ft_metrics,
            "post_ft_metrics": post_ft_metrics,
            "comparison": rows,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"Risultati JSON salvati in: {args.output_json}")

    if args.output_csv:
        df = pd.DataFrame(rows)
        df.to_csv(args.output_csv, index=False)
        print(f"Tabella CSV salvata in: {args.output_csv}")


if __name__ == "__main__":
    main()
