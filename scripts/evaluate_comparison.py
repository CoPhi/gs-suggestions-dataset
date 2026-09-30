import os
import sys
import gc
import json
import argparse
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForMaskedLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login

# Assicura che la root del progetto sia nel sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.bert.finetuning import (
    ModelRegistry,
    HF_TOKEN,
    WANDB_API_KEY,
    wandb_login,
    get_model_config,
)
from models.bert.finetuning.pipeline import (
    _load_eval_split,
    evaluate_metrics_on_test_set,
    generate_synthetic_cases,
)
from models.bert.evaluation.metrics import reset_scorer_cache
from models.bert.finetuning.model_card import (
    generate_model_card,
    save_model_card,
    push_model_card_to_hub,
)

load_dotenv()


def load_cases_by_policy(
    dataset_name: str,
    split: str = "test",
    max_cases: int = 300,
    policy: str = "default",
) -> dict[str, list]:
    """
    Carica i casi di valutazione dal dataset HuggingFace specificato.
    Se contiene già campi 'x' e 'y', usa _load_eval_split.
    Se è un dataset testuale grezzo (campo 'text'), genera casi sintetici con la policy richiesta.
    Restituisce un dizionario policy -> lista di DevCase.
    """
    print(f"Caricamento dataset di valutazione da '{dataset_name}' (split: {split})...")
    ds = load_dataset(dataset_name)

    target_split = split
    if target_split not in ds:
        if "train" in ds and len(ds) == 1:
            print(
                f"[{dataset_name}] contiene solo lo split 'train'. "
                f"Esecuzione split deterministico train/dev/test (seed=42)..."
            )
            split_1 = ds["train"].train_test_split(test_size=0.1, seed=42)
            split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
            ds = DatasetDict(
                {
                    "train": split_1["train"],
                    "dev": split_2["train"],
                    "test": split_2["test"],
                }
            )
            split_data = ds[target_split]
        else:
            available = list(ds.keys())
            target_split = available[0]
            print(f"[Avviso] Split '{split}' non trovato in {dataset_name}. Utilizzo '{target_split}'.")
            split_data = ds[target_split]
    else:
        split_data = ds[target_split]

    # Verifica se è un dataset di valutazione strutturato (x, y, gap_length)
    if "x" in split_data.column_names and "y" in split_data.column_names:
        cases = _load_eval_split(ds, target_split)
        if max_cases and len(cases) > max_cases:
            cases = cases[:max_cases]
        print(f"Caricati {len(cases)} casi di test pre-strutturati.")
        return {"default": cases}
    else:
        print(f"Il dataset contiene testo non mascherato. Generazione di casi sintetici (policy: {policy})...")
        policies = ["default", "word", "suffix"] if policy == "all" else [policy]
        cases_by_policy = {}
        for pol in policies:
            cases = generate_synthetic_cases(split_data, n=max_cases, max_gap=6, policy=pol)
            if max_cases and len(cases) > max_cases:
                cases = cases[:max_cases]
            cases_by_policy[pol] = cases
            print(f"Generati {len(cases)} casi per la policy '{pol}'.")
        return cases_by_policy


def main():
    parser = argparse.ArgumentParser(
        description="Confronto comparativo Pre-FT (Baseline) vs Post-FT (Fine-tuned) e pubblicazione Model Card"
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
        default="CNR-ILC/gs-dataset-tlg-uncased",
        help="Dataset Hugging Face da utilizzare per la valutazione (default: CNR-ILC/gs-dataset-tlg-uncased)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split da valutare ('test' o 'dev')",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="default",
        choices=["default", "word", "suffix", "all"],
        help="Policy di generazione lacuna per dataset testuali: 'default' (1-6 car.), 'word', 'suffix' o 'all'",
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
    parser.add_argument(
        "--update_model_card",
        action="store_true",
        help="Genera la Model Card arricchita con le metriche comparative calcolate",
    )
    parser.add_argument(
        "--output_model_card",
        type=str,
        default=None,
        help="Percorso opzionale dove salvare la Model Card generata (es. README.md)",
    )
    parser.add_argument(
        "--push_model_card",
        action="store_true",
        help="Carica direttamente la Model Card arricchita su Hugging Face Hub",
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
    print(f"Policy lacuna:                {args.policy}")
    print(f"Numero casi di test:          {args.max_cases}")
    print("=" * 80 + "\n")

    # 1. Caricamento dei casi di test divisi per policy
    policy_cases_dict = load_cases_by_policy(
        dataset_name=args.eval_dataset_name,
        split=args.split,
        max_cases=args.max_cases,
        policy=args.policy,
    )

    if not policy_cases_dict or not any(policy_cases_dict.values()):
        print("Errore: nessun caso di test estratto.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilizzato per l'inferenza: {device.upper()}\n")

    # Pulizia scorer BERTScore cache
    reset_scorer_cache()

    pre_ft_results: dict[str, dict] = {}
    post_ft_results: dict[str, dict] = {}

    # 2. Valutazione Modello Pre-FT
    print(f"--> [1/2] Caricamento modello baseline Pre-FT ({base_model})...")
    tokenizer_pre = AutoTokenizer.from_pretrained(base_model)
    model_pre = AutoModelForMaskedLM.from_pretrained(base_model).to(device)
    model_pre.eval()

    for pol_name, cases in policy_cases_dict.items():
        print(f"\n[Valutazione Pre-FT] Calcolo metriche per policy '{pol_name}' ({len(cases)} casi)...")
        pre_metrics = evaluate_metrics_on_test_set(
            split_name=f"pre_ft_{pol_name}",
            cases=cases,
            model=model_pre,
            tokenizer=tokenizer_pre,
            checkpoint=checkpoint,
            max_cases=args.max_cases,
        )
        pre_ft_results[pol_name] = pre_metrics

    del model_pre
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 3. Valutazione Modello Post-FT
    print(f"\n--> [2/2] Caricamento modello Post-FT ({checkpoint})...")
    tokenizer_post = AutoTokenizer.from_pretrained(checkpoint)
    model_post = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model_post.eval()

    for pol_name, cases in policy_cases_dict.items():
        print(f"\n[Valutazione Post-FT] Calcolo metriche per policy '{pol_name}' ({len(cases)} casi)...")
        post_metrics = evaluate_metrics_on_test_set(
            split_name=f"post_ft_{pol_name}",
            cases=cases,
            model=model_post,
            tokenizer=tokenizer_post,
            checkpoint=checkpoint,
            max_cases=args.max_cases,
        )
        post_ft_results[pol_name] = post_metrics

    del model_post
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 4. Tabella di confronto e calcolo dei delta
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

    all_rows = []
    for pol_name in policy_cases_dict.keys():
        pre_metrics = pre_ft_results.get(pol_name, {})
        post_metrics = post_ft_results.get(pol_name, {})

        print("\n" + "=" * 80)
        print(f"       RISULTATI CONFRONTO: PRE-FT VS POST-FT [POLICY: {pol_name.upper()}]")
        print("=" * 80)
        print(f"{'Metrica':<30} | {'Pre-FT':<12} | {'Post-FT':<12} | {'Delta':<12}")
        print("-" * 80)

        for key, label in comparison_keys:
            val_pre = float(pre_metrics.get(key, 0.0))
            val_post = float(post_metrics.get(key, 0.0))
            delta = val_post - val_pre

            is_pct = "(%)" in label
            delta_str = f"{delta:+.2f}%" if is_pct else f"{delta:+.4f}"
            pre_str = f"{val_pre:.2f}%" if is_pct else f"{val_pre:.4f}"
            post_str = f"{val_post:.2f}%" if is_pct else f"{val_post:.4f}"

            print(f"{label:<30} | {pre_str:>10} | {post_str:>10} | {delta_str:>10}")
            all_rows.append(
                {
                    "policy": pol_name,
                    "metric": label,
                    "metric_key": key,
                    "pre_ft": val_pre,
                    "post_ft": val_post,
                    "delta": delta,
                }
            )
        print("=" * 80 + "\n")

    # 5. Salvataggio su file JSON / CSV
    if args.output_json:
        out_data = {
            "checkpoint": checkpoint,
            "base_model": base_model,
            "eval_dataset": args.eval_dataset_name,
            "split": args.split,
            "policy": args.policy,
            "n_cases_per_policy": {p: len(c) for p, c in policy_cases_dict.items()},
            "pre_ft_metrics": pre_ft_results,
            "post_ft_metrics": post_ft_results,
            "comparison": all_rows,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"Risultati JSON salvati in: {args.output_json}")

    if args.output_csv:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output_csv, index=False)
        print(f"Tabella CSV salvata in: {args.output_csv}")

    # 6. Generazione e Pubblicazione della Model Card arricchita
    if args.update_model_card or args.output_model_card or args.push_model_card:
        print("\n" + "=" * 80)
        print("           GENERAZIONE MODEL CARD ARRICCHITA PER HUGGING FACE")
        print("=" * 80)

        model_cfg = get_model_config(checkpoint)
        raw_cfg = ModelRegistry().configs.get(checkpoint, {})
        hyperparams = raw_cfg.get("hyperparameters", {})

        # Formatta i dizionari di metriche (se una sola policy, usa la policy direttamente o passa il dizionario completo)
        pre_metrics_input = pre_ft_results if len(pre_ft_results) > 1 else next(iter(pre_ft_results.values()))
        post_metrics_input = post_ft_results if len(post_ft_results) > 1 else next(iter(post_ft_results.values()))

        card_content = generate_model_card(
            checkpoint=checkpoint,
            base_model=base_model,
            dataset_name=args.eval_dataset_name,
            pre_ft_metrics=pre_metrics_input,
            post_ft_metrics=post_metrics_input,
            hyperparameters=hyperparams,
            preprocessing_config=model_cfg,
        )

        output_card_path = args.output_model_card or f"README_{checkpoint.split('/')[-1]}.md"
        save_model_card(card_content, output_card_path)

        if args.push_model_card:
            print(f"\nCaricamento della Model Card su Hugging Face Hub [{checkpoint}]...")
            try:
                push_model_card_to_hub(
                    repo_id=checkpoint,
                    card_content=card_content,
                    token=HF_TOKEN,
                )
                print(f"Model Card aggiornata con successo su Hugging Face: https://huggingface.co/{checkpoint}")
            except Exception as e:
                print(f"[Errore Hub] Impossibile effettuare il push della Model Card: {e}")


if __name__ == "__main__":
    main()
