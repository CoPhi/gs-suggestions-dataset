import os
import sys
import gc
import json
import argparse
from pathlib import Path
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
    WANDB_PROJECT,
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

COMPARISON_KEYS = [
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


def log_bert_comparison_to_wandb(
    pre_ft_results: dict[str, dict],
    post_ft_results: dict[str, dict],
    split_prefix: str = "test",
) -> None:
    """
    Registra su Weights & Biases (se una run è attiva):
    1. Le metriche scalari `{split_prefix}_pre_ft_{policy}/*`, `{split_prefix}_post_ft_{policy}/*` e `delta_{policy}/*`.
    2. Le tabelle comparative `confronto_pre_post_ft_{policy}` con colonne `["Metrica", "Pre-FT", "Post-FT", "Delta"]`.
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    for pol_name in pre_ft_results.keys():
        pre_m = pre_ft_results.get(pol_name, {})
        post_m = post_ft_results.get(pol_name, {})
        if not pre_m and not post_m:
            continue

        pre_logs = {
            f"{split_prefix}_pre_ft_{pol_name}/{k}": float(v)
            for k, v in pre_m.items()
            if isinstance(v, (int, float))
        }
        post_logs = {
            f"{split_prefix}_post_ft_{pol_name}/{k}": float(v)
            for k, v in post_m.items()
            if isinstance(v, (int, float))
        }
        delta_logs = {
            f"delta_{pol_name}/{k}": float(post_m.get(k, 0.0)) - float(pre_m.get(k, 0.0))
            for k, _ in COMPARISON_KEYS
            if k in pre_m or k in post_m
        }
        wandb.log({**pre_logs, **post_logs, **delta_logs})

        wb_table = wandb.Table(columns=["Metrica", "Pre-FT", "Post-FT", "Delta"])
        for key, label in COMPARISON_KEYS:
            if key not in pre_m and key not in post_m:
                continue
            val_pre = float(pre_m.get(key, 0.0))
            val_post = float(post_m.get(key, 0.0))
            delta = val_post - val_pre
            wb_table.add_data(label, round(val_pre, 4), round(val_post, 4), round(delta, 4))

        wandb.log({f"confronto_pre_post_ft_{pol_name}": wb_table})


def print_and_log_multi_model_summary(
    reports: list[dict],
    log_to_wandb: bool = False,
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    """
    Costruisce una tabella riepilogativa multi-modello (Pre-FT, Post-FT e Delta per ciascun modello BERT)
    la stampa a terminale e, se richiesto, la pubblica su Weights & Biases in una run di riepilogo
    con sia la leaderboard ordinabile sia la tabella comparativa affiancata.
    """
    if not reports:
        return

    all_policies = []
    for r in reports:
        for pol in r.get("pre_ft_metrics", {}).keys():
            if pol not in all_policies:
                all_policies.append(pol)

    eval_ds = reports[0].get("eval_dataset", "eval")
    ds_short = eval_ds.split("/")[-1]

    for pol_name in all_policies:
        print("\n" + "=" * 110)
        print(f"         RIEPILOGO MULTI-MODELLO SU [{eval_ds}] - POLICY: {pol_name.upper()}")
        print("=" * 110)
        header = f"{'Modello':<24} | {'Stato':<8} | {'EM @1':>8} | {'EM @5':>8} | {'EM @20':>8} | {'BS F1@1':>8} | {'CosSim@1':>8} | {'Cluster%':>8}"
        print(header)
        print("-" * 110)

        for r in reports:
            ckpt_short = r.get("checkpoint", "").split("/")[-1]
            base_short = r.get("base_model", "").split("/")[-1]
            pre_m = r.get("pre_ft_metrics", {}).get(pol_name, {})
            post_m = r.get("post_ft_metrics", {}).get(pol_name, {})

            print(
                f"{base_short:<24} | {'Pre-FT':<8} | "
                f"{pre_m.get('top1', 0.0):>7.2f}% | "
                f"{pre_m.get('top5', 0.0):>7.2f}% | "
                f"{pre_m.get('top20', 0.0):>7.2f}% | "
                f"{pre_m.get('bertscore_f1_top1', 0.0):>7.2f}% | "
                f"{pre_m.get('cos_sim_top1_max', 0.0):>7.2f}% | "
                f"{pre_m.get('cluster_inclusion_rate', 0.0):>7.2f}%"
            )
            print(
                f"{ckpt_short:<24} | {'Post-FT':<8} | "
                f"{post_m.get('top1', 0.0):>7.2f}% | "
                f"{post_m.get('top5', 0.0):>7.2f}% | "
                f"{post_m.get('top20', 0.0):>7.2f}% | "
                f"{post_m.get('bertscore_f1_top1', 0.0):>7.2f}% | "
                f"{post_m.get('cos_sim_top1_max', 0.0):>7.2f}% | "
                f"{post_m.get('cluster_inclusion_rate', 0.0):>7.2f}%"
            )
            print("-" * 110)
        print("=" * 110 + "\n")

    if not log_to_wandb:
        return

    try:
        import wandb

        if WANDB_API_KEY:
            wandb_login()

        group_name = wandb_group or f"benchmark_{ds_short}"
        summary_run_name = wandb_run_name or f"summary_bert_models_{ds_short}"

        wandb.init(
            project=WANDB_PROJECT,
            name=summary_run_name,
            group=group_name,
            tags=[ds_short, "evaluation", "multi_model_comparison", "bert"],
            job_type="benchmark_summary",
            config={
                "eval_dataset": eval_ds,
                "models_compared": [r.get("checkpoint") for r in reports],
                "policies": all_policies,
            },
            reinit=True,
        )

        for pol_name in all_policies:
            # 1. Leaderboard Table (una riga per modello/stato, facilmente ordinabile su W&B)
            leaderboard_cols = [
                "Modello",
                "Checkpoint",
                "Stato (Pre/Post-FT)",
                *[label for _, label in COMPARISON_KEYS],
            ]
            leaderboard_table = wandb.Table(columns=leaderboard_cols)

            for r in reports:
                ckpt = r.get("checkpoint", "")
                ckpt_short = ckpt.split("/")[-1]
                base = r.get("base_model", "")
                pre_m = r.get("pre_ft_metrics", {}).get(pol_name, {})
                post_m = r.get("post_ft_metrics", {}).get(pol_name, {})

                pre_row = [
                    f"{ckpt_short} (Pre-FT)",
                    base,
                    "Pre-FT",
                    *[round(float(pre_m.get(k, 0.0)), 4) for k, _ in COMPARISON_KEYS],
                ]
                post_row = [
                    f"{ckpt_short} (Post-FT)",
                    ckpt,
                    "Post-FT",
                    *[round(float(post_m.get(k, 0.0)), 4) for k, _ in COMPARISON_KEYS],
                ]
                delta_row = [
                    f"{ckpt_short} (Delta)",
                    ckpt,
                    "Delta",
                    *[
                        round(float(post_m.get(k, 0.0)) - float(pre_m.get(k, 0.0)), 4)
                        for k, _ in COMPARISON_KEYS
                    ],
                ]
                leaderboard_table.add_data(*pre_row)
                leaderboard_table.add_data(*post_row)
                leaderboard_table.add_data(*delta_row)

            # 2. Tabella comparativa affiancata (Metrica sulle righe, Modelli sulle colonne)
            side_by_side_cols = ["Metrica"]
            for r in reports:
                ckpt_short = r.get("checkpoint", "").split("/")[-1]
                side_by_side_cols.extend(
                    [f"{ckpt_short} Pre-FT", f"{ckpt_short} Post-FT", f"{ckpt_short} Delta"]
                )
            side_by_side_table = wandb.Table(columns=side_by_side_cols)

            for key, label in COMPARISON_KEYS:
                row = [label]
                for r in reports:
                    pre_m = r.get("pre_ft_metrics", {}).get(pol_name, {})
                    post_m = r.get("post_ft_metrics", {}).get(pol_name, {})
                    v_pre = float(pre_m.get(key, 0.0))
                    v_post = float(post_m.get(key, 0.0))
                    row.extend([round(v_pre, 4), round(v_post, 4), round(v_post - v_pre, 4)])
                side_by_side_table.add_data(*row)

            wandb.log(
                {
                    f"leaderboard_modelli_bert_{pol_name}": leaderboard_table,
                    f"confronto_modelli_bert_{pol_name}": side_by_side_table,
                }
            )

        print(f"[W&B] Tabelle comparative pubblicate con successo nella run '{summary_run_name}'.")
        wandb.finish()
    except Exception as e:
        print(f"[Avviso W&B] Impossibile registrare il riepilogo multi-modello su W&B: {e}")


def evaluate_single_checkpoint(
    checkpoint: str,
    base_model: str | None,
    policy_cases_dict: dict[str, list],
    args: argparse.Namespace,
    device: str,
) -> dict:
    """Esegue la valutazione comparativa Pre-FT vs Post-FT per un singolo checkpoint."""
    if not base_model:
        base_model = ModelRegistry().base_model_map.get(checkpoint, checkpoint)

    ckpt_short = checkpoint.split("/")[-1]
    ds_short = args.eval_dataset_name.split("/")[-1]

    print("\n" + "=" * 80)
    print("               CONFRONTO METRICHE: PRE-FT VS POST-FT")
    print("=" * 80)
    print(f"Modello Pre-FT (Base):        {base_model}")
    print(f"Modello Post-FT (Fine-tuned): {checkpoint}")
    print(f"Dataset di valutazione:       {args.eval_dataset_name} [split: {args.split}]")
    print(f"Policy lacuna:                {args.policy}")
    print(f"Numero casi di test:          {args.max_cases}")
    print("=" * 80 + "\n")

    # Avvio eventuale run W&B prima dell'inferenza così da registrare anche le predictions_table
    wandb_active = False
    if args.log_to_wandb:
        try:
            import wandb

            if WANDB_API_KEY:
                wandb_login()
            run_name = (
                args.wandb_run_name
                if (args.wandb_run_name and args.checkpoint != "all")
                else f"compare_{ckpt_short}_{ds_short}"
            )
            group_name = args.wandb_group or f"benchmark_{ds_short}"
            wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                group=group_name,
                tags=[ckpt_short, ds_short, "evaluation", "pre_post_comparison"],
                job_type="evaluation",
                config={
                    "checkpoint": checkpoint,
                    "base_model": base_model,
                    "eval_dataset": args.eval_dataset_name,
                    "split": args.split,
                    "policy": args.policy,
                    "max_cases": args.max_cases,
                },
                reinit=True,
            )
            wandb_active = True
        except Exception as e:
            print(f"[Avviso W&B] Impossibile inizializzare Weights & Biases: {e}")

    reset_scorer_cache()

    pre_ft_results: dict[str, dict] = {}
    post_ft_results: dict[str, dict] = {}

    # 1. Valutazione Modello Pre-FT
    print(f"--> [1/2] Caricamento modello baseline Pre-FT ({base_model})...")
    tokenizer_pre = AutoTokenizer.from_pretrained(base_model)
    model_pre = AutoModelForMaskedLM.from_pretrained(base_model).to(device)
    model_pre.eval()

    for pol_name, cases in policy_cases_dict.items():
        print(f"\n[Valutazione Pre-FT] Calcolo metriche per policy '{pol_name}' ({len(cases)} casi)...")
        pre_metrics = evaluate_metrics_on_test_set(
            split_name=f"test_pre_ft_{pol_name}",
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

    # 2. Valutazione Modello Post-FT
    print(f"\n--> [2/2] Caricamento modello Post-FT ({checkpoint})...")
    tokenizer_post = AutoTokenizer.from_pretrained(checkpoint)
    model_post = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model_post.eval()

    for pol_name, cases in policy_cases_dict.items():
        print(f"\n[Valutazione Post-FT] Calcolo metriche per policy '{pol_name}' ({len(cases)} casi)...")
        post_metrics = evaluate_metrics_on_test_set(
            split_name=f"test_post_ft_{pol_name}",
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

    # 3. Tabella di confronto e calcolo dei delta
    all_rows = []
    for pol_name in policy_cases_dict.keys():
        pre_metrics = pre_ft_results.get(pol_name, {})
        post_metrics = post_ft_results.get(pol_name, {})

        print("\n" + "=" * 80)
        print(f"       RISULTATI CONFRONTO: PRE-FT VS POST-FT [{ckpt_short} - POLICY: {pol_name.upper()}]")
        print("=" * 80)
        print(f"{'Metrica':<30} | {'Pre-FT':<12} | {'Post-FT':<12} | {'Delta':<12}")
        print("-" * 80)

        for key, label in COMPARISON_KEYS:
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
                    "model": ckpt_short,
                    "checkpoint": checkpoint,
                    "base_model": base_model,
                    "policy": pol_name,
                    "metric": label,
                    "metric_key": key,
                    "pre_ft": val_pre,
                    "post_ft": val_post,
                    "delta": delta,
                }
            )
        print("=" * 80 + "\n")

    # 4. Registrazione su W&B della run corrente
    if wandb_active:
        try:
            import wandb

            log_bert_comparison_to_wandb(
                pre_ft_results=pre_ft_results,
                post_ft_results=post_ft_results,
                split_prefix="test",
            )
            wandb.finish()
        except Exception as e:
            print(f"[Avviso W&B] Errore durante il salvataggio delle metriche su W&B: {e}")

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

    # 5. Salvataggio su file JSON / CSV
    if args.output_json:
        out_json_path = Path(args.output_json)
        if args.checkpoint == "all":
            out_json_path = out_json_path.with_name(f"{out_json_path.stem}_{ckpt_short}{out_json_path.suffix}")
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"Risultati JSON salvati in: {out_json_path}")

    if args.output_csv:
        out_csv_path = Path(args.output_csv)
        if args.checkpoint == "all":
            out_csv_path = out_csv_path.with_name(f"{out_csv_path.stem}_{ckpt_short}{out_csv_path.suffix}")
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(out_csv_path, index=False)
        print(f"Tabella CSV salvata in: {out_csv_path}")

    # 6. Generazione e Pubblicazione della Model Card arricchita
    if args.update_model_card or args.output_model_card or args.push_model_card:
        print("\n" + "=" * 80)
        print(f"           GENERAZIONE MODEL CARD ARRICCHITA PER [{checkpoint}]")
        print("=" * 80)

        model_cfg = get_model_config(checkpoint)
        raw_cfg = ModelRegistry().configs.get(checkpoint, {})
        hyperparams = raw_cfg.get("hyperparameters", {})

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

        output_card_path = (
            args.output_model_card
            if (args.output_model_card and args.checkpoint != "all")
            else f"README_{ckpt_short}.md"
        )
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

    return out_data


def main():
    parser = argparse.ArgumentParser(
        description="Confronto comparativo Pre-FT (Baseline) vs Post-FT (Fine-tuned), pubblicazione W&B e Model Card"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="CNR-ILC/gs-GreBerta",
        help="Checkpoint del modello Fine-Tuned (es. CNR-ILC/gs-GreBerta) oppure 'all' per valutare tutti i modelli BERT",
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
        "--from_json",
        type=str,
        nargs="+",
        default=None,
        help="Uno o più file JSON già calcolati da caricare e pubblicare su W&B senza rieseguire l'inferenza",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Se specificato, pubblica le metriche e le tabelle comparative Pre-FT vs Post-FT su Weights & Biases",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Nome opzionale della run Weights & Biases",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Gruppo W&B per raggruppare i confronti sullo stesso dataset (default: benchmark_<dataset>)",
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

    # Modalità caricamento da JSON pre-calcolati (senza inferenza GPU)
    if args.from_json:
        loaded_reports = []
        for json_path in args.from_json:
            with open(json_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            loaded_reports.append(report)

            if args.log_to_wandb:
                try:
                    import wandb

                    if WANDB_API_KEY:
                        wandb_login()
                    ckpt = report.get("checkpoint", "unknown")
                    ckpt_short = ckpt.split("/")[-1]
                    eval_ds = report.get("eval_dataset", args.eval_dataset_name)
                    ds_short = eval_ds.split("/")[-1]
                    group_name = args.wandb_group or f"benchmark_{ds_short}"
                    run_name = (
                        args.wandb_run_name
                        if (args.wandb_run_name and len(args.from_json) == 1)
                        else f"compare_{ckpt_short}_{ds_short}"
                    )
                    wandb.init(
                        project=WANDB_PROJECT,
                        name=run_name,
                        group=group_name,
                        tags=[ckpt_short, ds_short, "evaluation", "pre_post_comparison", "from_json"],
                        job_type="evaluation",
                        config={
                            "checkpoint": ckpt,
                            "base_model": report.get("base_model"),
                            "eval_dataset": eval_ds,
                            "split": report.get("split", "test"),
                            "policy": report.get("policy", "default"),
                            "n_cases_per_policy": report.get("n_cases_per_policy", {}),
                        },
                        reinit=True,
                    )
                    log_bert_comparison_to_wandb(
                        pre_ft_results=report.get("pre_ft_metrics", {}),
                        post_ft_results=report.get("post_ft_metrics", {}),
                        split_prefix=report.get("split", "test"),
                    )
                    wandb.finish()
                    print(f"[W&B] Pubblicato report per '{ckpt}' da '{json_path}'.")
                except Exception as e:
                    print(f"[Avviso W&B] Errore durante il caricamento di '{json_path}' su W&B: {e}")

        if len(loaded_reports) > 1:
            print_and_log_multi_model_summary(
                reports=loaded_reports,
                log_to_wandb=args.log_to_wandb,
                wandb_group=args.wandb_group,
                wandb_run_name=args.wandb_run_name,
            )
        return

    if HF_TOKEN:
        login(token=HF_TOKEN)

    # 1. Caricamento dei casi di test divisi per policy (eseguito una sola volta per tutti i modelli)
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

    if args.checkpoint.lower() == "all":
        checkpoints = list(ModelRegistry().configs.keys())
    else:
        checkpoints = [c.strip() for c in args.checkpoint.split(",") if c.strip()]

    reports = []
    for ckpt in checkpoints:
        rep = evaluate_single_checkpoint(
            checkpoint=ckpt,
            base_model=args.base_model if len(checkpoints) == 1 else None,
            policy_cases_dict=policy_cases_dict,
            args=args,
            device=device,
        )
        reports.append(rep)

    if len(reports) > 1:
        print_and_log_multi_model_summary(
            reports=reports,
            log_to_wandb=args.log_to_wandb,
            wandb_group=args.wandb_group,
            wandb_run_name=args.wandb_run_name,
        )


if __name__ == "__main__":
    main()
