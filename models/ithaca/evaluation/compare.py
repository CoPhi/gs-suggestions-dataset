"""
Modulo di Valutazione Comparativa: Ithaca Pre-FT (Base) vs Ithaca Post-FT (TLG Fine-Tuned).

Valuta entrambi i checkpoint sul test set stratificato (policy 'default', 'word', 'suffix')
calcolando:
- Top-1, Top-5, Top-20 Exact Match Accuracy.
- Character Error Rate (CER) medio sui caratteri della lacuna.
- Delta % di miglioramento ottenuto dal fine-tuning.
Esporta i risultati in una tabella Markdown e in un file JSON pronto per la Model Card.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Assicura inclusione del progetto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from typing import Any
from models.ithaca.inference.predict import (
    fill_mask_ithaca,
    get_ithaca_contextual_embeddings,
)
from models.bert.evaluation.metrics import (
    evaluate_contextual_similarity,
    evaluate_cosine_similarity_topk,
    check_gold_in_dense_cluster,
    evaluate_dense_cluster_inclusion_batch,
    evaluate_bertscore_topk_text,
)


def compute_cer(pred: str, target: str) -> float:
    """Calcola il Character Error Rate (distanza di Levenshtein / lunghezza target)."""
    p, t = pred.upper(), target.upper()
    if not t:
        return 0.0 if not p else 1.0

    dp = [[0] * (len(t) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p) + 1):
        dp[i][0] = i
    for j in range(len(t) + 1):
        dp[0][j] = j

    for i in range(1, len(p) + 1):
        for j in range(1, len(t) + 1):
            if p[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    dist = dp[len(p)][len(t)]
    return dist / len(t)


def evaluate_checkpoint_on_cases(
    checkpoint_path: str,
    cases: list[dict],
    k_list: tuple[int, ...] = (1, 5, 10, 20),
    strategy: str = "hcb_best_to_worst",
    beam_size: int = 20,
    temperature: float = 1.0,
    params_override: Any | None = None,
    alphabet_override: Any | None = None,
    config_override: dict | None = None,
    forward_fn_override: Any | None = None,
    torso_fn_override: Any | None = None,
    compute_semantic_metrics: bool = True,
    compute_bertscore: bool = False,
    return_predictions: bool = False,
) -> dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], list[dict]]:
    """
    Esegue l'infilling su una lista di casi di test e raggruppa le metriche per policy,
    utilizzando HCB beam search.
    Calcola:
    - Top-1, Top-5, Top-10, Top-20 Exact Match (sia frazione 0..1 che percentuale 0..100)
    - MRR e Character Error Rate (CER) / Character Accuracy (1 - CER)
    - Metriche semantiche e di coesione geometrica (Cluster Inclusion Rate, Margin, CosSim @K)
    """
    results_by_policy = defaultdict(
        lambda: {
            "top1": [],
            "top5": [],
            "top10": [],
            "top20": [],
            "cer": [],
            "mrr": [],
            "similarities": [],
            "cluster_results": [],
            "preds_raw": [],
            "golds": [],
            "contexts": [],
        }
    )
    prediction_records: list[dict] = []

    print(
        f"Valutazione modello '{checkpoint_path}' su {len(cases)} casi "
        f"(strategia: {strategy}, beam_size: {beam_size}, temp: {temperature})..."
    )
    t0 = time.time()

    for idx, case in enumerate(cases, 1):
        text_input = case.get("text_leiden") or case.get("text_ithaca") or case.get("text")
        gold = str(case["gold_target"]).upper().strip()
        policy = case.get("policy", "default")

        try:
            preds = fill_mask_ithaca(
                text=text_input,
                checkpoint=checkpoint_path,
                K=max(k_list),
                beam_size=beam_size,
                strategy=strategy,
                temperature=temperature,
                params_override=params_override,
                alphabet_override=alphabet_override,
                config_override=config_override,
                forward_fn_override=forward_fn_override,
            )
            pred_strings = [p[0].upper().strip() for p in preds]
        except Exception as e:
            if idx <= 3:
                print(f"\n[Avviso] Errore di decodifica al caso {idx}: {e}")
            preds = []
            pred_strings = []

        # Exact Match Top-K
        t1 = 1.0 if (len(pred_strings) > 0 and pred_strings[0] == gold) else 0.0
        t5 = 1.0 if gold in pred_strings[:5] else 0.0
        t10 = 1.0 if gold in pred_strings[:10] else 0.0
        t20 = 1.0 if gold in pred_strings[:20] else 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        if gold in pred_strings:
            mrr = 1.0 / (pred_strings.index(gold) + 1)

        # CER sul primo suggerimento (o 1.0 se non predice)
        best_cand = pred_strings[0] if pred_strings else ""
        cer = compute_cer(best_cand, gold)

        # Calcolo embedding contestuali (Torso BigBird) per Cluster Inclusion e Cosine Similarity
        sims: list[float] = []
        cluster_info: dict[str, float | bool] = {
            "is_inside_cluster": False,
            "inclusion_margin": -1.0,
            "cluster_radius": 0.0,
            "gold_centroid_distance": 1.0,
            "gold_centroid_cosine_sim": 0.0,
        }
        if compute_semantic_metrics and pred_strings:
            try:
                all_cands_plus_gold = pred_strings + [gold]
                embs = get_ithaca_contextual_embeddings(
                    text_with_gap=text_input,
                    candidates=all_cands_plus_gold,
                    checkpoint=checkpoint_path,
                    params_override=params_override,
                    alphabet_override=alphabet_override,
                    config_override=config_override,
                    torso_fn_override=torso_fn_override,
                )
                if len(embs) >= 2:
                    gold_emb = embs[-1]
                    cand_embs = embs[:-1]
                    sims = evaluate_contextual_similarity(cand_embs, gold_emb)
                    cluster_info = check_gold_in_dense_cluster(cand_embs, gold_emb)
            except Exception as e:
                if idx <= 3:
                    print(f"\n[Avviso] Calcolo metriche semantiche saltato al caso {idx}: {e}")

        for target_pol in (policy, "overall"):
            bucket = results_by_policy[target_pol]
            bucket["top1"].append(t1)
            bucket["top5"].append(t5)
            bucket["top10"].append(t10)
            bucket["top20"].append(t20)
            bucket["mrr"].append(mrr)
            bucket["cer"].append(cer)
            bucket["similarities"].append(sims)
            bucket["cluster_results"].append(cluster_info)
            bucket["preds_raw"].append(preds)
            bucket["golds"].append(gold.lower())
            bucket["contexts"].append(text_input)

        prediction_records.append(
            {
                "policy": policy,
                "context": text_input,
                "gold": gold,
                "predictions": preds,
            }
        )

        if idx % 25 == 0 or idx == len(cases):
            elapsed = time.time() - t0
            print(f"\rProcessati {idx}/{len(cases)} casi ({elapsed:.1f}s)...", end="", flush=True)

    print("\nValutazione completata.")

    # Aggregazione delle medie
    aggregated = {}
    for pol, metrics in results_by_policy.items():
        count = max(len(metrics["top1"]), 1)
        top1_acc = float(sum(metrics["top1"]) / count)
        top5_acc = float(sum(metrics["top5"]) / count)
        top10_acc = float(sum(metrics["top10"]) / count)
        top20_acc = float(sum(metrics["top20"]) / count)
        mrr_val = float(sum(metrics["mrr"]) / count)
        mean_cer = float(sum(metrics["cer"]) / count)
        char_acc = max(0.0, 1.0 - mean_cer) * 100.0

        pol_dict = {
            # Chiavi frazionarie (0..1) per retrocompatibilità
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "top10_acc": top10_acc,
            "top20_acc": top20_acc,
            "mrr": mrr_val,
            "mean_cer": mean_cer,
            # Chiavi percentuali (0..100%) allineate con i modelli BERT e W&B
            "top1": top1_acc * 100.0,
            "top5": top5_acc * 100.0,
            "top10": top10_acc * 100.0,
            "top20": top20_acc * 100.0,
            "mrr_pct": mrr_val * 100.0,
            "char_acc": char_acc,
            "count": len(metrics["top1"]),
        }

        if compute_semantic_metrics and metrics["similarities"]:
            cos_metrics = evaluate_cosine_similarity_topk(
                metrics["similarities"], k_values=[1, 5, 10, 20]
            )
            cluster_metrics = evaluate_dense_cluster_inclusion_batch(
                metrics["cluster_results"]
            )
            pol_dict.update(cos_metrics)
            pol_dict.update(cluster_metrics)
        else:
            for k in (1, 5, 10, 20):
                pol_dict.setdefault(f"cos_sim_top{k}_max", 0.0)
                pol_dict.setdefault(f"cos_sim_top{k}_mean", 0.0)
            pol_dict.setdefault("cluster_inclusion_rate", 0.0)
            pol_dict.setdefault("mean_inclusion_margin", 0.0)
            pol_dict.setdefault("mean_gold_centroid_cosine_sim", 0.0)

        if compute_bertscore and metrics["preds_raw"]:
            try:
                bs_metrics = evaluate_bertscore_topk_text(
                    predictions_text=metrics["preds_raw"],
                    gold_labels=metrics["golds"],
                    contexts=metrics["contexts"],
                    k_values=[1, 5, 10, 20],
                    checkpoint="CNR-ILC/gs-ithaca-tlg",
                )
                pol_dict.update(bs_metrics)
            except Exception as e:
                print(f"[Avviso] BERTScore non disponibile per policy '{pol}': {e}")

        aggregated[pol] = pol_dict

    if return_predictions:
        return aggregated, prediction_records
    return aggregated


def log_ithaca_comparison_to_wandb(
    pre_metrics_by_policy: dict[str, dict[str, float]],
    post_metrics_by_policy: dict[str, dict[str, float]],
    predictions_records: list[dict] | None = None,
    split_prefix: str = "test",
) -> None:
    """
    Registra su Weights & Biases (se una run è attiva):
    1. Le metriche scalari `{split_prefix}_pre_ft_{policy}/*` e `{split_prefix}_post_ft_{policy}/*`.
    2. Le tabelle comparative `confronto_pre_post_ft_{policy}` con colonne `["Metrica", "Pre-FT", "Post-FT", "Delta"]`
       per ciascuna policy (`overall`, `default`, `word`, `suffix`), identiche a quelle dei modelli BERT.
    3. Le tabelle qualitative delle predizioni `{split_prefix}_post_ft_{policy}/predictions_table`
       evidenziando il suggerimento gold (`**parola** (score)`).
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    comparison_keys = [
        ("top1", "Exact Match @1"),
        ("top5", "Exact Match @5"),
        ("top10", "Exact Match @10"),
        ("top20", "Exact Match @20"),
        ("char_acc", "Character Accuracy (1-CER)"),
        ("mean_cer", "Character Error Rate (CER)"),
        ("mrr", "Mean Reciprocal Rank (MRR)"),
        ("cos_sim_top1_max", "CosSim Max @1"),
        ("cos_sim_top5_max", "CosSim Max @5"),
        ("cos_sim_top10_max", "CosSim Max @10"),
        ("cos_sim_top20_max", "CosSim Max @20"),
        ("cluster_inclusion_rate", "Cluster Inclusion Rate"),
        ("mean_inclusion_margin", "Mean Inclusion Margin"),
        ("mean_gold_centroid_cosine_sim", "Gold Centroid CosSim"),
    ]

    policy_order = ["overall", "default", "word", "suffix"]
    for policy_name in policy_order:
        pre_m = pre_metrics_by_policy.get(policy_name)
        post_m = post_metrics_by_policy.get(policy_name)
        if not pre_m or not post_m:
            continue

        # 1. Log scalari pre e post
        pre_logs = {
            f"{split_prefix}_pre_ft_{policy_name}/{k}": float(v)
            for k, v in pre_m.items()
            if isinstance(v, (int, float))
        }
        post_logs = {
            f"{split_prefix}_post_ft_{policy_name}/{k}": float(v)
            for k, v in post_m.items()
            if isinstance(v, (int, float))
        }
        wandb.log({**pre_logs, **post_logs})

        # 2. Costruzione wandb.Table comparativa Pre-FT vs Post-FT
        wb_table = wandb.Table(columns=["Metrica", "Pre-FT", "Post-FT", "Delta"])
        for key, label in comparison_keys:
            if key not in pre_m and key not in post_m:
                continue
            val_pre = float(pre_m.get(key, 0.0))
            val_post = float(post_m.get(key, 0.0))
            delta = val_post - val_pre
            wb_table.add_data(label, val_pre, val_post, delta)

        # Includi anche BERTScore se calcolato
        for k_bs in (1, 5, 10, 20):
            bs_key = f"bertscore_f1_top{k_bs}"
            if bs_key in pre_m or bs_key in post_m:
                val_pre = float(pre_m.get(bs_key, 0.0))
                val_post = float(post_m.get(bs_key, 0.0))
                wb_table.add_data(
                    f"BERTScore F1 @{k_bs}", val_pre, val_post, val_post - val_pre
                )

        wandb.log({f"confronto_pre_post_ft_{policy_name}": wb_table})

    # 3. Tabella qualitativa delle predizioni (Top-20 con gold evidenziato)
    if predictions_records:
        by_pol: dict[str, list[list[str]]] = defaultdict(list)
        for rec in predictions_records:
            pol = rec.get("policy", "default")
            ctx = str(rec.get("context", ""))
            gold = str(rec.get("gold", "")).strip()
            gold_upper = gold.upper()
            preds = rec.get("predictions", [])

            formatted_preds = []
            for cand_str, score in preds[:20]:
                if str(cand_str).strip().upper() == gold_upper:
                    formatted_preds.append(f"**{cand_str}** ({score:.4f})")
                else:
                    formatted_preds.append(f"{cand_str} ({score:.4f})")

            top_preds_str = " | ".join(formatted_preds)
            row = [ctx, gold, top_preds_str]
            by_pol[pol].append(row)
            by_pol["overall"].append([f"[{pol}] {ctx}", gold, top_preds_str])

        for pol_name, rows_data in by_pol.items():
            pred_table = wandb.Table(
                columns=["Context", "Gold Label", "Top Predictions"],
                data=rows_data[:100],
            )
            wandb.log(
                {f"{split_prefix}_post_ft_{pol_name}/predictions_table": pred_table}
            )


def render_comparison_table(
    pre_metrics: dict[str, dict[str, float]],
    post_metrics: dict[str, dict[str, float]],
) -> str:
    """Formatta i risultati comparativi in una tabella Markdown chiara e leggibile."""
    lines = []
    lines.append("| Policy | Metrica | Pre-FT (Ithaca Base) | Post-FT (Ithaca TLG) | Delta |")
    lines.append("|:---|:---|:---:|:---:|:---:|")

    policy_order = ["overall", "suffix", "word", "default"]
    for pol in policy_order:
        if pol not in pre_metrics or pol not in post_metrics:
            continue
        p_pre = pre_metrics[pol]
        p_post = post_metrics[pol]

        pol_label = pol.upper() if pol == "overall" else f"Policy '{pol}'"

        def _row(metric_name, pre_v, post_v, is_pct=True, lower_is_better=False, label=pol_label, already_100=False):
            if is_pct:
                scale = 1.0 if already_100 else 100.0
                pre_s = f"{pre_v * scale:.2f}%"
                post_s = f"{post_v * scale:.2f}%"
                diff = (post_v - pre_v) * scale
                diff_s = f"{diff:+.2f}%"
            else:
                pre_s = f"{pre_v:.4f}"
                post_s = f"{post_v:.4f}"
                diff = post_v - pre_v
                diff_s = f"{diff:+.4f}"

            # Indicatore visivo miglioramento
            if (diff > 0 and not lower_is_better) or (diff < 0 and lower_is_better):
                diff_s = f"🟢 **{diff_s}**"
            elif diff == 0:
                diff_s = f"⚪ {diff_s}"
            else:
                diff_s = f"🔴 {diff_s}"

            return f"| {label} | {metric_name} | {pre_s} | {post_s} | {diff_s} |"

        lines.append(_row("Top-1 Exact Match", p_pre["top1_acc"], p_post["top1_acc"]))
        lines.append(_row("Top-5 Exact Match", p_pre["top5_acc"], p_post["top5_acc"]))
        lines.append(_row("Top-20 Exact Match", p_pre["top20_acc"], p_post["top20_acc"]))
        lines.append(_row("Mean Reciprocal Rank (MRR)", p_pre["mrr"], p_post["mrr"], is_pct=False))
        lines.append(_row("Character Error Rate (CER)", p_pre["mean_cer"], p_post["mean_cer"], is_pct=False, lower_is_better=True))
        if "cluster_inclusion_rate" in p_pre and "cluster_inclusion_rate" in p_post:
            lines.append(
                _row(
                    "Cluster Inclusion Rate",
                    p_pre["cluster_inclusion_rate"],
                    p_post["cluster_inclusion_rate"],
                    is_pct=True,
                    already_100=True,
                )
            )
        if "cos_sim_top1_max" in p_pre and "cos_sim_top1_max" in p_post:
            lines.append(
                _row(
                    "CosSim Max @1",
                    p_pre["cos_sim_top1_max"],
                    p_post["cos_sim_top1_max"],
                    is_pct=True,
                    already_100=True,
                )
            )

    return "\n".join(lines)


def extract_comparison_rows(
    pre_metrics: dict[str, dict[str, float]],
    post_metrics: dict[str, dict[str, float]],
) -> list[dict]:
    """Estrae i record comparativi con policy, metrica, valori pre/post e delta."""
    metrics_meta = [
        ("top1_acc", "Top-1 Exact Match", True, False),
        ("top5_acc", "Top-5 Exact Match", True, False),
        ("top20_acc", "Top-20 Exact Match", True, False),
        ("mrr", "Mean Reciprocal Rank (MRR)", False, False),
        ("mean_cer", "Character Error Rate (CER)", False, True),
        ("cluster_inclusion_rate", "Cluster Inclusion Rate", True, False),
        ("cos_sim_top1_max", "CosSim Max @1", True, False),
    ]
    rows = []
    policy_order = ["overall", "suffix", "word", "default"]
    for pol in policy_order:
        if pol not in pre_metrics or pol not in post_metrics:
            continue
        p_pre = pre_metrics[pol]
        p_post = post_metrics[pol]
        for key, label, is_pct, lower_is_better in metrics_meta:
            if key not in p_pre and key not in p_post:
                continue
            pre_v = float(p_pre.get(key, 0.0))
            post_v = float(p_post.get(key, 0.0))
            delta = post_v - pre_v
            rows.append({
                "policy": pol,
                "metric": label,
                "metric_key": key,
                "pre_ft": pre_v,
                "post_ft": post_v,
                "delta": delta,
                "is_percentage": is_pct,
                "lower_is_better": lower_is_better,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Confronto comparativo Pre-FT vs Post-FT per Ithaca"
    )
    parser.add_argument(
        "--pre_checkpoint",
        type=str,
        default="checkpoints/ithaca/checkpoint_v1.pkl",
        help="Percorso al checkpoint base di Ithaca",
    )
    parser.add_argument(
        "--post_checkpoint",
        type=str,
        default="checkpoints/ithaca/checkpoint_tlg.pkl",
        help="Percorso al checkpoint fine-tunato di Ithaca",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/ithaca/tlg_test.jsonl",
        help="Percorso al dataset di test (JSONL)",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=300,
        help="Numero massimo di casi di test da valutare (default: 300)",
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
        help="Strategia di decodifica beam search (default: hcb_best_to_worst)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=20,
        help="Dimensione del beam per la decodifica (default: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura di decodifica per HCB Beam Search (default: 1.0)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="eval/results/eval_results_ithaca_comparison.json",
        help="Percorso per salvare il report di valutazione comparativa in formato JSON",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Percorso opzionale per salvare la tabella comparativa in formato CSV",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="Percorso opzionale per salvare la tabella comparativa in formato Markdown",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Se specificato, registra le tabelle comparative Pre-FT vs Post-FT su Weights & Biases",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ithaca_compare_pre_post_ft",
        help="Nome della run W&B quando --log_to_wandb è attivo",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_path):
        raise FileNotFoundError(
            f"File test set non trovato in '{args.test_path}'. "
            f"Esegui prima `python -m models.ithaca.dataset.prepare_tlg`"
        )

    # Carica i casi di test
    cases = []
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line.strip()))
            if args.max_cases and len(cases) >= args.max_cases:
                break

    print(f"Caricati {len(cases)} casi di test per la valutazione.")

    # 1. Valutazione Pre-FT
    pre_metrics = evaluate_checkpoint_on_cases(
        args.pre_checkpoint,
        cases,
        strategy=args.strategy,
        beam_size=args.beam_size,
        temperature=args.temperature,
    )

    # 2. Valutazione Post-FT
    post_metrics, post_preds = evaluate_checkpoint_on_cases(
        args.post_checkpoint,
        cases,
        strategy=args.strategy,
        beam_size=args.beam_size,
        temperature=args.temperature,
        return_predictions=True,
    )

    # 3. Tabella comparativa
    table_md = render_comparison_table(pre_metrics, post_metrics)
    print("\n" + "=" * 70)
    print("             RISULTATI COMPARATIVI PRE-FT VS POST-FT")
    print("=" * 70)
    print(table_md)
    print("=" * 70 + "\n")

    # 4. Estrazione righe tabulari
    rows = extract_comparison_rows(pre_metrics, post_metrics)

    # 5. Logging opzionale su W&B
    if args.log_to_wandb:
        try:
            import wandb
            from models.bert.finetuning import WANDB_PROJECT, WANDB_API_KEY, wandb_login

            if WANDB_API_KEY:
                wandb_login()
            if wandb.run is None:
                wandb.init(
                    project=WANDB_PROJECT,
                    name=args.wandb_run_name,
                    tags=["gs-ithaca-tlg", "evaluation", "pre_post_comparison"],
                    job_type="evaluation",
                    config={
                        "pre_checkpoint": args.pre_checkpoint,
                        "post_checkpoint": args.post_checkpoint,
                        "test_dataset": args.test_path,
                        "strategy": args.strategy,
                        "beam_size": args.beam_size,
                        "temperature": args.temperature,
                        "cases_evaluated": len(cases),
                    },
                )
            log_ithaca_comparison_to_wandb(
                pre_metrics_by_policy=pre_metrics,
                post_metrics_by_policy=post_metrics,
                predictions_records=post_preds,
                split_prefix="test",
            )
            wandb.finish()
        except Exception as e:
            print(f"[Avviso] Impossibile registrare il report su W&B: {e}")

    # 6. Esportazione JSON
    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "pre_checkpoint": args.pre_checkpoint,
            "post_checkpoint": args.post_checkpoint,
            "test_dataset": args.test_path,
            "strategy": args.strategy,
            "beam_size": args.beam_size,
            "temperature": args.temperature,
            "cases_evaluated": len(cases),
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "comparison": rows,
            "markdown_table": table_md,
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report di valutazione salvato in: {out_json_path}")

    # 7. Esportazione CSV opzionale
    if args.output_csv:
        import csv
        out_csv_path = Path(args.output_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["policy", "metric", "metric_key", "pre_ft", "post_ft", "delta"]
        with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Tabella comparativa CSV salvata in: {out_csv_path}")

    # 8. Esportazione Markdown opzionale
    if args.output_md:
        out_md_path = Path(args.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(table_md + "\n")
        print(f"Tabella comparativa Markdown salvata in: {out_md_path}")


if __name__ == "__main__":
    main()

