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

from models.ithaca.inference.predict import fill_mask_ithaca


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
    k_list: tuple[int, ...] = (1, 5, 20),
    strategy: str = "hcb_best_to_worst",
    beam_size: int = 20,
) -> dict[str, dict[str, float]]:
    """
    Esegue l'infilling su una lista di casi di test e raggruppa le metriche per policy,
    utilizzando HCB beam search.
    """
    results_by_policy = defaultdict(
        lambda: {"top1": [], "top5": [], "top20": [], "cer": [], "mrr": []}
    )

    print(
        f"Valutazione modello '{checkpoint_path}' su {len(cases)} casi "
        f"(strategia: {strategy}, beam_size: {beam_size})..."
    )
    t0 = time.time()

    for idx, case in enumerate(cases, 1):
        # Preferisci text_leiden o text_ithaca
        text_input = case.get("text_leiden") or case.get("text_ithaca") or case.get("text")
        gold = case["gold_target"].upper().strip()
        policy = case.get("policy", "default")

        try:
            preds = fill_mask_ithaca(
                text=text_input,
                checkpoint=checkpoint_path,
                K=max(k_list),
                beam_size=beam_size,
                strategy=strategy,
            )
            pred_strings = [p[0].upper().strip() for p in preds]
        except Exception as e:
            if idx <= 3:
                print(f"\n[Avviso] Errore di decodifica al caso {idx}: {e}")
            pred_strings = []

        # Exact Match Top-K
        t1 = 1.0 if (len(pred_strings) > 0 and pred_strings[0] == gold) else 0.0
        t5 = 1.0 if gold in pred_strings[:5] else 0.0
        t20 = 1.0 if gold in pred_strings[:20] else 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        if gold in pred_strings:
            mrr = 1.0 / (pred_strings.index(gold) + 1)

        # CER sul primo suggerimento (o 1.0 se non predice)
        best_cand = pred_strings[0] if pred_strings else ""
        cer = compute_cer(best_cand, gold)

        results_by_policy[policy]["top1"].append(t1)
        results_by_policy[policy]["top5"].append(t5)
        results_by_policy[policy]["top20"].append(t20)
        results_by_policy[policy]["mrr"].append(mrr)
        results_by_policy[policy]["cer"].append(cer)

        # Traccia anche le metriche globali aggregate
        results_by_policy["overall"]["top1"].append(t1)
        results_by_policy["overall"]["top5"].append(t5)
        results_by_policy["overall"]["top20"].append(t20)
        results_by_policy["overall"]["mrr"].append(mrr)
        results_by_policy["overall"]["cer"].append(cer)

        if idx % 25 == 0 or idx == len(cases):
            elapsed = time.time() - t0
            print(f"\rProcessati {idx}/{len(cases)} casi ({elapsed:.1f}s)...", end="", flush=True)

    print("\nValutazione completata.")

    # Aggregazione delle medie
    aggregated = {}
    for pol, metrics in results_by_policy.items():
        count = max(len(metrics["top1"]), 1)
        aggregated[pol] = {
            "top1_acc": float(sum(metrics["top1"]) / count),
            "top5_acc": float(sum(metrics["top5"]) / count),
            "top20_acc": float(sum(metrics["top20"]) / count),
            "mrr": float(sum(metrics["mrr"]) / count),
            "mean_cer": float(sum(metrics["cer"]) / count),
            "count": len(metrics["top1"]),
        }

    return aggregated


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

        def _row(metric_name, pre_v, post_v, is_pct=True, lower_is_better=False, label=pol_label):
            if is_pct:
                pre_s = f"{pre_v * 100:.2f}%"
                post_s = f"{post_v * 100:.2f}%"
                diff = (post_v - pre_v) * 100
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
    ]
    rows = []
    policy_order = ["overall", "suffix", "word", "default"]
    for pol in policy_order:
        if pol not in pre_metrics or pol not in post_metrics:
            continue
        p_pre = pre_metrics[pol]
        p_post = post_metrics[pol]
        for key, label, is_pct, lower_is_better in metrics_meta:
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
        "--output_json",
        type=str,
        default="eval/results/eval_results_ithaca_comparison.json",
        help="Percorso per salvare il report di valutazione comparativa in formato JSON (default: eval/results/eval_results_ithaca_comparison.json)",
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
        args.pre_checkpoint, cases, strategy=args.strategy, beam_size=args.beam_size
    )

    # 2. Valutazione Post-FT
    post_metrics = evaluate_checkpoint_on_cases(
        args.post_checkpoint, cases, strategy=args.strategy, beam_size=args.beam_size
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

    # 5. Esportazione JSON
    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "pre_checkpoint": args.pre_checkpoint,
            "post_checkpoint": args.post_checkpoint,
            "test_dataset": args.test_path,
            "strategy": args.strategy,
            "beam_size": args.beam_size,
            "cases_evaluated": len(cases),
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "comparison": rows,
            "markdown_table": table_md,
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report di valutazione salvato in: {out_json_path}")

    # 6. Esportazione CSV opzionale
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

    # 7. Esportazione Markdown opzionale
    if args.output_md:
        out_md_path = Path(args.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(table_md + "\n")
        print(f"Tabella comparativa Markdown salvata in: {out_md_path}")


if __name__ == "__main__":
    main()
