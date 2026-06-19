import json
import pathlib
import argparse
from collections import Counter
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

# Aggiungi la root del progetto al PYTHONPATH in modo da poter importare backend e models
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

from backend.core import UNK_TOKEN
from backend.core.preprocess import normalize_greek, remove_punctuation
from models.bert.finetuning import ModelRegistry, get_model_config
from datasets import load_dataset

MODELS = list(ModelRegistry().base_model_map.values())

DATA_DIR = pathlib.Path("data")
OUT_DIR = pathlib.Path("scripts/vocab")


def load_hf_texts(dataset_name: str) -> list[str]:
    print(f"Download del dataset '{dataset_name}' da HuggingFace Hub...")
    dataset = load_dataset(dataset_name, split="train")
    return [item["text"] for item in dataset]


def normalize_text(text: str, config: dict, unk_token: str) -> str:
    norm_text = normalize_greek(
        text,
        case_folding=config.get("case_folding", "upper"),
        strip_diacritics_flag=config.get("strip_diacritics", True),
    )

    if config.get("remove_punct"):
        norm_text = remove_punctuation(norm_text)

    # Calcoliamo la forma che UNK_TOKEN ha assunto dopo il case folding
    cf = config.get("case_folding", "upper")
    if cf == "lower":
        agnostic_unk = UNK_TOKEN.lower()
    elif cf == "upper":
        agnostic_unk = UNK_TOKEN.upper()
    elif cf == "fold":
        agnostic_unk = UNK_TOKEN.casefold()
    else:
        agnostic_unk = UNK_TOKEN

    # Ora rimpiazziamo la versione case-folded con il vero token del modello
    norm_text = norm_text.replace(agnostic_unk, unk_token)
    return norm_text


def analyze_model_vocab(model_name: str, texts: list[str], dataset_name: str) -> dict:
    print(f"\n--- Analisi Modello: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = get_model_config(model_name)

    total_words = 0
    total_tokens = 0
    unk_count = 0
    token_freqs = Counter()

    special_token_ids = set(tokenizer.all_special_ids)
    unk_token_id = tokenizer.unk_token_id

    # Tokenizzazione in batch per massimizzare le prestazioni
    batch_size = 500
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Processando {model_name.split('/')[-1]}",
    ):
        batch = texts[i : i + batch_size]
        norm_batch = []
        for text in batch:
            norm_text = normalize_text(text, config, tokenizer.unk_token)
            norm_batch.append(norm_text)
            total_words += len(norm_text.split())

        encoded_batch = tokenizer(norm_batch, add_special_tokens=False)
        for input_ids in encoded_batch["input_ids"]:
            total_tokens += len(input_ids)
            if unk_token_id is not None:
                unk_count += input_ids.count(unk_token_id)

            # Filtra i token speciali e aggiorna le frequenze usando gli ID (molto più veloce)
            filtered_ids = [tid for tid in input_ids if tid not in special_token_ids]
            token_freqs.update(filtered_ids)

    unk_rate = (unk_count / total_tokens * 100) if total_tokens > 0 else 0
    frag_index = (total_tokens / total_words) if total_words > 0 else 0

    # Calcolo Coperture percentuali
    sorted_freqs_ids = token_freqs.most_common()
    total_filtered_tokens = sum(count for _, count in sorted_freqs_ids)

    def get_percentile_coverage(target_percentage: float) -> int:
        target_sum = total_filtered_tokens * target_percentage
        current_sum = 0
        for i, (_, count) in enumerate(sorted_freqs_ids):
            current_sum += count
            if current_sum >= target_sum:
                return i + 1
        return len(sorted_freqs_ids)

    cov_50 = get_percentile_coverage(0.50)
    cov_90 = get_percentile_coverage(0.90)
    cov_95 = get_percentile_coverage(0.95)

    # Convertiamo gli ID dei token nelle rispettive stringhe in batch alla fine dell'analisi
    ids_to_convert = [tid for tid, _ in sorted_freqs_ids]
    token_strings = tokenizer.convert_ids_to_tokens(ids_to_convert)

    sorted_freqs = []
    for token_str, (_, count) in zip(token_strings, sorted_freqs_ids):
        if token_str is None:
            token_str = "[UNKNOWN_ID]"
        sorted_freqs.append((token_str, count))

    print(f"Metriche per {model_name}:")
    print(f" - Totale Parole:  {total_words:,}")
    print(f" - Totale Tokens:  {total_tokens:,}")
    print(f" - Indice di Frammentazione (Tokens/Word): {frag_index:.4f}")
    print(f" - UNK Token Count: {unk_count:,} ({unk_rate:.4f}%)")
    print(
        f" - Copertura Vocabolario: 50%={cov_50:,} tokens | 90%={cov_90:,} tokens | 95%={cov_95:,} tokens"
    )

    # Salvataggio distribuzione frequenza
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_dataset_name = dataset_name.replace("/", "_")
    safe_name = model_name.replace("/", "_")
    out_file = OUT_DIR / f"{safe_dataset_name}_{safe_name}_freq.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dict(sorted_freqs), f, ensure_ascii=False, indent=2)
    print(f"Distribuzione salvata in {out_file}")

    return {
        "model": model_name,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "frag_index": frag_index,
        "unk_count": unk_count,
        "unk_rate": unk_rate,
        "cov_50": cov_50,
        "cov_90": cov_90,
        "cov_95": cov_95,
        "top_30": sorted_freqs[:30],
    }


def main():
    parser = argparse.ArgumentParser(description="Analisi del vocabolario sui dataset HuggingFace")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CNR-ILC/gs-dataset-train",
        help="Nome del dataset HuggingFace da analizzare"
    )
    args = parser.parse_args()

    print(f"Avvio Analisi del Vocabolario per il dataset '{args.dataset}'...")
    texts = load_hf_texts(args.dataset)
    if not texts:
        print("Nessun testo scaricato dal dataset. Interruzione.")
        return
    print(f"Totale blocchi testuali caricati: {len(texts):,}\n")

    results = []
    for model in MODELS:
        res = analyze_model_vocab(model, texts, args.dataset)
        results.append(res)

    # Salva il file di riassunto MarkDown
    safe_dataset_name = args.dataset.replace("/", "_")
    summary_file = OUT_DIR / f"{safe_dataset_name}_analysis_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Riassunto Analisi del Vocabolario - Dataset: {args.dataset}\n\n")
        f.write(
            "| Modello | Totale Token | Tokens/Word | UNK Count | UNK Rate | 50% Coverage | 90% Coverage | 95% Coverage |\n"
        )
        f.write(
            "|---------|-------------|-------------|-----------|----------|--------------|--------------|--------------|\n"
        )
        for r in results:
            f.write(
                f"| `{r['model']}` | {r['total_tokens']:,} | {r['frag_index']:.4f} | {r['unk_count']:,} | {r['unk_rate']:.4f}% | {r['cov_50']:,} | {r['cov_90']:,} | {r['cov_95']:,} |\n"
            )

        f.write("\n## Top 30 Token più frequenti\n\n")
        for r in results:
            f.write(f"### {r['model']}\n")
            f.write("```json\n")
            f.write(json.dumps(dict(r["top_30"]), ensure_ascii=False, indent=2))
            f.write("\n```\n\n")

    print(f"\nAnalisi completata! Riassunto salvato in {summary_file}")


if __name__ == "__main__":
    main()
