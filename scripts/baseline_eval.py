"""
Valutazione baseline (pre-finetuning) dei modelli BERT sul test set.

Il flusso di preprocessing applicato al test set è **identico** a quello usato
durante il finetuning (model-specific):
  - normalize_greek (Unicode, case_folding, strip_diacritics)
  - remove_punctuation (se previsto dalla config)

Queste operazioni vengono applicate *dentro* `fill_mask` leggendo la config
tramite `get_model_config(checkpoint)`, quindi il confronto pre/post-FT è fair.

Le metriche calcolate (EM@K, BERTscore@K, BERTscore@K mean) per K=[1,5,10,20]
vengono stampate nel formato esatto atteso da `models/bert/evaluation/plot.py`
per popolare `PRE_FT_BASELINE`.
"""

import json
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset

from models.bert.dataset import EVAL_CHECKPOINT
from models.bert.dataset.dev_set import DevCase
from models.bert.finetuning import BASE_MODEL_MAP, GAP_TOKEN
from models.bert.finetuning.pipeline import _evaluate_hcb_on_split
from models.bert.evaluation.metrics import reset_scorer_cache

# K values usati durante la valutazione (allineati con plot.py e la pipeline di FT)
K_VALUES = [1, 5, 10, 20]


def load_test_cases() -> list[DevCase]:
    """
    Carica il test set dal checkpoint di valutazione su HuggingFace.

    I DevCase vengono caricati nel formato grezzo (model-agnostic): il
    preprocessing model-specific (case_folding, strip_diacritics, remove_punct)
    viene applicato **a runtime** dentro `fill_mask` tramite la config del
    checkpoint, garantendo lo stesso trattamento del finetuning.
    """
    print(f"Caricamento test set da '{EVAL_CHECKPOINT}'...")
    eval_dataset = load_dataset(EVAL_CHECKPOINT)

    cases = []
    for row in eval_dataset["test"].to_list():
        # Stesso filtro gap_length usato durante il finetuning (pipeline.py)
        if 1 <= row["gap_length"] <= 6:
            cases.append(
                DevCase(
                    x=row["x"],
                    y=row["y"],
                    gap_length=row["gap_length"],
                    corpus_id=row["corpus_id"],
                    file_id=row["file_id"],
                )
            )

    print(f"Test set caricato: {len(cases)} casi (gap_length 1-6).")
    return cases


def evaluate_all_baselines():
    """
    Valuta tutti i modelli base (pre-finetuning) e genera le entry per plot.py.

    Per ogni modello in BASE_MODEL_MAP:
      1. Carica tokenizer + modello base (pesi vergini)
      2. Esegue inferenza sul test set via `_evaluate_hcb_on_split`, che applica
         internamente il preprocessing model-specific attraverso `fill_mask`
      3. Calcola EM@K, BERTscore@K (max) e BERTscore@K (mean) per K=[1,5,10,20]
      4. Stampa le entry nel formato `PRE_FT_BASELINE` di plot.py
    """
    test_cases = load_test_cases()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}\n")

    all_results: list[dict] = []

    for finetuned_ckpt, base_model in BASE_MODEL_MAP.items():
        print(f"\n{'='*60}")
        print(f"Valutazione Baseline: {base_model}")
        print(f"Configurazione Target (per preprocessing): {finetuned_ckpt}")
        print(f"{'='*60}")

        # 1. Reset della cache BERTScore per evitare conflitti tra modelli
        reset_scorer_cache()

        # 2. Caricamento Tokenizer con aggiunta del GAP token
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.model_max_length = 512

        if GAP_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [GAP_TOKEN]})

        # 3. Caricamento Modello Base (pesi pre-training, non fine-tuned)
        model = AutoModelForMaskedLM.from_pretrained(base_model)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
        model.to(device)

        # 4. Inferenza + calcolo metriche sul test set
        #    Passiamo `checkpoint=finetuned_ckpt` così fill_mask legge la config
        #    model-specific (case_folding, strip_diacritics, remove_punct) corretta.
        #    Il dataset viene valutato con K=20 suggerimenti per coprire tutti i K.
        metrics = _evaluate_hcb_on_split(
            split_name="test_baseline",
            cases=test_cases,
            model=model,
            tokenizer=tokenizer,
            checkpoint=finetuned_ckpt,
        )

        # 5. Nome breve del modello per plot.py (es. "Logion", "GreBerta", "aristoBERTo")
        model_name_short = finetuned_ckpt.split("/")[-1].replace("gs-", "")

        # 6. Costruzione delle entry nel formato PRE_FT_BASELINE di plot.py
        print(f"\n[!] ENTRY PRE_FT_BASELINE per {model_name_short} [!]")
        for k in K_VALUES:
            em_val   = metrics.get(f"top{k}", 0.0)
            bs_max   = metrics.get(f"bertscore_f1_top{k}", 0.0)
            bs_mean  = metrics.get(f"bertscore_f1_top{k}_mean", 0.0)

            entry = {
                "Modello": model_name_short,
                "Stato":   "Pre-FT",
                "K":       k,
                "EM":      round(em_val, 4),
                "BS_Max":  round(bs_max, 4),
                "BS_Mean": round(bs_mean, 4),
            }
            all_results.append(entry)

            # Stringa pronta per copia-incolla in PRE_FT_BASELINE di plot.py
            print(
                f'    {{"Modello": "{model_name_short}", "Stato": "Pre-FT", "K": {k}, '
                f'"EM": {em_val:.4f}, "BS_Max": {bs_max:.4f}, "BS_Mean": {bs_mean:.4f}}},'
            )

        # 7. Liberazione della VRAM prima del prossimo modello
        del model
        torch.cuda.empty_cache()

    # 8. Dump JSON completo per uso programmatico
    print("\n\n=== RISULTATI COMPLETI (JSON) ===")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))

    return all_results


if __name__ == "__main__":
    evaluate_all_baselines()