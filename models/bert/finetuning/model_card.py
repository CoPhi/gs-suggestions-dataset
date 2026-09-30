"""
Modulo per la generazione, formattazione e pubblicazione della Model Card arricchita su Hugging Face Hub.

Questo modulo automatizza la creazione di una Model Card dettagliata e professionale conforme
agli standard di Hugging Face, evidenziando in modo esplicito:
1. Metadati YAML completi (tag, dataset, widget interattivi, pipeline fill-mask, model-index).
2. Contesto del progetto ERC GreekSchools e CNR-ILC.
3. Dettagli architetturali e pipeline di normalizzazione filologica (rimozione diacritici, casing).
4. Configurazione e iperparametri di fine-tuning (MLM / Span-MLM).
5. Tabella comparativa dettagliata Pre-FT vs Post-FT con misure delta (Δ) sul test set TLG
   (o dataset di valutazione specificato), suddivisa per le policy di lacuna (default, word, suffix).
6. Snippet di codice pronti all'uso (pipeline fill-mask e AutoModel).
7. Riferimenti bibliografici e citazioni.
"""

from __future__ import annotations

import io
import os
import json
from typing import Any, Mapping
from huggingface_hub import HfApi


METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    # key: (Etichetta display, Descrizione, Is Percentage)
    "top1": ("Exact Match @1", "Accuratezza lessicale esatta della prima predizione", True),
    "top5": ("Exact Match @5", "Presenza della gold label tra i primi 5 candidati", True),
    "top10": ("Exact Match @10", "Presenza della gold label tra i primi 10 candidati", True),
    "top20": ("Exact Match @20", "Presenza della gold label tra i primi 20 candidati", True),
    "bertscore_f1_top1": ("BERTScore F1 @1", "Plausibilità semantica contestuale della prima predizione", True),
    "bertscore_f1_top5": ("BERTScore F1 @5", "Plausibilità semantica massima tra i primi 5 candidati", True),
    "bertscore_f1_top10": ("BERTScore F1 @10", "Plausibilità semantica massima tra i primi 10 candidati", True),
    "bertscore_f1_top20": ("BERTScore F1 @20", "Plausibilità semantica massima tra i primi 20 candidati", True),
    "cos_sim_top1_max": ("Cosine Similarity Max @1", "Similarità coseno dell'embedding contestuale Top-1", True),
    "cos_sim_top5_max": ("Cosine Similarity Max @5", "Similarità coseno massima tra i primi 5 candidati", True),
    "cos_sim_top10_max": ("Cosine Similarity Max @10", "Similarità coseno massima tra i primi 10 candidati", True),
    "cos_sim_top20_max": ("Cosine Similarity Max @20", "Similarità coseno massima tra i primi 20 candidati", True),
    "cos_sim_top1_mean": ("Cosine Similarity Mean @1", "Similarità coseno media Top-1", True),
    "cos_sim_top5_mean": ("Cosine Similarity Mean @5", "Similarità coseno media tra i primi 5 candidati", True),
    "cos_sim_top10_mean": ("Cosine Similarity Mean @10", "Similarità coseno media tra i primi 10 candidati", True),
    "cos_sim_top20_mean": ("Cosine Similarity Mean @20", "Similarità coseno media tra i primi 20 candidati", True),
    "cluster_inclusion_rate": ("Cluster Inclusion Rate", "Tasso di inclusione della gold label nel cluster denso dei candidati", True),
    "mean_inclusion_margin": ("Mean Inclusion Margin", "Margine medio di inclusione nel cluster semantico", False),
    "mean_gold_centroid_cosine_sim": ("Gold Centroid CosSim", "Similarità coseno tra gold label e baricentro del cluster", True),
}

DEFAULT_WIDGET_EXAMPLES: dict[str, list[str]] = {
    "CNR-ILC/gs-GreBerta": [
        "περι [MASK] λεγομενων",
        "κατα την [MASK] των πραγματων",
        "ουδεν γαρ [MASK] γινεται",
    ],
    "CNR-ILC/gs-aristoBERTo": [
        "παντες ανθρωποι του [MASK] ορεγονται φυσει",
        "η δε [MASK] εστιν ενεργεια",
        "περι [MASK] λεγομενων",
    ],
    "CNR-ILC/gs-Logion": [
        "εν αρχη ην ο [MASK]",
        "και το φως εν τη [MASK] φαινει",
        "περι [MASK] λεγομενων",
    ],
}


def _format_metric_row(
    label: str,
    val_pre: float,
    val_post: float,
    is_pct: bool,
) -> tuple[str, str, str, str, str]:
    delta = val_post - val_pre
    if is_pct:
        pre_str = f"{val_pre:.2f}%"
        post_str = f"{val_post:.2f}%"
        delta_str = f"{delta:+.2f}%"
    else:
        pre_str = f"{val_pre:.4f}"
        post_str = f"{val_post:.4f}"
        delta_str = f"{delta:+.4f}"

    if delta > 0.05 if is_pct else delta > 0.001:
        badge = "🟢"
    elif delta < -0.05 if is_pct else delta < -0.001:
        badge = "🔻"
    else:
        badge = "⚪"

    return label, pre_str, post_str, delta_str, badge


def _render_metrics_table(
    pre_metrics: dict[str, float],
    post_metrics: dict[str, float],
    title: str | None = None,
) -> str:
    """Genera una tabella Markdown per il confronto Pre-FT vs Post-FT con colonna delta e badge."""
    lines = []
    if title:
        lines.append(f"#### {title}\n")

    lines.append("| Metric | Pre-FT (Baseline) | Post-FT (Fine-Tuned) | Delta (Δ) | Trend |")
    lines.append("|:-------|:------------------:|:--------------------:|:---------:|:-----:|")

    for key, (label, _desc, is_pct) in METRIC_DEFINITIONS.items():
        if key in pre_metrics or key in post_metrics:
            val_pre = float(pre_metrics.get(key, 0.0))
            val_post = float(post_metrics.get(key, 0.0))
            _, pre_str, post_str, delta_str, badge = _format_metric_row(
                label, val_pre, val_post, is_pct
            )
            lines.append(f"| **{label}** | {pre_str} | **{post_str}** | `{delta_str}` | {badge} |")

    lines.append("")
    return "\n".join(lines)


def _build_yaml_frontmatter(
    checkpoint: str,
    base_model: str,
    dataset_name: str,
    post_ft_metrics: dict[str, Any] | None = None,
    license_type: str = "apache-2.0",
) -> str:
    """Costruisce il frontmatter YAML per Hugging Face con tags, pipeline, dataset e model-index."""
    repo_name = checkpoint.split("/")[-1]
    widget_texts = DEFAULT_WIDGET_EXAMPLES.get(
        checkpoint,
        ["περι [MASK] λεγομενων", "κατα την [MASK] των πραγματων"],
    )

    tags = [
        "ancient-greek",
        "papyrology",
        "herculaneum-papyri",
        "lacuna-infilling",
        "text-restoration",
        "masked-language-modeling",
        "fill-mask",
        "greekschools",
        "erc-885222",
        "cnr-ilc",
    ]

    # Estrazione metriche per model-index (usando split default o principale)
    target_metrics = post_ft_metrics or {}
    if target_metrics and any(isinstance(v, dict) for v in target_metrics.values()):
        # Prendi la policy "default" se presente, altrimenti la prima policy
        target_metrics = target_metrics.get("default", next(iter(target_metrics.values())))

    model_index_metrics = []
    hf_metric_mapping = [
        ("top1", "Exact Match @1", "accuracy"),
        ("top5", "Exact Match @5", "accuracy"),
        ("top10", "Exact Match @10", "accuracy"),
        ("top20", "Exact Match @20", "accuracy"),
        ("bertscore_f1_top1", "BERTScore F1 @1", "bertscore"),
        ("bertscore_f1_top5", "BERTScore F1 @5", "bertscore"),
        ("cos_sim_top1_max", "Cosine Similarity Max @1", "cosine_similarity"),
        ("cluster_inclusion_rate", "Cluster Inclusion Rate", "cluster_inclusion"),
    ]

    for key, display_name, m_type in hf_metric_mapping:
        if key in target_metrics:
            val = float(target_metrics[key])
            model_index_metrics.append(
                f"      - name: {display_name}\n"
                f"        type: {m_type}\n"
                f"        value: {val:.2f}"
            )

    model_index_yaml = ""
    if model_index_metrics:
        metrics_block = "\n".join(model_index_metrics)
        model_index_yaml = f"""model-index:
- name: {repo_name}
  results:
  - task:
      type: fill-mask
      name: Masked Language Modeling (Ancient Greek Lacuna Infilling)
    dataset:
      name: {dataset_name.split('/')[-1]}
      type: {dataset_name}
    metrics:
{metrics_block}
"""
    else:
        model_index_yaml = f"""model-index:
- name: {repo_name}
  results: []
"""

    widget_yaml_lines = ["widget:"]
    for t in widget_texts:
        widget_yaml_lines.append(f"- text: \"{t}\"")
    widget_yaml = "\n".join(widget_yaml_lines)

    tags_yaml = "\n".join([f"- {tag}" for tag in tags])

    yaml = f"""---
language:
- grc
- el
license: {license_type}
library_name: transformers
pipeline_tag: fill-mask
base_model: {base_model}
tags:
{tags_yaml}
datasets:
- {dataset_name}
{widget_yaml}
{model_index_yaml}---
"""
    return yaml


def generate_model_card(
    checkpoint: str,
    base_model: str,
    dataset_name: str = "CNR-ILC/gs-dataset-tlg-uncased",
    pre_ft_metrics: dict[str, Any] | None = None,
    post_ft_metrics: dict[str, Any] | None = None,
    eval_metrics: dict[str, float] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    preprocessing_config: dict[str, Any] | None = None,
    license_type: str = "apache-2.0",
) -> str:
    """
    Genera il contenuto Markdown completo della Model Card per Hugging Face.
    
    Args:
        checkpoint: Identificativo del checkpoint fine-tuned (es. "CNR-ILC/gs-GreBerta").
        base_model: Checkpoint base dei pesi (es. "bowphs/GreBerta").
        dataset_name: Nome del dataset di addestramento / valutazione (default: "CNR-ILC/gs-dataset-tlg-uncased").
        pre_ft_metrics: Dizionario metriche Pre-FT (può essere diviso per policy o flat).
        post_ft_metrics: Dizionario metriche Post-FT (può essere diviso per policy o flat).
        eval_metrics: Metriche finali del Trainer (loss, perplexity).
        hyperparameters: Parametri di training (lr, batch_size, epochs, chunk_size, etc.).
        preprocessing_config: Configurazione normalizzazione (strip_diacritics, case_folding, etc.).
        license_type: Tipo di licenza software/modello (default: "apache-2.0").

    Returns:
        Stringa formattata con YAML frontmatter e Markdown completo.
    """
    repo_name = checkpoint.split("/")[-1]
    yaml_header = _build_yaml_frontmatter(
        checkpoint=checkpoint,
        base_model=base_model,
        dataset_name=dataset_name,
        post_ft_metrics=post_ft_metrics,
        license_type=license_type,
    )

    # Verifica se le metriche sono raggruppate per policy (es. default, word, suffix)
    is_multi_policy = False
    if pre_ft_metrics and isinstance(pre_ft_metrics, dict):
        first_val = next(iter(pre_ft_metrics.values()), None)
        if isinstance(first_val, dict):
            is_multi_policy = True

    # Calcolo dei KPI salienti per la sezione highlight
    highlight_cards = []
    ref_pre = None
    ref_post = None

    if is_multi_policy:
        ref_pre = pre_ft_metrics.get("default", next(iter(pre_ft_metrics.values())))
        if post_ft_metrics:
            ref_post = post_ft_metrics.get("default", next(iter(post_ft_metrics.values())))
    else:
        ref_pre = pre_ft_metrics
        ref_post = post_ft_metrics

    if ref_pre and ref_post:
        kpis = [
            ("top1", "Exact Match @1", "%"),
            ("top5", "Exact Match @5", "%"),
            ("bertscore_f1_top1", "BERTScore F1 @1", "%"),
            ("cluster_inclusion_rate", "Cluster Inclusion", "%"),
        ]
        for key, name, unit in kpis:
            if key in ref_pre and key in ref_post:
                v_pre = float(ref_pre[key])
                v_post = float(ref_post[key])
                diff = v_post - v_pre
                icon = "🚀" if diff > 0 else "📊"
                highlight_cards.append(
                    f"> **{icon} {name}**: **{v_post:.2f}{unit}** (`{diff:+.2f}{unit}` vs Pre-FT **{v_pre:.2f}{unit}**)"
                )

    highlights_block = ""
    if highlight_cards:
        highlights_block = (
            "### 🌟 Highlight dei Risultati (Progresso di Fine-Tuning)\n\n"
            + "\n>\n".join(highlight_cards)
            + "\n\n"
        )

    # Costruzione sezione tabelle di valutazione
    eval_tables_block = ""
    if pre_ft_metrics and post_ft_metrics:
        eval_tables_block += "### 📊 Risultati Comparativi sul Dataset di Test (TLG)\n\n"
        eval_tables_block += (
            f"La valutazione è stata condotta confrontando la baseline pre-addestrata (**Pre-FT: `{base_model}`**) "
            f"con il modello fine-tunato (**Post-FT: `{checkpoint}`**) sui casi di test estratti dal corpus "
            f"**TLG** (`{dataset_name}`).\n\n"
        )

        if is_multi_policy:
            policy_labels = {
                "default": "Policy 'Default' (Lacune casuali di 1-6 caratteri simulanti danni fisici ai papiri)",
                "word": "Policy 'Word' (Lacune a livello di parola intera)",
                "suffix": "Policy 'Suffix' (Lacune sulle terminazioni flessive / suffissi grammaticali)",
            }
            for p_name in pre_ft_metrics.keys():
                p_pre = pre_ft_metrics.get(p_name, {})
                p_post = post_ft_metrics.get(p_name, {}) if post_ft_metrics else {}
                if p_pre or p_post:
                    title = policy_labels.get(p_name, f"Policy '{p_name.capitalize()}'")
                    eval_tables_block += _render_metrics_table(p_pre, p_post, title=title)
        else:
            eval_tables_block += _render_metrics_table(pre_ft_metrics, post_ft_metrics)

        eval_tables_block += (
            "> [!NOTE]\n"
            "> - **Exact Match (Top-K)** misura l'accuratezza lessicale esatta della ricostruzione.\n"
            "> - **BERTScore F1 & CosSim** valutano la plausibilità semantica contestuale anche quando il completamento differisce dalla congettura gold.\n"
            "> - **Cluster Inclusion Rate** quantifica la densità e la coerenza dello spazio predittivo rispetto alla gold label.\n\n"
        )
    elif post_ft_metrics:
        eval_tables_block += "### 📊 Risultati di Valutazione Post-Fine-Tuning\n\n"
        eval_tables_block += "| Metrica | Valore Post-FT |\n|:---|:---:|\n"
        target_dict = post_ft_metrics.get("default", post_ft_metrics) if is_multi_policy else post_ft_metrics
        for k, (lbl, _, is_pct) in METRIC_DEFINITIONS.items():
            if k in target_dict:
                v = float(target_dict[k])
                v_str = f"{v:.2f}%" if is_pct else f"{v:.4f}"
                eval_tables_block += f"| **{lbl}** | {v_str} |\n"
        eval_tables_block += "\n"

    # Sezione Eval Loss e Perplexity (se disponibili)
    loss_block = ""
    if eval_metrics:
        eval_loss = eval_metrics.get("eval_loss")
        eval_ppl = eval_metrics.get("eval_perplexity")
        if eval_loss is not None:
            loss_block = (
                f"- **Validation Loss**: `{eval_loss:.4f}`\n"
                + (f"- **Validation Perplexity**: `{eval_ppl:.2f}`\n" if eval_ppl else "")
                + "\n"
            )

    # Parametri di addestramento
    hp_block = ""
    if hyperparameters:
        hp_rows = []
        for k, v in hyperparameters.items():
            if isinstance(v, float) and abs(v) < 1e-3:
                val_str = f"{v:.2e}"
            else:
                val_str = str(v)
            hp_rows.append(f"| `{k}` | `{val_str}` |")
        
        hp_table = "\n".join(hp_rows)
        hp_block = (
            "### ⚙️ Iperparametri di Addestramento\n\n"
            "| Iperparametro | Valore Configurato |\n"
            "|:---|:---|\n"
            f"{hp_table}\n\n"
        )

    # Configurazione di normalizzazione filologica
    norm_block = ""
    if preprocessing_config:
        case_fold = preprocessing_config.get("case_folding", "lower")
        strip_diacritics = preprocessing_config.get("strip_diacritics", True)
        remove_punct = preprocessing_config.get("remove_punct", True)
        norm_block = (
            "### 🔤 Pipeline di Preprocessing e Normalizzazione Filologica\n\n"
            "Per allineare il modello alle caratteristiche dei papiri documentari e letterari (in cui mancano accenti, "
            "spiriti e punteggiatura moderna), il testo viene normalizzato con le seguenti impostazioni:\n"
            f"- **Case Folding**: `{case_fold}` (conversione in minuscolo)\n"
            f"- **Rimozione Segni Diacritici**: `{'Abilitata (strip diacritics)' if strip_diacritics else 'Disabilitata'}` (rimozione di accenti, spiriti e iota sottoscritta)\n"
            f"- **Rimozione Punteggiatura**: `{'Abilitata' if remove_punct else 'Disabilitata'}`\n\n"
        )

    body = f"""# {repo_name}

[![ERC Project: GreekSchools](https://img.shields.io/badge/ERC%20Grant-885222-blue.svg)](https://greekschools.eu/)
[![CNR-ILC](https://img.shields.io/badge/CNR-ILC%20Pisa-darkgreen.svg)](https://www.ilc.cnr.it/)
[![Task](https://img.shields.io/badge/Task-Fill--Mask%20%7C%20Ancient%20Greek-orange.svg)](#)
[![Base Model](https://img.shields.io/badge/Base%20Model-{base_model.replace('-', '--')}-lightgrey.svg)](https://huggingface.co/{base_model})

**`{checkpoint}`** è un modello linguistico neurale basato su architettura Transformer per il **Greco Antico**, specializzato nel compito di **Masked Language Modeling (MLM)** e **infilling di lacune testuali** (integrazione di caratteri e parole mancanti in papiri ed iscrizioni).

Il modello è stato sviluppato nell'ambito del progetto **[ERC Advanced Grant GreekSchools](https://greekschools.eu/)** (Grant Agreement No. 885222) presso l'**[Istituto di Linguistica Computazionale "A. Zampolli" del CNR (CNR-ILC)](https://www.ilc.cnr.it/)** di Pisa.

---

## 🎯 Scopo del Modello e Casi d'Uso

Il modello è specificamente calibrato per supportare papirologi, epigrafisti e filologi classici nell'analisi e restauro dei testi frammentari dell'antichità greca, con particolare attenzione ai **Papiri di Ercolano** (es. testi filosofici epicurei di Filodemo di Gadara).

Caratteristiche chiave:
- **Lacuna Infilling Multicarattere**: integrazione di sequenze mancanti di lunghezza variabile mediante token `[MASK]`.
- **Adattamento di Dominio sul Corpus TLG**: continual pre-training sul corpus letterario greco antico (`{dataset_name}`).
- **Conformità Filologica**: normalizzazione model-specific diacritica ed epigrafica.

---

{highlights_block}{eval_tables_block}## 🛠️ Procedura di Addestramento

Il modello è stato addestrato eseguendo un continual pre-training con obiettivo **Masked Language Modeling (MLM)** a partire dai pesi pre-addestrati di [`{base_model}`](https://huggingface.co/{base_model}).

- **Corpus di Addestramento**: [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name}) (Thesaurus Linguae Graecae uncased).
{loss_block}{hp_block}{norm_block}---

## 💻 Come Utilizzare il Modello

### 1. Utilizzo ad alto livello con `pipeline` di Transformers

```python
from transformers import pipeline

# Inizializzazione della pipeline fill-mask
unmasker = pipeline("fill-mask", model="{checkpoint}")

# Frase in greco antico normalizzato con lacuna mascherata
text = "περι [MASK] λεγομενων"

# Estrazione dei migliori suggerimenti
suggestions = unmasker(text, top_k=5)
for s in suggestions:
    print(f"Token: {{s['token_str']:<15}} | Probabilità: {{s['score']:.4f}} | Testo completo: {{s['sequence']}}")
```

### 2. Inferenza con `AutoModelForMaskedLM` e `AutoTokenizer`

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

checkpoint = "{checkpoint}"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
model.eval()

text = "κατα την [MASK] των πραγματων"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Identifica la posizione del token [MASK]
mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
mask_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_logits, 5, dim=1).indices[0].tolist()

print("Candidati suggeriti:")
for token_id in top_5_tokens:
    print("-", tokenizer.decode([token_id]).strip())
```

---

## ⚠️ Limiti e Avvertenze Filologiche

- Il modello fornisce suggerimenti probabilistici basati sulla frequenza e co-occorrenza nel corpus TLG; non sostituisce l'autopsia del papiro, l'esame multispettrale o la perizia paleografica.
- Le predizioni devono essere vagliate criticamente dallo studioso rispetto alla compatibilità con lo spazio fisico della lacuna (*traccia delle lettere*) e con il *ductus* dello scriba.

---

## 🏛️ Riconoscimenti e Citazione

Il progetto **GreekSchools** ha ricevuto finanziamenti dall'**European Research Council (ERC)** nell'ambito del programma di ricerca e innovazione Horizon 2020 dell'Unione Europea (Grant Agreement No. 885222).

Se utilizzi questo modello nella tua ricerca, ti preghiamo di citarlo:

```bibtex
@misc{{greekschools_{repo_name.lower().replace('-', '_')},
  title        = {{{{{repo_name}: A Fine-Tuned Language Model for Ancient Greek Lacuna Infilling}}}},
  author       = {{ERC GreekSchools Project and CNR-ILC}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/{checkpoint}}}}},
  note         = {{ERC Advanced Grant No. 885222}}
}}
```
"""
    return yaml_header + body


def save_model_card(card_content: str, output_path: str) -> str:
    """Salva il contenuto della Model Card nel percorso specificato (es. directory/README.md)."""
    if os.path.isdir(output_path):
        target_file = os.path.join(output_path, "README.md")
    else:
        target_file = output_path

    os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"[ModelCard] Model Card salvata con successo in: {target_file}")
    return target_file


def push_model_card_to_hub(
    repo_id: str,
    card_content: str,
    token: str | None = None,
    commit_message: str = "docs: update rich model card with Pre/Post FT evaluation metrics and delta measures",
) -> str:
    """
    Pubblica direttamente il README.md arricchito sul repository Hugging Face Hub specificato.

    Args:
        repo_id: Identificativo del repository Hugging Face (es. "CNR-ILC/gs-GreBerta").
        card_content: Contenuto Markdown completo con YAML frontmatter.
        token: Token di accesso Hugging Face (se None, usa HF_TOKEN da ambiente).
        commit_message: Messaggio di commit per il caricamento su Hugging Face.

    Returns:
        URL del file caricato sull'Hub.
    """
    hf_token = token or os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)

    buffer = io.BytesIO(card_content.encode("utf-8"))
    upload_result = api.upload_file(
        path_or_fileobj=buffer,
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=commit_message,
        token=hf_token,
    )
    print(f"[ModelCard] Model Card caricata con successo su Hugging Face Hub [{repo_id}]!")
    return str(upload_result)
