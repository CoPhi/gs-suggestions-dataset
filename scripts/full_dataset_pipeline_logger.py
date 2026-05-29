import inspect
import json
from pathlib import Path
import sys
import os
import random

from transformers import AutoTokenizer

from backend.core.cleaner import load_abs, get_sentences
from backend.core.preprocess import (
    process_integrations,
    process_leiden_lb,
    process_unclear_signs,
    process_brackets,
    process_dactyl_patterns,
    process_vacat_text,
    process_double_obelisks,
    process_doubts,
    process_missing_lines,
    process_parentheses,
    process_markers,
    process_expunctions,
    process_parallel_text,
    process_dash_if_needed,
)
from models.bert.dataset.train_set import is_quality_sentence, _create_flat_record
from models.bert.dataset.dev_set import build_dev_cases
from models.bert.dataset.load import _normalize_example, _quality_filter_subword
from models.bert.finetuning import get_model_config

def get_function_signature(func):
    try:
        return f"{func.__name__}{inspect.signature(func)}"
    except ValueError:
        return f"{func.__name__}(...)"

def generate_full_log(output_file: str):
    print("Caricamento del corpus TLG e Herc per estrarre campioni di test...")
    sample_blocks = []
    
    # Preleviamo un paio di blocchi da TLG
    try:
        tlg_abs = load_abs(corpus_set=["tlg"])
        # Filtriamo blocchi che abbiano training_text
        tlg_valid = [ab for ab in tlg_abs if ab.get("training_text")]
        if tlg_valid:
            rng = random.Random(42)
            sample_blocks.extend(rng.sample(tlg_valid, min(3, len(tlg_valid))))
    except Exception as e:
        print(f"Errore caricamento TLG: {e}")

    # Preleviamo un paio di blocchi da Herc (che contengono sicuramente i supplementi per i dev cases)
    try:
        from backend.core.cleaner import load_specific_domain_abs
        all_abs = load_abs()
        herc_abs = load_specific_domain_abs(all_abs, "P.Herc.")
        herc_valid = [ab for ab in herc_abs if ab.get("training_text")]
        if herc_valid:
            # Cerchiamo specificamente blocchi che contengano supplementi (regex '[' e ']')
            herc_with_gaps = [ab for ab in herc_valid if "[" in ab["training_text"] and "]" in ab["training_text"]]
            rng = random.Random(43)
            sample_blocks.extend(rng.sample(herc_with_gaps, min(3, len(herc_with_gaps))))
    except Exception as e:
        print(f"Errore caricamento Herc: {e}")

    transformations = [
        process_integrations, process_leiden_lb, process_unclear_signs,
        process_brackets, process_dactyl_patterns, process_vacat_text,
        process_double_obelisks, process_doubts, process_missing_lines,
        process_parentheses, process_markers, process_expunctions,
        process_parallel_text, process_dash_if_needed,
    ]

    log_entries = []

    model_checkpoint = "CNR-ILC/gs-aristoBERTo"
    print(f"Caricamento tokenizer e config per {model_checkpoint}...")
    model_config = get_model_config(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    for i, ab in enumerate(sample_blocks):
        original_text = ab.get("training_text", "")
        entry = {"sample_id": i + 1, "original_text": original_text, "corpus_id": ab.get("corpus_id"), "steps": []}
        current_text = original_text
        
        # Fase 1: Pulizia Editoriale (MAAT)
        entry["steps"].append({"header": "Fase 1: Preprocessing di Base e Pulizia Editoriale (Agnostica)"})
        for transform in transformations:
            input_text = current_text
            current_text = transform(input_text)
            if input_text != current_text:
                entry["steps"].append({
                    "function": get_function_signature(transform),
                    "input": input_text,
                    "output": current_text
                })

        # Fase 2: Training Set (Model-Agnostic)
        entry["steps"].append({"header": "Fase 2: Costruzione Training Set (Model-Agnostic)"})
        
        # Passiamo il VERO blocco anonimo originale a get_sentences, come avviene in train_set.py
        sentences = list(get_sentences(
            [ab],
            case_folding="none",
            remove_punct=False,
            normalize=False,
            strip_diacritics=False,
            metadata=True
        ))
        
        train_records = []
        for record in sentences:
            tokens = record["sentence_tokens"]
            is_valid = is_quality_sentence(tokens)
            flat = _create_flat_record(record)
            
            entry["steps"].append({
                "function": "get_sentences -> is_quality_sentence",
                "input": f"Frase estratta: {tokens}",
                "output": f"Record Piatto: {flat}\nSupera il filtro di qualità (Word-level): {is_valid}"
            })
            if is_valid:
                train_records.append(flat)
                
       
        # Fase 3: Dev Set (Estrazione Supplementi)
        entry["steps"].append({"header": "Fase 3: Costruzione Evaluation Set (Dev Set)"})
        
        # build_dev_cases richiede in input l'intero blocco anonimo
        dev_cases = build_dev_cases(ab, normalize=False)
        if dev_cases:
            for case in dev_cases:
                entry["steps"].append({
                    "function": "build_dev_cases",
                    "input": original_text,
                    "output": f"Estratto DevCase:\n- X (con lacuna): {case.x}\n- Y (gold label): {case.y}\n- gap_length: {case.gap_length}"
                })
        else:
             entry["steps"].append({
                    "function": "build_dev_cases",
                    "input": original_text,
                    "output": "Nessun DevCase estratto (nessun supplemento valido trovato)."
             })

        
        # Fase 4: Normalizzazione Model-Specific
        entry["steps"].append({"header": f"Fase 4: Normalizzazione Model-Specific ({model_checkpoint})"})
        
        if not train_records:
            entry["steps"].append({
                "function": "N/A",
                "input": "-",
                "output": "Nessun record valido salvato dalla Fase 2 da poter processare."
            })
            
        for tr in train_records:
            norm = _normalize_example(tr, config=model_config, unk_token=tokenizer.unk_token)
            is_valid_subword = _quality_filter_subword(norm, tokenizer=tokenizer)
            tokens_generated = tokenizer.tokenize(norm["text"])
            
            entry["steps"].append({
                "function": "_normalize_example -> _quality_filter_subword -> tokenize",
                "input": f"Testo grezzo: {tr['text']}",
                "output": f"Testo Normalizzato: {norm['text']}\n- Subword Tokens generati: {tokens_generated}\n- Supera filtro qualità (Sub-word level): {is_valid_subword}"
            })

        log_entries.append(entry)

    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Log Completo: Pipeline di Creazione del Dataset (Fasi 1-4)\n\n")
        f.write("Questo documento traccia in modo trasparente l'intera pipeline di creazione dei dataset su blocchi reali di MAAT:\n")
        f.write("1. **Fase 1**: Pulizia editoriale MAAT\n")
        f.write("2. **Fase 2**: Estrazione e filtraggio del training set\n")
        f.write("3. **Fase 3**: Estrazione supplementi per il Dev/Test set\n")
        f.write("4. **Fase 4**: Normalizzazione e validazione model-specific\n\n")
        f.write("---\n\n")
        
        for entry in log_entries:
            f.write(f"## Sample {entry['sample_id']} (Corpus: {entry['corpus_id']})\n")
            f.write(f"**Testo Originale**: `{entry['original_text']}`\n\n")
            
            for step in entry["steps"]:
                if "header" in step:
                    f.write(f"### {step['header']}\n")
                else:
                    f.write(f"#### `{step['function']}`\n")
                    f.write(f"- **Input**: `{step['input']}`\n")
                    f.write(f"- **Output**: `{step['output']}`\n")
            
            f.write("\n---\n\n")

    print(f"Log completo generato con successo: {output_file}")

if __name__ == "__main__":
    out_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scratch')))
    out_dir.mkdir(exist_ok=True)
    log_file_path = out_dir / "full_dataset_operations_log.md"
    generate_full_log(log_file_path)
