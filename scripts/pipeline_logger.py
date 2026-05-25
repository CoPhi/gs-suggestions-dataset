import inspect
import json
from pathlib import Path
import sys
import os

import random

# Aggiunge la root del progetto al PYTHONPATH per consentire le importazioni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.cleaner import load_abs
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
    clean_tokens,
)

def get_function_signature(func):
    """Estrae la firma di una funzione come stringa."""
    try:
        return f"{func.__name__}{inspect.signature(func)}"
    except ValueError:
        return f"{func.__name__}(...)"

def generate_log(output_file: str):
    # Campioni di testo sintetici pensati per attivare tutte le varie regex e funzioni
    sample_texts = [
        "Oggi || è | una bella giornata + *",
        "Questo testo ha dei -uu- e poi un vac. e delle [parentesi]",
        "Frase con desunt versus 3 e parentesi (inutili) e aggiunte <marker>",
        "Un esempio †parallelo† con doppio obelisco ‡nota‡ e dubbi?",
        "Espunzioni {da rimuovere} e parole spezzate a fine riga che si- riuniscono",
        "Un testo estremamente sporco: || | + * -uu- vac. [gap] desunt versus 3 (abc) <def> {ghi} †lmn† ‡opq‡ ?"
    ]

    # Campionamento di training_text dal corpus TLG per testare casi reali
    try:
        print("Caricamento del corpus TLG per estrarre campioni di test...")
        tlg_abs = load_abs(corpus_set=["tlg"])
        valid_tlg = [ab["training_text"] for ab in tlg_abs if ab.get("training_text")]
        if valid_tlg:
            # Fissiamo un seed per la riproducibilità del log
            rng = random.Random(42)
            sampled_tlg = rng.sample(valid_tlg, min(5, len(valid_tlg)))
            sample_texts.extend(sampled_tlg)
            print(f"Aggiunti {len(sampled_tlg)} campioni reali dal TLG.")
    except Exception as e:
        print(f"Errore durante il caricamento del corpus TLG: {e}")

    # La pipeline esatta definita in process_editorial_marks
    transformations = [
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
    ]

    log_entries = []

    for i, text in enumerate(sample_texts):
        entry = {"sample_id": i + 1, "original_text": text, "steps": []}
        current_text = text
        
        # 1. Pipeline di process_editorial_marks
        for transform in transformations:
            input_text = current_text
            current_text = transform(input_text)
            
            step_log = {
                "function": get_function_signature(transform),
                "input": input_text,
                "output": current_text,
                "changed": input_text != current_text
            }
            entry["steps"].append(step_log)

        # 2. Pipeline finale di clean_tokens (normalizzazione, diacritici, split)
        input_text = current_text
        tokens = clean_tokens(
            input_text,
            case_folding="upper",
            strip_diacritics=True,
            normalize=True
        )
        final_text = " ".join(tokens).strip()
        
        entry["steps"].append({
            "function": "clean_tokens(text: str, case_folding: str = 'upper', strip_diacritics: bool = True, normalize: bool = True) -> list[str]",
            "input": input_text,
            "output": final_text,
            "changed": input_text != final_text
        })
        
        entry["final_text"] = final_text
        log_entries.append(entry)

    # Scrittura del file di log in formato Markdown
    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Log delle Operazioni: Pipeline di Creazione del Dataset\n\n")
        f.write("Questo documento traccia in modo trasparente e verificabile l'effetto di ogni funzione di preprocessing (la pipeline MAAT) sul testo grezzo, operazione per operazione.\n\n")
        
        for entry in log_entries:
            f.write(f"## Sample {entry['sample_id']}\n")
            f.write(f"**Testo Originale**: `{entry['original_text']}`\n\n")
            
            for step in entry["steps"]:
                f.write(f"### `{step['function']}`\n")
                f.write(f"- **Input**: `{step['input']}`\n")
                f.write(f"- **Output**: `{step['output']}`\n")
                if step['changed']:
                    f.write("- *Nota: Il testo è stato modificato in questa fase.*\n")
                f.write("\n")
            
            f.write(f"**Risultato Finale (Tokenizzato & Normalizzato)**: `{entry['final_text']}`\n")
            f.write("---\n\n")
            
    print(f"Log generato con successo: {output_file}")

if __name__ == "__main__":
    out_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scratch')))
    out_dir.mkdir(exist_ok=True)
    log_file_path = out_dir / "pipeline_operations_log.md"
    generate_log(log_file_path)
