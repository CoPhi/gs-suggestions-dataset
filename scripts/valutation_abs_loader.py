import json
import pandas as pd
import re 

def dump_json_abs_from_csv(file_path="test_abs.csv", output_path="test_abs.json"):
    """
    Convert a CSV file to a JSON file.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to the output JSON file.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Raggruppa per ID
    blocks = []
    grouped = df.groupby("ID")

    for group_id, group in grouped:
        training_text = group["testo con supplemento"].iloc[0]
        test_cases = [{"id": i, "test_case": row["testo"]} for i, (_, row) in enumerate(group.iterrows())]
        context = group["contesto ampio"].iloc[0] 
        blocks.append(
            {
                "id": group["ID"].iloc[0],
                "title": group["luogo"].iloc[0], 
                "material": "papyrus",
                "language": "grc",
                "suppl_text": training_text,
                "training_text": context,
                "test_cases": test_cases,
            }
        )

    # Write the list of dictionaries to a JSON file
    with open(f"data/{output_path}", "w", encoding="utf-8") as json_file:
        json.dump(blocks, json_file, indent=4, ensure_ascii=False)

def dump_test_cases_into_json_abs(file_path="data/test_abs.json"):
    blocks = json.load(open(file_path, "r", encoding="utf-8"))
    
    for block in blocks:
        training_text = block["training_text"]
        # Trova tutti i restauri tra quadre
        supplements = list(re.finditer(r"\[([^\[\]]+)\]", training_text))
        test_cases = []

        for idx, match in enumerate(supplements):
            start, end = match.span()
            supplement_content = match.group(1)  # contenuto interno alle quadre
            replacement = "[" + ("." * len(supplement_content)) + "]"
            offset = 0
            modified_text = training_text

            for i, other in enumerate(supplements):
                o_start, o_end = other.span()
                o_content = other.group(1)

                if i == idx:
                    # Sostituisci questo supplemento con [...], mantenendo le quadre
                    o_replacement = "[" + ("." * len(o_content)) + "]"
                else:
                    # Rimuovi le quadre per tutti gli altri
                    o_replacement = o_content

                # Calcola la posizione aggiornata tenendo conto degli offset precedenti
                mod_start = o_start + offset
                mod_end = o_end + offset
                modified_text = (
                    modified_text[:mod_start] + o_replacement + modified_text[mod_end:]
                )
                offset += len(o_replacement) - (o_end - o_start)

            test_cases.append({
                "case_index": idx + 1,
                "id": f'{block["id"]}/{idx + 1}',
                "test_case": modified_text
            })

        block["test_cases"] = test_cases

    # Salva su file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    #dump_json_abs_from_csv(file_path="test_abs.csv", output_path="test_abs.json")
    dump_test_cases_into_json_abs()
    print("Conversione completata con successo!")
