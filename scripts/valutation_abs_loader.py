import json
import pandas as pd
from utils import SUPPLEMENTS_REGEX

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
        supplements = list(SUPPLEMENTS_REGEX.finditer(training_text))
        modified_text = training_text
        offset = 0  # To handle shifting indices after replacements

        for idx, match in enumerate(supplements):
            start, end = match.span()
            content = match.group(0)
            if idx == 0:  # Sostituisci solo il primo supplemento trovato
                dots = "." * (end - start - 2)  # -2 per togliere le parentesi quadre
                replacement = f"[{dots}]"
                modified_text = (
                    modified_text[:start + offset]
                    + replacement
                    + modified_text[end + offset:]
                )
                offset += len(replacement) - len(content)
            else:
                # Rimuovi solo le parentesi quadre dagli altri supplementi
                inner_content = match.group(0)[1:-1]  # Rimuove la prima e l'ultima parentesi
                modified_text = (
                    modified_text[:start + offset]
                    + inner_content
                    + modified_text[end + offset:]
                )
                offset += len(inner_content) - (end - start)
            # Ora modified_text contiene il testo desiderato
            test_case = {"id": idx, "test_case": modified_text}
            block["test_cases"].append(test_case)
        
if __name__ == "__main__":
    #dump_json_abs_from_csv(file_path="test_abs.csv", output_path="test_abs.json")
    print("Conversione completata con successo!")
