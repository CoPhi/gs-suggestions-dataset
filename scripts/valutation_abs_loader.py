import json
import pandas as pd


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


if __name__ == "__main__":
    dump_json_abs_from_csv(file_path="test_abs.csv", output_path="test_abs.json")
    print("Conversione completata con successo!")
