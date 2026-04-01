import json
import pandas as pd
import re
from backend.core.preprocess import get_expanded_supplement
from collections import OrderedDict


def dump_json_abs_from_csv(file_path="test_abs.csv") -> list:
    """
    Converte gli esempi inediti presenti nel file .csv in un formato machine-actionable simile a MAAT.

    Args:
        file_path (str): Path to the input CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Raggruppa per ID
    blocks = []
    grouped = df.groupby("ID")

    for group_id, group in grouped:
        corpus = group["papiro herc"].iloc[0]
        training_text = group["testo con supplemento"].iloc[0]
        test_cases = [
            {"id": i, "test_case": row["testo"]}
            for i, (_, row) in enumerate(group.iterrows())
        ]
        context = group["contesto ampio"].iloc[0]
        blocks.append(
            {
                "corpus_id": corpus,
                "id": f"{corpus}/{group['ID'].iloc[0]}",
                "title": group["luogo"].iloc[0],
                "material": "papyrus",
                "language": "grc",
                "suppl_text": training_text,
                "context": context,
                "test_cases": test_cases,
            }
        )

    return blocks


def get_training_text_from_suppl_text_and_context():
    blocks = dump_json_abs_from_csv()
    for i, block in enumerate(blocks):
        suppl_text = block["suppl_text"]
        context = block["context"]
        if not isinstance(context, str):
            context = "" if pd.isna(context) else str(context)

        # Trova tutti i restauri tra quadre
        supplements = list(re.finditer(r"\[([^\[\]]+)\]", suppl_text))
        for suppl in supplements:
            start, end = suppl.span()
            expand = get_expanded_supplement(suppl_text, start, end)
            expand_suppl_no_brackets = expand.replace("[", "").replace("]", "")
            match_suppl = list(
                re.finditer(re.escape(expand_suppl_no_brackets), context)
            )
            if match_suppl:
                context = context.replace(match_suppl[0].group(0), expand)

        # Ricostruisci l'OrderedDict con il campo training_text prima di test_cases
        new_block = OrderedDict()
        new_block["corpus_id"] = block["corpus_id"]
        new_block["id"] = block["id"]
        new_block["title"] = block["title"]
        new_block["material"] = block["material"]
        new_block["language"] = block["language"]
        new_block["training_text"] = context
        new_block["test_cases"] = block["test_cases"]  # inizialmente vuoto
        blocks[i] = new_block

    return blocks


def dump_test_cases_into_json_abs(file_path="data/test_abs.json"):
    blocks = get_training_text_from_suppl_text_and_context()
    for block in blocks:
        training_text = block["training_text"]
        # Trova tutti i restauri tra quadre
        supplements = list(re.finditer(r"\[([^\[\]]+)\]", training_text))
        test_cases = []

        for idx, match in enumerate(supplements):
            supplement_content = match.group(1)  # contenuto interno alle quadre
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

            test_cases.append(
                {
                    "case_index": idx + 1,
                    "id": f'{block["id"]}/{idx + 1}',
                    "test_case": modified_text,
                }
            )

        block["test_cases"] = test_cases

    # Salva su file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    dump_test_cases_into_json_abs()
    print("Conversione completata con successo!")
