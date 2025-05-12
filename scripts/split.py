import os
import json
from pathlib import Path
from config.settings import CORPUS_PATHS, LIM, INDENT

def write_output_to_multiple_files(output, max_size_mb=LIM):
    """
    Funzione usata per splittare il risultato della conversione dei documenti in più file se si supera una certa dimensione in MB (specificata da LIM)
    è possibile rappresentare il livello di indentazione degli oggetti nei file con INDENT
    """

    Path("data").mkdir(parents=True, exist_ok=True)  # si crea la dir `data` se non esiste

    file_index = 0
    current_file_size = 0

    arr_objs = []

    for line in output:
        line_size = len(line.encode("utf-8"))

        if current_file_size + line_size > max_size_mb * 1024 * 1024:

            with open(f"data/maat_{file_index}.json", "w") as current_file:
                json.dump(arr_objs, current_file, ensure_ascii=False, indent=INDENT)

            file_index += 1
            arr_objs = []
            current_file_size = 0  # Reset della dimensione del file

        arr_objs.append(json.loads(line))
        current_file_size += line_size

    # se è rimasto qualcosa nell'array lo scrivo in un ultimo file
    if arr_objs:
        with open(f"data/maat_{file_index}.json", "w") as current_file:
            json.dump(arr_objs, current_file, ensure_ascii=False, indent=INDENT)


def main():
    for path in CORPUS_PATHS:
        output = []
        with os.popen(f"poetry run python -m maat.scripts.convert {path}", "r") as pipe:
            for line in pipe:
                output.append(line.strip())
        write_output_to_multiple_files(output)

if __name__ == "__main__":
    main()
