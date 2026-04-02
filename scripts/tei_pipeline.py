import os
import json
import sys
from pathlib import Path
from scripts.tei_converter import convert_tei_to_json
from backend.config.settings import LIM, INDENT

def write_output_to_multiple_files(output_objs, corpus_id, max_size_mb=LIM):
    Path("data").mkdir(parents=True, exist_ok=True)

    file_index = 0
    current_file_size = 0
    arr_objs = []
    
    # Check if INDENT is non-zero
    indent_level = INDENT if INDENT else None

    for obj in output_objs:
        line_str = json.dumps(obj, ensure_ascii=False, indent=indent_level)
        line_size = len(line_str.encode("utf-8"))

        if current_file_size + line_size > max_size_mb * 1024 * 1024:
            with open(f"data/{corpus_id}_{file_index}.json", "w", encoding="utf-8") as current_file:
                json.dump(arr_objs, current_file, ensure_ascii=False, indent=indent_level)

            file_index += 1
            arr_objs = []
            current_file_size = 0

        arr_objs.append(obj)
        current_file_size += line_size

    if arr_objs:
        with open(f"data/{corpus_id}_{file_index}.json", "w", encoding="utf-8") as current_file:
            json.dump(arr_objs, current_file, ensure_ascii=False, indent=indent_level)

def process_directory(root_dir):
    all_objs_by_corpus = {}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                objs = convert_tei_to_json(file_path)
                for obj in objs:
                    c_id = obj['corpus_id']
                    if c_id not in all_objs_by_corpus:
                        all_objs_by_corpus[c_id] = []
                    all_objs_by_corpus[c_id].append(obj)

    # Scrivi i risultati raggruppati per corpus
    for c_id, objs in all_objs_by_corpus.items():
        write_output_to_multiple_files(objs, c_id)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.tei_pipeline <directory_path>")
        sys.exit(1)
        
    directory = sys.argv[1]
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        sys.exit(1)
        
    print(f"Processing TEI XML files in {directory}...")
    process_directory(directory)
    print("Done!")

if __name__ == "__main__":
    main()
