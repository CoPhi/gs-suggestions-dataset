import os
import json
import sys
from pathlib import Path
from scripts.tei_converter import convert_tei_to_json
from backend.config.settings import LIM, INDENT


class CorpusWriter:
    """
    Classe responsabile della scrittura dei dati su file JSON,
    suddivisi per corpus e dimensionati secondo un limite massimo (LIM).
    """

    def __init__(self, corpus_id, max_size_mb=LIM):
        """
        Inizializza un CorpusWriter.

        Args:
            corpus_id (str): ID del corpus.
            max_size_mb (int): Dimensione massima di ogni file in MB.
        """
        self.corpus_id = corpus_id
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.file_index = 0
        self.current_file_size = 0
        self.arr_objs = []
        self.indent_level = INDENT if INDENT else None
        Path("data").mkdir(
            parents=True, exist_ok=True
        )  # Inizializza la cartella `data/` se non presente dentro la root del progetto

    def add(self, obj):
        """
        Aggiunge un oggetto al writer.
        Se il file è pieno, lo salva e ne apre uno nuovo.
        """
        line_str = json.dumps(obj, ensure_ascii=False, indent=self.indent_level)
        line_size = len(line_str.encode("utf-8"))

        if self.current_file_size + line_size > self.max_size_bytes and self.arr_objs:
            self.flush()

        self.arr_objs.append(obj)
        self.current_file_size += line_size

    def flush(self):
        """
        Scrive su file gli oggetti accumulati.
        Se il file è pieno, ne apre uno nuovo e continua ad aggiungere oggetti, incrementando l'indice del file.
        """
        if not self.arr_objs:
            return
        file_path = f"data/{self.corpus_id}_{self.file_index}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.arr_objs, f, ensure_ascii=False, indent=self.indent_level)
        self.file_index += 1
        self.arr_objs = []
        self.current_file_size = 0


def process_directory(root_dir):
    """
    Elabora directory contenente file TEI, creando file JSON per ogni corpus.

    Args:
        root_dir (str): Directory da elaborare.

    Returns:
        dict: Dizionario di CorpusWriter, uno per ogni corpus trovato.
    """
    writers = {}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                objs = convert_tei_to_json(file_path)
                for obj in objs:
                    c_id = obj["corpus_id"]
                    if c_id not in writers:
                        writers[c_id] = CorpusWriter(c_id)
                    writers[c_id].add(obj)

    # Scrivi i risultati finali raggruppati per corpus
    for writer in writers.values():
        writer.flush()


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
