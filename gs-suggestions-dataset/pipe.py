import json
import os
import subprocess
from pathlib import Path
from cltk.nlp import GreekPipeline
from convert import main as convert

"""
    Per eseguire questo script entrare nella dir del package ed eseguire : poetry run python pipe.py

    "/home/gabriele/cltk_data/grc/corpora/idp.data/DDB_EpiDoc_XML/",
    "/home/gabriele/cltk_data/grc/corpora/First1KGreek", 
    "/home/gabriele/cltk_data/grc/corpora/canonical-greekL/data/",
    "/home/gabriele/cltk_data/grc/corpora/LSJLogeion/", 

"""

PATHS = [ 
    "/home/gabriele/cltk_data/grc/corpora/idp.data/DCLP/",
]

def paginate_objs(objs, page_size=50):
    """
    Divide una lista di oggetti in pagine di dimensione fissa.
    
    Args:
        objs (list): Lista di oggetti da paginare.
        page_size (int): Numero massimo di oggetti per pagina.
    
    Returns:
        list: Una lista di liste (pagine).
    """
    return [objs[i:i + page_size] for i in range(0, len(objs), page_size)]

def save_json(data, output_dir, base_filename):
    """
    Salva i dati in più file JSON.
    
    Args:
        data (list): Lista di liste (pagine) da salvare.
        output_dir (str): Directory di output.
        base_filename (str): Nome base dei file JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, page in enumerate(data, start=1):
        output_file = Path(output_dir) / f"{base_filename}_page_{i}.json"
        with open(output_file, "w") as json_file:
            json.dump(page, json_file, indent=4, ensure_ascii=False)
        print(f"Salvato: {output_file}")

def capture_stdout(path):
    """
        Lancia un sottoprocesso che cattura lo stdout di convert.py applicato ad un path (che si riferisce ad un corpus) 
    """
    result = subprocess.run(
        ["python", "convert.py", path],  
        capture_output=True,
        text=True, 
        check=True, 
    )

    if result.returncode != 0:
        print(f"Errore: {result.stderr}")
        return []
    
    arr_objs = []
    for line in result.stdout.splitlines():
        try:
            json_object = json.loads(line.strip())
            arr_objs.append(json_object)
        except json.JSONDecodeError:
            print(f"Errore di decodifica JSON per la linea: {line}")
   
    return arr_objs

    
    
def main():
    for path in PATHS:
        print(f"Processing path: {path}")
        try:
            objs = capture_stdout(path)
            if objs:
                pag = paginate_objs(objs)
                save_json(pag, 'results', 'doc')
            else:
                print("La conversione non ha prodotto nessun risultato")
        except Exception as e:
            print(f"Errore durante la conversione del percorso {path}: {e}")

if __name__ == "__main__":
    main()