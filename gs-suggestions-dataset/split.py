import os
import json
from pathlib import Path

PATHS = [ 
    "/home/gabriele/cltk_data/grc/corpora/idp.data/DCLP/",
    "/home/gabriele/cltk_data/grc/corpora/idp.data/DDB_EpiDoc_XML/",
    "/home/gabriele/cltk_data/grc/corpora/First1KGreek/", 
    "/home/gabriele/cltk_data/grc/corpora/canonical-greekL/",
    "/home/gabriele/cltk_data/grc/corpora/LSJLogeion/",
]

LIM=50  # MAX MB per file
INDENT=0 #indentazione 


def write_to_multiple_files(output, base_filename, max_size_mb=LIM):
    """
         Funzione usata per splittare il risultato della conversione dei documenti in più file se si supera una certa dimensione in MB (specificata da LIM)
         è possibile rappresentare il livello di indentazione degli oggetti nei file con INDENT
    """

    Path("data").mkdir(parents=True, exist_ok=True) #creo la dir 'data' se non esiste

    file_index = 0
    current_file_size = 0
    
    arr_objs = []

    for line in output:
        line_size = len(line.encode('utf-8'))  

        if current_file_size + line_size > max_size_mb * 1024 * 1024:

            with open(f"data/{base_filename}_{file_index}.json", 'w') as current_file:
                json.dump(arr_objs, current_file, ensure_ascii=False, indent=INDENT)

            file_index += 1
            arr_objs = []
            current_file_size = 0  # Reset della dimensione del file


        arr_objs.append(json.loads(line))
        current_file_size += line_size

    #se è rimasto qualcosa nell'array lo scrivo in un ultimo file
    if arr_objs:
        with open(f"data/{base_filename}_{file_index}.json", 'w') as current_file:
            json.dump(arr_objs, current_file, ensure_ascii=False, indent=INDENT)


for path in PATHS:
    output = []
    with os.popen(f'python convert.py {path}', 'r') as pipe:
        for line in pipe:
            output.append(line.strip())
 
    write_to_multiple_files(output, Path(path).name)
