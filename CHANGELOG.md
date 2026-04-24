# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [Unreleased]

### Added


### Fixed

## [0.5.0] - 2026-04-24

### Added
- creazione di una nuova pipeline di conversione dedicata ai file TEI XML generici per renderli machine-actionable (`scripts/tei_converter.py`, `scripts/tei_pipeline.py`):
    - il convertitore estrae automaticamente i metadati (`corpus_id`, `title`, `language`, `material`) dall'intestazione TEI.
    - viene effettuato lo `strip()` del tag body dei file TEI XML per ripulire il rumore presente dalla formattazione XML;
    - Vengono rimossi i tag label dai file TEI XML; 
    - i gap di lunghezza nota vengono sostituiti con una sequenza di punti (`.`), quelli di lunghezza ignota vengono preservati come `<gap/>`.
    - sono generati i `test_cases`, uno per ogni integrazione presente nel training text (le integrazioni sono identificate con le parentesi quadrate). Ogni test case è un oggetto JSON che contiene: 
		- case_index (indice del caso di test);
		- id (concatenazione corpus_id,file_id e case_index);
		- test_case (testo con la lacuna al posto dell'integrazione);
    - Separazione del testo prelevato dai file TEI XML e presente nei campi `training_text` dei blocchi anonimi MAAT usando come criterio di separazione la punteggiatura ( `.`, `;`, `°` e `·`)
    - i file TLG (nome file con prefisso `tlg_<numero>`) vengono raggruppati sotto il corpus_id unificato `tlg`.
    - l'output rispetta il formato JSON machine-actionable standard MAAT.
- creazione della batteria di test per i nuovi moduli (`tests/test_tei_converter.py`, `tests/test_tei_pipeline.py`) con 13 test totali.
- aggiunta la documentazione relativa alla grammatica MAAT Leiden (`core/maat_leiden_grammar.md`) e alla specifica del transpiler (`core/transpiler_spec.md`).
- aggiunta la batteria di test per il transpiler: comprende test di idempotenza, test end-to-end e test delle post-condizioni/invarianti per ogni fase (`tests/backend/core/test_transpiler.py`).
- creazione dei test di unità per la validazione della fase di preprocessing dei dati. (`tests/backend/core/test_preprocess.py`)
- creazione package per implementazione della pipeline di creazione del dataset (`models/bert/dataset`)
    - consolidato il preprocessing con `transpile()` per preservare gap liberi da markup editoriale MAAT prima della conversione (`backend/core/preprocess.py`).
    - costruzione dev set: `models/bert/dataset/dev_set.py` (valido anche per costruire il test set)
    - costruzione train set: `models/bert/dataset/train_set.py`
- creazione della pipeline di text-infilling tramite HCB (`models/bert/inference/predict.py`) capace di gestire range adattivi di maschere consecutive attraverso HCB beam search.
    - integrazione di `HCBEvaluationCallback` in `models/bert/finetuning/run.py` per valutare le metriche reali `top-K` via HCB su un pool di lacune estratte durante le validazioni cross-epoch dell'addestramento.
    - creazione dei wrapper per evaluation HCB (`models/bert/evaluation/topk.py`) e script dedicato al test di ablazione sulla dimensione del contesto (`models/bert/evaluation/ablation.py`).
    - creazione di un DataCollator custom (`models/bert/finetuning/collator.py`) per il continual-pretraining in cui si mascherano porzioni di testo contingue (da 1 fino a 3 token mascherati massimo) dinamicamente. 
- Introduzione di un nuovo modulo `utils.py` nella cartella `backend/core` dedicato alla creazione di utility relative ai modelli.
    - implementazione di una pipeline di creazione di plot che mostrano statistiche descrittive (media, varianza) relative ai token presenti nei corpus usando i diversi tokenizer dei modelli (`backend/core/utils.py`)
### Changed
- Modifica alla struttura del progetto per migliorare la separazione delle responsabilità (SoC)
- Ottimizzazione della logica principale in `cleaner.py` (models/ngrams/train/cleaner.py).
- Fix riproducibilità della divisione dei dataset in `cleaner.py` (models/ngrams/train/cleaner.py). 
- Ottimizzazione della logica di creazione della funzione obiettivo in `start.py` (models/bert/tuning/start.py): adesso i dataset sono pre-calcolati all'esterno della funzione obiettivo per migliorare le performance.
- Ottimizzazione `preprocess.py` (backend/core/preprocess.py): apportati cambiamenti per migliorare le performance conservando la semantica delle operazioni.
    - Esternalizzazione delle chiusure (trasformazioni) presenti dentro la funzione `process_editorial_marks` per evitare crescite dello stack non desiderate (adesso le funzioni sono state create una sola volta e riutilizzate ad ogni chiamata).
    - Rimossi inutili overhead introdotti da operazioni ridondanti sulle stringhe (es; Sostituzioni concatenate e ricorsioni non ottimali).
- Aggiornamento `.gitignore`: la cartella `data/` è ora esclusa dal versioning ad eccezione dei file `tlg_*.json` (dati TLG distribuiti nel repository in formato machine-actionable).
- Aggiornamento `README.md`: rimozione della parte relativa all'addestramento e valutazione del modello n-grammi, aggiunta sezione di integrazione dei dati con le istruzioni per ricostruire l'ambiente dati in locale.
- Refactoring `models/bert/finetuning/run.py`: disaccoppiato il caricamento dei dati utilizzando due versioni dal repository HF:
    - `gs-dataset-train`: testo grezzo per continual pre-training MLM (https://huggingface.co/datasets/CNR-ILC/gs-dataset-train)
    - `gs-dataset-eval`: dataset che fornisce campioni di test case affiancati alle gold labels per la valutazione dei modelli (https://huggingface.co/datasets/CNR-ILC/gs-dataset-eval).

