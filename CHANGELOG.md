# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [0.7.0] - 2026-10-09

### Added
- **Supporto al modello Ithaca (Google DeepMind) & Restauro Character-Level**:
  - Integrazione completa del modello Ithaca (BigBird character-level) e dell'infrastruttura di training/inferenza basata su JAX, Flax Linen e Optax (`models/ithaca`).
  - Pipeline di preprocessing e normalizzazione del corpus TLG (*Thesaurus Linguae Graecae*) per l'adattamento all'alfabeto epigrafico e letterario greco (`models/ithaca/dataset/prepare_tlg.py`).
  - Fine-tuning su TLG con allineamento della sequenza a 768 caratteri e freezing selettivo dei rami di attribuzione geografica e temporale per focalizzare l'ottimizzazione sul restauro testuale.
  - Implementazione della decodifica condizionata **HCB (Hammersley-Clifford-Besag) Beam Search** con strategia non-sequenziale Best-to-Worst e sottrazione del pivot del token maschera (`models/ithaca/inference/predict.py`).
  - Integrazione di Ithaca nei servizi FastAPI (`backend/api`): registrazione e memorizzazione del modello in MongoDB (`POST /models`), esecuzione delle inferenze (`GET /predictions`), esempi interattivi OpenAPI/Swagger e documentazione dedicata (`backend/api/README.md`).
  - Suite di valutazione comparativa Pre-FT vs Post-FT per Ithaca (`models/ithaca/evaluation/compare.py`) con calcolo di Top-1/5/20 Accuracy, Character Error Rate (CER) e Mean Reciprocal Rank (MRR).
  - Script di pubblicazione del modello Ithaca e della relativa Model Card su Hugging Face Hub (`scripts/publish_ithaca_hub.py`).
  - Suite di test dedicata al decoding HCB di Ithaca (`tests/test_ithaca_hcb.py`).

- **Ottimizzazione Multi-Obiettivo degli Iperparametri (Optuna NSGA-II)**:
  - Framework di HPO multi-obiettivo con algoritmo genetico NSGA-II (`scripts/sweep/optuna_nsga2.py`) per bilanciare simultaneamente loss di validazione, accuratezza Top-K e coesione semantica.
  - Modulo di analisi del fronte di Pareto e selezione della soluzione di compromesso (*Knee point selection*) (`scripts/sweep/optuna_pareto_analysis.py`).
  - Integrazione degli studi Optuna con logging e sincronizzazione su Weights & Biases (W&B).
  - Nuovi target nel `Makefile`: `make opt` e `make opt-pareto`.

- **Benchmark Comparativo & Generazione Automatica di Model Card**:
  - Script di benchmark per il confronto sistematico Pre-FT vs Post-FT su dataset di test (`scripts/evaluate_comparison.py`), con esportazione dei risultati in formati JSON, CSV e tabelle Markdown.
  - Modulo per la generazione automatica di **Model Card** su Hugging Face Hub (`models/bert/finetuning/model_card.py`, `scripts/publish_model_card.py`) complete di metadati YAML per Digital Classics, iperparametri e snippet di inferenza Python.
  - Suite di test dedicata `tests/test_model_card.py`.

- **Stratificazione del Dev/Test Set su 2 Livelli & Metriche di Cluster Cohesion**:
  - Stratificazione del dataset di valutazione in base alla policy di lacuna: `default` (sottostringa casuale), `word` (parola intera) e `suffix` (desinenza/terminazione flessiva) (`models/bert/dataset/dev_set.py`).
  - Supporto per lacune di lunghezza arbitraria in fase di inferenza BERT (`models/bert/inference/predict.py`).
  - Introduzione delle metriche di coesione dei cluster (UMAP, silhouette score, cluster inclusion per le gold label) integrate nel callback di validazione (`HCBEvaluationCallback`) e nei test di regressione (`tests/test_umap_cohesion.py`).

- **Gestione Unificata e Pubblicazione Dataset Papyrologici**:
  - Integrazione e pubblicazione su Hugging Face Hub del dataset dei Papiri di Ercolano (`CNR-ILC/gs-dataset-pherc-uncased`) e del dataset di valutazione stratificato (`models/bert/dataset/load.py`, `models/bert/dataset/README.md`).
  - Nuovi comandi nel `Makefile` per la pubblicazione selettiva dei dataset (`make publish-dataset-pherc`, `make publish-dataset-eval`, `make publish-datasets`).

### Changed
- Refactoring del `Makefile` con comandi modulari e parametrici per training BERT, HPO Optuna, valutazione comparativa, pubblicazione di dataset/model card e gestione di Ithaca (`make ithaca`, `make ithaca-compare`).
- Rimozione del composite score scalare dal tracciamento degli esperimenti in favore del monitoraggio multi-obiettivo delle metriche native.
- Aggiornamento della documentazione `README.md` con le guide complete a HPO Optuna, benchmarking comparativo e fine-tuning di Ithaca.

### Fixed
- Allineamento delle dipendenze e configurazione JAX per librerie GPU NVIDIA con gestore di pacchetti `uv`.
- Correzione dell'estrazione dei logit e dei tuple di output nell'architettura Ithaca.
- Gestione della mappatura degli alfabeti e delle etichette dei parametri Optax per il caricamento dei pesi DeepMind originali.

## [0.6.0] - 2026-06-22

### Added

- Implementazione di un sistema di LRU cache per i modelli BERT, in modo da ridurne il tempo di caricamento. Ora, il sistema mantiene in memoria gli ultimi 3 modelli BERT caricati, consentendo un recupero quasi istantaneo in caso di riutilizzo.
- Ottimizzazione fase di decoding: implementazione di filtri più stringenti per l'accettazione dei suggerimenti del modello BERT. Vengono ora scartati i candidati che contengono caratteri latini, numeri o punteggiatura estesa, oltre ai falsi positivi dovuti a lacune rappresentate da puntini.
- Aumento della beam size per considerare più candidati ad ogni generazione. 
- Refactoring del frontend per migliorare la manutenibilità e l'usabilità.
    - Model selection tramite modal box.
    - Suggestion box ridefinita per migliorare la UX: i token suggeriti dal modello sono ora sempre visualizzati in corsivo.
- Hyperparameter tuning tramite wandb sweep sui principali iperparametri per il finetuning dei modello: learning rate, numero di layers da freezare, chunk size, batch size, mlm probability, epoche, etc.     


### Changed

- Rimossi prefissi di tokenizzazione (`##`) dalle predizioni effettuate da `models/bert/inference/predict.py`.

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
- Sostituzione della "request-box" nel frontend con una "prompt-box" standalone per favorire modularità ed interoperabilità. 
- Migrazione della gestione delle dipendenze da Poetry a uv  