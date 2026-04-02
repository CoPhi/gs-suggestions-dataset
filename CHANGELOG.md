# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
-- creazione dei test di unità per la validazione della fase di preprocessing dei dati. (tests/backend/core/test_preprocess.py)


### Changed
- Ottimizzazione della logica principale in `cleaner.py` (models/ngrams/train/cleaner.py).
- Fix riproducibilità della divisione dei dataset in `cleaner.py` (models/ngrams/train/cleaner.py). 
- Ottimizzazione della logica di creazione della funzione obiettivo in `start.py` (models/bert/tuning/start.py): adesso i dataset sono pre-calcolati all'esterno della funzione obiettivo per migliorare le performance.
- Ottimizzazione `preprocess.py` (backend/core/preprocess.py): apportati cambiamenti per migliorare le performance conservando la semantica delle operazioni.
    - Esternalizzazione delle chiusure (trasformazioni) presenti dentro la funzione `process_editorial_marks` per evitare crescite dello stack non desiderate (adesso le funzioni sono state create una sola volta e riutilizzate ad ogni chiamata).
    - Rimossi inutili overhead introdotti da operazioni ridondanti sulle stringhe (es; Sostituzioni concatenate e ricorsioni non ottimali).
- Rimozione della parte relativa all'addestramento e valutazione del modello n-grammi dal README.
- Modifica alla struttura del progetto per migliorare la separazione delle responsabilità (SoC)
