[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/CoPhi/gs-suggestions-dataset)

# gs-suggestions-dataset

[![GreekSchools Logo][gs-logo]][gs]

Questo progetto mira a fornire un sistema di suggerimenti automatico per supplementi volte a colmare le lacune nei Papiri di Ercolano e a supportare il processo di creazione di nuove edizioni critiche.

## Prerequisiti e Requisiti di Sistema

Per installare ed eseguire il progetto localmente, avrai bisogno dei seguenti strumenti installati sul tuo sistema:

### 1. Docker
Questo è il modo più semplice per eseguire l'intero stack (API, Frontend e MongoDB) in modo fluido e integrato.
- **Docker** e **Docker Compose**: [Installa Docker](https://docs.docker.com/get-docker/)

### 2. Sviluppo Locale 
Se preferisci eseguire i servizi manualmente o sviluppare localmente senza Docker:
- **Python**
- **uv**: Gestore di pacchetti e progetti Python. [Installa uv](https://docs.astral.sh/uv/)
- **Node.js** e **npm**: Richiesti per il frontend Angular. [Installa Node.js](https://nodejs.org/)
- **Angular CLI**: Da installare globalmente tramite `npm install -g @angular/cli`.

---

## Per Iniziare

Segui questi passaggi per configurare ed eseguire il progetto sulla tua macchina.

### 1. Clonare il Repository
```bash
git clone https://github.com/CoPhi/gs-suggestions-dataset.git
cd gs-suggestions-dataset
```

### 2. Configurazione delle Variabili d'Ambiente
Il progetto utilizza le variabili d'ambiente per configurare i servizi. Nel repository è fornito un file modello chiamato `.env.example`.

Per configurare il tuo ambiente, copia il file `.env.example` in un nuovo file chiamato `.env` e modificalo:
   ```bash
   cp .env.example .env
   ```

*(Nota: Quando si esegue localmente al di fuori di Docker, assicurarsi che `MONGO_HOST=localhost`)*.

Se desideri addestrare nuovi modelli, dovrai impostare le variabili `WANDB_API_KEY` e `HF_TOKEN` nel file `.env`.

## 3. Pipeline di Integrazione dei Dati

Per mantenere il repository leggero, i grandi dataset testuali analizzati memorizzati nella cartella `data/` sono esclusi dal tracciamento Git (tramite `.gitignore`). **Tutti i collaboratori devono ricostruire autonomamente l'ambiente dei dati a livello locale dopo aver clonato il repository.**

### Dataset Inclusi:
- [MAAT Corpus](https://zenodo.org/records/12553283)
- [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek)
- [PDL-canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)

### Esecuzione della Preparazione dei Dati
Prima di utilizzare i modelli o le API in modo significativo, è necessario popolare i dati. Assicurati innanzitutto che le dipendenze del backend siano installate tramite `uv`:

```bash
uv sync
```

> [!IMPORTANTE]
> **Configurazione CLTK per corpora personalizzati**
> Prima di eseguire lo script di download dei dati (`make data`), è necessario configurare o creare il file `distributed_corpora.yaml` all'interno della cartella `cltk_data` del proprio utente (solitamente situata in `~/cltk_data/distributed_corpora.yaml`).
>
> Questo file è richiesto da CLTK per mappare e scaricare i repository remoti contenenti i corpora. Assicurati che il file contenga la seguente sintassi:
>
> ```yaml
> DDbDP-DCLP:
>   origin: https://github.com/papyri/idp.data
>   language: grc
>   type: corpora
> 
> EDH:
>   origin: https://github.com/epigraphic-database-heidelberg/data
>   language: grc
>   type: corpora
> 
> First1KGreek:
>   origin: https://github.com/OpenGreekAndLatin/First1KGreek
>   language: grc
>   type: corpora
> 
> PerseusDL:
>   origin: https://github.com/PerseusDL/canonical-greekLit
>   language: grc
>   type: corpora
> ```

**Passo 1: Scaricare e integrare i corpora**
Esegui la pipeline automatizzata per scaricare, elaborare e inserire i corpora nella cartella `data/`:
```bash
make data
```

**Passo 2: Analisi dei file XML TEI standard (Opzionale)**
Se disponi di archivi di testo aggiuntivi che utilizzano il formato TEI standard (senza lacune complesse in formato EpiDoc), puoi compilarli utilizzando il convertitore autonomo:
```bash
uv run python -m scripts.tei.pipeline <percorso_della_tua_cartella_tei>
```

*Nota: Entrambi i comandi popoleranno la directory `data/` in blocchi di file isolati (fino a 50 MB) in un formato JSON fruibile da codice, pronto per le attività successive.*


## 4. Esecuzione e Test dei Servizi

È possibile eseguire e testare i servizi in due modi: tramite Docker (consigliato per uno stack completo e pronto all'uso) o avviando il backend e il frontend in locale per lo sviluppo attivo.

### Opzione A: Eseguire lo Stack tramite Docker (Consigliata)
Questo è il modo più semplice per testare l'intera applicazione integrata (API Backend, Frontend Angular e MongoDB) senza installare manualmente le dipendenze di sviluppo.

1. **Avviare l'ambiente**:
   ```bash
   make run
   ```
   *(Questo avvia tutti i servizi in background tramite `docker compose up`)*.

2. **Arrestare l'ambiente**:
   ```bash
   make stop
   ```

3. **Riavviare l'ambiente**:
   ```bash
   make restart
   ```

Una volta avviato, puoi accedere ai servizi ai seguenti indirizzi:
- **Applicazione Frontend**: [http://localhost:4200](http://localhost:4200)
- **API Backend**: [http://localhost:8000](http://localhost:8000) (Documentazione interattiva Swagger su [http://localhost:8000/docs](http://localhost:8000/docs))
- **MongoDB**: `localhost:27017`

---

### Opzione B: Eseguire i Servizi in Locale (Per lo Sviluppo Attivo)
Se stai sviluppando attivamente o testando modifiche al codice del backend o del frontend, è più veloce eseguire i servizi localmente.

1. **Eseguire l'API Backend**:
   ```bash
   make run-api
   ```
   Questo avvierà il server Uvicorn su [http://localhost:8000](http://localhost:8000) con ricaricamento automatico (auto-reload) attivo.

2. **Eseguire l'Applicazione Frontend**:
   ```bash
   make run-frontend
   ```
   Questo verificherà e installerà automaticamente eventuali dipendenze npm mancanti e avvierà il server di sviluppo di Angular su [http://localhost:4200](http://localhost:4200) con Hot Module Replacement (HMR) attivo.

*(Nota: Consulta `frontend/README.md` per comandi avanzati e test specifici di Angular).*

---

## 5. Addestramento e Ottimizzazione dei Modelli BERT

Il framework supporta il pre-addestramento (MLM) e il fine-tuning di modelli linguistici BERT specializzati per il greco antico su frammenti e lacune papirologiche.

I modelli gestiti centralmente tramite `ModelRegistry` sono:
- `CNR-ILC/gs-GreBerta` (base: `bowphs/GreBerta`)
- `CNR-ILC/gs-aristoBERTo` (base: `Jacobo/aristoBERTo`)
- `CNR-ILC/gs-Logion` (base: `cabrooks/LOGION-50k_wordpiece`)

Per un riepilogo rapido di tutti i comandi disponibili è possibile digitare in qualsiasi momento:
```bash
make help
```

---

### 5.1 Addestramento Modelli (Fine-Tuning)

È possibile addestrare i modelli sia con le configurazioni ottimali predefinite (definite in `ModelRegistry`) sia sovrascrivendo i parametri da riga di comando.

#### Addestramento Standard o con Iperparametri Personalizzati:
```bash
# Esempio 1: Addestramento di GreBERTa con configurazione di default
make train MODEL=gs-GreBerta

# Esempio 2: Personalizzazione di epoche, learning rate e batch size
make train MODEL=gs-aristoBERTo EPOCHS=3 LR=1.5e-5 BATCH_SIZE=64

# Esempio 3: Addestramento senza caricamento su Hugging Face Hub
make train MODEL=gs-Logion NO_PUSH=true
```

#### Smoke-Test Rapido di Verifica:
Per verificare che l'ambiente, la GPU e i dataset funzionino correttamente senza attendere tutte le epoche e senza pubblicare su Hugging Face (esegue 1 epoca, batch 32, `--no_push_to_hub`):
```bash
make train-test MODEL=gs-GreBerta
```

---

### 5.2 Ottimizzazione Multi-Obiettivo con NSGA-II (Optuna)

Nel restauro di testi antichi, l'accuratezza lessicale rigida (**Top-1 Exact Match**) e la plausibilità semantica/coesione contestuale (**Cluster Inclusion Rate**) sono obiettivi concorrenti. Il progetto integra l'algoritmo genetico **NSGA-II** (*Non-dominated Sorting Genetic Algorithm II*, Deb et al., 2002) tramite **Optuna**, permettendo di trovare la **Frontiera di Pareto** senza imporre pesi arbitrari a priori.

#### Passo 1: Avviare la Ricerca NSGA-II
```bash
# Avvio standard (25 trial, popolazione di 8 individui, obiettivi 2D: Top-1 EM vs Cluster Inc.)
make hpo-nsga2 MODEL=gs-GreBerta

# Con parametri personalizzati
make hpo-nsga2 MODEL=gs-GreBerta TRIALS=30 POPULATION=10 OBJECTIVES=2d
```

> [!NOTE]
> Tutti i trial vengono salvati in un database SQLite locale (`optuna_nsga_studies.db`). L'esecuzione può essere interrotta e ripresa in qualsiasi momento. Al termine, viene generato automaticamente un grafico interattivo della frontiera in formato HTML (`pareto_front_*.html`).

#### Passo 2: Analisi della Frontiera di Pareto e Identificazione del Knee Point
Ispeziona la frontiera di Pareto estratta da NSGA-II e identifica automaticamente il punto di massimo compromesso matematico (**Knee Point**, minima distanza euclidea dall'Utopia Point $[1, 1]$ nello spazio normalizzato):
```bash
make pareto MODEL=gs-GreBerta SELECTION=knee
```

Strategie di selezione supportate (`SELECTION`):
- `knee`: Miglior compromesso tra accuratezza lessicale e plausibilità semantica (consigliato).
- `max_em`: Massima precisione lessicale (modello conservativo su congetture storiche).
- `max_cluster`: Massima coesione semantica ed inclusione delle gold label nel cluster predittivo.

#### Passo 3: Addestramento Finale del Modello Pareto-Ottimale
Una volta individuata la configurazione desiderata, puoi addestrare il modello finale a regime e caricarlo su Hugging Face Hub con un unico comando:
```bash
make pareto-train MODEL=gs-GreBerta SELECTION=knee
```

---

### 5.3 Confronto Diretto Pre-FT vs Post-FT e Pubblicazione Model Card

Per confrontare le prestazioni del modello base (**Pre-FT**, es. `bowphs/GreBerta`) rispetto alla versione fine-tunata (**Post-FT**, es. `CNR-ILC/gs-GreBerta`) su un test set specifico (senza dover riavviare un addestramento) e generare o pubblicare la **Model Card arricchita** su Hugging Face Hub:

```bash
# Esempio 1: Confronto sul dataset di testing TLG per tutte le policy di lacuna (default, word, suffix)
make compare MODEL=gs-GreBerta EVAL_DATASET=CNR-ILC/gs-dataset-tlg-uncased POLICY=all

# Esempio 2: Confronto con generazione della Model Card in locale (README.md)
make compare MODEL=gs-GreBerta EVAL_DATASET=CNR-ILC/gs-dataset-tlg-uncased POLICY=all UPDATE_CARD=true OUTPUT_CARD=README_gs-GreBerta.md

# Esempio 3: Confronto e pubblicazione diretta della Model Card su Hugging Face Hub
make compare MODEL=gs-GreBerta EVAL_DATASET=CNR-ILC/gs-dataset-tlg-uncased POLICY=all PUSH_CARD=true

# Esempio 4: Pubblicazione / aggiornamento della Model Card da un file JSON precedentemente salvato
make model-card MODEL=gs-GreBerta FROM_JSON=comparison_greberta.json PUSH_CARD=true
```

Il confronto produce sia a terminale sia nella Model Card una tabella dettagliata che evidenzia:
- **Exact Match**: Top-1, Top-5, Top-10, Top-20
- **BERTScore F1**: @1, @5, @10, @20
- **Cosine Similarity (Max e Mean)**: @1, @5, @10, @20
- **Cluster Inclusion (Plausibilità Semantica)**: In-Cluster Rate, Margin, Gold Centroid CosSim
- **Delta ($\Delta$)**: variazione netta tra modello pre-addestrato e modello post-finetuning con indicatori di progresso (`+X.XX% 🟢`).

> [!TIP]
> Durante il fine-tuning standard (`make train` o `pipeline_finetuning`), se `push_to_hub=True`, la pipeline genera e carica automaticamente su Hugging Face Hub la Model Card arricchita comprensiva dei metadati completi, iperparametri e della tabella comparativa Pre vs Post-FT con i delta su TLG!

---

### 5.4 Ottimizzazione con Weights & Biases Sweeps (Single-Objective)

In alternativa a NSGA-II, è possibile utilizzare l'ottimizzatore bayesiano su metrica composita tramite W&B Sweeps:

```bash
# 1. Inizializzare lo sweep (restituisce SWEEP_ID)
make sweep SWEEP_YAML=models/bert/finetuning/sweep_greBERTa.yaml

# 2. Avviare l'agente per eseguire le run
make sweep-agent SWEEP_ID=<tuo_sweep_id>

# 3. Visualizzare a terminale la miglior configurazione trovata
make sweep-best SWEEP_ID=<tuo_sweep_id>
```

---

### 5.5 Esecuzione su Macchina Remota (Best Practice)

Quando si eseguono addestramenti o sweep su server remoti via SSH:

1. **Variabili d'ambiente**: assicurarsi che il file `.env` sulla macchina remota contenga:
   ```bash
   HF_TOKEN=tuo_token_huggingface
   WANDB_API_KEY=tuo_token_wandb
   ```
2. **Sessione persistente con `tmux`** (per evitare interruzioni in caso di disconnessione SSH):
   ```bash
   # Avviare una sessione persistente
   tmux new -s training

   # Selezionare la GPU ed eseguire
   CUDA_VISIBLE_DEVICES=0 make train MODEL=gs-GreBerta

   # Per staccarsi dalla sessione: premere Ctrl+B, poi D
   # Per riconnettersi:
   tmux attach -t training
   ```
3. **Selezione specifica della GPU**: anteporre `CUDA_VISIBLE_DEVICES=<id_gpu>` (es. `CUDA_VISIBLE_DEVICES=0 make hpo-nsga2 ...`).

---

## 6. Fine-Tuning e Valutazione di Ithaca (JAX/Flax)

Il framework supporta l'adattamento (fine-tuning) del modello **Ithaca** (Google DeepMind, Assael et al., *Nature* 2022) per il restauro del testo greco sui papiri ercolanesi e sulla letteratura greca antica tramite il corpus **TLG (Thesaurus Linguae Graecae)**.

A differenza dell'addestramento originale su iscrizioni epigrafiche (corpus PHI), il modello viene adattato al registro linguistico letterario e filosofico congelando le teste di attribuzione geografica e temporale per concentrare la capacità del modello sul restauro testuale a livello di carattere (*character-level masked language modeling*).

Per visualizzare tutti i comandi disponibili digitare:
```bash
make help
# oppure per la guida rapida a Ithaca:
make ithaca
```

### Pipeline Completa:

1. **Configurazione dell'ambiente e checkpoint base**:
   Scarica il checkpoint pre-addestrato di DeepMind (`checkpoint_v1.pkl`), clona il motore Ithaca e configura l'ambiente JAX/Flax:
   ```bash
   make ithaca-setup
   ```

2. **Preparazione del dataset TLG**:
   Normalizza il corpus TLG in maiuscolo epigrafico senza diacritici e genera lacune sintetiche stratificate (`default`, `word`, `suffix`):
   ```bash
   make ithaca-data
   # Oppure specificando un dataset personalizzato:
   make ithaca-data DATASET=CNR-ILC/gs-dataset-tlg-uncased
   ```

3. **Esecuzione del Fine-Tuning**:
   Allena il Transformer BigBird e la testa di restauro MLM congelando le teste di datazione e localizzazione:
   ```bash
   make ithaca-train
   # Con parametri personalizzati:
   make ithaca-train ITHACA_EPOCHS=3 ITHACA_BATCH_SIZE=8 ITHACA_LR=2e-5
   ```

4. **Valutazione comparativa (Pre-FT vs Post-FT)**:
   Confronta le prestazioni di infilling del checkpoint base rispetto a quello fine-tunato calcolando Top-1/5/20 Accuracy, MRR e Character Error Rate (CER), salvando il report JSON, la tabella Markdown e il CSV:
   ```bash
   make ithaca-compare
   # Con personalizzazione dei file di destinazione (JSON, CSV, Markdown):
   make ithaca-compare ITHACA_OUTPUT_JSON=eval/results/eval_results_ithaca_comparison.json ITHACA_OUTPUT_CSV=eval/results/eval_results_ithaca_comparison.csv ITHACA_OUTPUT_MD=eval/results/eval_results_ithaca_comparison.md
   ```

5. **Pubblicazione su Hugging Face Hub**:
   Carica il checkpoint fine-tunato, la configurazione e la Model Card arricchita con la tabella dei benchmark sul repository di Hugging Face Hub:
   ```bash
   make ithaca-publish
   # Oppure specificando il repository target:
   make ithaca-publish ITHACA_REPO_ID=CNR-ILC/gs-ithaca-tlg
   ```

---

## Changelog

Per monitorare lo stato di avanzamento del progetto, incluse nuove funzionalità, correzioni di bug, refactoring e aggiornamenti dei pacchetti, puoi fare riferimento al file [CHANGELOG.md](CHANGELOG.md).

---

[gs]: https://greekschools.eu
[gs-logo]: https://greekschools.eu/wp-content/uploads/2021/01/logo-gs.png