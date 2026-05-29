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

**Passo 1: Scaricare e integrare i corpora**
Esegui la pipeline automatizzata per scaricare, elaborare e inserire i corpora nella cartella `data/`:
```bash
make data
```

**Passo 2: Analisi dei file XML TEI standard (Opzionale)**
Se disponi di archivi di testo aggiuntivi che utilizzano il formato TEI standard (senza lacune complesse in formato EpiDoc), puoi compilarli utilizzando il convertitore autonomo:
```bash
uv run python -m scripts.tei_pipeline <percorso_della_tua_cartella_tei>
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

## Changelog

Per monitorare lo stato di avanzamento del progetto, incluse nuove funzionalità, correzioni di bug, refactoring e aggiornamenti dei pacchetti, puoi fare riferimento al file [CHANGELOG.md](CHANGELOG.md).

---

[gs]: https://greekschools.eu
[gs-logo]: https://greekschools.eu/wp-content/uploads/2021/01/logo-gs.png