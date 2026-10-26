# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [Unreleased]

### Added
- **Studio di Ablation della Finestra di Contesto e Resilienza Text-Infilling (`models/bert/evaluation/ablation.py`)**:
  - Implementata la funzione `truncate_context_by_words` in [`models/bert/evaluation/ablation.py`](./models/bert/evaluation/ablation.py) per il ritaglio simmetrico e asimmetrico basato su parole greche attorno alla lacuna (`[.....]`), preservando i prefissi/suffissi intra-parola e isolando la lacuna da $W$ parole a $0$ parole di contesto esterno.
  - Creato il benchmark CLI multi-modello `run_ablation_study` per valutare la resilienza delle architetture BERT Pre-FT (`Jacobo/aristoBERTo`, `bowphs/GreBerta`, `cabrooks/LOGION-50k_wordpiece`) e Post-FT (`CNR-ILC/gs-aristoBERTo`, `CNR-ILC/gs-GreBerta`, `CNR-ILC/gs-Logion`) su scale di finestre esponenziali ($W \in [2, 5, 10, 25, 50, 100, \text{Full}]$).
  - Implementata la generazione di grafici a linee comparativi `plot_context_ablation_curves` (Top-1 EM, Top-5 EM, BERTScore F1 vs Window size per modello), report Markdown tabellare con calcolo del delta drop da Full Context a $W=2$ e sincronizzazione opzionale su Weights & Biases (`wandb.Table` e `wandb.Image`).
  - Aggiunti il target `make eval-context-ablation` in [`Makefile`](./Makefile) e la suite di test unitari [`tests/test_context_ablation.py`](./tests/test_context_ablation.py).
- **Sistema di Text-Infilling basato su DSPy RAG e LLM Hugging Face (`models/rag/`)**:
  - Creato il package [`models/rag/`](./models/rag/) per il restauro testuale in greco antico tramite Retrieval-Augmented Generation (RAG) orchestrato con **DSPy (`dspy>=3.4.0`)** e modelli LLM da Hugging Face (es. `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`).
  - Implementato il supporto dual-mode agli LLM in [`models/rag/lm.py`](./models/rag/lm.py) (`build_dspy_lm`, `HuggingFaceLocalLM`, `TransformersLocalEngine` canonico `dspy.lm15`): esecuzione locale su GPU/CPU tramite `AutoModelForCausalLM` con cache LRU e quantizzazione opzionale 4-bit, Hugging Face Inference API via `HF_TOKEN` ed endpoint OpenAI-compatibili (vLLM, Ollama, TGI).
  - Implementato il retriever ibrido `GreekHybridRetriever` in [`models/rag/retriever.py`](./models/rag/retriever.py) sui corpora in `data/` (`tlg_*.json` e `maat_*.json`), combinando ricerca lessicale/morfologica BM25 su word token e character n-grammi (`GreekLexicalBM25Index`) con ricerca semantica densa tramite `dspy.Embedder` + `dspy.Embeddings` e i modelli BERT greci del progetto (`GreekBertEmbedder` con `CNR-ILC/gs-GreBerta`), fusione Reciprocal Rank Fusion (RRF), persistenza su disco e filtro anti-data-leakage.
  - Definite in [`models/rag/signatures.py`](./models/rag/signatures.py) le Signature dichiarative `GreekInfillingSignature` e `ConstraintRefinementSignature` con schema Pydantic `GreekCandidate` e gestione esplicita dei vincoli di lunghezza e della coesistenza di prefisso (`word_prefix`) e/o suffisso (`word_suffix`) intra-parola (`format_gap_constraints`).
  - Creato il modulo componibile `GreekRAGInfiller(dspy.Module)` in [`models/rag/modules.py`](./models/rag/modules.py) e la pipeline di inferenza `fill_mask_rag` in [`models/rag/inference.py`](./models/rag/inference.py) con sanitizzazione filologica, rimozione automatica degli affissi ripetuti dall'LLM e integrazione delle attestazioni dirette dai *loci similes* recuperati.
  - Implementata la compilazione automatica del prompt (`BootstrapFewShot` / `MIPROv2`) in [`models/rag/optimizer.py`](./models/rag/optimizer.py) tramite la metrica multi-criterio `greek_infilling_dspy_metric` (Top-1/5 EM, $1 - \text{CER}$ e validità della lunghezza) e il modulo di valutazione stratificata in [`models/rag/evaluation.py`](./models/rag/evaluation.py).
  - Integrata la famiglia di modelli `RAG` (`RAGModel`, `ModelType.RAG`) nei servizi FastAPI ([`backend/api/models.py`](./backend/api/models.py), [`backend/api/services/model_service.py`](./backend/api/services/model_service.py), [`backend/api/services/suggestions_service.py`](./backend/api/services/suggestions_service.py), [`backend/api/routes/models.py`](./backend/api/routes/models.py)) e nelle impostazioni [`backend/config/settings.py`](./backend/config/settings.py).
  - Aggiunti i target `make rag`, `make rag-index`, `make rag-compile`, `make rag-eval` e `make rag-predict` in [`Makefile`](./Makefile), la documentazione in [`README.md`](./README.md) e la suite di test [`tests/test_dspy_rag.py`](./tests/test_dspy_rag.py).
- **Skill Riusabile per Gestione Commit (`conventional-commits`)**:
  - Creata la skill di workspace [`.agents/skills/conventional-commits/SKILL.md`](./.agents/skills/conventional-commits/SKILL.md) e lo script di ispezione [`.agents/skills/conventional-commits/scripts/commit_inspect.sh`](./.agents/skills/conventional-commits/scripts/commit_inspect.sh) per standardizzare la creazione di commit atomici secondo Conventional Commits e l'allineamento con `CHANGELOG.md`.
- **Pubblicazione su Weights & Biases e Confronto Multi-Modello BERT (`make compare`)**:
  - Esteso [`scripts/evaluate_comparison.py`](./scripts/evaluate_comparison.py) con i flag `--log_to_wandb`, `--wandb_group`, `--wandb_run_name`, `--from_json` e il supporto a `--checkpoint all` per valutare e confrontare tutti i modelli del `ModelRegistry` (`gs-GreBerta`, `gs-aristoBERTo`, `gs-Logion`) in un'unica esecuzione.
  - Implementate `log_bert_comparison_to_wandb` e `print_and_log_multi_model_summary` per pubblicare su W&B le metriche scalari (`test_pre_ft_*`, `test_post_ft_*`, `delta_*`), le tabelle delle predizioni qualitative (`predictions_table`), le tabelle Pre/Post-FT per singolo modello (`confronto_pre_post_ft_*`), la leaderboard ordinabile (`leaderboard_modelli_bert_*`) e la tabella comparativa affiancata (`confronto_modelli_bert_*`).
  - Aggiornato il target `make compare` in [`Makefile`](./Makefile) con il supporto a `MODEL=all`, `WANDB=true`, `WANDB_GROUP`, `WANDB_RUN_NAME` e `FROM_JSON`.
- **Hyperparameter Tuning Multi-Obiettivo (NSGA-II) e Tabelle W&B Pre/Post-FT per Ithaca**:
  - Esteso [`scripts/sweep/optuna_nsga2.py`](./scripts/sweep/optuna_nsga2.py) con `create_ithaca_objective` per ottimizzare congiuntamente gli iperparametri di training JAX/Optax (`lr`, `epochs`, `batch_size`, `num_layers_to_freeze` sui blocchi `encoderblock_0..5`, `weight_decay`, `warmup_ratio`, `lr_scheduler_type`, `dropout_rate`) e di decodifica HCB Beam Search (`strategy`, `temperature`).
  - Aggiunto il supporto alle modalità obiettivo `--objectives` `2d` (`Top-1 EM`, `Cluster Inclusion Rate`), `3d` (`+ CosSim Max @1`), `em_cer` (`Top-1 EM`, `Character Accuracy 1-CER`) ed `em_cer_mrr` (`+ MRR`).
  - Implementato il caching in memoria della valutazione baseline Pre-FT (`checkpoint_v1.pkl`) all'inizio dello studio Optuna e la funzione `log_ithaca_comparison_to_wandb` in [`models/ithaca/evaluation/compare.py`](./models/ithaca/evaluation/compare.py) per pubblicare su Weights & Biases (`wandb.Table`) le tabelle comparative `confronto_pre_post_ft_{policy}` (`overall`, `default`, `word`, `suffix`), le tabelle qualitative `predictions_table` (con la gold label evidenziata) e la `pareto_frontier_table`.
  - Aggiunta `get_ithaca_contextual_embeddings` in [`models/ithaca/inference/predict.py`](./models/ithaca/inference/predict.py) per estrarre dal torso BigBird (`output_return_emb=True`) gli embedding contestuali 1D dei candidati e della gold label per il calcolo di `cluster_inclusion_rate` e `cos_sim_topK_max`.
  - Esteso `create_optimizer_with_freezing` in [`models/ithaca/finetuning/checkpoint.py`](./models/ithaca/finetuning/checkpoint.py) per supportare il congelamento selettivo dei primi $N$ blocchi dell'encoder BigBird (`num_layers_to_freeze`) e lo scheduler lineare oltre al coseno.
  - Estratta `pipeline_ithaca_finetuning` in [`models/ithaca/finetuning/train.py`](./models/ithaca/finetuning/train.py), `publish_ithaca_to_hub` in [`scripts/publish_ithaca_hub.py`](./scripts/publish_ithaca_hub.py) ed esteso [`scripts/sweep/optuna_pareto_analysis.py`](./scripts/sweep/optuna_pareto_analysis.py) per eseguire automaticamente, al termine di `--train_selected`, anche il push su Hugging Face Hub (`CNR-ILC/gs-ithaca-tlg`) di `checkpoint_tlg.pkl`, `config.json` e della Model Card (`README.md`) con gli iperparametri NSGA-II e la tabella Pre/Post-FT.
  - Aggiunti i target `make ithaca-hpo-nsga2`, `make ithaca-pareto` e `make ithaca-pareto-train` in [`Makefile`](./Makefile) e i relativi test unitari in [`tests/test_ithaca_hcb.py`](./tests/test_ithaca_hcb.py).

### Changed
- **Configurabilità Parametri Beam Search in `evaluate_metrics_on_test_set`**:
  - Aggiunti i parametri opzionali `K: int = 20` e `beam_size: int = 50` alla firma di `evaluate_metrics_on_test_set` in [`models/bert/finetuning/pipeline.py`](./models/bert/finetuning/pipeline.py), permettendo di personalizzare la profondità di ricerca per esecuzioni rapide di ablation o sweep ad alta fedeltà.
- **Unificazione Split del Dataset `gs-dataset-herc`**:
  - Aggiornata `build_herc_eval_dataset` in [`models/bert/dataset/load.py`](./models/bert/dataset/load.py) per pubblicare `CNR-ILC/gs-dataset-herc` con un unico split `test` contenente tutti i casi estratti dai Papiri di Ercolano (sia i blocchi MAAT `P.Herc.` sia i casi controllati da `test_abs.json`), rimuovendo lo split `dev` non impiegato per l'ottimizzazione degli iperparametri.
  - Aggiornata la descrizione della Dataset Card `_HERC_DESCRIPTION` in [`models/bert/dataset/__init__.py`](./models/bert/dataset/__init__.py) e la documentazione in [`models/bert/dataset/README.md`](./models/bert/dataset/README.md).
  - Reso resiliente `prepare_data` in [`models/bert/finetuning/pipeline.py`](./models/bert/finetuning/pipeline.py) ai dataset di valutazione privi di split `dev`, generando automaticamente casi di validazione sintetici dal corpus di addestramento quando `dev` non è presente.
- **Esclusione Checkpoint N-gram da Git**:
  - Aggiunto `*.pkl.zlib` in [`.gitignore`](./.gitignore) per mantenere i checkpoint compressi (`models/ngrams/checkpoints/*.pkl.zlib`) esclusivamente in locale per le build Docker senza appesantire la cronologia del repository Git.

## [0.7.1] - 2026-10-19

### Added
- **Checkpoint Pre-addestrati N-gram e Script di Export Offline**:
  - Creato [`scripts/export_ngram_checkpoints.py`](./scripts/export_ngram_checkpoints.py) e target `make export-ngrams` in [`Makefile`](./Makefile) per addestrare offline i modelli N-gram (Lidstone, MLE) e serializzarli compressi con zlib.
  - Generati i checkpoint pre-addestrati di produzione in `models/ngrams/checkpoints/` (`ngram_lidstone_n3_gamma0.001.pkl.zlib` e `ngram_mle_n3.pkl.zlib`), inclusi automaticamente nell'immagine Docker di `gs-api`.
  - Introdotti `NGRAM_CHECKPOINTS_DIR` e l'helper `get_ngram_checkpoint_path` in [`backend/config/settings.py`](./backend/config/settings.py) per la risoluzione deterministica dei percorsi di checkpoint.
  - Aggiunta l'eccezione `DatasetNotFoundError` in [`backend/api/exceptions.py`](./backend/api/exceptions.py) con rispettivo handler HTTP 400 in [`backend/api/main.py`](./backend/api/main.py) per gestire richieste di training personalizzato quando il dataset `data/` non è disponibile nell'ambiente.
  - Creata la suite di test unitari [`tests/test_model_service_checkpoints.py`](./tests/test_model_service_checkpoints.py), integrata nel target `make test-api` di [`Makefile`](./Makefile).

### Changed
- Sincronizzazione della versione di progetto a `0.7.1` in [`pyproject.toml`](./pyproject.toml), [`uv.lock`](./uv.lock) e [`frontend/package.json`](./frontend/package.json).
- **Disaccoppiamento Dataset `data/` e Autonomia Immagine Docker `gs-api`**:
  - Rimosso `VOLUME ["/app/data"]` da [`Dockerfile`](./Dockerfile) e il volume bind mount `./data:/app/data` da [`docker-compose.yaml`](./docker-compose.yaml), rendendo il container `gs-api` 100% autonomo e leggero per l'inferenza senza dipendenze dall'host.
  - Aggiornato `ModelService._train_and_persist_ngram` in [`backend/api/services/model_service.py`](./backend/api/services/model_service.py) per verificare in prima battuta la presenza dei checkpoint serializzati su disco prima di tentare il training da zero.
  - Estesa la signature di `pipeline_train` in [`models/ngrams/train/training.py`](./models/ngrams/train/training.py) per supportare il passaggio facoltativo di blocchi pre-caricati (`train_abs`), ottimizzando la memoria durante export multipli.

### Fixed
- **Risoluzione Timeout `UV_HTTP_TIMEOUT` e Disaccoppiamento Dipendenze di Training da `gs-api`**:
  - Separati i pacchetti di addestramento, HPO, valutazione e visualizzazione (`maat`, `datasets`, `transformers[torch]`, `bert-score`, `wandb`, `optuna`, `dspy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, `tabulate`, `umap-learn`, `scipy`, `scikit-learn`, `lxml`, `chex`, `jax[cuda12]`) nel gruppo `[dependency-groups] train` (incluso in `dev`) in [`pyproject.toml`](./pyproject.toml) e [`uv.lock`](./uv.lock), lasciando in `[project.dependencies]` esclusivamente i pacchetti richiesti per l'esposizione dell'API e l'inferenza (N-grammi, BERT e Ithaca).
  - Resi lazy gli import di `wandb` in [`models/bert/finetuning/__init__.py`](./models/bert/finetuning/__init__.py), `jax`/`optax` in [`models/ithaca/finetuning/checkpoint.py`](./models/ithaca/finetuning/checkpoint.py) e `train_test_split` in [`backend/core/cleaner.py`](./backend/core/cleaner.py) per evitare il caricamento di moduli di training durante l'avvio e l'inferenza dell'API.

### Optimized
- **Avvio Istantaneo, Build Rapida e Riduzione Impronta `gs-api`**:
  - Eliminato il collo di bottiglia dell'inizializzazione al bootstrap dell'API (oltre 7 minuti di training ed elaborazione di 1.4 GB di JSON ad ogni setup iniziale di MongoDB), sostituito da un caricamento dei checkpoint da file zlib in pochi millisecondi.
  - Ottimizzato [`Dockerfile`](./Dockerfile) con binario ufficiale `ghcr.io/astral-sh/uv`, `UV_HTTP_TIMEOUT=600`, `uv sync --frozen --no-dev --no-cache` con esclusione delle librerie GPU NVIDIA/CUDA (`nvidia-cudnn-cu13`, `nvidia-*`, `cuda-*`, `triton`), installazione di PyTorch CPU-only (`https://download.pytorch.org/whl/cpu`) ed esecuzione diretta di `uvicorn` dal virtualenv.
  - Snellito il contesto di build in [`.dockerignore`](./.dockerignore) (escludendo `scripts`, artefatti di training/valutazione e dati pesanti in `packages/ithaca_engine/train`) e impostati `DOCKER_PLATFORMS ?= linux/amd64` e `DOCKER_NO_CACHE ?= false` come default in [`Makefile`](./Makefile).

## [0.7.0] - 2026-10-09

### Added
- **Pipeline di Pubblicazione e Rilascio Docker Hub (API & Frontend)**:
  - Implementate in [`Makefile`](./Makefile) le pipeline di automazione complete (`make publish`, `make publish-api`, `make publish-frontend`) per la pubblicazione delle immagini container multi-arch (`linux/amd64,linux/arm64`) su Docker Hub (`gabrielegiannessi/gs-api` e `gabrielegiannessi/gs-frontend`).
  - Integrata suite di Quality Assurance (QA) pre-release con blocco automatico in caso di fallimento: test suite API/backend (`make test-api`) e unit test con production build Angular (`make test-frontend`).
  - Aggiunti helper per Docker Buildx (`make docker-builder`, `make docker-binfmt`), login Docker Hub (`make docker-login`), diagnostica dell'ambiente di rilascio (`make docker-check`), sincronizzazione della versione di progetto (`make set-version`) e ispezione dei manifest remoti (`make verify-release`).
  - Ottimizzati i contesti di build Docker con l'introduzione di [`frontend/.dockerignore`](./frontend/.dockerignore) e l'esclusione di `.venv`, cache e artefatti pesanti in [`.dockerignore`](./.dockerignore).
- **Frontend / Chiusura e Abort dei Suggerimenti (Suggestion Box)**:
  - Introdotto un pulsante di chiusura nell'header della card dei suggerimenti ([`frontend/src/app/components/suggests-box/suggests-box.component.html`](./frontend/src/app/components/suggests-box/suggests-box.component.html), [`frontend/src/app/components/suggests-box/suggests-box.component.ts`](./frontend/src/app/components/suggests-box/suggests-box.component.ts), [`frontend/src/app/components/suggests-box/suggests-box.component.css`](./frontend/src/app/components/suggests-box/suggests-box.component.css)) con emissione dell'output `onClose`.
  - Implementata la chiusura su richiesta della finestra dei suggerimenti (`closeSuggestions`), l'interruzione della chiamata HTTP asincrona attiva e la possibilità di chiudere anche il box in caso di nessun suggerimento trovato ([`frontend/src/app/app.component.html`](./frontend/src/app/app.component.html), [`frontend/src/app/app.component.ts`](./frontend/src/app/app.component.ts)).
- **Suite di Test per Lacune Multiple**:
  - Test unitari dedicati (`tests/test_multiple_lacunae.py`) per verificare la gestione uniforme di contesti contenenti più lacune (`[...]`, `<UNK>`, `....`) attraverso N-grammi, MLM/BERT e Ithaca.
- **Supporto al modello Ithaca (Google DeepMind) & Restauro Character-Level**:
  - Integrazione completa del modello Ithaca (BigBird character-level) e dell'infrastruttura di training/inferenza basata su JAX, Flax Linen e Optax (`models/ithaca`).
  - Pipeline di preprocessing e normalizzazione del corpus TLG (*Thesaurus Linguae Graecae*) per l'adattamento all'alfabeto epigrafico e letterario greco (`models/ithaca/dataset/prepare_tlg.py`).
  - Fine-tuning su TLG con allineamento della sequenza a 768 caratteri e freezing selettivo dei rami di attribuzione geografica e temporale per focalizzare l'ottimizzazione sul restauro testuale.
  - Implementazione della decodifica condizionata **HCB (Hammersley-Clifford-Besag) Beam Search** con strategia non-sequenziale Best-to-Worst e sottrazione del pivot del token maschera (`models/ithaca/inference/predict.py`).
  - Integrazione di Ithaca nei servizi FastAPI (`backend/api`): registrazione e memorizzazione del modello in MongoDB (`POST /models`), esecuzione delle inferenze (`GET /predictions`), esempi interattivi OpenAPI/Swagger e documentazione dedicata (`backend/api/README.md`).
  - Suite di valutazione comparativa Pre-FT vs Post-FT per Ithaca (`models/ithaca/evaluation/compare.py`) con calcolo di Top-1/5/20 Accuracy, Character Error Rate (CER) e Mean Reciprocal Rank (MRR).
  - Script di pubblicazione del modello Ithaca e della relativa Model Card su Hugging Face Hub (`scripts/publish_ithaca_hub.py`).
  - Suite di test dedicata al decoding HCB di Ithaca (`tests/test_ithaca_hcb.py`).
- **Aggiornamento Frontend: Supporto Ithaca, Feedback Predizioni e Alert Rosso Eliminazione**:
  - Integrazione dell'interfaccia `IthacaModelInterface` e del tipo `modelType` in [`frontend/src/app/services/api.service.ts`](./frontend/src/app/services/api.service.ts).
  - Implementazione di card dedicate con iconografia e metadati per modelli Ithaca nel modale di selezione ([`frontend/src/app/components/model/model.component.html`](./frontend/src/app/components/model/model.component.html), [`model.component.ts`](./frontend/src/app/components/model/model.component.ts)).
  - Sistema di notifiche reattive con segnali Angular in [`frontend/src/app/services/notification.service.ts`](./frontend/src/app/services/notification.service.ts) e container alert in [`app.component.html`](./frontend/src/app/app.component.html).
  - Feedback visuale di successo in-app con conteggio delle predizioni al termine della generazione dei suggerimenti ([`app.component.ts`](./frontend/src/app/app.component.ts), [`suggests-box.component.html`](./frontend/src/app/components/suggests-box/suggests-box.component.html)).
  - Alert rosso (`alert-danger`) che conferma esplicitamente l'eliminazione di un modello dal database MongoDB con reset della selezione attiva ([`model.component.ts`](./frontend/src/app/components/model/model.component.ts)).
  - Estensione del badge del modello selezionato e della validazione della lacuna (supporto a `[...]` e `[---]`) in [`prompt-box.component.html`](./frontend/src/app/components/prompt-box/prompt-box.component.html) e [`prompt-box.component.ts`](./frontend/src/app/components/prompt-box/prompt-box.component.ts).
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
- **Aggiornamento Frontend: Menù a Tendina Modelli, Ingrandimento e Raffinamento Card**:
  - Implementazione del menù a tendina compatto per la selezione del modello collocato direttamente accanto al pulsante di invio inferenza in [`frontend/src/app/components/prompt-box/prompt-box.component.html`](./frontend/src/app/components/prompt-box/prompt-box.component.html) e [`prompt-box.component.ts`](./frontend/src/app/components/prompt-box/prompt-box.component.ts).
  - Aggiunta della funzionalità di espansione/ingrandimento ("Ingrandisci") sia tramite icona dedicata affiancata al menù a tendina sia tramite voce nel menù per esporre la model box completa con tutti i dettagli dei modelli ([`prompt-box.component.html`](./frontend/src/app/components/prompt-box/prompt-box.component.html), [`app.component.html`](./frontend/src/app/app.component.html)).
  - Suite di test unitari per il componente prompt box in [`frontend/src/app/components/prompt-box/prompt-box.component.spec.ts`](./frontend/src/app/components/prompt-box/prompt-box.component.spec.ts).

### Changed
- Sincronizzazione della versione di progetto a `0.7.0` in [`pyproject.toml`](./pyproject.toml), [`uv.lock`](./uv.lock) e [`frontend/package.json`](./frontend/package.json).
- **Uniformazione Inferenza su Contesti con Lacune Multiple**:
  - Aggiornata la validazione delle lacune in [`backend/core/preprocess.py`](./backend/core/preprocess.py) e [`backend/api/services/suggestions_service.py`](./backend/api/services/suggestions_service.py): `test_case_contains_lacuna` accetta ora contesti contenenti una o più lacune, designando la prima come target e prevenendo rifiuti indebiti con `InvalidContextError`.
  - Modelli MLM / BERT ([`models/bert/inference/predict.py`](./models/bert/inference/predict.py)): la sola lacuna target viene sostituita con `GAP_TOKEN` per l'espansione e decodifica delle maschere `[MASK]`, mentre le lacune secondarie (`[...]`, `....`, `<UNK>`) vengono normalizzate in `[UNK]` / `<unk>`, evitando la cancellazione accidentale dei puntini da `remove_punctuation`.
  - Modelli N-grammi ([`models/ngrams/metrics/utils.py`](./models/ngrams/metrics/utils.py)): aggiornato `get_context_from_test_case` per consentire testi con lacune multiple, estraendo il contesto a sinistra della lacuna target e normalizzando i gap secondari a `<UNK>`.
  - Modello Ithaca ([`models/ithaca/inference/predict.py`](./models/ithaca/inference/predict.py)): isolamento dei `mask_indices` unicamente sulla lacuna target e conversione preventiva di tutte le lacune secondarie nel carattere sconosciuto dell'alfabeto Ithaca (`^` / `alphabet.unk`), prevenendo artefatti sul delimitatore SOS `<` ed evitando che i punti vengano scambiati per punteggiatura letterale.
- **Frontend / Model Box & Menù a Tendina**:
  - Rimosse le icone associate alle tipologie di modello dal menù a tendina (`prompt-box`) e dalle card informative nella model box (`model.component`) ([`frontend/src/app/components/prompt-box/prompt-box.component.html`](./frontend/src/app/components/prompt-box/prompt-box.component.html), [`frontend/src/app/components/prompt-box/prompt-box.component.ts`](./frontend/src/app/components/prompt-box/prompt-box.component.ts), [`frontend/src/app/components/model/model.component.html`](./frontend/src/app/components/model/model.component.html)).
  - Rimosso il metadato sulla tecnica di "Infilling" (Token Masking) dalle schede dei modelli BERT in [`frontend/src/app/components/model/model.component.html`](./frontend/src/app/components/model/model.component.html).
- Refactoring e redesign grafico della model box ([`frontend/src/app/components/models-box/models-box.component.html`](./frontend/src/app/components/models-box/models-box.component.html), [`models-box.component.css`](./frontend/src/app/components/models-box/models-box.component.css)): adozione di un layout CSS Grid responsive (`models-grid`) per assicurare altezza e larghezza uniformi tra tutte le card dei modelli.
- Raffinamento delle card modello ([`frontend/src/app/components/model/model.component.html`](./frontend/src/app/components/model/model.component.html), [`model.component.css`](./frontend/src/app/components/model/model.component.css)): rimozione di wrapper ridondanti, allineamento orizzontale dei footer con bottoni fissati alla base, badge distintivi per architetture (Ithaca, BERT, N-grammi), visualizzazione monospazio e protetta dei checkpoint, e stato attivo evidenziato.
- Inclusione di `bootstrap.bundle.min.js` in [`frontend/angular.json`](./frontend/angular.json) per garantire il supporto a Popper per menù a tendina e modali Bootstrap.
- Refactoring del `Makefile` con comandi modulari e parametrici per training BERT, HPO Optuna, valutazione comparativa, pubblicazione di dataset/model card e gestione di Ithaca (`make ithaca`, `make ithaca-compare`).
- Rimozione del composite score scalare dal tracciamento degli esperimenti in favore del monitoraggio multi-obiettivo delle metriche native.
- Aggiornamento della documentazione `README.md` con le guide complete a HPO Optuna, benchmarking comparativo e fine-tuning di Ithaca.

### Fixed
- **Correzione Estrazione Prefisso e Suffisso Intra-Parola**:
  - Risolto un bug in `get_head_supplement` e `get_tail_supplement` in [`backend/core/preprocess.py`](./backend/core/preprocess.py), garantendo che lacune inter-parola separate da spazi restituiscano `None` ed evitando penalità improprie di edit distance nel calcolo del ranking degli n-grammi.
- **Frontend / Apertura Menù a Tendina e Selezione Modello**:
  - Ripristinata la corretta reattività al click del menù a tendina dei modelli: eliminato il blocco della propagazione degli eventi (`stopPropagation`) e integrata la gestione dello stato di apertura tramite Signal Angular (`isDropdownOpen`), con supporto alla chiusura su selezione, click esterno (`document:click`) ed Escape ([`frontend/src/app/components/prompt-box/prompt-box.component.html`](./frontend/src/app/components/prompt-box/prompt-box.component.html), [`frontend/src/app/components/prompt-box/prompt-box.component.ts`](./frontend/src/app/components/prompt-box/prompt-box.component.ts), [`frontend/src/app/components/prompt-box/prompt-box.component.css`](./frontend/src/app/components/prompt-box/prompt-box.component.css)).
  - Corretto il dismiss del modal di selezione modello (`data-bs-dismiss="modal"`) eliminando `stopPropagation` dal pulsante "Seleziona" in [`frontend/src/app/components/model/model.component.html`](./frontend/src/app/components/model/model.component.html).
- Risolto `ModuleNotFoundError: No module named 'ithaca'` configurando l'engine in `packages/ithaca_engine` e inserendo il percorso in `sys.path` in [`models/ithaca/finetuning/checkpoint.py`](./models/ithaca/finetuning/checkpoint.py) prima della deserializzazione pickle.
- Correzione del target `run-api` nel [`Makefile`](./Makefile) invocando Uvicorn tramite `python -m uvicorn` per garantire la portabilità dell'ambiente virtuale ed evitare errori di shebang del binario (`os error 2`).
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