VERSION := $(shell python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || .venv/bin/python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "0.6.0")
API_CONTAINER_NAME := gabrielegiannessi/gs-api
FRONTEND_CONTAINER_NAME := gabrielegiannessi/gs-frontend

# ---------------------------------------------------------
# Parametri per Training & HPO (personalizzabili da CLI)
# ---------------------------------------------------------
# Checkpoint target (es. CNR-ILC/gs-GreBerta, CNR-ILC/gs-aristoBERTo, CNR-ILC/gs-Logion)
CHECKPOINT ?= CNR-ILC/gs-GreBerta
ifdef MODEL
    CHECKPOINT := $(if $(findstring /,$(MODEL)),$(MODEL),CNR-ILC/$(MODEL))
endif

DATASET ?= CNR-ILC/gs-dataset-tlg-uncased
EPOCHS ?=
BATCH_SIZE ?=
LR ?=
NO_PUSH ?= false

TRAIN_ARGS := --checkpoint "$(CHECKPOINT)" --dataset_name "$(DATASET)"
ifneq ($(strip $(EPOCHS)),)
    TRAIN_ARGS += --epochs $(EPOCHS)
endif
ifneq ($(strip $(BATCH_SIZE)),)
    TRAIN_ARGS += --batch_size $(BATCH_SIZE)
endif
ifneq ($(strip $(LR)),)
    TRAIN_ARGS += --lr $(LR)
endif
ifeq ($(NO_PUSH),true)
    TRAIN_ARGS += --no_push_to_hub
endif

# Parametri HPO Optuna NSGA-II
TRIALS ?= 25
POPULATION ?= 8
OBJECTIVES ?= 2d
MAX_EVAL_CASES ?= 300
STORAGE ?= sqlite:///optuna_nsga_studies.db
STUDY_SLUG := $(shell echo "$(CHECKPOINT)" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
STUDY_NAME ?= nsga_$(STUDY_SLUG)_$(OBJECTIVES)
SELECTION ?= knee

# Parametri W&B Sweep
SWEEP_YAML ?= models/bert/finetuning/sweep_greBERTa.yaml
SWEEP_ID ?=

.PHONY: help data requirements requirements-api \
        run-api run-frontend \
        run stop restart \
        build-api build-frontend \
        tag-api tag-frontend push-api push-frontend \
        release-api release-frontend release \
        train train-test \
        hpo-nsga2 pareto pareto-train \
        sweep sweep-agent sweep-best

# ---------------------------------------------------------
# 0. Help / Riepilogo Comandi
# ---------------------------------------------------------

help:
	@echo "=========================================================================="
	@echo "                        GS-SUGGESTIONS-DATASET"
	@echo "=========================================================================="
	@echo "Addestramento Modelli BERT:"
	@echo "  make train                 Addestra il modello con configurazione di default o CLI"
	@echo "                             (es. make train MODEL=gs-GreBerta EPOCHS=3 LR=2e-5)"
	@echo "  make train-test            Smoke-test rapido (1 epoca, bs 32, no push)"
	@echo ""
	@echo "Ottimizzazione Multi-Obiettivo (NSGA-II con Optuna):"
	@echo "  make hpo-nsga2             Avvia la ricerca NSGA-II (Exact Match vs Cluster Inclusion)"
	@echo "                             (es. make hpo-nsga2 MODEL=gs-GreBerta TRIALS=25 POPULATION=8)"
	@echo "  make pareto                Ispeziona la frontiera di Pareto e mostra il Knee Point"
	@echo "                             (es. make pareto MODEL=gs-GreBerta SELECTION=knee)"
	@echo "  make pareto-train          Addestra ed esegue il push del modello selezionato da Pareto"
	@echo ""
	@echo "WandB Sweeps:"
	@echo "  make sweep                 Inizializza lo sweep WandB (SWEEP_YAML=...)"
	@echo "  make sweep-agent           Lancia l'agente WandB (SWEEP_ID=...)"
	@echo "  make sweep-best            Mostra la miglior run dello sweep WandB"
	@echo ""
	@echo "Dati e Servizi:"
	@echo "  make data                  Download ed elaborazione dei corpora"
	@echo "  make run                   Avvia stack Docker Compose (API + Frontend + Mongo)"
	@echo "  make run-api               Avvia server FastAPI locale con reload"
	@echo "  make run-frontend          Avvia server Angular locale con HMR"
	@echo "=========================================================================="

# ---------------------------------------------------------
# 1. Addestramento Modelli (Training)
# ---------------------------------------------------------

train:
	@echo "=== Avvio Finetuning per [$(CHECKPOINT)] ==="
	uv run python -m models.bert.finetuning.run $(TRAIN_ARGS)

train-test:
	@echo "=== Avvio Smoke-Test Finetuning per [$(CHECKPOINT)] (no push) ==="
	uv run python -m models.bert.finetuning.run \
		--checkpoint "$(CHECKPOINT)" \
		--dataset_name "$(DATASET)" \
		--epochs 1 \
		--batch_size 32 \
		--logging_steps 10 \
		--no_push_to_hub

# ---------------------------------------------------------
# 2. Ottimizzazione Multi-Obiettivo NSGA-II (Optuna)
# ---------------------------------------------------------

hpo-nsga2:
	@echo "=== Avvio Ottimizzazione NSGA-II per [$(CHECKPOINT)] ==="
	@echo "Trials: $(TRIALS) | Population: $(POPULATION) | Obiettivi: $(OBJECTIVES)"
	@echo "Studio: $(STUDY_NAME) | Storage: $(STORAGE)"
	uv run python -m scripts.sweep.optuna_nsga2 \
		--checkpoint "$(CHECKPOINT)" \
		--dataset_name "$(DATASET)" \
		--n_trials $(TRIALS) \
		--population_size $(POPULATION) \
		--objectives $(OBJECTIVES) \
		--max_eval_cases $(MAX_EVAL_CASES) \
		--storage "$(STORAGE)" \
		--study_name "$(STUDY_NAME)"

pareto:
	@echo "=== Analisi Frontiera di Pareto per [$(STUDY_NAME)] ==="
	uv run python -m scripts.sweep.optuna_pareto_analysis \
		--study_name "$(STUDY_NAME)" \
		--storage "$(STORAGE)" \
		--selection_strategy $(SELECTION)

pareto-train:
	@echo "=== Addestramento Finale del Modello Pareto [$(STUDY_NAME)] (Strategia: $(SELECTION)) ==="
	uv run python -m scripts.sweep.optuna_pareto_analysis \
		--study_name "$(STUDY_NAME)" \
		--storage "$(STORAGE)" \
		--selection_strategy $(SELECTION) \
		--train_selected

# ---------------------------------------------------------
# 3. W&B Sweeps (Alternativa Single-Objective)
# ---------------------------------------------------------

sweep:
	@echo "=== Inizializzazione WandB Sweep da [$(SWEEP_YAML)] ==="
	wandb sweep $(SWEEP_YAML)

sweep-agent:
	@if [ -z "$(SWEEP_ID)" ]; then \
		echo "Errore: specifica SWEEP_ID (es. make sweep-agent SWEEP_ID=username/project/xxxxxx)"; \
		exit 1; \
	fi
	uv run wandb agent $(SWEEP_ID)

sweep-best:
	@if [ -z "$(SWEEP_ID)" ]; then \
		echo "Errore: specifica SWEEP_ID (es. make sweep-best SWEEP_ID=username/project/xxxxxx)"; \
		exit 1; \
	fi
	uv run python -m scripts.sweep.get_best_run --sweep_id $(SWEEP_ID)

# ---------------------------------------------------------
# 4. Data Ingestion
# ---------------------------------------------------------

data:
	uv run python -m scripts.data.corpus_downloader
	uv run python -m scripts.data.split

requirements:
	uv export --format requirements-txt -o requirements.txt --no-hashes
	sed -i "s|file://$(PWD)/packages/|file:./packages/|g" requirements.txt

requirements-api:
	pipreqs . --force --ignore tests,migrations,docs
	mv requirements.txt requirements.txt.tmp
	grep -v "pkg-resources" requirements.txt.tmp > requirements.txt
	rm requirements.txt.tmp

# ---------------------------------------------------------
# 5. Run Services Locally (Development)
# ---------------------------------------------------------

run-api:
	uv run uvicorn backend.api.main:app --reload

frontend/node_modules: frontend/package.json
	cd frontend && npm install
	@touch frontend/node_modules

run-frontend: frontend/node_modules
	cd frontend && npm run start

# ---------------------------------------------------------
# 6. Multi-Container Environment (Docker Compose)
# ---------------------------------------------------------

run:
	docker compose up

stop: 
	docker compose down

restart: stop run

# ---------------------------------------------------------
# 7. Docker Build Images (Local/Multi-Arch)
# ---------------------------------------------------------

build-api: requirements
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--no-cache \
		-t $(API_CONTAINER_NAME):$(VERSION) \
		-t $(API_CONTAINER_NAME):latest \
		-f ./Dockerfile \
		--push \
		.

build-frontend:
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--no-cache \
		-t $(FRONTEND_CONTAINER_NAME):$(VERSION) \
		-t $(FRONTEND_CONTAINER_NAME):latest \
		-f ./frontend/Dockerfile \
		--push \
		./frontend

# ---------------------------------------------------------
# 8. Docker Image Deploy & Release
# ---------------------------------------------------------

tag-api:
	docker tag $(API_CONTAINER_NAME):latest $(API_CONTAINER_NAME):$(VERSION)

tag-frontend:
	docker tag $(FRONTEND_CONTAINER_NAME):latest $(FRONTEND_CONTAINER_NAME):$(VERSION)

push-api:
	@if ! docker image inspect $(API_CONTAINER_NAME):$(VERSION) > /dev/null 2>&1; then \
		echo "Image $(API_CONTAINER_NAME):$(VERSION) not found."; \
		exit 1; \
	fi
	docker push $(API_CONTAINER_NAME):$(VERSION)

push-frontend:
	@if ! docker image inspect $(FRONTEND_CONTAINER_NAME):$(VERSION) > /dev/null 2>&1; then \
		echo "Image $(FRONTEND_CONTAINER_NAME):$(VERSION) not found."; \
		exit 1; \
	fi
	docker push $(FRONTEND_CONTAINER_NAME):$(VERSION)

release-api: build-api
	@echo "API $(VERSION) pubblicata su Docker Hub"

release-frontend: build-frontend
	@echo "Frontend $(VERSION) pubblicato su Docker Hub"

release: release-api release-frontend
	@echo "Release $(VERSION) completata"