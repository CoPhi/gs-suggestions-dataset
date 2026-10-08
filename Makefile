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

# Parametri Valutazione & Confronto Pre-FT vs Post-FT
EVAL_DATASET ?= CNR-ILC/gs-dataset-eval
SPLIT ?= test
BASE_MODEL ?=
OUTPUT_JSON ?=
OUTPUT_CSV ?=
POLICY ?= default
UPDATE_CARD ?= false
PUSH_CARD ?= false
OUTPUT_CARD ?=
FROM_JSON ?=

# Parametri Gestione & Pubblicazione Dataset
DATASET_TARGET ?= herc
MIN_GAP ?= 1
MAX_GAP ?= 6
DATASET_TEST_SIZE ?= 0.2
DATASET_REPO ?=
PUSH_DATASET ?= true

DATASET_ARGS := --min-gap $(MIN_GAP) --max-gap $(MAX_GAP) --test-size $(DATASET_TEST_SIZE)
ifneq ($(strip $(DATASET_REPO)),)
    DATASET_ARGS += --repo-id "$(DATASET_REPO)"
endif

.PHONY: help data requirements requirements-api \
        run-api run-frontend \
        run stop restart \
        build-api build-frontend \
        tag-api tag-frontend push-api push-frontend \
        release-api release-frontend release \
        train train-test \
        hpo-nsga2 pareto pareto-train \
        compare model-card \
        sweep sweep-agent sweep-best \
        dataset-herc dataset-eval dataset-train dataset-tlg dataset-all dataset-dry-run \
        ithaca ithaca-setup ithaca-data ithaca-train ithaca-compare ithaca-publish

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
	@echo "Pipeline Ithaca (Fine-Tuning & Valutazione JAX/Flax):"
	@echo "  make ithaca                Mostra la panoramica dei comandi della pipeline Ithaca"
	@echo "  make ithaca-setup          Configura l'ambiente JAX/Flax e scarica il checkpoint base Ithaca"
	@echo "  make ithaca-data           Prepara il dataset TLG in formato epigrafico per Ithaca"
	@echo "                             (Opzioni: DATASET=CNR-ILC/gs-dataset-tlg-uncased)"
	@echo "  make ithaca-train          Avvia il fine-tuning di Ithaca (freezing teste attribuzione)"
	@echo "                             (es. make ithaca-train ITHACA_EPOCHS=3 ITHACA_BATCH_SIZE=8 ITHACA_LR=2e-5)"
	@echo "                             (Opzioni: ITHACA_BASE_CKPT=... ITHACA_FT_CKPT=...)"
	@echo "  make ithaca-compare        Confronta Ithaca Pre-FT vs Post-FT su test set stratificato"
	@echo "                             (policy: default, word, suffix con metriche Top-1/5/20 e CER)"
	@echo "                             (Opzioni: ITHACA_OUTPUT_JSON=... ITHACA_OUTPUT_CSV=... ITHACA_OUTPUT_MD=...)"
	@echo "  make ithaca-publish        Pubblica checkpoint, configurazione e Model Card su Hugging Face Hub"
	@echo "                             (es. make ithaca-publish ITHACA_REPO_ID=CNR-ILC/gs-ithaca-tlg)"
	@echo ""
	@echo "Ottimizzazione Multi-Obiettivo (NSGA-II con Optuna):"
	@echo "  make hpo-nsga2             Avvia la ricerca NSGA-II (Exact Match vs Cluster Inclusion)"
	@echo "                             (es. make hpo-nsga2 MODEL=gs-GreBerta TRIALS=25 POPULATION=8)"
	@echo "  make pareto                Ispeziona la frontiera di Pareto e mostra il Knee Point"
	@echo "                             (es. make pareto MODEL=gs-GreBerta SELECTION=knee)"
	@echo "  make pareto-train          Addestra ed esegue il push del modello selezionato da Pareto"
	@echo ""
	@echo "Confronto, Valutazione (Pre-FT vs Post-FT) e Model Card:"
	@echo "  make compare               Confronta baseline Pre-FT e modello Post-FT su un test set"
	@echo "                             (es. make compare MODEL=gs-GreBerta EVAL_DATASET=CNR-ILC/gs-dataset-tlg-uncased POLICY=all)"
	@echo "                             Opzioni: UPDATE_CARD=true, PUSH_CARD=true, OUTPUT_CARD=README.md"
	@echo "  make model-card            Genera / pubblica su Hugging Face Hub la Model Card con metriche"
	@echo "                             (es. make model-card MODEL=gs-GreBerta FROM_JSON=results.json PUSH_CARD=true)"
	@echo ""
	@echo "Dataset Hugging Face (Composizione & Pubblicazione):"
	@echo "  make dataset-herc          Genera e pubblica il benchmark Ercolano (CNR-ILC/gs-dataset-herc)"
	@echo "  make dataset-eval          Genera e pubblica il dataset di valutazione generale (CNR-ILC/gs-dataset-eval)"
	@echo "  make dataset-train         Genera e pubblica il corpus di addestramento MAAT (CNR-ILC/gs-dataset-train)"
	@echo "  make dataset-tlg           Genera e pubblica i dataset TLG (CNR-ILC/gs-dataset-tlg-uncased/cased)"
	@echo "  make dataset-all           Genera e pubblica tutti i dataset su Hugging Face Hub"
	@echo "  make dataset-dry-run       Verifica in locale senza caricare su HF (DATASET_TARGET=herc|eval|train|tlg)"
	@echo "                             (Opzioni: PUSH_DATASET=false, MIN_GAP=1, MAX_GAP=6, DATASET_REPO=...)"
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
# 3. Confronto e Valutazione (Pre-FT vs Post-FT)
# ---------------------------------------------------------

compare:
	@echo "=== Confronto Pre-FT vs Post-FT per [$(CHECKPOINT)] sul test set [$(EVAL_DATASET)] (Split: $(SPLIT)) ==="
	uv run python -m scripts.evaluate_comparison \
		--checkpoint "$(CHECKPOINT)" \
		$(if $(strip $(BASE_MODEL)),--base_model "$(BASE_MODEL)",) \
		--eval_dataset_name "$(EVAL_DATASET)" \
		--split "$(SPLIT)" \
		--policy "$(POLICY)" \
		--max_cases $(MAX_EVAL_CASES) \
		$(if $(strip $(OUTPUT_JSON)),--output_json "$(OUTPUT_JSON)",) \
		$(if $(strip $(OUTPUT_CSV)),--output_csv "$(OUTPUT_CSV)",) \
		$(if $(filter true,$(UPDATE_CARD)),--update_model_card,) \
		$(if $(filter true,$(PUSH_CARD)),--push_model_card,) \
		$(if $(strip $(OUTPUT_CARD)),--output_model_card "$(OUTPUT_CARD)",)

model-card:
	@echo "=== Generazione / Pubblicazione Model Card per [$(CHECKPOINT)] ==="
	uv run python -m scripts.publish_model_card \
		--checkpoint "$(CHECKPOINT)" \
		$(if $(strip $(BASE_MODEL)),--base_model "$(BASE_MODEL)",) \
		$(if $(strip $(FROM_JSON)),--from_json "$(FROM_JSON)",) \
		$(if $(strip $(DATASET)),--dataset_name "$(DATASET)",) \
		$(if $(strip $(OUTPUT_CARD)),--output "$(OUTPUT_CARD)",) \
		$(if $(filter true,$(PUSH_CARD)),--push_to_hub,)

# ---------------------------------------------------------
# 4. Gestione e Pubblicazione Dataset (Hugging Face Hub)
# ---------------------------------------------------------

dataset-herc:
	@echo "=== Generazione e Pubblicazione Dataset Ercolano [CNR-ILC/gs-dataset-herc] ==="
	uv run python -m models.bert.dataset.load --target herc $(if $(filter false,$(PUSH_DATASET)),,--push) $(DATASET_ARGS)

dataset-eval:
	@echo "=== Generazione e Pubblicazione Dataset Valutazione Generale [CNR-ILC/gs-dataset-eval] ==="
	uv run python -m models.bert.dataset.load --target eval $(if $(filter false,$(PUSH_DATASET)),,--push) $(DATASET_ARGS)

dataset-train:
	@echo "=== Generazione e Pubblicazione Corpus Addestramento MAAT [CNR-ILC/gs-dataset-train] ==="
	uv run python -m models.bert.dataset.load --target train $(if $(filter false,$(PUSH_DATASET)),,--push) $(DATASET_ARGS)

dataset-tlg:
	@echo "=== Generazione e Pubblicazione Corpus TLG [CNR-ILC/gs-dataset-tlg-*] ==="
	uv run python -m models.bert.dataset.load --target tlg $(if $(filter false,$(PUSH_DATASET)),,--push) $(DATASET_ARGS)

dataset-all:
	@echo "=== Generazione e Pubblicazione di TUTTI i Dataset su Hugging Face Hub ==="
	uv run python -m models.bert.dataset.load --target all $(if $(filter false,$(PUSH_DATASET)),,--push) $(DATASET_ARGS)

dataset-dry-run:
	@echo "=== Verifica in Locale (Dry Run) Dataset [$(DATASET_TARGET)] (no push) ==="
	uv run python -m models.bert.dataset.load --target $(DATASET_TARGET) $(DATASET_ARGS)

# ---------------------------------------------------------
# 5. Data Ingestion
# ---------------------------------------------------------

data:
	uv run python -m scripts.data.corpus_downloader
	uv run python -m scripts.data.split

# ---------------------------------------------------------
# 6. W&B Sweeps (Alternativa Single-Objective)
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

requirements:
	uv export --format requirements-txt -o requirements.txt --no-hashes
	sed -i "s|file://$(PWD)/packages/|file:./packages/|g" requirements.txt

requirements-api:
	pipreqs . --force --ignore tests,migrations,docs
	mv requirements.txt requirements.txt.tmp
	grep -v "pkg-resources" requirements.txt.tmp > requirements.txt
	rm requirements.txt.tmp

# ---------------------------------------------------------
# 7. Run Services Locally (Development)
# ---------------------------------------------------------

run-api:
	uv run uvicorn backend.api.main:app --reload

frontend/node_modules: frontend/package.json
	cd frontend && npm install
	@touch frontend/node_modules

run-frontend: frontend/node_modules
	cd frontend && npm run start

# ---------------------------------------------------------
# 8. Multi-Container Environment (Docker Compose)
# ---------------------------------------------------------

run:
	docker compose up

stop: 
	docker compose down

restart: stop run

# ---------------------------------------------------------
# 9. Docker Build Images (Local/Multi-Arch)
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
# 10. Docker Image Deploy & Release
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

# ---------------------------------------------------------
# 11. Ithaca Fine-Tuning & Evaluation Pipeline
# ---------------------------------------------------------
ITHACA_BASE_CKPT ?= checkpoints/ithaca/checkpoint_v1.pkl
ITHACA_FT_CKPT ?= checkpoints/ithaca/checkpoint_tlg.pkl
ITHACA_REPO_ID ?= CNR-ILC/gs-ithaca-tlg
ITHACA_EPOCHS ?= 3
ITHACA_BATCH_SIZE ?= 8
ITHACA_LR ?= 2e-5
ITHACA_STRATEGY ?= hcb_best_to_worst
ITHACA_BEAM_SIZE ?= 20
ITHACA_TEST_PATH ?= data/ithaca/tlg_test.jsonl
ITHACA_MAX_CASES ?= 300
ITHACA_OUTPUT_JSON ?= eval/results/eval_results_ithaca_comparison.json
ITHACA_OUTPUT_CSV ?=
ITHACA_OUTPUT_MD ?=

.PHONY: ithaca ithaca-setup ithaca-data ithaca-train ithaca-compare ithaca-publish

ithaca:
	@echo "=========================================================================="
	@echo "          PIPELINE ITHACA: FINE-TUNING & VALUTAZIONE (JAX/FLAX)"
	@echo "=========================================================================="
	@echo "Passaggi per eseguire la pipeline completa:"
	@echo ""
	@echo "1. Setup ambiente JAX e checkpoint base:"
	@echo "   make ithaca-setup"
	@echo ""
	@echo "2. Preparazione dataset TLG epigrafico [----]:"
	@echo "   make ithaca-data [DATASET=$(DATASET)]"
	@echo ""
	@echo "3. Avvio fine-tuning Ithaca su TLG (freezing teste attribuzione):"
	@echo "   make ithaca-train [ITHACA_EPOCHS=$(ITHACA_EPOCHS) ITHACA_BATCH_SIZE=$(ITHACA_BATCH_SIZE) ITHACA_LR=$(ITHACA_LR)]"
	@echo ""
	@echo "4. Valutazione e confronto Pre-FT vs Post-FT (HCB Beam Search):"
	@echo "   make ithaca-compare [ITHACA_STRATEGY=$(ITHACA_STRATEGY) ITHACA_BEAM_SIZE=$(ITHACA_BEAM_SIZE)]"
	@echo "                       [ITHACA_OUTPUT_JSON=...] [ITHACA_OUTPUT_CSV=...] [ITHACA_OUTPUT_MD=...]"
	@echo ""
	@echo "5. Pubblicazione su Hugging Face Hub (checkpoint, config, Model Card):"
	@echo "   make ithaca-publish [ITHACA_REPO_ID=$(ITHACA_REPO_ID)]"
	@echo "=========================================================================="

ithaca-setup:
	@echo "Configurazione ambiente e download checkpoint Ithaca..."
	bash models/ithaca/finetuning/setup_env.sh

ithaca-data:
	@echo "Preparazione dataset TLG per Ithaca..."
	uv run --no-sync python -m models.ithaca.dataset.prepare_tlg --dataset_name "$(DATASET)" --output_dir data/ithaca

ithaca-train:
	@echo "Avvio fine-tuning Ithaca su TLG con freezing teste di attribuzione..."
	uv run --no-sync python -m models.ithaca.finetuning.train \
		--train_path data/ithaca/tlg_train.jsonl \
		--val_path data/ithaca/tlg_val.jsonl \
		--checkpoint_path $(ITHACA_BASE_CKPT) \
		--output_dir checkpoints/ithaca \
		--epochs $(ITHACA_EPOCHS) \
		--batch_size $(ITHACA_BATCH_SIZE) \
		--lr $(ITHACA_LR)

ithaca-compare:
	@echo "Valutazione comparativa Ithaca Pre-FT vs Post-FT (HCB Beam Search, policy: default, word, suffix)..."
	uv run --no-sync python -m models.ithaca.evaluation.compare \
		--pre_checkpoint $(ITHACA_BASE_CKPT) \
		--post_checkpoint $(ITHACA_FT_CKPT) \
		--test_path $(ITHACA_TEST_PATH) \
		--max_cases $(ITHACA_MAX_CASES) \
		--strategy $(ITHACA_STRATEGY) \
		--beam_size $(ITHACA_BEAM_SIZE) \
		--output_json $(ITHACA_OUTPUT_JSON) \
		$(if $(strip $(ITHACA_OUTPUT_CSV)),--output_csv "$(ITHACA_OUTPUT_CSV)",) \
		$(if $(strip $(ITHACA_OUTPUT_MD)),--output_md "$(ITHACA_OUTPUT_MD)",)

ithaca-publish:
	@echo "Pubblicazione modello Ithaca fine-tunato su Hugging Face Hub ($(ITHACA_REPO_ID))..."
	uv run --no-sync python scripts/publish_ithaca_hub.py \
		--repo_id "$(ITHACA_REPO_ID)" \
		--checkpoint_path $(ITHACA_FT_CKPT) \
		--config_path checkpoints/ithaca/config.json \
		--eval_json $(ITHACA_OUTPUT_JSON)