[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/CoPhi/gs-suggestions-dataset)

# gs-suggestions-dataset

[![GreekSchools Logo][gs-logo]][gs]

This project aims to provide an automatic suggestion system for supplements to fill the gaps in the Herculaneum Papyri and support the process of creating new critical editions.

## Prerequisites & System Requirements

To install and run the project locally, you will need the following tools installed on your system:

### 1. For Dockerized Setup (Recommended)
This is the easiest way to run the entire stack (API, Frontend, and MongoDB) seamlessly.
- **Docker** and **Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/)

### 2. For Local Development (Optional)
If you prefer to run services manually or develop locally without Docker:
- **Python**
- **uv**: Dependency and project python manager. [Install uv](https://docs.astral.sh/uv/)
- **Node.js** & **npm**: Required for the Angular frontend. [Install Node.js](https://nodejs.org/)
- **Angular CLI**: Install globally via `npm install -g @angular/cli`.

---

## Getting Started

Follow these steps to set up and run the project on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/CoPhi/gs-suggestions-dataset.git
cd gs-suggestions-dataset
```

### 2. Environment Variables Configuration
The project uses environment variables to configure services. A template file `.env.example` is provided in the repository.

To set up your environment, copy the `.env.example` file to a new file named `.env` and edit it:
   ```bash
   cp .env.example .env
   ```

*(Note: When running locally outside of Docker, ensure `MONGO_HOST=localhost`)*.

If you want to train new models, you will need to set the `WANDB_API_KEY` and `HF_TOKEN` variables in the `.env` file.

## 3. Data Integration Pipeline

To keep the repository lightweight, the large parsed textual datasets stored in the `data/` folder are excluded from Git tracking (via `.gitignore`). **All collaborators must independently rebuild the data environment locally after cloning the repository.**

### Datasets Included:
- [MAAT Corpus](https://zenodo.org/records/12553283)
- [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek)
- [PDL-canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)

### Running the Data Preparation
Before utilizing the models or the API meaningfully, you need to populate the data. Make sure backend dependencies are installed through `uv` first:

```bash
uv sync
```

**Step 1: Download and integrate corpora**
Run the automated pipeline to download, process, and inject the corpora into the `data/` folder:
```bash
make data
```

**Step 2: Parsing standard TEI XML files (Optional)**
If you have additional text archives using standard TEI format (without complex gaps in EpiDoc format), you can compile them using the standalone converter:
```bash
uv run python -m scripts.tei_pipeline <path_to_your_tei_folder>
```

*Note: Both commands will populate the `data/` directory in isolated file chunks (up to 50 MB) in a machine-actionable JSON format, ready for subsequent tasks.*


## 4. Running and Testing the Services

You can run and test the services in two ways: via Docker (recommended for a full, ready-to-use stack) or by launching the backend and frontend locally for active development.

### Option A: Running the Stack via Docker (Recommended)
This is the easiest way to test the entire integrated application (Backend API, Angular Frontend, and MongoDB) without manually installing development dependencies.

1. **Start the environment**:
   ```bash
   make run
   ```
   *(This starts all services in the background using `docker compose up`)*.

2. **Stop the environment**:
   ```bash
   make stop
   ```

3. **Restart the environment**:
   ```bash
   make restart
   ```

Once running, you can access the services at:
- **Frontend App**: [http://localhost:4200](http://localhost:4200)
- **Backend API**: [http://localhost:8000](http://localhost:8000) (Interactive Swagger Docs at [http://localhost:8000/docs](http://localhost:8000/docs))
- **MongoDB**: `localhost:27017`

---

### Option B: Running Services Locally (For Active Development)
If you are actively developing or testing changes in the backend or frontend code, it is faster to run the services locally.

1. **Run the Backend API**:
   ```bash
   make run-api
   ```
   This will start the Uvicorn server on [http://localhost:8000](http://localhost:8000) with auto-reload active.

2. **Run the Frontend App**:
   ```bash
   make run-frontend
   ```
   This will automatically verify and install any missing npm dependencies and start the Angular development server on [http://localhost:4200](http://localhost:4200) with Hot Module Replacement (HMR) active.

*(Note: Check `frontend/README.md` for specific Angular testing and advanced commands).*

---

## Changelog

To track the progress of the project, including new features, bug fixes, refactoring, and package updates, you can refer to the [CHANGELOG.md](CHANGELOG.md) file.

---

[gs]: https://greekschools.eu
[gs-logo]: https://greekschools.eu/wp-content/uploads/2021/01/logo-gs.png