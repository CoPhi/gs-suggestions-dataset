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
- **Poetry**: Dependency management for the Python backend. [Install Poetry](https://python-poetry.org/docs/#installation)
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
The project uses environment variables to configure services. Create a `.env` file in the root directory (if not already present) and ensure it has the following variables (example values):
```env
MONGO_USERNAME=admin
MONGO_PASSWORD=greekSchools123
MONGO_HOST=mongodb
```
*(Note: When running locally outside of Docker, you must set `MONGO_HOST=localhost`)*.

## 3. Data Integration Pipeline

To keep the repository lightweight, the large parsed textual datasets stored in the `data/` folder are excluded from Git tracking (via `.gitignore`). **All collaborators must independently rebuild the data environment locally after cloning the repository.**

### Datasets Included:
- [MAAT Corpus](https://zenodo.org/records/12553283)
- [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek)
- [PDL-canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)

### Running the Data Preparation
Before utilizing the models or the API meaningfully, you need to populate the data. Make sure backend dependencies are installed through Poetry first:

```bash
poetry install
```

**Step 1: Download and integrate corpora**
Run the automated pipeline to download, process, and inject the corpora into the `data/` folder:
```bash
make data
```

**Step 2: Parsing standard TEI XML files (Optional)**
If you have additional text archives using standard TEI format (without complex gaps in EpiDoc format), you can compile them using the standalone converter:
```bash
poetry run python -m scripts.tei_pipeline <path_to_your_tei_folder>
```

*Note: Both commands will populate the `data/` directory in isolated file chunks (up to 50 MB) in a machine-actionable JSON format, ready for subsequent tasks.*


### 4. Running the Project with Docker
To build and start the entire application (Backend API, Angular Frontend, and MongoDB), simply use the provided `Makefile` command:
```bash
make run
```
*(This corresponds to running `docker compose up` under the hood).*

Once the containers are running, you can access the services at:
- **Frontend App**: [http://localhost:4200](http://localhost:4200)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **MongoDB**: `localhost:27017`

---

## Local Development Commands

The `Makefile` includes additional convenient commands for development:

- **Run the Backend API Locally**: `make run-api` (runs on `http://localhost:8000` via Uvicorn).
- **Export Python Requirements**: `make requirements`
- **Build Docker Images manually**: `make build-api` or `make build-frontend`
- **Frontend details**: Check `frontend/README.md` for specific Angular frontend commands like testing and building.

---

[gs]: https://greekschools.eu
[gs-logo]: https://greekschools.eu/wp-content/uploads/2021/01/logo-gs.png