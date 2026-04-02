# gs-suggestions-dataset

This project aim to provide an automatic suggestion system for supplements for fill the gaps in the Herculaneum Papyri and support the process of creating new critical editions.  
`NLTK` and `CLTK` are used for the ngrams models implementations.  

`Poetry` 2.3 is used for managing the projects dependencies. Here there is the [documentation](https://python-poetry.org/docs/). 

## Dataset

The dataset includes:
- [MAAT Corpus](https://zenodo.org/records/12553283)
- [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek)
- [PDL-canonical-greekLit](https://github.com/PerseusDL/canonical-greekLit)

## Data Integration Pipeline

To keep the repository lightweight, the large extracted files generated in the `data/` folder are excluded from Git tracking (via `.gitignore`).
All collaborators, once the repository has been cloned, must independently rebuild the data environment locally.

### Steps to download and integrate data:

1. **Install Dependencies and Download Original Corpora**:
Make sure you have `poetry install`, then run the automatic downloader that retrieves the necessary XML documents for the configured databases:
```bash
poetry run python -m scripts.corpus_downloader
```

2. **Parsing EpiDoc Files**:
Use this command to process classic files mapped to the EpiDoc schema. The script will automatically create chunks in the `data/` directory:
```bash
poetry run python -m scripts.split
```

3. **Parsing standard TEI XML files**:
If you have text archives that use standard TEI without complex gaps in EpiDoc format, use the dedicated converter by pointing it to the directory:
```bash
poetry run python -m scripts.tei_pipeline <path_to_your_tei_folder>
```

*Note: Both pipelines will populate the `data/` directory in sections separated by up to 50 MB in machine-actionable JSON format, ready for use in subsequent tasks.*