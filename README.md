# gs-suggestions-dataset

[![GreekSchools Logo][gs-logo]][gs]

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

1. **Donwload and integrate corpora**:
Make sure you have `poetry install`, then run the automatic pipeline that downloads and integrates the corpora in the `data/` folder:
```bash
make data
```

2. **Parsing standard TEI XML files**:
If you have text archives that use standard TEI without complex gaps in EpiDoc format, use the dedicated converter by pointing it to the directory:
```bash
poetry run python -m scripts.tei_pipeline <path_to_your_tei_folder>
```

*Note: Both pipelines will populate the `data/` directory in sections separated by up to 50 MB in machine-actionable JSON format, ready for use in subsequent tasks.*

[gs]: https://greekschools.eu
[gs-logo]: https://greekschools.eu/wp-content/uploads/2021/01/logo-gs.png