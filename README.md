# gs-suggestions-dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12553283.svg)](https://doi.org/10.5281/zenodo.12553283)

Questo progetto sviluppa un sistema di autosuggerimento per la ricostruzione di lacune nei testi in greco antico.  
I dati provengono dal [MAAT Corpus](https://zenodo.org/records/12553283), e il modello utilizza le librerie `nltk` e `cltk` per l'elaborazione del linguaggio naturale.  

## Installazione

Questo progetto utilizza `Poetry`. 
Per installare le dipendenze, eseguire: 

```sh
$ poetry install
```

Per utilizzare la shell creata dall'ambiente virtuale: 

```sh
$ poetry shell
```

## Dataset

Il dataset è basato sul [MAAT Corpus](https://zenodo.org/records/12553283) ed è stato arricchito con i testi:
- [primi 1000 anni del greco antico](https://github.com/OpenGreekAndLatin/First1KGreek). 
- [Perseus Digital Library](https://github.com/PerseusDL/canonical-greekLit).