# gs-suggestions-dataset

Questo progetto sviluppa un sistema di autosuggerimento per la ricostruzione di lacune nei testi in greco antico.  
Si utilizzano le librerie `NLTK` e `CLTK` per l'implementazione dei modelli probabilistici basati sugli n-grammi.  

Per la gestione delle dipendenze del progetto si utilizza la versione 2.1.2 di `Poetry`. Si consiglia di seguire la [documentazione ufficiale](https://python-poetry.org/docs/) per creare l'ambiente virtuale e installare le dipendenze. 

## Dataset

Il dataset si basa sul [MAAT Corpus](https://zenodo.org/records/12553283) ed è stato arricchito con i seguenti testi:
- [Primi 1000 anni del greco antico](https://github.com/OpenGreekAndLatin/First1KGreek)
- [Perseus Digital Library](https://github.com/PerseusDL/canonical-greekLit)

## Addestramento e creazione del modello

Per avviare l'addestramento, eseguire il comando:  
```bash
make training
```

Per personalizzare l'addestramento impostando i parametri di creazione personali modificare i parametri presenti nella cartella `config`. 

## Valutazione del modello 

Eseguire: 
```bash
make assessment
```
