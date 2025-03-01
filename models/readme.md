# Modelli di Markov

Questi script permettono la generazione di modelli probabilistici basati su ngrammi con dimensione a nostra scelta,
oltre a fornire anche delle metriche con cui valutare i modelli generati (perplessità e accuracy).

## Addestramento del modello:
Esegui il modulo `training` per addestrare il modello sui dati presenti nella cartella specificata (data/).
Esempio: 

    python -m models.training

## Generazione di parole:
Esegui il modulo `infer` per generare parole utilizzando un modello pre-addestrato.
È necessario specificare il contesto e il numero di parole da generare.
Esempio: 

    python -m models.infer --context="καὶ" --num_words=2

Argomenti:
    `context`: Contesto per la generazione di parole (richiesto in modalità "infer").
    `num_words`: Numero di parole da generare (richiesto in modalità "infer").

## Valutazione di un modello: 
Esegui il modulo `evaluate` per valutare il modello secondo le metriche di perplessità e accuracy: 
Esempio: 

    python -m models.evaluate
       


       