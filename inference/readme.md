## Generazione di parole:
Esegui il modulo `infer` per generare parole utilizzando un modello pre-addestrato.
È necessario specificare il contesto e il numero di parole da generare.
Esempio: 

    poetry run python -m models.infer --context="καὶ" --num_words=2

Argomenti:
    `context`: Contesto per la generazione di parole (richiesto in modalità "infer").
    `num_words`: Numero di parole da generare (richiesto in modalità "infer").



       