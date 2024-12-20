 # Addestramento del modello:
        Esegui lo script con l'argomento "train" per addestrare il modello sui dati presenti nella cartella specificata (data/).
        Esempio: python trigram_lm.py train
 # Generazione di parole:
        Esegui lo script con l'argomento "infer" per generare parole utilizzando un modello pre-addestrato.
        È necessario specificare il contesto e il numero di parole da generare.
        Esempio: python trigram_lm.py infer --context "parole di esempio" --num_words 5
    Argomenti:
    - mode: Modalità di esecuzione dello script ("train" per addestrare, "infer" per generare parole).
    - context: Contesto per la generazione di parole (richiesto in modalità "infer").
    - num_words: Numero di parole da generare (richiesto in modalità "infer").