"""
    Script per la valutazione dell'accuracy dei modelli BERT su MAAT, dividendo il dataset in: 
    - training set: frasi pulite in greco antico senza la presenza di token sconosciuti, usate per fare finetuning.
    - dev set: dataset usato per monitorare la loss durante il processo di finetuning, e per regolare gli iperparametri dei modelli BERT.
    - test set: dataset di test, frasi pulite in greco antico con la presenza di token mascherati, usate per valutare l'accuracy del modello.
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from config.settings import K_PRED
import torch
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

import torch
import itertools

from finetuning.utils import get_db, get_model, get_tokenizer

#La lunghezza della sequenza da generare è la lunghezza della gold label (esprime il numero di parola da predire)
#Calcolare l'accuracy del modello BERT sul test_set
#L'accuracy è calcolata come il numero di predizioni corrette diviso il numero totale di predizioni
#Una predizione è corretta se la parola predetta è uguale alla gold label
#Restituire l'accuracy

def evaluate_accuracy(k_pred=K_PRED) -> float:
    
    fill_masker = pipeline("fill-mask", model="Jacobo/aristoBERTo")
    test_db = load_dataset('GabrieleGiannessi/maat-corpus', split='test')
    tp = 0
    tn = 0

    # Iteriamo su ogni esempio nel dataset con una progress bar
    for example in tqdm(test_db, desc="Processing examples", leave=False):
        text = example['text']

        if len(text) > 512: #se il testo è troppo lungo per ora si gestisce ignorando il caso
          continue

        predictions = fill_masker(text, top_k=k_pred)
        found = any(pred['token_str'].upper() in example['gold_label'] for pred in predictions)

        if found:
            tp += 1
        else:
            tn += 1

    return tp / (tp + tn)

def get_top_k_mask_sequences(model, tokenizer, sentence: str, k_pred=K_PRED) -> list[tuple[list[str], float]]:
    """
    Prende una frase con token mascherati e restituisce le top-k sequenze più probabili per i token mascherati.
    Si assume che i [MASK] token formino una sequenza continua.
    
    Args: 
        model: modello BERT
        tokenizer: tokenizer BERT
        sentence: frase con token mascherati
        k_pred: numero di sequenze più probabili da restituire, default `K_PRED`

    Returns:
    list[tuple] : (lista_di_token_predetti, probabilità_logaritmica).
    """
    # Tokenizzazione della frase
    token_ids = tokenizer.encode(sentence, return_tensors='pt')

    # Trova le posizioni dei token mascherati
    masked_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    
    if len(masked_positions) == 0:
        return []  # Nessun token mascherato nella frase

    # Ottieni output dal modello
    with torch.no_grad():
        output = model(token_ids).logits  # Otteniamo i logits

    # Applica softmax per ottenere probabilità
    probs = torch.nn.functional.softmax(output[0, masked_positions], dim=1)

    # Trova i top-k token e le loro probabilità
    top_k_probs, top_k_indices = torch.topk(probs, k=k_pred, dim=1)  # (num_masks, k_pred)

    # Converti gli indici in token testuali
    top_k_tokens = [[tokenizer.decode(idx.item()).strip().replace('*', '') for idx in indices] for indices in top_k_indices]

    # Genera tutte le possibili combinazioni delle parole predette
    possible_sequences = list(itertools.product(*top_k_tokens))  # Lista di tuple di parole

    # Calcola la probabilità moltiplicando i valori delle probabilità
    sequence_probs = [torch.prod(torch.tensor(probs)) for probs in itertools.product(*top_k_probs.tolist())]

    # Ordina le sequenze in base alla probabilità
    sorted_sequences = sorted(zip(possible_sequences, sequence_probs), key=lambda x: x[1], reverse=True)

    # Formatta il risultato in liste di token senza asterischi
    best_sequences = [(list(seq), prob.item()) for seq, prob in sorted_sequences[:k_pred]]

    return best_sequences


def calculate_accuracy (model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, test_set: list, k_pred=K_PRED):
    pass

def main (): 
    pass
    
    
if __name__ == "__main__":  
    print (evaluate_accuracy())


    