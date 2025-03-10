"""
    Script per la valutazione dell'accuracy dei modelli BERT su MAAT, dividendo il dataset in: 
    - training set: frasi pulite in greco antico senza la presenza di token sconosciuti, usate per fare finetuning.
    - dev set: dataset usato per monitorare la loss durante il processo di finetuning, e per regolare gli iperparametri dei modelli BERT.
    - test set: dataset di test, frasi pulite in greco antico con la presenza di token mascherati, usate per valutare l'accuracy del modello.
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, DataCollatorForLanguageModeling
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
    total_pred = 0
    correct_pred = 0

    # Iteriamo su ogni esempio nel dataset con una progress bar
    for example in tqdm(test_db, desc="Processing examples", leave=False):
        text = example['text']

        if len(text) > 512: #se il testo è troppo lungo per ora si gestisce ignorando il caso
          continue

        predictions = fill_masker(text, top_k=k_pred)
        found = any(pred['token_str'].upper() in example['gold_label'] for pred in predictions)

        if found:
            correct_pred += 1

        total_pred += 1

    return correct_pred / total_pred


def hcb_beam_search(model_name: str, masked_sentences: list[str], k:int=K_PRED, beam_size:int=K_PRED):
    """
    Implementa l'HCB Beam Search per prevedere sequenze di token mascherati con un MLM,
    restituendo solo le top-k predizioni per i token sostituiti.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    results = []
    
    for masked_text in masked_sentences:
        inputs = tokenizer(masked_text, return_tensors="pt")
        mask_indices = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1].tolist()
        
        beam = [(inputs.input_ids.clone(), 0)]  # (sequence, score)
        
        for mask_idx in mask_indices:
            new_beam = []
            for seq, score in beam:
                with torch.no_grad():
                    outputs = model(input_ids=seq)
                logits = outputs.logits[0, mask_idx, :]
                top_k_ids = torch.topk(logits, k=k, dim=-1).indices.squeeze(0)
                top_k_probs = torch.topk(logits, k=k, dim=-1).values.squeeze(0)
                
                for token_id, prob in zip(top_k_ids, top_k_probs):
                    new_seq = seq.clone()
                    new_seq[0, mask_idx] = token_id
                    new_score = score + prob.item()
                    new_beam.append((new_seq, new_score))
            
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
        
        top_k_predictions = [tokenizer.decode([b[0][0, idx].item() for idx in mask_indices], skip_special_tokens=True).upper() for b in beam[:k]]
        results.append(top_k_predictions)
    
    return results


#Implementazione per GPU

def hcb_beam_search_batch(model_name, masked_sentences, ground_truths, k=5, beam_size=5, device="cuda"):
    """
    Implementa l'HCB Beam Search per prevedere sequenze di token mascherati con un MLM,
    utilizzando batch per l'elaborazione efficiente su GPU e calcolando la top-k accuracy.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    
    # Tokenizzazione in batch
    inputs = tokenizer(masked_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    
    correct_predictions = 0
    total_predictions = len(mask_indices[0])
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = []
    for i, (batch_idx, mask_idx) in enumerate(zip(mask_indices[0], mask_indices[1])):
        logits = outputs.logits[batch_idx, mask_idx, :]
        top_k_ids = torch.topk(logits, k=beam_size, dim=-1).indices.tolist()
        top_k_probs = torch.topk(logits, k=beam_size, dim=-1).values.tolist()
        
        beam = [(inputs.input_ids[batch_idx].clone(), 0)]  # (sequence, score)
        
        new_beam = []
        for seq, score in beam:
            for token_id, prob in zip(top_k_ids, top_k_probs):
                new_seq = seq.clone()
                new_seq[mask_idx] = token_id
                new_score = score + prob
                new_beam.append((new_seq, new_score))
        
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
        
        top_k_predictions = [tokenizer.decode([b[0][0, idx].item() for idx in mask_indices], skip_special_tokens=True).upper() for b in beam[:k]]
        results.append(top_k_predictions)
        
        if ground_truths[i] in top_k_predictions:
            correct_predictions += 1
    
    top_k_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return results, top_k_accuracy

def main (): 
    pass
    
    
if __name__ == "__main__":  
    test_sentences = ["[MASK] Λ ΙΓΡ ΕΝ ΠΕΡΙΟΔΕΎΟΥ ΣΙ ΤῊΝ ΤΩ͂Ν ΔΟΞΩ͂ ' Ν ΓΈΝΕΣΙΝ· ἘΠΕῚ ΓᾺΡ ΑἸΕῚ ΤᾺ ΜῈΝ ἜΝΓΕΙΟΝ ΠΡΟΠΕΊ ΠΤΟΝΤΑ ΤΡΑΝΌΤΕΡΑ ΒΛΈ ΠΕΤΑΙ , ΤᾺ ΔῈ ΠΟΡΡΏΤΕΡΑ ΠΆΝΤΩΣ ΟΥ̓Κ Ἀ ΚΟΛΟΥΘΟΥ͂ΣΙ ΤΟΙ͂Σ ΖΩΙ ΓΡΆΦΟΙΣ ἘΝ ΠΑΡΑΤΗΡΉ ΣΕΣΙ ΠΙ ΔΙΌΤΙ"]
    predictions = hcb_beam_search(model_name='Jacobo/aristoBERTo', masked_sentences=test_sentences, k=10, beam_size=10)
    print (len(predictions))
    for sent, pred in zip(test_sentences, predictions):
        print(f"Input: {sent}\nPredicted: {pred}\n")


    