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

from finetuning.utils import get_db, get_model, get_tokenizer

def hcb_beam_search(model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, masked_text: str, k:int=K_PRED, beam_size:int=4*K_PRED) -> list[str]:
    """
    Implementa l'HCB Beam Search per prevedere sequenze di token mascherati con un MLM,
    restituendo solo le top-k predizioni per i token sostituiti.
    """
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
                new_score = score + prob.item() #Probabilità logaritmica
                new_beam.append((new_seq, new_score))
        
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
    
    result = []
    for seq, score in beam[:k]:
        decoded_sequence = tokenizer.decode([seq[0, idx].item() for idx in mask_indices], skip_special_tokens=True)
        normalized_score = score / len(mask_indices)  # Normalizza lo score
        result.append((decoded_sequence, normalized_score))
    
    return result

def main (): 
    pass
      
if __name__ == "__main__":  
    test = "[MASK][MASK][MASK][MASK] Λ ΙΓΡ ΕΝ ΠΕΡΙΟΔΕΎΟΥ ΣΙ ΤῊΝ ΤΩ͂Ν ΔΟΞΩ͂ ' Ν ΓΈΝΕΣΙΝ· ἘΠΕῚ ΓᾺΡ ΑἸΕῚ ΤᾺ ΜῈΝ ἜΝΓΕΙΟΝ ΠΡΟΠΕΊ ΠΤΟΝΤΑ ΤΡΑΝΌΤΕΡΑ ΒΛΈ ΠΕΤΑΙ , ΤᾺ ΔῈ ΠΟΡΡΏΤΕΡΑ ΠΆΝΤΩΣ ΟΥ̓Κ Ἀ ΚΟΛΟΥΘΟΥ͂ΣΙ ΤΟΙ͂Σ ΖΩΙ ΓΡΆΦΟΙΣ ἘΝ ΠΑΡΑΤΗΡΉ ΣΕΣΙ ΠΙ ΔΙΌΤΙ"
    model_checkpoint="CNR-ILC/gs-aristoBERTo"
    model = get_model(model_checkpoint)
    tokenizer = get_tokenizer(model_checkpoint)
    print (hcb_beam_search(model=model, tokenizer=tokenizer, masked_text=test))
    