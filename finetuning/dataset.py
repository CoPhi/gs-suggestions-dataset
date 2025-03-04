"""
    Script per rendere disponibile l'insieme delle frasi pre-processsate del MAAT corpus su hugging face
    
    access token: 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
    nome: hf-maat-upload
    
    Questo dataset serve per fare il finetuning dei modelli BERT per il greco antico sul task MLM 
    
    Il train_set è composto da frasi pulite in greco antico senza la presenza di token sconosciuti, in cui è stato applicato il case folding.
    Il test_set è composto da frasi pulite in greco antico con la presenza di token mascherati, in cui è stato applicato il case folding.
    
    Le gold label sono usate per confrontare l'accuracy del modello BERT sul test_set rispetto ad altri modelli. 
"""

from models.training import load_abs, split_abs, get_sentences
from datasets import Dataset, DatasetDict
from finetuning.utils import get_test_cases_from_abs    


def get_sent_from_tokens(tokens: list[str]):
    return " ".join(tokens) 

def push_to_huggingface_hub(dataset: DatasetDict): 
    dataset.push_to_hub("GabrieleGiannessi/maat-corpus", 
                        commit_message="v5: fix bug gold_label nel test_set ")
    pass


def main ():
    abs = load_abs()
    temp_abs, test_abs = split_abs(abs, 0.1)
    train_abs, dev_abs = split_abs(temp_abs, 0.1)
    test_set = get_test_cases_from_abs(test_abs)
    train_set = [get_sent_from_tokens(tokens) for tokens in get_sentences(train_abs) if '<UNK>' not in tokens]  
    dev_set = [get_sent_from_tokens(tokens) for tokens in get_sentences(dev_abs) if '<UNK>' not in tokens] #uso il dev set per aggiustare gli iperparametri durante il finetuning
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "text": train_set, 
            "gold_label": [[""]] * len(train_set) #gold_label è una sequenza vuota
        }),
        "dev": Dataset.from_dict({
            "text": dev_set, 
            "gold_label": [[""]] * len(dev_set) #gold_label è una sequenza vuota
        }),
        "test": Dataset.from_dict({
            "text": [obj['text'] for obj in test_set], 
            "gold_label": [obj['gold_label'] for obj in test_set]  # Ensure gold_label is a sequence
        })
    })
    
    push_to_huggingface_hub(dataset)
            
if __name__ == "__main__":
    main()