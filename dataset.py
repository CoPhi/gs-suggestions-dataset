"""
    Script per rendere disponibile l'insieme delle frasi pre-processsate del MAAT corpus su hugging face
    
    access token: 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
    nome: hf-maat-upload
    
    Questo dataset serve per fare il finetuning dei modelli BERT per il greco antico sul task MLM 
    
"""

from models.training import load_abs, split_abs, get_sentences
from datasets import Dataset, DatasetDict
def get_sent_from_tokens(tokens: list[str]):
    return " ".join(tokens) 

def push_to_huggingface_hub(dataset: DatasetDict): 
    dataset.push_to_hub("GabrieleGiannessi/maat-corpus")
    pass

def main ():
    abs = load_abs()
    sentences = [get_sent_from_tokens(tokens) for tokens in get_sentences(abs) if '<UNK>' not in tokens]  
    temp_sentences, test_sentences = split_abs(sentences)
    train_sentences, dev_sentences = split_abs(temp_sentences)
    
    dataset = DatasetDict({"train": Dataset.from_dict({"text": train_sentences}),
                       "validation": Dataset.from_dict({"text": dev_sentences}),
                       "test": Dataset.from_dict({"text": test_sentences}
                                                 )})
    
    push_to_huggingface_hub(dataset)
            
if __name__ == "__main__":
    main()