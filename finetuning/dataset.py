"""
Script per rendere disponibile l'insieme delle frasi pre-processsate del MAAT corpus su hugging face

access token: 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
nome: hf-maat-upload

token dell'organizzazione CNR-ILC: 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

Questo dataset serve per fare il finetuning dei modelli BERT per il greco antico sul task MLM

Il train_set è composto da frasi pulite in greco antico senza la presenza di token sconosciuti, in cui è stato applicato il case folding.
Il test_set è composto da frasi pulite in greco antico con la presenza di token mascherati, in cui è stato applicato il case folding.

Le gold label sono usate per confrontare l'accuracy del modello BERT sul test_set rispetto ad altri modelli.
"""

from datasets import DatasetDict
from huggingface_hub import notebook_login
from finetuning.utils import load_and_split_abs, get_train_set, get_test_cases_from_abs
from finetuning import TRAIN_DATASET_CHECKPOINT, TEST_DATASET_CHECKPOINT

def push_trainset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    
    dataset.push_to_hub(TRAIN_DATASET_CHECKPOINT, commit_message=message)
    
def push_testset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub(TEST_DATASET_CHECKPOINT, commit_message=message)


def main():

    train_abs, dev_abs, test_abs = load_and_split_abs()
    train_eval_dataset = DatasetDict(
        {
            "train": get_train_set(train_abs),
        }
    )

    push_trainset_to_huggingface_hub(train_eval_dataset, "New split")
    
    test_dataset = DatasetDict({
        "dev": get_test_cases_from_abs(dev_abs), 
        "test": get_test_cases_from_abs(test_abs)
    })
    
    push_testset_to_huggingface_hub(test_dataset, "New split")
    

if __name__ == "__main__":
    main()
