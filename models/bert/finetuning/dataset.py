"""
Script per rendere disponibile l'insieme delle frasi pre-processsate del MAAT corpus su hugging face
Questo dataset serve per fare il finetuning dei modelli BERT per il greco antico sul task MLM

Il train_set è composto da frasi pulite in greco antico senza la presenza di token sconosciuti, in cui è stato applicato il case folding.
Il test_set è composto da frasi pulite in greco antico con la presenza di token mascherati, in cui è stato applicato il case folding.

Le gold label sono usate per confrontare l'accuracy del modello BERT sul test_set rispetto ad altri modelli.
"""

from datasets import DatasetDict, Dataset
from tqdm import tqdm
from backend.core.preprocess import clean_tokens, process_editorial_marks
from models.bert.finetuning import (
    load_train_and_dev_set,
    get_processed_sentences,
    sentence_tokenizer, 
)

from backend.core.cleaner import load_test_set, get_sentences
from models.bert.finetuning import (
    TRAIN_DATASET_CHECKPOINT,
    TEST_DATASET_CHECKPOINT,
    get_cast_unk_tokens_text,
    get_sent_from_tokens,
)

def push_trainset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub(TRAIN_DATASET_CHECKPOINT, commit_message=message)


def push_testset_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub(TEST_DATASET_CHECKPOINT, commit_message=message)


def push_set_to_huggingface_hub(dataset: DatasetDict, message: str) -> None:
    dataset.push_to_hub("CNR-ILC/gs-maat-corpus", commit_message=message)

def build_raw_dataset(abs: list) -> Dataset:
    """
    Produce il dataset grezzo: markup editoriale rimosso, lacune → <UNK>,
    ma NESSUNA normalizzazione model-specific (case, diacritici, punteggiatura).
    """
    sentences = []
    for obj in tqdm(abs, desc="Building raw dataset", unit="ab", leave=False):
        if obj.get("language") != "grc" or not obj.get("training_text"):
            continue
        # Fase A: solo markup editoriale + clean lacune
        text = process_editorial_marks(obj["training_text"])
        tokens = clean_tokens(text)           # lacune → <UNK>
        text = " ".join(tokens).strip()
        if not text:
            continue
        for sent in sentence_tokenizer.tokenize(text):
            if sent:
                sentences.append(sent)

    return Dataset.from_dict({"text": sentences})

def main():
    train_abs, dev_abs = load_train_and_dev_set(test_size=0.2)
    test_abs = load_test_set()
    dataset = DatasetDict(
        {
            "train": get_processed_sentences(train_abs),
            "dev": get_processed_sentences(dev_abs),
            "test": Dataset.from_dict(
                {
                    "text": [
                        get_cast_unk_tokens_text(get_sent_from_tokens(sent_tkns))
                        for sent_tkns in get_sentences(
                            test_abs, case_folding=True, remove_punct=True
                        )
                    ],
                }
            ),
        }
    )

    push_set_to_huggingface_hub(
        dataset,
        "Unlabeled ancient greek sentences for fill-mask task, folded to uppercase, remove punct, with dev & test set",
    )
    
if __name__ == "__main__":
    main()