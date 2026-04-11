import os
import math
import torch
import numpy as np
from itertools import chain
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    DataCollatorForLanguageModeling, 
    AutoModelForMaskedLM, 
    TrainingArguments, 
    Trainer,
)


from models.bert.finetuning import OUTPUT_DIR, LOGS_DIR

from models.bert.finetuning.callback import HCBEvaluationCallback

def prepare_data(tokenizer):
    CHUNK_SIZE = 128
    
    print("Loading textual corpus for MLM...")
    corpus_dataset = load_dataset("CNR-ILC/gs-maat-corpus")
    
    print("Loading structured HCB Dev/Test set...")
    eval_dataset = load_dataset("CNR-ILC/gs-maat-eval")
    
    from models.bert.dataset.dev_set import DevCase
    dev_cases = []
    # Usiamo il 'dev' split dell'eval_dataset per la validation
    for row in eval_dataset["dev"].to_list():
        # Limitiamo a gap compresi fra 1 e 5 caratteri/token (più rapidi per validation check durante train)
        if 1 <= row["gap_length"] <= 5:
            dev_cases.append(
                DevCase(
                    x=row["x"],
                    y=row["y"],
                    gap_length=row["gap_length"],
                    corpus_id=row["corpus_id"],
                    file_id=row["file_id"]
                )
            )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, add_special_tokens=False)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples["input_ids"])
        if total_length % CHUNK_SIZE != 0:
            padding_length = CHUNK_SIZE - (total_length % CHUNK_SIZE)
            for key in concatenated_examples.keys():
                if key == "attention_mask":
                    pad_value = 0
                else:
                    pad_value = tokenizer.pad_token_id
                concatenated_examples[key] += [pad_value] * padding_length

        total_length = (len(concatenated_examples["input_ids"]) // CHUNK_SIZE) * CHUNK_SIZE
        result = {
            k: [t[i : i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def process_data(examples):
        tokenized_output = tokenize_function(examples)
        return group_texts(tokenized_output)

    lm_datasets = corpus_dataset.map(process_data, batched=True, remove_columns=['text'])
    lm_datasets = lm_datasets.select_columns(['input_ids', 'attention_mask', 'labels'])
    
    return lm_datasets, dev_cases

def main():
    torch.cuda.empty_cache()
    
    # Init wandb project if needed
    # wandb.init(project="huggingface")

    model_name = "bowphs/GreBerta"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
    
    print("Preparazione Dataset...")
    lm_datasets, hcb_dev_cases = prepare_data(tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGS_DIR, 
        report_to="wandb",         
        eval_strategy="epoch",     
        save_strategy="no", 
        learning_rate=5e-5,      
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=10,             
        seed=42,                         
        fp16=True,                       
        logging_strategy="epoch",        
        eval_accumulation_steps=4,       
        torch_empty_cache_steps=5000,     
        dataloader_drop_last=True,       
    )

    hcb_callback = HCBEvaluationCallback(
        dev_cases_pool=hcb_dev_cases, 
        tokenizer=tokenizer, 
        max_eval_cases=50 # Pool di campioni reale limitato per mantenere il check veloce
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["dev"],
        data_collator=data_collator,
        callbacks=[hcb_callback] # Iniettiamo l'evaluation custom
    ) 

    print("Avvio Finetuning MLM (con check HCB epochs callback)")
    trainer.train()

if __name__ == "__main__":
    main()
