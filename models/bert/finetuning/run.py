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
    TrainerCallback
)
import wandb

# Custom Imports per HCB
from models.bert.dataset.dev_set import build_dev_set
from models.bert.inference.predict import fill_mask
from models.bert.evaluation.topk import evaluate_topK

from models.bert.finetuning import OUTPUT_DIR, LOGS_DIR

class HCBEvaluationCallback(TrainerCallback):
    """
    Callback personalizzato per calcolare le metriche TopK tramite HCB durante la fase di eval.
    Invece di valutare l'intero corpus con HCB (che rallenterebbe enormemente il training),
    valutiamo un sottoinsieme (pool) di casi reali annotati ad ogni ciclo di on_evaluate.
    """
    def __init__(self, dev_cases_pool, tokenizer, max_eval_cases=50):
        super().__init__()
        # Limitiamo il pool per non far durare ore ogni validazione
        self.dev_cases_pool = dev_cases_pool[:max_eval_cases]
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model, **kwargs):
        """
        Esegue la validazione HCB sul pool di casi di test.
        """

        # Assicuriamoci che il modello sia in eval mode
        model.eval()
        device = next(model.parameters()).device
        
        predictions_ids = []
        true_ids_batch = []
        
        if len(self.dev_cases_pool) == 0:
            return

        for case in self.dev_cases_pool:
            try:
                suggestions = fill_mask(
                    case=case,
                    tokenizer=self.tokenizer,
                    model=model,
                    num_suggestions=10,
                    mask_token=self.tokenizer.mask_token or "[MASK]",
                    method="modified_best_to_worst",
                    device=device
                )
                
                y_tokens_ids = self.tokenizer(case.y, add_special_tokens=False)["input_ids"]
                
                beam_preds = []
                for s in suggestions:
                    beam_preds.append([s[0]] + s[1])
                    
                predictions_ids.append(beam_preds)
                true_ids_batch.append(y_tokens_ids)
            except Exception as e:
                pass # skip se c'è un errore formativo su qualche lacuna particolare

        if predictions_ids:
            top_k_metrics = evaluate_topK(
                predictions_hcb_format=predictions_ids,
                true_ids=true_ids_batch,
                tokenizer=self.tokenizer
            )
            
            # Log su wandb se abilitato e stampiamo su CLI
            print(f"[HCB Val] Top1: {top_k_metrics.get('top1', 0):.2f}% | Top5: {top_k_metrics.get('top5', 0):.2f}%")
            
            logs = { f"eval_hcb_{k}": v for k, v in top_k_metrics.items() }
            logs["step"] = state.global_step
            if wandb.run is not None:
                wandb.log(logs)

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
