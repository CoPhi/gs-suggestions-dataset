from dataclasses import dataclass
from typing import Any
import random
import re
import torch
from transformers import PreTrainedTokenizer

from models.bert.finetuning import GAP_TOKEN


@dataclass
class DataCollatorForSyntheticGapMLM:
    """
    Data collator che allinea l'addestramento MLM con l'inferenza di HCB:
    1. Genera lacune sintetiche a livello di caratteri (non token) all'interno delle parole,
       replicando esattamente la logica di `generate_synthetic_cases`.
    2. Utilizza il GAP_TOKEN per mantenere i prefissi sub-word esatti.
    3. Usa l'offset mapping sul testo originale per determinare le vere label (gold tokens).
    """

    tokenizer: PreTrainedTokenizer
    mlm_probability: float = 0.15
    min_gap_length: int = 1
    max_gap_length: int = 6

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        gap_token_id = self.tokenizer.convert_tokens_to_ids(GAP_TOKEN)

        for feature in features:
            text = feature["text"]

            # Troviamo parole con caratteri alfabetici (stessa policy di generate_synthetic_cases)
            words_matches = list(re.finditer(r"[^\W\d_]+", text))
            valid_matches = [
                m for m in words_matches if len(m.group()) >= self.min_gap_length
            ]

            # Decidiamo quali parole mascherare
            # Approssimiamo la probabilità MLM: mascheriamo in media una percentuale delle parole candidate.
            words_to_mask = [
                m for m in valid_matches if random.random() < self.mlm_probability
            ]

            # Ordiniamo al contrario per non sballare gli indici durante le modifiche alla stringa
            words_to_mask.sort(key=lambda x: x.start(), reverse=True)

            modified_text = text
            gaps_info = []  # Conterrà dict con {start_char_orig, end_char_orig}

            for match in words_to_mask:
                word = match.group()
                max_possible_gap = min(self.max_gap_length, len(word))
                gap_len = random.randint(self.min_gap_length, max_possible_gap)
                start_in_word = random.randint(0, len(word) - gap_len)

                start_char = match.start() + start_in_word
                end_char = start_char + gap_len

                modified_text = (
                    modified_text[:start_char] + GAP_TOKEN + modified_text[end_char:]
                )

                gaps_info.append(
                    {
                        "start_char_orig": start_char,
                        "end_char_orig": end_char,
                    }
                )

            # 2. Tokenizzazione del testo originale con offset mapping
            orig_encoding = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            orig_input_ids = orig_encoding["input_ids"]
            orig_offsets = orig_encoding["offset_mapping"]

            # 3. Determinazione delle gold labels
            # gap_true_labels manterrà le liste di token originali corrispondenti ai gap, in ordine decrescente (da destra a sinistra)
            gap_true_labels = []
            for gap in gaps_info:
                start_orig = gap["start_char_orig"]
                end_orig = gap["end_char_orig"]

                true_tokens = []
                for idx, (os_start, os_end) in enumerate(orig_offsets):
                    if os_start == os_end == 0:
                        continue
                    # Condizione di overlap tra [os_start, os_end) e [start_orig, end_orig)
                    if max(os_start, start_orig) < min(os_end, end_orig):
                        true_tokens.append(orig_input_ids[idx])

                if not true_tokens:
                    # Fallback sicuro: se l'offset mapping fallisce per caratteri invisibili
                    true_tokens = [self.tokenizer.unk_token_id]

                gap_true_labels.append(true_tokens)

            # 4. Tokenizzazione del testo modificato
            mod_encoding = self.tokenizer(
                modified_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
            mod_input_ids = mod_encoding["input_ids"]

            # 5. Sostituzione di GAP_TOKEN con le vere labels mascherate
            final_input_ids = []
            final_labels = []

            # Invertiamo gap_true_labels per scandirli da sinistra a destra, in accordo con mod_input_ids
            gap_true_labels = list(reversed(gap_true_labels))
            gap_idx = 0

            for token_id in mod_input_ids:
                if token_id == gap_token_id:
                    if gap_idx < len(gap_true_labels):
                        true_tokens = gap_true_labels[gap_idx]

                        # Schema BERT: 80% [MASK], 10% random, 10% unchanged
                        for t in true_tokens:
                            prob = random.random()
                            if prob < 0.8:
                                final_input_ids.append(self.tokenizer.mask_token_id)
                            elif prob < 0.9:
                                random_word = random.randint(0, len(self.tokenizer) - 1)
                                final_input_ids.append(random_word)
                            else:
                                final_input_ids.append(t)
                            final_labels.append(t)

                        gap_idx += 1
                    else:
                        # Fallback nel raro caso in cui la corrispondenza fallisca
                        final_input_ids.append(self.tokenizer.mask_token_id)
                        final_labels.append(-100)
                else:
                    final_input_ids.append(token_id)
                    final_labels.append(-100)  # -100 per i token non mascherati

            batch_input_ids.append(torch.tensor(final_input_ids))
            batch_labels.append(torch.tensor(final_labels))
            batch_attention_mask.append(torch.ones(len(final_input_ids)))

        # 6. Padding manuale
        max_len = max(len(x) for x in batch_input_ids)
        max_len = min(max_len, self.tokenizer.model_max_length)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        for i in range(len(batch_input_ids)):
            ids = batch_input_ids[i][:max_len]
            lbls = batch_labels[i][:max_len]
            att = batch_attention_mask[i][:max_len]

            pad_len = max_len - len(ids)

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_token_id)])
                lbls = torch.cat([lbls, torch.full((pad_len,), -100)])
                att = torch.cat([att, torch.zeros(pad_len)])

            padded_input_ids.append(ids)
            padded_labels.append(lbls)
            padded_attention_mask.append(att)

        return {
            "input_ids": torch.stack(padded_input_ids).long(),
            "labels": torch.stack(padded_labels).long(),
            "attention_mask": torch.stack(padded_attention_mask).long(),
        }
