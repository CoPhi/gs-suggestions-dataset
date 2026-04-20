from dataclasses import dataclass
from typing import Optional

import torch
from transformers import DataCollatorForLanguageModeling
from models.bert.finetuning import MAX_SPAN_LENGTH

@dataclass
class DataCollatorForSpanMLM(DataCollatorForLanguageModeling):
    """
    Data collator per Masked Language Modeling con span masking contigui di lunghezza
    variabile (1..max_span_length).

    - Mantiene l'obiettivo MLM standard (loss token-wise).
    - Maschera blocchi contigui invece che singoli token indipendenti.
    """

    max_span_length: int = MAX_SPAN_LENGTH 

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replica la logica HF, ma scegliendo i token da mascherare per span.
        """
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        batch_size, seq_len = labels.size()

        # Mask matrix inizialmente vuota
        masked_indices = torch.zeros_like(labels, dtype=torch.bool)

        # Per ogni sequenza del batch scegliamo span contigui
        for i in range(batch_size):
            # Posizioni candidate (non special token)
            candidate_positions = torch.nonzero(
                ~special_tokens_mask[i], as_tuple=False
            ).view(-1)

            if candidate_positions.numel() == 0:
                continue

            # Target approssimativo di token da mascherare
            num_to_mask = max(
                1,
                int(self.mlm_probability * candidate_positions.numel()),
            )

            # Mescoliamo le posizioni candidate per non avere bias di ordine
            perm = torch.randperm(candidate_positions.numel())
            candidate_positions = candidate_positions[perm]

            masked_count = 0
            idx_ptr = 0

            while masked_count < num_to_mask and idx_ptr < candidate_positions.numel():
                start_pos = candidate_positions[idx_ptr].item()
                idx_ptr += 1

                if masked_indices[i, start_pos]:
                    continue

                # Campiona lunghezza span
                span_len = torch.randint(1, self.max_span_length + 1, (1,)).item()
                end_pos = min(start_pos + span_len, seq_len)

                # Evita di mascherare special token dentro lo span
                span_positions = torch.arange(start_pos, end_pos, device=labels.device)
                if special_tokens_mask[i, span_positions].any():
                    continue

                # Applica mask sullo span
                masked_indices[i, span_positions] = True
                masked_count += span_positions.numel()

                if masked_count >= num_to_mask:
                    break

        # Label = token gold solo dove mascheriamo, altrove -100 (ignorato dalla loss)
        labels[~masked_indices] = -100

        # Applica schema BERT 80% [MASK], 10% random, 10% unchanged
        # come in DataCollatorForLanguageModeling originale
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long,
            device=labels.device,
        )
        inputs[indices_random] = random_words[indices_random]

        # Il resto dei masked_indices rimane il token originale (10%)
        return inputs, labels
