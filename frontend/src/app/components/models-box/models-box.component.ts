import { Component, inject, model, signal } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { ApiService, CreateRAGModelPayload, modelType, RagBackendType, RAGModelInterface } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';
import { ModelComponent } from "../model/model.component";

@Component({
  selector: 'app-models-box',
  imports: [ModelComponent, ReactiveFormsModule],
  templateUrl: './models-box.component.html',
  styleUrl: './models-box.component.css'
})
export class ModelsBoxComponent {
  api = inject(ApiService);
  notifications = inject(NotificationService);

  models = model.required<modelType[]>();
  curr_id = model.required<string | null>();

  showAddRagForm = signal<boolean>(false);
  isCreating = signal<boolean>(false);

  readonly ragPresets: { label: string; value: string }[] = [
    { label: 'Qwen 2.5 7B Instruct (Consigliato)', value: 'Qwen/Qwen2.5-7B-Instruct' },
    { label: 'Qwen 2.5 3B Instruct (Leggero)', value: 'Qwen/Qwen2.5-3B-Instruct' },
    { label: 'Gemma 2 9B Instruct (Google DeepMind)', value: 'google/gemma-2-9b-it' },
    { label: 'Llama 3.1 8B Instruct (Meta)', value: 'meta-llama/Llama-3.1-8B-Instruct' },
    { label: 'Gemma 4 E4B (Ollama locale)', value: 'gemma4:e4b' }
  ];

  ragForm = new FormGroup({
    checkpoint: new FormControl<string>('Qwen/Qwen2.5-7B-Instruct', {
      nonNullable: true,
      validators: [Validators.required, Validators.minLength(2)]
    }),
    backend: new FormControl<RagBackendType>('auto', {
      nonNullable: true,
      validators: [Validators.required]
    }),
    topKPassages: new FormControl<number>(5, {
      nonNullable: true,
      validators: [Validators.required, Validators.min(0), Validators.max(20)]
    })
  });

  toggleAddRagForm() {
    this.showAddRagForm.update(open => !open);
  }

  applyPreset(checkpoint: string) {
    this.ragForm.controls.checkpoint.setValue(checkpoint);
  }

  submitRagModel() {
    this.ragForm.markAllAsTouched();
    if (this.ragForm.invalid || this.isCreating()) {
      return;
    }

    const raw = this.ragForm.getRawValue();
    const payload: CreateRAGModelPayload = {
      CHECKPOINT: raw.checkpoint.trim(),
      BACKEND: raw.backend,
      TOP_K_PASSAGES: Number(raw.topKPassages),
      COMPILED_PROGRAM_PATH: null,
      TYPE: 'RAG'
    };

    this.isCreating.set(true);
    // Registra il modello sul backend e aggiorna reattivamente lista modelli e selezione corrente
    this.api.createModel(payload).subscribe({
      next: (createdId: string) => {
        const createdModel: RAGModelInterface = {
          _id: createdId,
          ...payload
        };
        this.models.set([...this.models(), createdModel]);
        this.curr_id.set(createdId);
        this.isCreating.set(false);
        this.showAddRagForm.set(false);
        this.notifications.showSuccess(`Modello RAG (${payload.CHECKPOINT}) registrato e selezionato con successo!`);
      },
      error: (err) => {
        this.isCreating.set(false);
        const detail = err?.error?.detail || 'Impossibile registrare il modello RAG (potrebbe essere già presente).';
        this.notifications.showDanger(detail);
      }
    });
  }
}
