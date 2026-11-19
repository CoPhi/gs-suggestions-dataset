import { Component, computed, inject, model, OnInit, signal, ChangeDetectionStrategy } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { ApiService, CreateRAGModelPayload, isBertCheckpoint, modelType, RagBackendType, RAGModelInterface, SupportedRAGModelInterface } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';
import { ModelComponent } from "../model/model.component";

@Component({
  selector: 'app-models-box',
  imports: [ModelComponent, ReactiveFormsModule],
  templateUrl: './models-box.component.html',
  changeDetection: ChangeDetectionStrategy.Eager,
  styleUrl: './models-box.component.css'
})
export class ModelsBoxComponent implements OnInit {
  api = inject(ApiService);
  notifications = inject(NotificationService);

  models = model.required<modelType[]>();
  curr_id = model.required<string | null>();
  // Segnali condivisi con PromptBoxComponent per sincronizzare la modalità confronto anche dalla modale
  isCompareMode = model<boolean>(false);
  selectedModelIds = model<string[]>([]);

  showAddRagForm = signal<boolean>(false);
  isCreating = signal<boolean>(false);

  effectiveSelectedIds = computed<string[]>(() => {
    if (this.isCompareMode()) {
      const multi = this.selectedModelIds().filter(id => this.models().some(m => m._id === id));
      if (multi.length > 0) {
        return multi;
      }
    }
    const single = this.curr_id();
    return single ? [single] : [];
  });

  // Catalogo dei preset verificati (BERT Neuro-lessicale + RAG, Hugging Face Router e Ollama locale)
  readonly defaultRagPresets: SupportedRAGModelInterface[] = [
    {
      id: 'CNR-ILC/gs-GreBerta',
      name: 'gs-GreBerta + RAG (Neuro-lessicale locale)',
      category: 'BERT + RAG',
      recommended: true,
      description: 'Modello BERT neuro-lessicale accostato al sistema RAG (retrieval ibrido BM25+GreBerta, attestazioni TLG e HCB Beam Search in-process)'
    },
    {
      id: 'CNR-ILC/gs-aristoBERTo',
      name: 'gs-aristoBERTo + RAG (Neuro-lessicale locale)',
      category: 'BERT + RAG',
      recommended: false,
      description: 'Modello BERT (aristoBERTo) accostato al sistema RAG con retrieval ibrido BM25+GreBerta e attestazioni TLG'
    },
    {
      id: 'CNR-ILC/gs-Logion',
      name: 'gs-Logion + RAG (Neuro-lessicale locale)',
      category: 'BERT + RAG',
      recommended: false,
      description: 'Modello BERT (Logion) accostato al sistema RAG con retrieval ibrido BM25+GreBerta e attestazioni TLG'
    },
    {
      id: 'Qwen/Qwen2.5-72B-Instruct',
      name: 'Qwen 2.5 72B Instruct (Qwen)',
      category: 'Qwen',
      recommended: true,
      description: 'Massima profondità di ragionamento filologico e analisi contestuale epica (HF Router)'
    },
    {
      id: 'meta-llama/Llama-3.3-70B-Instruct',
      name: 'Llama 3.3 70B Instruct (Meta)',
      category: 'Llama',
      recommended: false,
      description: 'Modello 70B di ultima generazione Meta per il completamento e CoT (HF Router)'
    },
    {
      id: 'meta-llama/Llama-3.1-8B-Instruct',
      name: 'Llama 3.1 8B Instruct (Meta)',
      category: 'Llama',
      recommended: false,
      description: 'Modello compatto da 8B, veloce e con ottimo rispetto dei vincoli morfologici (HF Router)'
    },
    {
      id: 'google/gemma-3-12b-it',
      name: 'Gemma 3 12B Instruct (Google)',
      category: 'Gemma',
      recommended: false,
      description: 'Modello multilingue Google DeepMind da 12B (HF Router)'
    },
    {
      id: 'google/gemma-3-27b-it',
      name: 'Gemma 3 27B Instruct (Google)',
      category: 'Gemma',
      recommended: false,
      description: 'Modello 27B Google DeepMind ad alta competenza linguistica (HF Router)'
    },
    {
      id: 'google/gemma-4-31B-it',
      name: 'Gemma 4 31B Instruct (Google)',
      category: 'Gemma',
      recommended: false,
      description: 'Modello di frontiera Gemma 4 da 31B (HF Router)'
    },
    {
      id: 'deepseek-ai/DeepSeek-V3',
      name: 'DeepSeek V3 (DeepSeek)',
      category: 'DeepSeek',
      recommended: false,
      description: 'Architettura MoE 671B con routing efficiente (HF Router)'
    },
    {
      id: 'gemma4:e4b',
      name: 'Gemma 4 E4B (Ollama locale)',
      category: 'Ollama',
      recommended: false,
      description: 'Esecuzione locale offline su CPU/GPU tramite demone Ollama'
    }
  ];

  supportedRagModels = signal<SupportedRAGModelInterface[]>(this.defaultRagPresets);

  ragForm = new FormGroup({
    checkpoint: new FormControl<string>('Qwen/Qwen2.5-72B-Instruct', {
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

  ngOnInit(): void {
    // Carica dinamicamente dal backend i modelli supportati e verificati dal router
    this.api.getSupportedRagLms().subscribe({
      next: (models) => {
        if (models && models.length > 0) {
          this.supportedRagModels.set(models);
        }
      },
      error: () => {
        // Fallback resiliente ai preset statici predefiniti
      }
    });
  }

  toggleAddRagForm() {
    this.showAddRagForm.update(open => !open);
  }

  toggleCompareMode() {
    const nextState = !this.isCompareMode();
    this.isCompareMode.set(nextState);
    if (!nextState) {
      const primary = this.curr_id();
      this.selectedModelIds.set(primary ? [primary] : []);
    } else if (this.selectedModelIds().length === 0 && this.curr_id()) {
      this.selectedModelIds.set([this.curr_id()!]);
    }
  }

  isModelInComparison(id: string): boolean {
    return this.effectiveSelectedIds().includes(id);
  }

  toggleModelComparison(id: string) {
    this.isCompareMode.set(true);
    const current = [...this.effectiveSelectedIds()];
    const idx = current.indexOf(id);
    if (idx >= 0) {
      current.splice(idx, 1);
    } else {
      current.push(id);
    }
    this.selectedModelIds.set(current);
    this.curr_id.set(current.length > 0 ? current[0] : null);
  }

  selectAllModelsForComparison() {
    this.isCompareMode.set(true);
    const allIds = this.models().map(m => m._id);
    if (this.effectiveSelectedIds().length === allIds.length && allIds.length > 0) {
      this.selectedModelIds.set([]);
      this.curr_id.set(null);
    } else {
      this.selectedModelIds.set(allIds);
      this.curr_id.set(allIds[0] || null);
    }
  }

  onSingleModelSelected(id: string) {
    this.isCompareMode.set(false);
    this.selectedModelIds.set([id]);
  }

  applyPreset(checkpoint: string) {
    this.ragForm.controls.checkpoint.setValue(checkpoint);
    if (isBertCheckpoint(checkpoint)) {
      this.ragForm.controls.backend.setValue('neuro_lexical');
    } else if (this.ragForm.controls.backend.value === 'neuro_lexical') {
      this.ragForm.controls.backend.setValue('auto');
    }
  }

  onSelectModelChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    if (target && target.value) {
      this.applyPreset(target.value);
    }
  }

  submitRagModel() {
    this.ragForm.markAllAsTouched();
    if (this.ragForm.invalid || this.isCreating()) {
      return;
    }

    const raw = this.ragForm.getRawValue();
    const trimmedCheckpoint = raw.checkpoint.trim();
    const resolvedBackend: RagBackendType =
      raw.backend === 'auto' && isBertCheckpoint(trimmedCheckpoint)
        ? 'neuro_lexical'
        : raw.backend;

    const payload: CreateRAGModelPayload = {
      CHECKPOINT: trimmedCheckpoint,
      BACKEND: resolvedBackend,
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
        if (!this.isCompareMode()) {
          this.selectedModelIds.set([createdId]);
        }
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
