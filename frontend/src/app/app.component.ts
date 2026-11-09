import { Component, computed, inject, signal } from '@angular/core';
import { ApiService, HitlRefinementOptions, isBertCheckpoint, modelType, SuggestionInterface } from './services/api.service';
import { SuggestsBoxComponent } from './components/suggests-box/suggests-box.component';
import { SuggestionFeedbackEvent } from './components/suggest/suggest.component';
import { ModelsBoxComponent } from "./components/models-box/models-box.component";
import { NotificationService } from './services/notification.service';
import { PromptBoxComponent } from './components/prompt-box/prompt-box.component';

import { catchError, forkJoin, map, Observable, of, Subscription } from 'rxjs';

export interface ModelSuggestionBoxItem {
  modelId: string;
  modelLabel: string;
  modelType: string;
  checkpoint?: string | null;
  use_rag?: boolean | null;
  suggestions: SuggestionInterface[];
  error?: string | null;
}

@Component({
  selector: 'app-root',
  imports: [SuggestsBoxComponent, ModelsBoxComponent, PromptBoxComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'gs-api';

  api = inject(ApiService);
  notifications = inject(NotificationService);

  // Segnale scrivibile condiviso in two-way binding tra menù a tendina e modale modelli
  models = signal<modelType[]>([]);
  suggestions = signal<SuggestionInterface[] | null>(null);
  // Lista di suggestion box per la consultazione orizzontale multi-modello
  modelSuggestionBoxes = signal<ModelSuggestionBoxItem[] | null>(null);
  curr_id = signal<string | null>(null);

  // Stato dell'ultima richiesta di generazione per supportare il log di feedback (Fase 1)
  // e il raffinamento guidato in-session (Fase 2) senza reinserire il prompt
  lastGenerationPayload = signal<{
    text: string;
    modelID: string;
    modelIDs?: string[];
    num_tokens: number;
    num_predictions: number;
    use_rag?: boolean;
  } | null>(null);
  sessionHumanFeedback = signal<string>('');
  sessionRejectedCandidates = signal<string[]>([]);

  constructor() {
    this.api.getModels().subscribe({
      next: (loadedModels) => this.models.set(loadedModels),
      error: () => {}
    });
  }

  selectedModel = computed(() => {
    return this.curr_id() ? this.models().find((model) => model._id === this.curr_id()) || null : null;
  });

  // Etichetta e tipo del modello mostrati nell'intestazione del suggestion box singolo
  activeSingleModelLabel = computed<string | null>(() => {
    const boxes = this.modelSuggestionBoxes();
    if (boxes && boxes.length === 1) {
      return boxes[0].modelLabel;
    }
    const last = this.lastGenerationPayload();
    if (last?.modelID) {
      return this.resolveModelLabel(last.modelID, last.use_rag);
    }
    return this.curr_id() ? this.resolveModelLabel(this.curr_id()!) : null;
  });

  activeSingleModelType = computed<string | null>(() => {
    const boxes = this.modelSuggestionBoxes();
    if (boxes && boxes.length === 1) {
      return boxes[0].modelType;
    }
    const last = this.lastGenerationPayload();
    if (last?.modelID) {
      return this.resolveModelTypeBadge(last.modelID, last.use_rag);
    }
    return this.curr_id() ? this.resolveModelTypeBadge(this.curr_id()!) : null;
  });

  activeSingleModelCheckpoint = computed<string | null>(() => {
    const boxes = this.modelSuggestionBoxes();
    if (boxes && boxes.length === 1 && boxes[0].checkpoint) {
      return boxes[0].checkpoint;
    }
    const last = this.lastGenerationPayload();
    if (last?.modelID) {
      return this.resolveModelCheckpoint(last.modelID);
    }
    return this.curr_id() ? this.resolveModelCheckpoint(this.curr_id()!) : null;
  });

  isGenerating = signal<boolean>(false);
  private currentRequestSubscription: Subscription | null = null;

  resolveModelCheckpoint(modelId: string): string | null {
    const m = this.models().find(item => item._id === modelId);
    if (!m) return null;
    return 'CHECKPOINT' in m ? m.CHECKPOINT : null;
  }

  resolveModelLabel(modelId: string, useRag?: boolean): string {
    const m = this.models().find(item => item._id === modelId);
    if (!m) return modelId;
    if (m.TYPE === 'RAG' || m.TYPE === 'Ithaca' || m.TYPE === 'BERT') {
      const shortName = (m.CHECKPOINT.includes('/') ? m.CHECKPOINT.split('/').pop() : m.CHECKPOINT) || m.CHECKPOINT;
      const isNeuroLex = m.TYPE === 'RAG' && (m.BACKEND === 'neuro_lexical' || isBertCheckpoint(m.CHECKPOINT));
      if (isNeuroLex || (m.TYPE === 'BERT' && useRag)) {
        return `${shortName} + RAG`;
      }
      return shortName;
    }
    return `N-grammi${m.N ? ` (n=${m.N})` : ''}`;
  }

  resolveModelTypeBadge(modelId: string, useRag?: boolean): string {
    const m = this.models().find(item => item._id === modelId);
    if (!m) return 'Modello';
    if (m.TYPE === 'RAG' && (m.BACKEND === 'neuro_lexical' || isBertCheckpoint(m.CHECKPOINT))) {
      return 'BERT + RAG';
    }
    if (m.TYPE === 'BERT' && useRag) {
      return 'BERT + RAG';
    }
    if (m.TYPE === 'RAG') return 'RAG';
    if (m.TYPE === 'BERT') return 'BERT';
    if (m.TYPE === 'Ithaca') return 'Ithaca';
    return 'N-grammi';
  }

  setCurrentID($event: Event) {
    const target = $event.target as HTMLInputElement;
    this.curr_id.set(target.value);
  }

  showAlert(message: string, type: 'success' | 'danger' | 'warning' | 'info' = 'info') {
    this.notifications.showAlert(message, type);
  }

  closeSuggestions() {
    if (this.currentRequestSubscription) {
      this.currentRequestSubscription.unsubscribe();
      this.currentRequestSubscription = null;
    }
    this.suggestions.set(null);
    this.modelSuggestionBoxes.set(null);
    this.sessionHumanFeedback.set('');
    this.sessionRejectedCandidates.set([]);
  }

  closeModelBox(modelId: string) {
    const current = this.modelSuggestionBoxes();
    if (!current) {
      this.closeSuggestions();
      return;
    }
    const remaining = current.filter(b => b.modelId !== modelId);
    if (remaining.length === 0) {
      this.closeSuggestions();
    } else {
      this.modelSuggestionBoxes.set(remaining);
      this.suggestions.set(remaining[0].suggestions);
    }
  }

  generateSuggestions(
    payload: {
      text: string;
      modelID: string;
      modelIDs?: string[];
      num_tokens: number;
      num_predictions: number;
      use_rag?: boolean;
    },
    hitlOptions?: HitlRefinementOptions
  ) {
    if (this.currentRequestSubscription) {
      this.currentRequestSubscription.unsubscribe();
      this.currentRequestSubscription = null;
    }
    // Se è una nuova richiesta dal PromptBox (senza hitlOptions), azzera i vincoli di sessione precedenti
    if (!hitlOptions) {
      this.sessionHumanFeedback.set('');
      this.sessionRejectedCandidates.set([]);
    }
    this.lastGenerationPayload.set(payload);
    this.isGenerating.set(true);
    const { text: rawText, modelID, modelIDs, num_tokens, num_predictions, use_rag } = payload;
    // Pre-processing a livello di coordinamento: garantisce che la lacuna con asterischi (*...*)
    // sia sempre convertita nel formato atteso dalle API ([...])
    const text = (rawText || '').replace(/\*(\.{1,}|\-{1,})\*/g, '[$1]');

    const mergedOptions: HitlRefinementOptions | undefined =
      hitlOptions || use_rag
        ? { ...hitlOptions, ...(use_rag ? { use_rag: true } : {}) }
        : undefined;

    const targetIds = modelIDs && modelIDs.length > 1
      ? Array.from(new Set(modelIDs))
      : [modelID];

    // Caso multi-modello: lancia le richieste in parallelo e attende sulla barriera (forkJoin)
    // che tutti i modelli consultati abbiano risposto prima di esporre le suggestion box orizzontali.
    if (targetIds.length > 1) {
      const parallelRequests: Observable<ModelSuggestionBoxItem>[] = targetIds.map(id => {
        const req$ = mergedOptions
          ? this.api.generateSuggestion(id, text, num_tokens, num_predictions, mergedOptions)
          : this.api.generateSuggestion(id, text, num_tokens, num_predictions);
        return req$.pipe(
          map((preds): ModelSuggestionBoxItem => ({
            modelId: id,
            modelLabel: this.resolveModelLabel(id, use_rag),
            modelType: this.resolveModelTypeBadge(id, use_rag),
            checkpoint: this.resolveModelCheckpoint(id),
            use_rag: use_rag,
            suggestions: preds || [],
            error: null
          })),
          catchError((err): Observable<ModelSuggestionBoxItem> => of({
            modelId: id,
            modelLabel: this.resolveModelLabel(id, use_rag),
            modelType: this.resolveModelTypeBadge(id, use_rag),
            checkpoint: this.resolveModelCheckpoint(id),
            use_rag: use_rag,
            suggestions: [],
            error: err?.message || 'Errore durante la consultazione del modello'
          }))
        );
      });

      this.currentRequestSubscription = forkJoin(parallelRequests).subscribe({
        next: (boxes) => {
          this.modelSuggestionBoxes.set(boxes);
          this.suggestions.set(boxes[0]?.suggestions || []);
          this.isGenerating.set(false);
          const totalPreds = boxes.reduce((acc, b) => acc + (b.suggestions?.length || 0), 0);
          this.notifications.showSuccess(
            `Confronto pronto! Ricevute ${totalPreds} predizioni complessive da ${boxes.length} modelli.`
          );
          this.notifications.showLocalNotification();
        },
        error: () => {
          this.notifications.showDanger('Errore durante la generazione parallela dei suggerimenti');
          this.isGenerating.set(false);
        },
        complete: () => {
          this.isGenerating.set(false);
        }
      });
      return;
    }

    const request$ = mergedOptions
      ? this.api.generateSuggestion(modelID, text, num_tokens, num_predictions, mergedOptions)
      : this.api.generateSuggestion(modelID, text, num_tokens, num_predictions);

    this.currentRequestSubscription = request$.subscribe({
      next: (response) => {
        this.suggestions.set(response);
        this.modelSuggestionBoxes.set([
          {
            modelId: modelID,
            modelLabel: this.resolveModelLabel(modelID, use_rag),
            modelType: this.resolveModelTypeBadge(modelID, use_rag),
            checkpoint: this.resolveModelCheckpoint(modelID),
            use_rag: use_rag,
            suggestions: response || []
          }
        ]);
        this.isGenerating.set(false);
        const count = response?.length || 0;
        this.notifications.showSuccess(`Suggerimenti pronti! Generate con successo ${count} predizioni.`);
        this.notifications.showLocalNotification();
      },
      error: (err) => {
        const detail = err?.error?.detail || err?.message;
        this.notifications.showDanger(detail || 'Errore durante la generazione dei suggerimenti');
        this.isGenerating.set(false);
      },
      complete: () => {
        this.isGenerating.set(false);
      }
    });
  }

  /**
   * Invia al backend l'annotazione del filologo (HITL e predisposizione per RLHF / DPO),
   * associando i metadati del modello, i dati intrinseci del suggerimento e le esclusioni.
   */
  recordFeedback(
    event: SuggestionFeedbackEvent,
    boxModelId?: string | null,
    boxModelLabel?: string | null,
    boxModelType?: string | null,
    boxCheckpoint?: string | null,
    boxUseRag?: boolean | null
  ) {
    const lastReq = this.lastGenerationPayload();
    // Priorità al model_id specifico della colonna consultata, poi all'evento, poi all'ultima richiesta
    const modelId = boxModelId || event.model_id || lastReq?.modelID || this.curr_id() || '';
    const context = lastReq?.text || '';

    if (!event.is_accepted) {
      this.sessionRejectedCandidates.update(list =>
        list.includes(event.token_str) ? list : [...list, event.token_str]
      );
    } else {
      this.sessionRejectedCandidates.update(list =>
        list.filter(item => item !== event.token_str)
      );
    }

    if (!modelId || !context) {
      return;
    }

    const modelLabel = boxModelLabel || event.model_label || this.activeSingleModelLabel() || null;
    const modelType = boxModelType || event.model_type || this.activeSingleModelType() || null;
    const checkpoint = boxCheckpoint || event.checkpoint || this.activeSingleModelCheckpoint() || null;
    const useRag = boxUseRag !== undefined && boxUseRag !== null
      ? boxUseRag
      : (event.use_rag !== undefined && event.use_rag !== null ? event.use_rag : (lastReq?.use_rag ?? null));

    this.api.submitFeedback({
      model_id: modelId,
      context,
      token_str: event.token_str,
      sentence: event.sentence,
      is_accepted: event.is_accepted,
      notes: this.sessionHumanFeedback() || null,
      // Dati intrinseci del suggerimento proposto
      score: event.score,
      rank: event.rank,
      reconstructed_word: event.reconstructed_word,
      tlg_attested: event.tlg_attested,
      candidate_reasoning: event.candidate_reasoning,
      philological_reasoning: event.philological_reasoning,
      attestations: event.attestations,
      // Metadati sul modello generatore
      model_label: modelLabel,
      model_type: modelType,
      checkpoint: checkpoint,
      use_rag: useRag,
      top_k: event.top_k || lastReq?.num_predictions || null,
      // Supporto per RLHF (coppie di preferenza per DPO / Reward Modeling)
      alternative_candidates: event.alternative_candidates
    }).subscribe({
      next: () => {
        const label = event.is_accepted ? 'validato' : 'scartato';
        this.notifications.showAlert(`Supplemento "${event.token_str}" registrato come ${label}.`, 'info');
      },
      error: () => {
        this.notifications.showDanger('Impossibile salvare il feedback filologico sul server');
      }
    });
  }

  /**
   * Rigenera i suggerimenti applicando le indicazioni filologiche e le esclusioni (Fase 2 - In-Session HITL).
   */
  refineSuggestions(options: HitlRefinementOptions) {
    const lastReq = this.lastGenerationPayload();
    if (!lastReq) {
      return;
    }
    this.sessionHumanFeedback.set(options.human_feedback || '');
    this.sessionRejectedCandidates.set(options.rejected_candidates || []);
    this.generateSuggestions(lastReq, options);
  }
}

