import { Component, computed, inject, signal } from '@angular/core';
import { ApiService, HitlRefinementOptions, modelType, SuggestionInterface } from './services/api.service';
import { SuggestsBoxComponent } from './components/suggests-box/suggests-box.component';
import { SuggestionFeedbackEvent } from './components/suggest/suggest.component';
import { ModelsBoxComponent } from "./components/models-box/models-box.component";
import { NotificationService } from './services/notification.service';
import { PromptBoxComponent } from './components/prompt-box/prompt-box.component';

import { Subscription } from 'rxjs';

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
  curr_id = signal<string | null>(null);

  // Stato dell'ultima richiesta di generazione per supportare il log di feedback (Fase 1)
  // e il raffinamento guidato in-session (Fase 2) senza reinserire il prompt
  lastGenerationPayload = signal<{ text: string; modelID: string; num_tokens: number; num_predictions: number } | null>(null);
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

  isGenerating = signal<boolean>(false);
  private currentRequestSubscription: Subscription | null = null;

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
    this.sessionHumanFeedback.set('');
    this.sessionRejectedCandidates.set([]);
  }

  generateSuggestions(
    payload: { text: string; modelID: string; num_tokens: number; num_predictions: number },
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
    const { text, modelID, num_tokens, num_predictions } = payload;

    const request$ = hitlOptions
      ? this.api.generateSuggestion(modelID, text, num_tokens, num_predictions, hitlOptions)
      : this.api.generateSuggestion(modelID, text, num_tokens, num_predictions);

    this.currentRequestSubscription = request$.subscribe({
      next: (response) => {
        this.suggestions.set(response);
        this.isGenerating.set(false);
        const count = response?.length || 0;
        this.notifications.showSuccess(`Suggerimenti pronti! Generate con successo ${count} predizioni.`);
        this.notifications.showLocalNotification();
      },
      error: () => {
        this.notifications.showDanger('Errore durante la generazione dei suggerimenti');
        this.isGenerating.set(false);
      },
      complete: () => {
        this.isGenerating.set(false);
      }
    });
  }

  /**
   * Invia al backend l'annotazione del filologo (Fase 1 - Dataset Curation) e aggiorna
   * i candidati esclusi nella sessione corrente.
   */
  recordFeedback(event: SuggestionFeedbackEvent) {
    const lastReq = this.lastGenerationPayload();
    const modelId = lastReq?.modelID || this.curr_id() || '';
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

    this.api.submitFeedback({
      model_id: modelId,
      context,
      token_str: event.token_str,
      sentence: event.sentence,
      is_accepted: event.is_accepted,
      notes: this.sessionHumanFeedback() || null
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

