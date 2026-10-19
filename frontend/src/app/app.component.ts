import { Component, computed, inject, signal } from '@angular/core';
import { Signal } from '@angular/core';
import { ApiService, modelType, SuggestionInterface } from './services/api.service';
import { toSignal } from '@angular/core/rxjs-interop';
import { SuggestsBoxComponent } from './components/suggests-box/suggests-box.component';
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

  models = toSignal(this.api.getModels(), { initialValue: [] }) as Signal<modelType[]>;
  suggestions = signal<SuggestionInterface[] | null>(null);
  curr_id = signal<string | null>(null);

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
  }

  generateSuggestions(payload: { text: string, modelID: string, num_tokens: number, num_predictions: number }) {
    if (this.currentRequestSubscription) {
      this.currentRequestSubscription.unsubscribe();
      this.currentRequestSubscription = null;
    }
    this.isGenerating.set(true);
    const { text, modelID, num_tokens, num_predictions } = payload;

    this.currentRequestSubscription = this.api.generateSuggestion(modelID, text, num_tokens, num_predictions).subscribe({
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
}

