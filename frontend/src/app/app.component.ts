import { Component, computed, inject, signal } from '@angular/core';
import { Signal } from '@angular/core';
import { ApiService, modelType, SuggestionInterface } from './services/api.service';
import { toSignal } from '@angular/core/rxjs-interop';
import { SuggestsBoxComponent } from './components/suggests-box/suggests-box.component';
import { ModelsBoxComponent } from "./components/models-box/models-box.component";
import { NotificationService } from './services/notification.service';
import { PromptBoxComponent } from './components/prompt-box/prompt-box.component';

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

  setCurrentID($event: Event) {
    const target = $event.target as HTMLInputElement;
    this.curr_id.set(target.value);
  }

  toggleModels() {
    const models = document.querySelectorAll('#modelsCard');
    models.forEach((model) => {
      if ((model as HTMLElement).style.display === 'none') {
        (model as HTMLElement).style.display = 'block';
      } else {
        (model as HTMLElement).style.display = 'none';
      }
    });
  }

  showAlert(message: string, type: string) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show mt-3 fixed-top w-50 mx-auto shadow-sm`;
    alert.style.zIndex = '9999';
    alert.innerHTML = `
          ${message}
          <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
      `;

    document.body.appendChild(alert);

    setTimeout(() => {
      alert.classList.remove('show');
      setTimeout(() => alert.remove(), 150);
    }, 5000);
  }

  generateSuggestions(payload: { text: string, modelID: string, num_tokens: number, num_predictions: number }) {
    this.isGenerating.set(true);
    const { text, modelID, num_tokens, num_predictions } = payload;

    this.api.generateSuggestion(modelID, text, num_tokens, num_predictions).subscribe({
      next: (response) => {
        this.suggestions.set(response);
        this.isGenerating.set(false);
        this.notifications.showLocalNotification()
      },
      error: () => {
        this.showAlert('Errore durante la generazione dei suggerimenti', 'danger');
        this.isGenerating.set(false);
      },
      complete: () => {
        this.isGenerating.set(false);
      }
    });
  }
}

