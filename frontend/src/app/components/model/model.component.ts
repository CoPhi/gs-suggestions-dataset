import { Component, computed, effect, inject, input, model } from '@angular/core';
import { ApiService, BERTModelInterface, IthacaModelInterface, modelType, NgramsModelInterface } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';

@Component({
  selector: 'app-model',
  imports: [],
  templateUrl: './model.component.html',
  styleUrl: './model.component.css'
})
export class ModelComponent {

  api = inject(ApiService);
  notifications = inject(NotificationService);

  curr_id = model.required<string | null>(); // ID del modello corrente usato nella richiesta
  model = input.required<modelType>(); // Modello da visualizzare
  models = model.required<modelType[]>(); // Lista dei modelli disponibili

  id = computed(() => this.model()._id); // ID del modello mostrato
  isBERT = computed(() => this.model().TYPE === 'BERT'); // Verifica se il modello è di tipo BERT
  isIthaca = computed(() => this.model().TYPE === 'Ithaca'); // Verifica se il modello è di tipo Ithaca
  isNgrams = computed(() => this.model().TYPE === 'Ngrams'); // Verifica se il modello è di tipo N-grams
  bertmodel = computed(() => <BERTModelInterface>this.model()); // Modello BERT
  ithacamodel = computed(() => <IthacaModelInterface>this.model()); // Modello Ithaca
  ngramsmodel = computed(() => <NgramsModelInterface>this.model()); // Modello N-grams

  deleteModel() {
    const modelLabel = this.isIthaca()
      ? `Ithaca (${this.ithacamodel().CHECKPOINT})`
      : this.isBERT()
      ? `BERT (${this.bertmodel().CHECKPOINT})`
      : `N-grammi (${this.ngramsmodel().LM_SCORE})`;

    this.api.deleteModel(this.id()).subscribe({
      next: () => {
        this.models.set(this.models().filter((m) => m._id !== this.id()));
        if (this.curr_id() === this.id()) {
          this.curr_id.set(null);
        }
        this.notifications.showDanger(`Modello ${modelLabel} eliminato con successo dal database MongoDB!`);
      },
      error: (e) => {
        console.error('Errore durante eliminazione modello:', e);
        this.notifications.showDanger(`Errore durante l'eliminazione del modello dal database: ${e?.message || 'Errore del server'}`);
      }
    });
  }

  // Imposta il modello corrente nell'input dell'app principale per la produzione dei suggerimenti
  set_currentID() {
    this.curr_id.set(this.id());
  }

}
