import { Component, computed, effect, inject, input, model } from '@angular/core';
import { ApiService, BERTModelInterface, modelType, NgramsModelInterface } from '../../services/api.service';

@Component({
  selector: 'app-model',
  imports: [],
  templateUrl: './model.component.html',
  styleUrl: './model.component.css'
})
export class ModelComponent {

  api = inject(ApiService)

  curr_id = model.required<string | null>(); // ID del modello corrente usato nella richiesta
  model = input.required<modelType>(); // Modello da visualizzare
  models = model.required<modelType[]>() // Lista dei modelli disponibili

  id = computed(() => this.model()._id); // ID del modello mostrato
  isBERT = computed(() => this.model().TYPE === 'BERT'); // Verifica se il modello è di tipo BERT
  bertmodel = computed(() => <BERTModelInterface>this.model()); // Modello BERT
  ngramsmodel = computed(() => <NgramsModelInterface>this.model()); // Modello N-grams

  showCopy(element: Event) {
    const target = <HTMLElement>element.target
    const modelId = target.getAttribute('data-model-id')
    if (modelId) {
      navigator.clipboard.writeText(modelId).then(() => {
        const originalText = target.innerHTML;
        target.innerHTML = '<i class="bi bi-check2"></i> Copiato!';
        target.classList.remove('btn-outline-primary');
        target.classList.add('btn-outline-success');
        setTimeout(() => {
          target.innerHTML = originalText;
          target.classList.remove('btn-outline-success');
          target.classList.add('btn-outline-primary');
        }, 2000);
      });
    }
  }

  deleteModel() {
    this.api.deleteModel(this.id()).subscribe({
      next: () => this.models.set(this.models().filter((m) => m._id !== this.id())),
      error: (e) => console.log(e)
    });
  }

  set_currentID() {
    this.curr_id.set(this.id());
  }

}
