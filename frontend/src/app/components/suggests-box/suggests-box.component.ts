import { Component, computed, effect, inject, input, signal } from '@angular/core';
import { AbstractControl, FormControl, FormGroup, ReactiveFormsModule, FormsModule, ValidationErrors, Validators } from '@angular/forms';
import { ApiService, modelType, SuggestionInterface } from '../../services/api.service';
import { ModelComponent } from '../model/model.component';
import { SuggestComponent } from '../suggest/suggest.component';

@Component({
  selector: 'app-suggests-box',
  imports: [ReactiveFormsModule, FormsModule, ModelComponent, SuggestComponent],
  templateUrl: './suggests-box.component.html',
  styleUrl: './suggests-box.component.css'
})
export class SuggestsBoxComponent {

  api = inject(ApiService);
  models = input.required<modelType[]>()
  form: FormGroup;

  suggestions = signal<SuggestionInterface[]>([]);
  curr_id = signal<string | null>(null);
  selectedModel = computed(() => {
    return this.curr_id() ? this.models().find((model) => model._id === this.curr_id()) || null : null;
  });
  curr_type_model = computed(() => this.selectedModel()?.TYPE)
  isGenerating =  signal<boolean>(false);

  constructor() {
    this.form = new FormGroup({
      text: new FormControl<string>('', { validators: [Validators.required, this.isContextValid, Validators.minLength(10)] }),
      modelID: new FormControl<string>('', {
        validators: [Validators.required, Validators.minLength(24),
        Validators.maxLength(24)]
      }),
      num_tokens: new FormControl<number>(1, { validators: [Validators.min(1), Validators.max(10)] }), 
      num_predictions: new FormControl<number>(1, { validators: [Validators.required]} )
    })
  }

  setCurrentID($event: Event) {
    const target = $event.target as HTMLInputElement;
    this.curr_id.set(target.value);
  }

  setNumPredictions($event: Event) {
    const target = $event.target as HTMLInputElement;
    console.log(target.value);
    }

  isContextValid = (c: AbstractControl): ValidationErrors | null => {
    if (!c.value) return { notvalid: true };
    return c.value.search("[...]") !== -1 ? null : { notMasked: true };
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

  toggleSpinner() {
    const spinner = <HTMLElement>document.querySelector('#loadingSpinner');
    if (spinner.classList.contains('d-none')) {
      spinner.classList.remove('d-none')
    } else {
      spinner.classList.add('d-none')
    }
  }

  clearForm() {
    this.form.reset();
    this.form.get('modelID')?.setValue('');
    this.form.get('text')?.setValue('');
    this.form.get('text')?.setErrors(null);
    this.form.get('modelID')?.setErrors(null);
  }

  showAlert(message: string, type: string) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show mt-3`;
    alert.innerHTML = `
          ${message}
          <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
      `;

    const container = document.querySelector('.card-body');
    container!.appendChild(alert);

    setTimeout(() => {
      alert.classList.remove('show');
      setTimeout(() => alert.remove(), 150);
    }, 5000);
  }
  generateSuggestions() {
    this.form.markAllAsTouched(); // forza la validazione
    if (this.form.invalid) {
      const textErrors = this.form.controls['text'].errors;
      const modelErrors = this.form.controls['modelID'].errors;
      const tokenErrors = this.form.controls['num_tokens'].errors;
  
      if (textErrors?.['required']) {
        this.showAlert('Il campo testo è obbligatorio', 'danger');
        return;
      }
  
      if (textErrors?.['notMasked']) {
        this.showAlert('Il testo non contiene la lacuna [...]', 'warning');
        return;
      }
  
      if (textErrors?.['minlength']) {
        this.showAlert('Il testo è troppo corto', 'danger');
        return;
      }
  
      if (modelErrors?.['required']) {
        this.showAlert('Seleziona un modello', 'danger');
        return;
      }
  
      if (modelErrors?.['minlength'] || modelErrors?.['maxlength']) {
        this.showAlert('L\'ID del modello deve essere di 24 caratteri', 'danger');
        return;
      }
  
      if (tokenErrors?.['required']) {
        this.showAlert('Specifica il numero di token', 'danger');
        return;
      }
  
      if (tokenErrors?.['min'] || tokenErrors?.['max']) {
        this.showAlert('Il numero di token deve essere tra 1 e 10', 'danger');
        return;
      }
  
      return;
    }
    this.isGenerating.set(true);
    const { text, modelID, num_tokens, num_predictions } = this.form.getRawValue();
    this.toggleSpinner();
  
    this.api.generateSuggestion(modelID, text, num_tokens, Number(num_predictions)).subscribe({
      next: (response) => {
        this.suggestions.set(response);
        this.showAlert('Suggerimenti generati con successo', 'success');
        this.isGenerating.set(false);
      },
      error: (error: any) => {
        console.error('Errore:', error);
        this.showAlert('Errore durante la generazione dei suggerimenti', 'danger');
        this.isGenerating.set(false);
        this.toggleSpinner();
      },
      complete: () => {
        this.toggleSpinner();
      }
    });
  }
}
