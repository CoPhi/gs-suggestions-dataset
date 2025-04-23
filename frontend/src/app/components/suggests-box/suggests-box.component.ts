import { Component, inject, input, model } from '@angular/core';
import { setThrowInvalidWriteToSignalError } from '@angular/core/primitives/signals';
import { AbstractControl, FormControl, FormGroup, ReactiveFormsModule, ValidationErrors, Validators } from '@angular/forms';
import { ApiService, modelType } from '../../services/api.service';
import { ModelComponent } from '../model/model.component';

@Component({
  selector: 'app-suggests-box',
  imports: [ReactiveFormsModule, ModelComponent],
  templateUrl: './suggests-box.component.html',
  styleUrl: './suggests-box.component.css'
})
export class SuggestsBoxComponent {

  api = inject(ApiService);
  models = input.required<modelType[]>()

  isContextValid = (c: AbstractControl): ValidationErrors | null => {
    if (!c.value) return { notvalid: true };
    return c.value.search("[...]") !== -1 ? null : { notMasked: true };
  }

  form = new FormGroup({
    text: new FormControl<string>('', { validators: [Validators.required, this.isContextValid, Validators.minLength(10)] }),
    modelID: new FormControl<string>('', {
      validators: [Validators.required, Validators.minLength(24),
      Validators.maxLength(24)]
    }),
  })

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
    if (spinner.style.display === 'none') {
      spinner.style.display = 'block';
    } else {
      spinner.style.display = 'none';
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
    this.form.updateValueAndValidity(); // aggiorna tutti i validator

    

    if (this.form.invalid) {
      const textErrors = this.form.controls['text'].errors;
      const modelErrors = this.form.controls['modelID'].errors;

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

      return;
    }

    const { text, modelID } = this.form.getRawValue();

    this.toggleSpinner();

    this.api.generateSuggestion(modelID!, text!).subscribe({
      next: (response) => {
        console.log(response);
        this.showAlert('Suggerimenti generati con successo', 'success');
        this.clearForm();
      },
      error: (error: any) => {
        console.error('Errore:', error);
        this.showAlert('Errore durante la generazione dei suggerimenti', 'danger');
      },
      complete: () => {
        this.toggleSpinner();
      }
    });
  }

}
