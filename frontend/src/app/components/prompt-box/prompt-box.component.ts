import { Component, computed, effect, input, model, output } from '@angular/core';
import { AbstractControl, FormControl, FormGroup, ReactiveFormsModule, ValidationErrors, Validators } from '@angular/forms';
import { modelType } from '../../services/api.service';

@Component({
  selector: 'app-prompt-box',
  imports: [ReactiveFormsModule],
  templateUrl: './prompt-box.component.html',
  styleUrl: './prompt-box.component.css'
})
export class PromptBoxComponent {
  models = input.required<modelType[]>();
  curr_id = model.required<string | null>();
  isGenerating = input.required<boolean>();

  toggleModels = output<void>();
  onGenerate = output<{ text: string, modelID: string, num_tokens: number, num_predictions: number }>();

  selectedModel = computed(() => {
    return this.curr_id() ? this.models().find((model) => model._id === this.curr_id()) || null : null;
  });

  curr_type_model = computed(() => this.selectedModel()?.TYPE);

  isFocused = false;
  form: FormGroup;

  constructor() {
    this.form = new FormGroup({
      text: new FormControl<string>('', { validators: [Validators.required, this.isContextValid] }),
      modelID: new FormControl<string>('', {
        validators: [Validators.required, Validators.minLength(24), Validators.maxLength(24)]
      }),
      num_tokens: new FormControl<number>(1, { validators: [Validators.required, Validators.min(1), Validators.max(10)] }),
      num_predictions: new FormControl<number>(1, { validators: [Validators.required] })
    });

    effect(() => {
      this.form.controls['modelID'].setValue(this.curr_id() || '');
    });
  }

  isContextValid = (c: AbstractControl): ValidationErrors | null => {
    if (!c.value || c.value.trim() === '') return { notvalid: true };
    const regex = /\[\.{1,}\]/;
    return regex.test(c.value) ? null : { notMasked: true };
  }

  focusTextarea() {
    const textarea = document.getElementById('inputText');
    if (textarea) {
      textarea.focus();
    }
  }

  clearForm() {
    this.form.get('text')?.reset('');
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

  generateSuggestions() {
    this.form.markAllAsTouched();
    if (this.form.invalid) {
      const textErrors = this.form.controls['text'].errors;
      const modelErrors = this.form.controls['modelID'].errors;
      const tokenErrors = this.form.controls['num_tokens'].errors;

      if (textErrors?.['required']) {
        this.showAlert('Il campo testo è obbligatorio', 'danger');
        return;
      }
      if (textErrors?.['notMasked']) {
        this.showAlert('Il testo non contiene la lacuna', 'warning');
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

    const { text, modelID, num_tokens, num_predictions } = this.form.getRawValue();
    this.onGenerate.emit({ text, modelID, num_tokens, num_predictions: Number(num_predictions) });
  }

  triggerToggleModels() {
    this.toggleModels.emit();
  }
}
