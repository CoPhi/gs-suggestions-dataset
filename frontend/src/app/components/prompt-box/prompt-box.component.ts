import { Component, ElementRef, HostListener, computed, effect, inject, input, model, output, signal } from '@angular/core';
import { AbstractControl, FormControl, FormGroup, ReactiveFormsModule, ValidationErrors, Validators } from '@angular/forms';
import { modelType } from '../../services/api.service';
import { NotificationService } from '../../services/notification.service';

@Component({
  selector: 'app-prompt-box',
  imports: [ReactiveFormsModule],
  templateUrl: './prompt-box.component.html',
  styleUrl: './prompt-box.component.css'
})
export class PromptBoxComponent {
  notifications = inject(NotificationService);
  private elementRef = inject(ElementRef);

  models = input.required<modelType[]>();
  curr_id = model.required<string | null>();
  isGenerating = input.required<boolean>();

  onGenerate = output<{ text: string, modelID: string, num_tokens: number, num_predictions: number }>();

  isDropdownOpen = signal<boolean>(false);

  selectedModel = computed(() => {
    return this.curr_id() ? this.models().find((model) => model._id === this.curr_id()) || null : null;
  });

  curr_type_model = computed(() => this.selectedModel()?.TYPE);

  selectedModelLabel = computed(() => {
    const m = this.selectedModel();
    if (!m) return null;
    if (m.TYPE === 'Ithaca') {
      const name = m.CHECKPOINT.includes('/') ? m.CHECKPOINT.split('/').pop() : m.CHECKPOINT;
      return `Ithaca: ${name}`;
    }
    if (m.TYPE === 'BERT') {
      const name = m.CHECKPOINT.includes('/') ? m.CHECKPOINT.split('/').pop() : m.CHECKPOINT;
      return `BERT: ${name}`;
    }
    return `N-grammi (${m.LM_SCORE})`;
  });

  getModelLabel(m: modelType): string {
    if (m.TYPE === 'Ithaca') {
      const name = m.CHECKPOINT.includes('/') ? m.CHECKPOINT.split('/').pop() : m.CHECKPOINT;
      return `Ithaca (${name})`;
    }
    if (m.TYPE === 'BERT') {
      const name = m.CHECKPOINT.includes('/') ? m.CHECKPOINT.split('/').pop() : m.CHECKPOINT;
      return `BERT (${name})`;
    }
    return `N-grammi (${m.LM_SCORE}${m.N ? `, n=${m.N}` : ''})`;
  }

  toggleDropdown(event?: Event) {
    if (event) {
      event.stopPropagation();
    }
    this.isDropdownOpen.update(open => !open);
  }

  closeDropdown() {
    this.isDropdownOpen.set(false);
  }

  selectModel(id: string) {
    this.curr_id.set(id);
    this.closeDropdown();
  }

  removeModelSelection() {
    this.curr_id.set(null);
    this.closeDropdown();
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent) {
    if (!this.isDropdownOpen()) return;
    const target = event.target as HTMLElement | null;
    if (target && !this.elementRef.nativeElement.querySelector('.model-dropdown')?.contains(target)) {
      this.closeDropdown();
    }
  }

  @HostListener('document:keydown.escape')
  onEscape() {
    if (this.isDropdownOpen()) {
      this.closeDropdown();
    }
  }

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
    const regex = /\[(\.{1,}|\-{1,})\]/;
    return regex.test(c.value) ? null : { notMasked: true };
  }

  focusTextarea(event?: MouseEvent) {
    if (event) {
      const target = event.target as HTMLElement | null;
      if (target?.closest('.prompt-toolbar')) {
        return;
      }
    }
    const textarea = document.getElementById('inputText');
    if (textarea) {
      textarea.focus();
    }
  }

  clearForm() {
    this.form.get('text')?.reset('');
  }

  showAlert(message: string, type: 'success' | 'danger' | 'warning' | 'info') {
    this.notifications.showAlert(message, type);
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

}
