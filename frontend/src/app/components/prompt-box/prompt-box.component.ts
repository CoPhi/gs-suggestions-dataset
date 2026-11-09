import { Component, ElementRef, HostListener, computed, effect, inject, input, model, output, signal } from '@angular/core';
import { AbstractControl, FormControl, FormGroup, ReactiveFormsModule, ValidationErrors, Validators } from '@angular/forms';
import { isBertCheckpoint, modelType } from '../../services/api.service';
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

  onGenerate = output<{
    text: string;
    modelID: string;
    modelIDs?: string[];
    num_tokens: number;
    num_predictions: number;
    use_rag?: boolean;
  }>();

  isDropdownOpen = signal<boolean>(false);
  // Attiva/disattiva la modalità di confronto multi-modello con esposizione della checklist nel menù a tendina
  isCompareMode = signal<boolean>(false);
  // Permette di accostare esplicitamente un modello BERT al sistema RAG (modalità neuro-lessicale a priori)
  coupleBertWithRag = signal<boolean>(false);
  // Lista opzionale di modelli selezionati per l'inferenza parallela e il confronto orizzontale
  selectedModelIds = signal<string[]>([]);

  // Restituisce l'elenco ordinato degli ID effettivamente consultati (1 o più modelli)
  effectiveSelectedIds = computed<string[]>(() => {
    const multi = this.selectedModelIds().filter(id => this.models().some(m => m._id === id));
    if (multi.length > 0) {
      return multi;
    }
    const single = this.curr_id();
    return single ? [single] : [];
  });

  selectedModel = computed(() => {
    return this.curr_id() ? this.models().find((model) => model._id === this.curr_id()) || null : null;
  });

  curr_type_model = computed(() => this.selectedModel()?.TYPE);

  // Riconosce se un modello RAG registrato usa un checkpoint BERT (motore neuro-lessicale locale)
  isNeuroLexicalRag(m: modelType): boolean {
    return m.TYPE === 'RAG' && (m.BACKEND === 'neuro_lexical' || isBertCheckpoint(m.CHECKPOINT));
  }

  // Etichetta del modello selezionato: esplicita '+ RAG' o il conteggio dei modelli in confronto parallelo
  selectedModelLabel = computed(() => {
    const ids = this.effectiveSelectedIds();
    if (ids.length > 1) {
      return `Confronto (${ids.length} modelli)`;
    }
    const m = this.selectedModel();
    if (!m) return null;
    const base = this.getModelLabel(m);
    if (m.TYPE === 'BERT' && this.coupleBertWithRag()) {
      return `${base} + RAG`;
    }
    return base;
  });

  getModelLabel(m: modelType): string {
    if (m.TYPE === 'RAG' || m.TYPE === 'Ithaca' || m.TYPE === 'BERT') {
      const shortName = (m.CHECKPOINT.includes('/') ? m.CHECKPOINT.split('/').pop() : m.CHECKPOINT) || m.CHECKPOINT;
      if (this.isNeuroLexicalRag(m)) {
        return `${shortName} + RAG`;
      }
      return shortName;
    }
    return `N-grammi${m.N ? ` (n=${m.N})` : ''}`;
  }

  getModelBadge(m: modelType): string {
    if (this.isNeuroLexicalRag(m)) {
      return 'BERT + RAG';
    }
    if (m.TYPE === 'RAG') {
      return 'RAG';
    }
    if (m.TYPE === 'BERT') {
      return 'BERT';
    }
    if (m.TYPE === 'Ithaca') {
      return 'Ithaca';
    }
    return 'N-grammi';
  }

  toggleBertRag(event?: Event) {
    if (event) {
      event.stopPropagation();
    }
    this.coupleBertWithRag.update(v => !v);
  }

  selectBertWithRag(id: string, event?: Event) {
    if (event) {
      event.stopPropagation();
    }
    this.curr_id.set(id);
    this.selectedModelIds.set([id]);
    this.coupleBertWithRag.set(true);
    this.closeDropdown();
  }

  toggleCompareMode(event?: Event) {
    if (event) {
      event.stopPropagation();
    }
    const nextState = !this.isCompareMode();
    this.isCompareMode.set(nextState);
    if (!nextState) {
      // Uscendo dalla modalità confronto si ripristina la selezione singola sul modello primario corrente
      const primary = this.curr_id();
      this.selectedModelIds.set(primary ? [primary] : []);
    } else if (this.selectedModelIds().length === 0 && this.curr_id()) {
      this.selectedModelIds.set([this.curr_id()!]);
    }
  }

  isModelInComparison(id: string): boolean {
    return this.effectiveSelectedIds().includes(id);
  }

  toggleModelComparison(id: string, event?: Event) {
    if (event) {
      event.stopPropagation();
    }
    this.isCompareMode.set(true);
    const current = [...this.effectiveSelectedIds()];
    const idx = current.indexOf(id);
    if (idx >= 0) {
      current.splice(idx, 1);
    } else {
      current.push(id);
    }
    this.selectedModelIds.set(current);
    this.curr_id.set(current.length > 0 ? current[0] : null);
  }

  selectAllModelsForComparison(event?: Event) {
    if (event) {
      event.stopPropagation();
    }
    this.isCompareMode.set(true);
    const allIds = this.models().map(m => m._id);
    if (this.effectiveSelectedIds().length === allIds.length) {
      this.selectedModelIds.set([]);
      this.curr_id.set(null);
    } else {
      this.selectedModelIds.set(allIds);
      this.curr_id.set(allIds[0] || null);
    }
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
    if (this.isCompareMode()) {
      this.toggleModelComparison(id);
      return;
    }
    const chosen = this.models().find(m => m._id === id);
    if (chosen && chosen.TYPE !== 'BERT') {
      this.coupleBertWithRag.set(false);
    }
    this.curr_id.set(id);
    this.selectedModelIds.set([id]);
    this.closeDropdown();
  }

  removeModelSelection() {
    this.curr_id.set(null);
    this.selectedModelIds.set([]);
    this.isCompareMode.set(false);
    this.coupleBertWithRag.set(false);
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
    // Rifiuta stringhe contenenti punti sublineari (U+0323 combinatorio o decomposto da NFD)
    if (typeof c.value === 'string' && c.value.normalize('NFD').includes('\u0323')) {
      return { unknownCharacters: true };
    }
    // Riconosce la lacuna delimitata da asterischi (es. *...* o *---*) oppure da parentesi quadre ([...])
    const regex = /(\*(\.{1,}|\-{1,})\*|\[(\.{1,}|\-{1,})\])/;
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
      if (textErrors?.['unknownCharacters']) {
        this.showAlert('la stringa contiene carattere sconosciuti', 'danger');
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
    const ids = this.effectiveSelectedIds();
    // Pre-processing frontend della lacuna: converte la rappresentazione con asterischi (es. *...*)
    // nel formato canonico dell'API ([...]) senza toccare la gestione lato backend
    const processedText = (text || '').replace(/\*(\.{1,}|\-{1,})\*/g, '[$1]');

    const payload: {
      text: string;
      modelID: string;
      modelIDs?: string[];
      num_tokens: number;
      num_predictions: number;
      use_rag?: boolean;
    } = {
      text: processedText,
      modelID,
      num_tokens,
      num_predictions: Number(num_predictions)
    };
    if (ids.length > 1) {
      payload.modelIDs = ids;
    }
    if ((this.curr_type_model() === 'BERT' || ids.some(id => this.models().find(m => m._id === id)?.TYPE === 'BERT')) && this.coupleBertWithRag()) {
      payload.use_rag = true;
    }
    this.onGenerate.emit(payload);
  }

}
