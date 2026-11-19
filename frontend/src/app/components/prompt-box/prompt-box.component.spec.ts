import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PromptBoxComponent } from './prompt-box.component';
import { BERTModelInterface, IthacaModelInterface, NgramsModelInterface, RAGModelInterface } from '../../services/api.service';

describe('PromptBoxComponent', () => {
  let component: PromptBoxComponent;
  let fixture: ComponentFixture<PromptBoxComponent>;

  const mockIthacaModel: IthacaModelInterface = {
    _id: '67066928e4414eb1d8109bf1',
    CHECKPOINT: 'CNR-ILC/gs-ithaca-tlg',
    TYPE: 'Ithaca'
  };

  const mockBertModel: BERTModelInterface = {
    _id: '67066928e4414eb1d8109bf2',
    CHECKPOINT: 'pranaydeeps/Ancient-Greek-BERT',
    TYPE: 'BERT'
  };

  const mockNgramsModel: NgramsModelInterface = {
    _id: '67066928e4414eb1d8109bf3',
    LM_SCORE: 'LIDSTONE',
    GAMMA: 0.1,
    N: 3,
    CORPUS_NAMES: ['TLG'],
    TYPE: 'Ngrams'
  };

  const mockRagModel: RAGModelInterface = {
    _id: '67066928e4414eb1d8109bf4',
    CHECKPOINT: 'Qwen/Qwen2.5-7B-Instruct',
    BACKEND: 'auto',
    TOP_K_PASSAGES: 5,
    TYPE: 'RAG'
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PromptBoxComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(PromptBoxComponent);
    fixture.componentRef.setInput('curr_id', null);
    fixture.componentRef.setInput('models', [mockIthacaModel, mockBertModel, mockNgramsModel, mockRagModel]);
    fixture.componentRef.setInput('isGenerating', false);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create prompt-box component', () => {
    expect(component).toBeTruthy();
  });

  it('should render model labels correctly in dropdown', () => {
    expect(component.getModelLabel(mockIthacaModel)).toBe('gs-ithaca-tlg');
    expect(component.getModelLabel(mockBertModel)).toBe('Ancient-Greek-BERT');
    expect(component.getModelLabel(mockNgramsModel)).toBe('N-grammi (n=3)');
    expect(component.getModelLabel(mockRagModel)).toBe('Qwen2.5-7B-Instruct');
  });

  it('should update curr_id when a model is selected', () => {
    component.selectModel(mockBertModel._id);
    fixture.detectChanges();
    expect(component.curr_id()).toBe(mockBertModel._id);
    expect(component.selectedModel()?._id).toBe(mockBertModel._id);
    expect(component.form.get('modelID')?.value).toBe(mockBertModel._id);
  });

  it('should render the dropdown trigger and expand button in DOM', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const trigger = compiled.querySelector('#modelSelectDropdown');
    const expandBtn = compiled.querySelector('.model-expand-btn');

    expect(trigger).toBeTruthy();
    expect(expandBtn).toBeTruthy();
    expect(expandBtn?.getAttribute('data-bs-target')).toBe('#modelsModal');
  });

  it('should toggle dropdown on trigger click and close when a model is selected', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const trigger = compiled.querySelector('#modelSelectDropdown') as HTMLButtonElement;
    const dropdownMenu = compiled.querySelector('.model-dropdown-menu') as HTMLElement;

    expect(component.isDropdownOpen()).toBeFalse();
    expect(dropdownMenu.classList.contains('show')).toBeFalse();

    // Click to open
    trigger.click();
    fixture.detectChanges();
    expect(component.isDropdownOpen()).toBeTrue();
    expect(dropdownMenu.classList.contains('show')).toBeTrue();

    // Select a model from items
    const firstItem = compiled.querySelector('.model-dropdown-item') as HTMLButtonElement;
    expect(firstItem).toBeTruthy();
    firstItem.click();
    fixture.detectChanges();

    expect(component.isDropdownOpen()).toBeFalse();
    expect(dropdownMenu.classList.contains('show')).toBeFalse();
    expect(component.curr_id()).toBe(mockIthacaModel._id);
  });

  it('should not contain model icons in the dropdown trigger or menu items', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const trigger = compiled.querySelector('#modelSelectDropdown');
    expect(trigger?.querySelectorAll('i').length).toBe(0);

    const dropdownItems = compiled.querySelectorAll('.model-dropdown-item');
    expect(dropdownItems.length).toBe(4);
    dropdownItems.forEach((item) => {
      // Model items should not contain model icons
      expect(item.querySelectorAll('i').length).toBe(0);
    });
  });

  it('should close dropdown on escape and on outside click', () => {
    component.isDropdownOpen.set(true);
    fixture.detectChanges();

    // Escape key
    component.onEscape();
    expect(component.isDropdownOpen()).toBeFalse();

    // Reopen and test outside click
    component.isDropdownOpen.set(true);
    fixture.detectChanges();

    const outsideElement = document.createElement('div');
    document.body.appendChild(outsideElement);
    const mouseEvent = new MouseEvent('click', { bubbles: true });
    Object.defineProperty(mouseEvent, 'target', { value: outsideElement });
    component.onDocumentClick(mouseEvent);
    expect(component.isDropdownOpen()).toBeFalse();
    document.body.removeChild(outsideElement);
  });

  it('should explicitly label BERT + RAG neuro-lexical models and allow coupling BERT with RAG', () => {
    const neuroLexicalRagModel: RAGModelInterface = {
      _id: '67066928e4414eb1d8109bf5',
      CHECKPOINT: 'CNR-ILC/gs-GreBerta',
      BACKEND: 'neuro_lexical',
      TOP_K_PASSAGES: 5,
      TYPE: 'RAG'
    };
    expect(component.getModelLabel(neuroLexicalRagModel)).toBe('gs-GreBerta + RAG');
    expect(component.getModelBadge(neuroLexicalRagModel)).toBe('BERT + RAG');

    // Accostamento di un modello BERT puro al sistema RAG tramite selectBertWithRag
    component.selectBertWithRag(mockBertModel._id);
    fixture.detectChanges();
    expect(component.curr_id()).toBe(mockBertModel._id);
    expect(component.coupleBertWithRag()).toBeTrue();
    expect(component.selectedModelLabel()).toBe('Ancient-Greek-BERT + RAG');
  });

  it('should expose the comparison checklist only after pressing the Confronta button', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    expect(component.isCompareMode()).toBeFalse();
    expect(compiled.querySelectorAll('.model-compare-checkbox').length).toBe(0);

    const compareBtn = compiled.querySelector('.compare-mode-btn') as HTMLButtonElement;
    expect(compareBtn).toBeTruthy();
    expect(compareBtn.textContent?.trim()).toBe('Confronta');

    compareBtn.click();
    fixture.detectChanges();

    expect(component.isCompareMode()).toBeTrue();
    expect(compiled.querySelectorAll('.model-compare-checkbox').length).toBe(4);

    component.toggleModelComparison(mockBertModel._id);
    component.toggleModelComparison(mockRagModel._id);
    fixture.detectChanges();

    expect(component.effectiveSelectedIds()).toEqual([mockBertModel._id, mockRagModel._id]);
    expect(component.selectedModelLabel()).toBe('Confronto (2 modelli)');
    expect(component.isModelInComparison(mockBertModel._id)).toBeTrue();
    expect(component.isModelInComparison(mockRagModel._id)).toBeTrue();
  });

  it('should accept context with asterisk lacuna (*...*) and convert to brackets on generate', () => {
    spyOn(component.onGenerate, 'emit');
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ κατ\' αὐτὰ *...* ὑπὲρ');
    component.form.controls['num_tokens'].setValue(1);
    component.form.controls['num_predictions'].setValue(5);
    fixture.detectChanges();

    expect(component.form.controls['text'].valid).toBeTrue();
    expect(component.form.valid).toBeTrue();

    component.generateSuggestions();

    expect(component.onGenerate.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      text: 'τὰ κατ\' αὐτὰ [...] ὑπὲρ',
      modelID: mockBertModel._id,
      num_tokens: 1,
      num_predictions: 5
    }));
  });

  it('should reject text containing sublinear dots with unknownCharacters error and alert message', () => {
    spyOn(component, 'showAlert');
    // Stringa con punto sublineare U+0323 (es. α̣)
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ κατ\' α\u0323ὐτὰ *...* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('unknownCharacters')).toBeTrue();
    expect(component.form.valid).toBeFalse();

    component.generateSuggestions();

    expect(component.showAlert).toHaveBeenCalledWith('la stringa contiene carattere sconosciuti', 'danger');
  });

  it('should reject text containing editor additions (<>) with unknownCharacters error', () => {
    spyOn(component, 'showAlert');
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ <κατ\'> αὐτὰ *...* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('unknownCharacters')).toBeTrue();
    expect(component.form.valid).toBeFalse();

    component.generateSuggestions();

    expect(component.showAlert).toHaveBeenCalledWith('la stringa contiene carattere sconosciuti', 'danger');
  });

  it('should reject text containing editor removals ({}) with unknownCharacters error', () => {
    spyOn(component, 'showAlert');
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ {κατ\'} αὐτὰ *...* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('unknownCharacters')).toBeTrue();
    expect(component.form.valid).toBeFalse();

    component.generateSuggestions();

    expect(component.showAlert).toHaveBeenCalledWith('la stringa contiene carattere sconosciuti', 'danger');
  });

  it('should accept contextual square-bracket lacuna and convert it to loose dots', () => {
    spyOn(component.onGenerate, 'emit');
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ κατ\' [...] αὐτὰ *...* ὑπὲρ');
    component.form.controls['num_tokens'].setValue(1);
    component.form.controls['num_predictions'].setValue(5);
    fixture.detectChanges();

    expect(component.form.controls['text'].valid).toBeTrue();
    expect(component.form.valid).toBeTrue();

    component.generateSuggestions();

    // La lacuna contestuale diventa puntini sciolti (contesto non vincolante),
    // mentre solo quella target con asterischi diventa `[...]` per l'API.
    expect(component.onGenerate.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      text: 'τὰ κατ\' ... αὐτὰ [...] ὑπὲρ',
      modelID: mockBertModel._id
    }));
  });

  it('should reject contextual hyphen lacuna ([---]) with unknownCharacters error', () => {
    component.form.controls['text'].setValue('τὰ [---] αὐτὰ *...* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('unknownCharacters')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should reject text with only contextual square-bracket lacuna and no target', () => {
    component.form.controls['text'].setValue('τὰ κατ\' [...] αὐτὰ ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('notMasked')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should reject square brackets not representing a contextual lacuna', () => {
    component.form.controls['text'].setValue('τὰ [κατ\'] αὐτὰ *...* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('unknownCharacters')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should reject text containing latin characters with unknownCharacters error', () => {
    component.form.controls['text'].setValue('τὰ κατ\' text *...* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('unknownCharacters')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should reject text containing digits or arbitrary symbols with unknownCharacters error', () => {
    component.form.controls['text'].setValue('τὰ 123 # *...* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('unknownCharacters')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should reject hyphen target lacuna (*---*) with notMasked error', () => {
    component.form.controls['text'].setValue('«τὰ, κατ\'· αὐτὰ;» *---* ὑπὲρ.');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('notMasked')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should extract prefix characters from inside the lacuna and move them outside', () => {
    spyOn(component.onGenerate, 'emit');
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ κατ\' *α...* ὑπὲρ');
    component.form.controls['num_tokens'].setValue(1);
    component.form.controls['num_predictions'].setValue(5);
    fixture.detectChanges();

    expect(component.form.controls['text'].valid).toBeTrue();

    component.generateSuggestions();

    // Verifica che 'α' sia stato spostato prima della lacuna sia nel form sia nel payload
    expect(component.form.controls['text'].value).toBe('τὰ κατ\' α*...* ὑπὲρ');
    expect(component.onGenerate.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      text: 'τὰ κατ\' α[...] ὑπὲρ'
    }));
  });

  it('should extract suffix characters from inside the lacuna and move them outside', () => {
    spyOn(component.onGenerate, 'emit');
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ κατ\' *...β* ὑπὲρ');
    component.form.controls['num_tokens'].setValue(1);
    component.form.controls['num_predictions'].setValue(5);
    fixture.detectChanges();

    expect(component.form.controls['text'].valid).toBeTrue();

    component.generateSuggestions();

    expect(component.form.controls['text'].value).toBe('τὰ κατ\' *...*β ὑπὲρ');
    expect(component.onGenerate.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      text: 'τὰ κατ\' [...]β ὑπὲρ'
    }));
  });

  it('should extract both prefix and suffix characters from inside the lacuna (*κα...τος*)', () => {
    spyOn(component.onGenerate, 'emit');
    component.form.controls['modelID'].setValue(mockBertModel._id);
    component.curr_id.set(mockBertModel._id);
    component.form.controls['text'].setValue('τὰ *κα...τος* ὑπὲρ');
    component.form.controls['num_tokens'].setValue(1);
    component.form.controls['num_predictions'].setValue(5);
    fixture.detectChanges();

    expect(component.form.controls['text'].valid).toBeTrue();

    component.generateSuggestions();

    expect(component.form.controls['text'].value).toBe('τὰ κα*...*τος ὑπὲρ');
    expect(component.onGenerate.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      text: 'τὰ κα[...]τος ὑπὲρ'
    }));
  });

  it('should move boundary characters outside on textarea blur', () => {
    component.form.controls['text'].setValue('τὰ κατ\' *α...* ὑπὲρ');
    component.isFocused = true;

    component.handleBlur();

    expect(component.isFocused).toBeFalse();
    expect(component.form.controls['text'].value).toBe('τὰ κατ\' α*...* ὑπὲρ');
  });

  it('should reject lacuna with letters trapped in the middle of dots (*..α..*) with notMasked error', () => {
    component.form.controls['text'].setValue('τὰ κατ\' *..α..* ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('notMasked')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should reject text without lacuna with notMasked error', () => {
    component.form.controls['text'].setValue('τὰ κατ\' αὐτὰ ὑπὲρ');
    fixture.detectChanges();

    expect(component.form.controls['text'].hasError('notMasked')).toBeTrue();
    expect(component.form.valid).toBeFalse();
  });

  it('should support refinement mode with natural language feedback and rejected candidate chips', () => {
    const refineSpy = spyOn(component.onRefineSubmit, 'emit');
    const exitSpy = spyOn(component.onExitRefinement, 'emit');
    const removeSpy = spyOn(component.onRemoveRejectedCandidate, 'emit');

    component.setSingleModelSelection(mockRagModel._id);
    fixture.componentRef.setInput('isRefinementMode', true);
    fixture.componentRef.setInput('refinementContextText', 'τειχεσι[...]');
    fixture.componentRef.setInput('refinementRejectedCandidates', ['πλητα']);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.refinement-mode-banner')).toBeTruthy();
    expect(compiled.querySelector('.refinement-context-preview')?.textContent).toContain('τειχεσι[...]');
    expect(compiled.querySelectorAll('.rejected-chip').length).toBe(1);

    // Accetta indicazioni filologiche in linguaggio naturale (caratteri latini) senza errori di validazione greca
    component.form.controls['text'].setValue('Preferisci forme omeriche al vocativo');
    fixture.detectChanges();
    expect(component.canSubmit()).toBeTrue();

    component.generateSuggestions();
    fixture.detectChanges();

    expect(refineSpy).toHaveBeenCalledWith(jasmine.objectContaining({
      human_feedback: 'Preferisci forme omeriche al vocativo',
      rejected_candidates: ['πλητα'],
      modelID: mockRagModel._id
    }));
    // Dopo l'invio del raffinamento, la textarea viene svuotata
    expect(component.form.controls['text'].value).toBe('');

    // Anche con feedback vuoto resta inviabile finché c'è almeno un candidato scartato
    expect(component.canSubmit()).toBeTrue();

    // Rimozione chip escluso e uscita dalla modalità raffinamento
    const chipCloseBtn = compiled.querySelector('.rejected-chip .btn-close') as HTMLButtonElement;
    chipCloseBtn.click();
    expect(removeSpy).toHaveBeenCalledWith('πλητα');

    const exitBtn = compiled.querySelector('.exit-refinement-btn') as HTMLButtonElement;
    exitBtn.click();
    expect(exitSpy).toHaveBeenCalled();
  });

  it('should pre-select single model and restore prompt text via helper methods', () => {
    component.toggleCompareMode();
    component.toggleModelComparison(mockIthacaModel._id);
    component.toggleModelComparison(mockBertModel._id);
    expect(component.isCompareMode()).toBeTrue();

    component.setSingleModelSelection(mockBertModel._id, true);
    expect(component.isCompareMode()).toBeFalse();
    expect(component.curr_id()).toBe(mockBertModel._id);
    expect(component.coupleBertWithRag()).toBeTrue();

    component.setPromptText('τὰ κατ\' αὐτὰ *...* ὑπὲρ');
    expect(component.form.controls['text'].value).toBe('τὰ κατ\' αὐτὰ *...* ὑπὲρ');
  });

  it('should render "per ulteriori approfondimenti" link and toggle the prompt engineering manual modal', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const toggleLink = compiled.querySelector('.prompt-manual-toggle-link') as HTMLButtonElement;
    expect(toggleLink).toBeTruthy();
    expect(toggleLink.textContent?.trim()).toBe('per ulteriori approfondimenti');
    expect(compiled.querySelector('#promptManualModal')).toBeNull();

    // Apertura al click sul link
    toggleLink.click();
    fixture.detectChanges();

    const modal = compiled.querySelector('#promptManualModal') as HTMLElement;
    expect(modal).toBeTruthy();
    expect(component.isManualModalOpen()).toBeTrue();
    expect(modal.textContent).toContain('Manuale di Prompt Engineering Filologico');
    expect(modal.textContent).toContain('*...*');
    expect(modal.textContent).toContain('[...]');
    expect(modal.textContent).toContain('U+0323');
    expect(modal.textContent).toContain('Leiden+');

    // Chiusura tramite Escape
    component.onEscape();
    fixture.detectChanges();
    expect(component.isManualModalOpen()).toBeFalse();
    expect(compiled.querySelector('#promptManualModal')).toBeNull();

    // Riapertura tramite link e chiusura tramite pulsante footer e backdrop
    toggleLink.click();
    fixture.detectChanges();
    expect(compiled.querySelector('#promptManualModal')).toBeTruthy();

    const closeFooterBtn = compiled.querySelector('.close-manual-btn') as HTMLButtonElement;
    closeFooterBtn.click();
    fixture.detectChanges();
    expect(compiled.querySelector('#promptManualModal')).toBeNull();

    toggleLink.click();
    fixture.detectChanges();
    const reopenedModal = compiled.querySelector('#promptManualModal') as HTMLElement;
    reopenedModal.click();
    fixture.detectChanges();
    expect(compiled.querySelector('#promptManualModal')).toBeNull();
  });
});
