import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PromptBoxComponent } from './prompt-box.component';
import { BERTModelInterface, IthacaModelInterface, NgramsModelInterface } from '../../services/api.service';

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

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PromptBoxComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(PromptBoxComponent);
    fixture.componentRef.setInput('curr_id', null);
    fixture.componentRef.setInput('models', [mockIthacaModel, mockBertModel, mockNgramsModel]);
    fixture.componentRef.setInput('isGenerating', false);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create prompt-box component', () => {
    expect(component).toBeTruthy();
  });

  it('should render model labels correctly in dropdown', () => {
    expect(component.getModelLabel(mockIthacaModel)).toBe('Ithaca (gs-ithaca-tlg)');
    expect(component.getModelLabel(mockBertModel)).toBe('BERT (Ancient-Greek-BERT)');
    expect(component.getModelLabel(mockNgramsModel)).toBe('N-grammi (LIDSTONE, n=3)');
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
    expect(dropdownItems.length).toBe(3);
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
});
