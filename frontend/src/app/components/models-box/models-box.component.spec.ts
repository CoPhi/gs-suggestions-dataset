import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of } from 'rxjs';

import { ModelsBoxComponent } from './models-box.component';
import { ApiService } from '../../services/api.service';

describe('ModelsBoxComponent', () => {
  let component: ModelsBoxComponent;
  let fixture: ComponentFixture<ModelsBoxComponent>;
  let apiService: ApiService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelsBoxComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelsBoxComponent);
    apiService = TestBed.inject(ApiService);
    fixture.componentRef.setInput('curr_id', null);
    fixture.componentRef.setInput('models', []);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should toggle RAG creation form and register a new RAG model', () => {
    expect(component.showAddRagForm()).toBeFalse();
    component.toggleAddRagForm();
    fixture.detectChanges();
    expect(component.showAddRagForm()).toBeTrue();

    spyOn(apiService, 'createModel').and.returnValue(of('67066928e4414eb1d8109bfa'));
    component.applyPreset('Qwen/Qwen2.5-3B-Instruct');
    component.submitRagModel();
    fixture.detectChanges();

    expect(apiService.createModel).toHaveBeenCalledWith({
      CHECKPOINT: 'Qwen/Qwen2.5-3B-Instruct',
      BACKEND: 'auto',
      TOP_K_PASSAGES: 5,
      COMPILED_PROGRAM_PATH: null,
      TYPE: 'RAG'
    });
    expect(component.models().length).toBe(1);
    expect(component.curr_id()).toBe('67066928e4414eb1d8109bfa');
    expect(component.showAddRagForm()).toBeFalse();
  });

  it('should set backend to neuro_lexical when a BERT preset is selected for RAG', () => {
    component.toggleAddRagForm();
    fixture.detectChanges();

    spyOn(apiService, 'createModel').and.returnValue(of('67066928e4414eb1d8109bfb'));
    component.applyPreset('CNR-ILC/gs-GreBerta');
    expect(component.ragForm.controls.backend.value).toBe('neuro_lexical');

    component.submitRagModel();
    fixture.detectChanges();

    expect(apiService.createModel).toHaveBeenCalledWith({
      CHECKPOINT: 'CNR-ILC/gs-GreBerta',
      BACKEND: 'neuro_lexical',
      TOP_K_PASSAGES: 5,
      COMPILED_PROGRAM_PATH: null,
      TYPE: 'RAG'
    });
  });

  it('should toggle Confronta mode inside the modal and select multiple models for comparison', () => {
    fixture.componentRef.setInput('models', [
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-ithaca-tlg', TYPE: 'Ithaca' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf3', LM_SCORE: 'LIDSTONE', GAMMA: 0.1, N: 3, CORPUS_NAMES: ['TLG'], TYPE: 'Ngrams' }
    ]);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const compareBtn = compiled.querySelector('.models-compare-mode-btn') as HTMLButtonElement;
    expect(compareBtn).toBeTruthy();
    expect(compareBtn.textContent).toContain('Confronta');
    expect(compiled.querySelector('.models-compare-toolbar')).toBeNull();

    // Attiva modalità confronto nella modale
    compareBtn.click();
    fixture.detectChanges();

    expect(component.isCompareMode()).toBeTrue();
    expect(compiled.querySelector('.models-compare-toolbar')).toBeTruthy();
    const cardBtns = compiled.querySelectorAll('.compare-card-toggle-btn');
    expect(cardBtns.length).toBe(3);

    (cardBtns[0] as HTMLButtonElement).click();
    (cardBtns[2] as HTMLButtonElement).click();
    fixture.detectChanges();

    expect(component.selectedModelIds()).toEqual([
      '67066928e4414eb1d8109bf1',
      '67066928e4414eb1d8109bf3'
    ]);
    expect(compiled.querySelector('.apply-models-compare-btn')?.textContent).toContain('Applica confronto (2)');

    // Seleziona tutti / Deseleziona tutti
    const selectAllBtn = compiled.querySelector('.models-compare-all-btn') as HTMLButtonElement;
    selectAllBtn.click();
    fixture.detectChanges();
    expect(component.selectedModelIds().length).toBe(3);
  });
});
