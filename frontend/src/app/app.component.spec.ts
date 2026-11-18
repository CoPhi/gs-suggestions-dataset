import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of, Subject } from 'rxjs';
import { AppComponent } from './app.component';
import { ApiService, SuggestionInterface } from './services/api.service';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it(`should have the 'gs-api' title`, () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app.title).toEqual('gs-api');
  });

  it('should render title', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Generatore di Suggerimenti');
  });

  it('should reset suggestions to null when closeSuggestions is called', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    app.suggestions.set([{ sentence: 'test', token_str: 'test', score: 1.0 }]);
    expect(app.suggestions()).not.toBeNull();
    app.closeSuggestions();
    expect(app.suggestions()).toBeNull();
  });

  it('should execute parallel requests on barrier and render horizontal suggestion boxes with model provenance', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    spyOn(api, 'generateSuggestion').and.callFake((modelId: string) => {
      if (modelId === '67066928e4414eb1d8109bf1') {
        return of([{ sentence: 'τειχεσιβλητα', token_str: 'λητα', score: 0.9 }]);
      }
      return of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλητα', score: 0.95 }]);
    });

    app.generateSuggestions({
      text: 'τειχεσι[....]',
      modelID: '67066928e4414eb1d8109bf1',
      modelIDs: ['67066928e4414eb1d8109bf1', '67066928e4414eb1d8109bf2'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    expect(app.modelSuggestionBoxes()?.length).toBe(2);
    expect(app.modelSuggestionBoxes()?.[0].modelLabel).toBe('gs-GreBerta + RAG');
    expect(app.modelSuggestionBoxes()?.[0].modelType).toBe('BERT + RAG');
    expect(app.modelSuggestionBoxes()?.[1].modelLabel).toBe('Qwen2.5-72B-Instruct');
    expect(app.modelSuggestionBoxes()?.[1].modelType).toBe('RAG');

    const compiled = fixture.nativeElement as HTMLElement;
    const horizontalCols = compiled.querySelectorAll('.horizontal-suggestion-col');
    expect(horizontalCols.length).toBe(2);
    const originBadges = compiled.querySelectorAll('.model-origin-badge');
    expect(originBadges.length).toBe(2);
    expect(originBadges[0].textContent).toContain('gs-GreBerta + RAG');
    expect(originBadges[1].textContent).toContain('Qwen2.5-72B-Instruct');
  });

  it('should show centered early consultation banner when a model has candidates while another is still pending on barrier', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const fastModel$ = new Subject<SuggestionInterface[]>();
    const slowModel$ = new Subject<SuggestionInterface[]>();

    spyOn(api, 'generateSuggestion').and.callFake((modelId: string) => {
      return modelId === '67066928e4414eb1d8109bf1' ? fastModel$.asObservable() : slowModel$.asObservable();
    });

    app.generateSuggestions({
      text: 'τειχεσι[....]',
      modelID: '67066928e4414eb1d8109bf1',
      modelIDs: ['67066928e4414eb1d8109bf1', '67066928e4414eb1d8109bf2'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(app.isGenerating()).toBeTrue();
    expect(compiled.querySelector('.early-consult-section')).toBeNull();

    // Il primo modello termina con candidati mentre il secondo è ancora sulla barriera
    fastModel$.next([{ sentence: 'τειχεσιβλητα', token_str: 'λητα', score: 0.9, tlg_attested: true }]);
    fastModel$.complete();
    fixture.detectChanges();

    expect(app.isGenerating()).toBeTrue();
    const consultSection = compiled.querySelector('.early-consult-section');
    expect(consultSection).not.toBeNull();
    expect(compiled.querySelector('.early-consult-text')?.textContent).toContain('(1/2)');
    const readyBadges = compiled.querySelectorAll('.early-ready-model-badge');
    expect(readyBadges.length).toBe(1);
    expect(readyBadges[0].textContent).toContain('gs-GreBerta + RAG');
    expect(compiled.querySelector('.early-consult-btn')?.textContent).toContain('Consulta');
    expect(compiled.querySelector('.multi-model-comparison-wrapper')).toBeNull();
  });

  it('should render ready box and pending placeholder column on Consulta click, then replace placeholder when slow model completes', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const slowModel$ = new Subject<SuggestionInterface[]>();

    spyOn(api, 'generateSuggestion').and.callFake((modelId: string) => {
      if (modelId === '67066928e4414eb1d8109bf1') {
        return of([{ sentence: 'τειχεσιβλητα', token_str: 'λητα', score: 0.9, tlg_attested: true }]);
      }
      return slowModel$.asObservable();
    });

    app.generateSuggestions({
      text: 'τειχεσι[....]',
      modelID: '67066928e4414eb1d8109bf1',
      modelIDs: ['67066928e4414eb1d8109bf1', '67066928e4414eb1d8109bf2'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const consultBtn = compiled.querySelector('.early-consult-btn') as HTMLButtonElement;
    expect(consultBtn).not.toBeNull();
    consultBtn.click();
    fixture.detectChanges();

    // Dopo il click su Consulta il banner sparisce e mostra colonna pronta + placeholder in attesa
    expect(compiled.querySelector('.early-consult-section')).toBeNull();
    expect(compiled.querySelector('.multi-model-comparison-wrapper')).not.toBeNull();
    const cols = compiled.querySelectorAll('.horizontal-suggestion-col');
    expect(cols.length).toBe(2);
    expect(cols[0].querySelector('app-suggests-box')).not.toBeNull();
    expect(cols[1].querySelector('.model-pending-placeholder-card')).not.toBeNull();
    expect(cols[1].querySelector('.model-pending-placeholder-card')?.textContent).toContain('Qwen2.5-72B-Instruct');
    expect(cols[1].querySelector('.spinner-border')).not.toBeNull();
    expect(app.isGenerating()).toBeTrue();

    // Al completamento del secondo modello, il placeholder viene sostituito dai suggerimenti reali
    slowModel$.next([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλητα', score: 0.95, tlg_attested: true }]);
    slowModel$.complete();
    fixture.detectChanges();

    expect(app.isGenerating()).toBeFalse();
    expect(compiled.querySelector('.model-pending-placeholder-card')).toBeNull();
    const updatedCols = compiled.querySelectorAll('.horizontal-suggestion-col');
    expect(updatedCols.length).toBe(2);
    expect(updatedCols[0].querySelector('app-suggests-box')).not.toBeNull();
    expect(updatedCols[1].querySelector('app-suggests-box')).not.toBeNull();
  });

  it('should not show early consultation banner when completed model returned 0 candidates while another is still pending', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const slowModel$ = new Subject<SuggestionInterface[]>();

    spyOn(api, 'generateSuggestion').and.callFake((modelId: string) => {
      if (modelId === '67066928e4414eb1d8109bf1') {
        return of([]);
      }
      return slowModel$.asObservable();
    });

    app.generateSuggestions({
      text: 'τειχεσι[....]',
      modelID: '67066928e4414eb1d8109bf1',
      modelIDs: ['67066928e4414eb1d8109bf1', '67066928e4414eb1d8109bf2'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(app.isGenerating()).toBeTrue();
    expect(compiled.querySelector('.early-consult-section')).toBeNull();

    // Quando la barriera termina completamente, apre comunque la vista di confronto finale
    slowModel$.next([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλητα', score: 0.95, tlg_attested: true }]);
    slowModel$.complete();
    fixture.detectChanges();

    expect(app.isGenerating()).toBeFalse();
    expect(compiled.querySelector('.early-consult-section')).toBeNull();
    expect(compiled.querySelector('.multi-model-comparison-wrapper')).not.toBeNull();
    const cols = compiled.querySelectorAll('.horizontal-suggestion-col');
    expect(cols.length).toBe(2);
    expect(cols[0].textContent).toContain('Nessun suggerimento disponibile');
    expect(cols[1].querySelector('app-suggests-box')).not.toBeNull();
  });

  it('should allow dismissing a pending placeholder column and resolve generating state if no pending models remain', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf3', LM_SCORE: 'srilm', GAMMA: null, N: 3, CORPUS_NAMES: ['tlg'], TYPE: 'Ngrams' }
    ]);

    const slowModel$ = new Subject<SuggestionInterface[]>();

    spyOn(api, 'generateSuggestion').and.callFake((modelId: string) => {
      if (modelId === '67066928e4414eb1d8109bf2') {
        return slowModel$.asObservable();
      }
      return of([{ sentence: 'τειχεσιβλητα', token_str: 'λητα', score: 0.9, tlg_attested: true }]);
    });

    app.generateSuggestions({
      text: 'τειχεσι[....]',
      modelID: '67066928e4414eb1d8109bf1',
      modelIDs: ['67066928e4414eb1d8109bf1', '67066928e4414eb1d8109bf2', '67066928e4414eb1d8109bf3'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.early-consult-text')?.textContent).toContain('(2/3)');
    (compiled.querySelector('.early-consult-btn') as HTMLButtonElement).click();
    fixture.detectChanges();

    const closePendingBtn = compiled.querySelector('.model-pending-placeholder-card .close-btn') as HTMLButtonElement;
    expect(closePendingBtn).not.toBeNull();
    closePendingBtn.click();
    fixture.detectChanges();

    expect(app.isGenerating()).toBeFalse();
    expect(app.modelSuggestionBoxes()?.length).toBe(2);
    expect(compiled.querySelector('.model-pending-placeholder-card')).toBeNull();
    expect(compiled.querySelectorAll('.horizontal-suggestion-col').length).toBe(2);
  });
});
