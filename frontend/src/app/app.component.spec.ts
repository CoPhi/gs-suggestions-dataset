import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of } from 'rxjs';
import { AppComponent } from './app.component';
import { ApiService } from './services/api.service';

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
});
