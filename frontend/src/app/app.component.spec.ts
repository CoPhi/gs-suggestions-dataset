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
    expect(compiled.querySelector('.initial-screen-layout')).not.toBeNull();
  });

  it('should reset suggestions and return to initial screen when closeSuggestions is called', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    app.suggestions.set([{ sentence: 'test', token_str: 'test', score: 1.0 }]);
    expect(app.suggestions()).not.toBeNull();
    app.closeSuggestions();
    expect(app.suggestions()).toBeNull();
    expect(app.conversationTurns().length).toBe(0);
  });

  it('should transition from initial screen to loading state and then to suggestions for a single model', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const model$ = new Subject<SuggestionInterface[]>();
    spyOn(api, 'generateSuggestion').and.returnValue(model$.asObservable());

    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: '67066928e4414eb1d8109bf1',
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.initial-screen-layout')).toBeNull();
    expect(compiled.querySelector('.conversation-screen-layout')).not.toBeNull();
    expect(compiled.querySelector('.bottom-prompt-dock')).not.toBeNull();
    expect(compiled.querySelector('.turn-lacuna-text')?.textContent).toContain('τειχεσι[....]');
    expect(compiled.querySelector('.turn-loading-state')?.textContent).toContain('loading...');

    model$.next([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }]);
    model$.complete();
    fixture.detectChanges();

    expect(compiled.querySelector('.turn-loading-state')).toBeNull();
    expect(compiled.querySelectorAll('app-suggest').length).toBe(1);
    expect(compiled.querySelector('.model-origin-badge')?.textContent).toContain('gs-GreBerta + RAG');
  });

  it('should unlock comparison view automatically on first ready model, show scrollable model tabs with live spinners, and switch tabs', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const slowModel$ = new Subject<SuggestionInterface[]>();
    const fastModel$ = new Subject<SuggestionInterface[]>();

    spyOn(api, 'generateSuggestion').and.callFake((modelId: string) => {
      return modelId === '67066928e4414eb1d8109bf1' ? slowModel$.asObservable() : fastModel$.asObservable();
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
    expect(compiled.querySelector('.turn-loading-state')).not.toBeNull();
    expect(compiled.querySelector('.scrollable-models-bar')).toBeNull();

    // Il secondo modello risponde per primo: sblocca subito la schermata confronto selezionando il suo tab
    fastModel$.next([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.95, tlg_attested: true }]);
    fastModel$.complete();
    fixture.detectChanges();

    expect(app.isGenerating()).toBeTrue();
    expect(compiled.querySelector('.turn-loading-state')).toBeNull();
    expect(compiled.querySelector('.scrollable-models-bar')).not.toBeNull();
    const tabs = compiled.querySelectorAll('.model-comparison-tab');
    expect(tabs.length).toBe(2);
    // Il primo tab è ancora pending con mini-spinner e bottone disabilitato; il secondo è attivo
    expect(tabs[0].classList.contains('pending')).toBeTrue();
    expect(tabs[0].querySelector('.model-tab-spinner')).not.toBeNull();
    expect((tabs[0].querySelector('.model-tab-select-btn') as HTMLButtonElement).disabled).toBeTrue();
    expect(tabs[1].classList.contains('active')).toBeTrue();
    expect(compiled.textContent).toContain('βλῆτα');

    // Al completamento del modello lento, sparisce lo spinner dal suo tab e diventa cliccabile
    slowModel$.next([{ sentence: 'τειχεσιπλητα', token_str: 'πλητα', score: 0.89, tlg_attested: true }]);
    slowModel$.complete();
    fixture.detectChanges();

    expect(app.isGenerating()).toBeFalse();
    expect(compiled.querySelector('.model-tab-spinner')).toBeNull();
    const firstTabBtn = compiled.querySelectorAll('.model-tab-select-btn')[0] as HTMLButtonElement;
    expect(firstTabBtn.disabled).toBeFalse();

    // Cliccando sul primo tab si commutano i suggerimenti visualizzati
    firstTabBtn.click();
    fixture.detectChanges();

    expect(compiled.querySelectorAll('.model-comparison-tab')[0].classList.contains('active')).toBeTrue();
    expect(compiled.textContent).toContain('πλητα');
  });

  it('should allow dismissing a pending model tab from the scrollable bar and resolve generating state when no pending models remain', () => {
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
    expect(app.isGenerating()).toBeTrue();
    const tabs = compiled.querySelectorAll('.model-comparison-tab');
    expect(tabs.length).toBe(3);

    const closePendingTabBtn = tabs[1].querySelector('.close-model-tab-btn') as HTMLButtonElement;
    expect(closePendingTabBtn).not.toBeNull();
    closePendingTabBtn.click();
    fixture.detectChanges();

    expect(app.isGenerating()).toBeFalse();
    expect(app.conversationTurns()[0].modelBoxes.length).toBe(2);
    expect(compiled.querySelectorAll('.model-comparison-tab').length).toBe(2);
  });

  it('should append multiple turns in conversation stream, support text reuse, and return to initial screen when last turn is closed', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);
    spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.95, tlg_attested: true }])
    );

    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: '67066928e4414eb1d8109bf1',
      num_tokens: 1,
      num_predictions: 5
    });
    app.generateSuggestions({
      text: 'ἄλλο *...* κείμενον',
      modelID: '67066928e4414eb1d8109bf1',
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const turns = compiled.querySelectorAll('.conversation-turn');
    expect(turns.length).toBe(2);

    // Riutilizzo del testo del primo turno nella barra in basso (riconversione a *....*)
    const reuseBtn = turns[0].querySelector('.reuse-turn-text-btn') as HTMLButtonElement;
    reuseBtn.click();
    fixture.detectChanges();
    expect(app.promptBox()?.form.controls['text'].value).toBe('τειχεσι*....*');

    // Chiusura di entrambi i turni -> ritorno alla schermata iniziale
    const closeTurnBtns = compiled.querySelectorAll('.close-turn-btn');
    (closeTurnBtns[0] as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(app.conversationTurns().length).toBe(1);

    (compiled.querySelector('.close-turn-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(app.conversationTurns().length).toBe(0);
    expect(compiled.querySelector('.initial-screen-layout')).not.toBeNull();
  });

  it('should toggle refinement mode on a turn, pre-select active model, accumulate rejections across tabs, and append a refinement turn', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const genSpy = spyOn(api, 'generateSuggestion').and.callFake((modelId: string) => {
      if (modelId === '67066928e4414eb1d8109bf1') {
        return of([{ sentence: 'τειχεσιλητα', token_str: 'λητα', score: 0.9, tlg_attested: true }]);
      }
      return of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.95, tlg_attested: true }]);
    });
    spyOn(api, 'submitFeedback').and.returnValue(of({ id: 'fb-1', status: 'saved' }));

    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: '67066928e4414eb1d8109bf1',
      modelIDs: ['67066928e4414eb1d8109bf1', '67066928e4414eb1d8109bf2'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const firstTurn = app.conversationTurns()[0];

    // Scarto su modello 1
    (compiled.querySelector('.feedback-reject-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(app.conversationTurns()[0].turnRejectedCandidates).toEqual(['λητα']);

    // Cambio tab su modello 2 e scarto anche su modello 2 -> accumulato sul turno
    (compiled.querySelectorAll('.model-tab-select-btn')[1] as HTMLButtonElement).click();
    fixture.detectChanges();
    (compiled.querySelector('.feedback-reject-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(app.conversationTurns()[0].turnRejectedCandidates).toEqual(['λητα', 'βλῆτα']);

    // Click sul toggle "Raffina": attiva modalità raffinamento e preseleziona il modello del tab attivo (modello 2)
    (compiled.querySelector('.hitl-toggle-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(app.activeRefinementTurnId()).toBe(firstTurn.turnId);
    expect(app.curr_id()).toBe('67066928e4414eb1d8109bf2');
    expect(compiled.querySelector('.refinement-mode-banner')).not.toBeNull();

    // Invio del raffinamento dalla barra in basso
    app.promptBox()!.form.controls['text'].setValue('Solo epiteti epici attestati');
    fixture.detectChanges();
    app.promptBox()!.generateSuggestions();
    fixture.detectChanges();

    expect(app.conversationTurns().length).toBe(2);
    const secondTurn = app.conversationTurns()[1];
    expect(secondTurn.isRefinement).toBeTrue();
    expect(secondTurn.sessionId).toBe(firstTurn.sessionId);
    expect(secondTurn.appliedHumanFeedback).toBe('Solo epiteti epici attestati');
    expect(secondTurn.appliedRejectedCandidates).toEqual(['λητα', 'βλῆτα']);
    expect(app.activeRefinementTurnId()).toBeNull();
    expect(compiled.querySelectorAll('.conversation-turn')[1].querySelector('.turn-refinement-constraints')?.textContent)
      .toContain('Solo epiteti epici attestati');
    expect(genSpy.calls.mostRecent().args[0]).toBe('67066928e4414eb1d8109bf2');
    expect(genSpy.calls.mostRecent().args[4]?.session_id).toBe(firstTurn.sessionId);
  });

  it('should generate a shared per-lacuna session_id across multi-model calls, HITL refinements, and feedback', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: 'm1', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: 'm2', CHECKPOINT: 'meta-llama/Llama-3.3-70B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const genSpy = spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλητα', token_str: 'λητα', score: 0.9, tlg_attested: true }])
    );
    const fbSpy = spyOn(api, 'submitFeedback').and.returnValue(of({ id: 'fb-1', status: 'saved' }));

    // 1. Nuova ricerca multi-modello: stesso session_id per tutte le colonne
    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: 'm1',
      modelIDs: ['m1', 'm2'],
      num_tokens: 1,
      num_predictions: 5
    });

    const firstSessionId = app.currentSessionId();
    expect(firstSessionId).toBeTruthy();
    expect(genSpy.calls.count()).toBe(2);
    expect(genSpy.calls.argsFor(0)[4]?.session_id).toBe(firstSessionId!);
    expect(genSpy.calls.argsFor(1)[4]?.session_id).toBe(firstSessionId!);

    // 2. Feedback: propaga lo stesso session_id su MongoDB
    app.recordFeedback(
      {
        token_str: 'λητα',
        sentence: 'τειχεσιβλητα',
        is_accepted: true,
        score: 0.9,
        rank: 1
      },
      'm1'
    );
    expect(fbSpy).toHaveBeenCalled();
    expect(fbSpy.calls.mostRecent().args[0].session_id).toBe(firstSessionId!);

    // 3. Raffinamento HITL: conserva lo stesso session_id della lacuna
    app.refineSuggestions({ human_feedback: 'Solo forme omeriche', rejected_candidates: ['πλητα'] });
    expect(app.currentSessionId()).toBe(firstSessionId!);
    expect(genSpy.calls.mostRecent().args[4]?.session_id).toBe(firstSessionId!);

    // 4. Nuova ricerca indipendente: genera un nuovo session_id
    app.generateSuggestions({
      text: 'ἄλλο *...* κείμενον',
      modelID: 'm1',
      num_tokens: 1,
      num_predictions: 5
    });
    const secondSessionId = app.currentSessionId();
    expect(secondSessionId).toBeTruthy();
    expect(secondSessionId).not.toBe(firstSessionId!);
    expect(genSpy.calls.mostRecent().args[4]?.session_id).toBe(secondSessionId!);

    // 5. Chiusura: resetta currentSessionId
    app.closeSuggestions();
    expect(app.currentSessionId()).toBeNull();
  });

  it('should synchronize Confronta mode and selected models between #modelsModal and PromptBox', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-ithaca-tlg', TYPE: 'Ithaca' },
      { _id: '67066928e4414eb1d8109bf2', CHECKPOINT: 'pranaydeeps/Ancient-Greek-BERT', TYPE: 'BERT' }
    ]);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const modalCompareBtn = compiled.querySelector('.models-compare-mode-btn') as HTMLButtonElement;
    expect(modalCompareBtn).toBeTruthy();

    modalCompareBtn.click();
    fixture.detectChanges();

    const cardToggleBtns = compiled.querySelectorAll('.compare-card-toggle-btn');
    (cardToggleBtns[0] as HTMLButtonElement).click();
    (cardToggleBtns[1] as HTMLButtonElement).click();
    fixture.detectChanges();

    expect(app.isCompareMode()).toBeTrue();
    expect(app.selectedModelIds()).toEqual(['67066928e4414eb1d8109bf1', '67066928e4414eb1d8109bf2']);
    expect(app.promptBox()?.isCompareMode()).toBeTrue();
    expect(app.promptBox()?.selectedModelLabel()).toBe('Confronto (2 modelli)');
  });

  it('should collapse and expand a conversation turn via the attached turn-collapse-btn', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);
    spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }])
    );

    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: '67066928e4414eb1d8109bf1',
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const collapseBtn = compiled.querySelector('.turn-collapse-btn') as HTMLButtonElement;
    expect(collapseBtn).toBeTruthy();
    expect(collapseBtn.getAttribute('aria-expanded')).toBe('true');
    expect(compiled.querySelector('app-suggests-box')).not.toBeNull();
    expect(compiled.querySelector('.turn-collapsed-summary')).toBeNull();

    // Comprime il turno
    collapseBtn.click();
    fixture.detectChanges();

    expect(app.conversationTurns()[0].isCollapsed).toBeTrue();
    expect(collapseBtn.getAttribute('aria-expanded')).toBe('false');
    expect(compiled.querySelector('app-suggests-box')).toBeNull();
    const summaryBtn = compiled.querySelector('.turn-collapsed-summary') as HTMLButtonElement;
    expect(summaryBtn).toBeTruthy();
    expect(summaryBtn.textContent).toContain('gs-GreBerta + RAG');
    expect(summaryBtn.textContent).toContain('1 predizione');

    // Riespande il turno
    summaryBtn.click();
    fixture.detectChanges();
    expect(app.conversationTurns()[0].isCollapsed).toBeFalse();
    expect(compiled.querySelector('app-suggests-box')).not.toBeNull();
  });
});
