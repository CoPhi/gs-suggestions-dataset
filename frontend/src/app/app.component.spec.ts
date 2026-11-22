import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withXhr } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of, Subject } from 'rxjs';
import { AppComponent } from './app.component';
import { ApiService, SuggestionInterface } from './services/api.service';
import { CHAT_HISTORY_TTL_MS, ChatHistoryService } from './services/chat-history.service';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [
        provideHttpClient(withXhr()),
        provideHttpClientTesting()
      ]
    }).compileComponents();
    await TestBed.inject(ChatHistoryService).clearAllChats();
  });

  afterEach(async () => {
    await TestBed.inject(ChatHistoryService).clearAllChats();
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

  it('should enable sticky lacuna header only when turn has generated suggestions and is expanded', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: '67066928e4414eb1d8109bf1', CHECKPOINT: 'CNR-ILC/gs-GreBerta', BACKEND: 'neuro_lexical', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);

    const pending$ = new Subject<SuggestionInterface[]>();
    const empty$ = new Subject<SuggestionInterface[]>();
    let callIdx = 0;
    spyOn(api, 'generateSuggestion').and.callFake(() => {
      callIdx += 1;
      return callIdx === 1 ? pending$.asObservable() : empty$.asObservable();
    });

    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: '67066928e4414eb1d8109bf1',
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const stickyHeader = compiled.querySelector('.turn-sticky-header') as HTMLElement;
    expect(stickyHeader).toBeTruthy();
    // 1. Durante loading... lo sticky non è attivo
    expect(stickyHeader.classList.contains('is-sticky')).toBeFalse();
    expect(stickyHeader.querySelector('.turn-collapse-btn')).toBeTruthy();
    expect(stickyHeader.querySelector('.turn-lacuna-text')).toBeTruthy();
    expect(stickyHeader.querySelector('.reuse-turn-text-btn')).toBeTruthy();
    expect(stickyHeader.querySelector('.close-turn-btn')).toBeTruthy();

    // 2. Al completamento con suggerimenti pronti si attiva .is-sticky
    pending$.next([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }]);
    pending$.complete();
    fixture.detectChanges();
    expect(stickyHeader.classList.contains('is-sticky')).toBeTrue();

    // 3. Quando il turno è compresso, .is-sticky viene disattivato; al ripristino ritorna attivo
    (stickyHeader.querySelector('.turn-collapse-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(stickyHeader.classList.contains('is-sticky')).toBeFalse();

    (stickyHeader.querySelector('.turn-collapse-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(stickyHeader.classList.contains('is-sticky')).toBeTrue();

    // 4. Turno con 0 suggerimenti (empty state): .is-sticky rimane disattivo
    app.generateSuggestions({
      text: 'ἄλλο *...* κείμενον',
      modelID: '67066928e4414eb1d8109bf1',
      num_tokens: 1,
      num_predictions: 5
    });
    empty$.next([]);
    empty$.complete();
    fixture.detectChanges();

    const headers = compiled.querySelectorAll('.turn-sticky-header');
    expect(headers.length).toBe(2);
    expect(headers[1].classList.contains('is-sticky')).toBeFalse();
  });

  it('should display turn identifier (number, refinement parent, timestamp) and render models inside the sticky header instead of footer', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: 'm1', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: 'm2', CHECKPOINT: 'CNR-ILC/gs-ithaca-tlg', TYPE: 'Ithaca' }
    ]);
    spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }])
    );

    // Turno #1 (singolo modello LLM)
    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: 'm1',
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const turn1Header = compiled.querySelectorAll('.turn-sticky-header')[0] as HTMLElement;
    expect(turn1Header.querySelector('.turn-number-badge')?.textContent?.trim()).toBe('Turno #1');
    expect(turn1Header.querySelector('.turn-timestamp')?.textContent?.trim()).toBeTruthy();
    // Modello visibile nella sezione sticky, assente dal footer di app-suggests-box
    expect(turn1Header.querySelector('.turn-header-models .model-origin-badge')?.textContent).toContain('Qwen2.5-72B-Instruct');
    expect(compiled.querySelectorAll('.conversation-turn')[0].querySelector('app-suggests-box .model-origin-badge')).toBeNull();

    // Turno #2 (raffinamento multi-modello)
    const firstTurnId = app.conversationTurns()[0].turnId;
    app.refineSuggestions({ human_feedback: 'Solo epico', rejected_candidates: [] }, firstTurnId);
    fixture.detectChanges();

    // Turno #2 con confronto multi-modello per verificare .scrollable-models-bar dentro .turn-sticky-header
    app.generateSuggestions({
      text: 'ἄλλο *...*',
      modelID: 'm1',
      modelIDs: ['m1', 'm2'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const turn2Header = compiled.querySelectorAll('.turn-sticky-header')[1] as HTMLElement;
    expect(turn2Header.querySelector('.turn-number-badge')?.textContent?.trim()).toBe('Turno #2');
    expect(turn2Header.querySelector('.turn-refinement-badge')?.textContent).toContain('Raffinamento di #1');

    const turn3Header = compiled.querySelectorAll('.turn-sticky-header')[2] as HTMLElement;
    expect(turn3Header.querySelector('.scrollable-models-bar')).not.toBeNull();
    expect(compiled.querySelectorAll('.conversation-turn')[2].querySelector('.turn-body-content .scrollable-models-bar')).toBeNull();
  });

  it('should disable Raffina on non-LLM models and exit refinement mode when switching to a non-LLM tab', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: 'llm1', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: 'ith1', CHECKPOINT: 'CNR-ILC/gs-ithaca-tlg', TYPE: 'Ithaca' }
    ]);
    spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }])
    );

    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: 'llm1',
      modelIDs: ['llm1', 'ith1'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const refineBtnOnLlm = compiled.querySelector('.hitl-toggle-btn') as HTMLButtonElement;
    expect(refineBtnOnLlm.disabled).toBeFalse();

    // Attiva raffinamento sul tab LLM
    refineBtnOnLlm.click();
    fixture.detectChanges();
    expect(app.activeRefinementTurnId()).toBe(app.conversationTurns()[0].turnId);

    // Commuta al tab Ithaca -> disabilita Raffina e chiude la modalità raffinamento attiva
    const tabBtns = compiled.querySelectorAll('.model-tab-select-btn');
    (tabBtns[1] as HTMLButtonElement).click();
    fixture.detectChanges();

    expect(app.activeRefinementTurnId()).toBeNull();
    const refineBtnOnIthaca = compiled.querySelector('.hitl-toggle-btn') as HTMLButtonElement;
    expect(refineBtnOnIthaca.disabled).toBeTrue();
  });

  it('should move focus and scroll to the new turn on generate and to the suggestions list on model tab click', async () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: 'm1', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' },
      { _id: 'm2', CHECKPOINT: 'meta-llama/Llama-3.3-70B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);
    spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }])
    );

    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: 'm1',
      modelIDs: ['m1', 'm2'],
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();
    await Promise.resolve();

    const compiled = fixture.nativeElement as HTMLElement;
    const turnSection = compiled.querySelector('.conversation-turn') as HTMLElement;
    const turnBody = compiled.querySelector('.turn-body-content') as HTMLElement;
    const turnScrollSpy = spyOn(turnSection, 'scrollIntoView');
    const turnFocusSpy = spyOn(turnSection, 'focus');
    const bodyScrollSpy = spyOn(turnBody, 'scrollIntoView');
    const bodyFocusSpy = spyOn(turnBody, 'focus');

    // Al cambio tab modello lo scroll e il focus vanno in cima alla lista suggerimenti (.turn-body-content)
    const tabBtns = compiled.querySelectorAll('.model-tab-select-btn');
    (tabBtns[1] as HTMLButtonElement).click();
    await Promise.resolve();
    expect(bodyScrollSpy).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });
    expect(bodyFocusSpy).toHaveBeenCalledWith({ preventScroll: true });

    // Alla creazione di un secondo turno lo scroll e il focus vanno sul nuovo turno
    app.generateSuggestions({
      text: 'ἄλλο *...*',
      modelID: 'm1',
      num_tokens: 1,
      num_predictions: 5
    });
    fixture.detectChanges();
    const secondTurnEl = compiled.querySelectorAll('.conversation-turn')[1] as HTMLElement;
    const secondFocusSpy = spyOn(secondTurnEl, 'focus');
    const secondScrollSpy = spyOn(secondTurnEl, 'scrollIntoView');
    // Verifica anche invocazione diretta su nuovo turno successivo (prima del microtask)
    (app as unknown as { focusAndScrollToTurn: (id: string) => void }).focusAndScrollToTurn(app.conversationTurns()[1].turnId);
    await Promise.resolve();
    expect(secondScrollSpy).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });
    expect(secondFocusSpy).toHaveBeenCalledWith({ preventScroll: true });
    expect(turnScrollSpy).not.toHaveBeenCalled();
    expect(turnFocusSpy).not.toHaveBeenCalled();
  });

  it('should persist chats in IndexedDB sidebar, purge chats older than 7 days on new request, and allow restoring/deleting chats', async () => {
    const historyService = TestBed.inject(ChatHistoryService);
    const oldUpdatedAt = new Date(Date.now() - CHAT_HISTORY_TTL_MS - 3600_000).toISOString();
    const recentUpdatedAt = new Date(Date.now() - 3600_000).toISOString();

    await historyService.saveChat({
      chatId: 'old-chat',
      title: 'Παλαιὰ *...*',
      createdAt: oldUpdatedAt,
      updatedAt: oldUpdatedAt,
      turns: []
    });
    await historyService.saveChat({
      chatId: 'recent-chat',
      title: 'Πρόσφατος *...*',
      createdAt: recentUpdatedAt,
      updatedAt: recentUpdatedAt,
      turns: [
        {
          turnId: 't-recent',
          sessionId: 's-recent',
          createdAt: recentUpdatedAt,
          rawText: 'Πρόσφατος *...*',
          normalizedText: 'Πρόσφατος [...]',
          numTokens: 1,
          numPredictions: 5,
          isRefinement: false,
          parentTurnId: null,
          appliedHumanFeedback: '',
          appliedRejectedCandidates: [],
          modelBoxes: [
            {
              modelId: 'm1',
              modelLabel: 'Qwen2.5-72B-Instruct',
              modelType: 'RAG',
              suggestions: [{ sentence: 'Πρόσφατος λόγος', token_str: 'λόγος', score: 0.9 }],
              pending: false,
              error: null,
              submittedFeedback: {}
            }
          ],
          activeModelId: 'm1',
          userSelectedTab: false,
          isCollapsed: false,
          turnRejectedCandidates: [],
          turnHumanFeedback: ''
        }
      ]
    });

    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);
    app.models.set([
      { _id: 'm1', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);
    spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }])
    );

    await app.flushHistoryPersistence();
    fixture.detectChanges();

    // All'avvio l'app resta sulla schermata iniziale e la sidebar mostra le chat salvate
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.initial-screen-layout')).not.toBeNull();
    expect(app.savedChats().length).toBe(2);

    // Apre la sidebar dalla schermata iniziale
    (compiled.querySelector('.sidebar-toggle-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(compiled.querySelector('.chat-history-sidebar')).not.toBeNull();
    expect(compiled.querySelectorAll('.chat-history-item').length).toBe(2);

    // Nuova richiesta: elimina automaticamente la chat scaduta (> 7 giorni) e salva quella corrente
    app.generateSuggestions({
      text: 'τειχεσι*....*',
      modelID: 'm1',
      num_tokens: 1,
      num_predictions: 5
    });
    await app.flushHistoryPersistence();
    fixture.detectChanges();

    expect(app.savedChats().some(c => c.chatId === 'old-chat')).toBeFalse();
    expect(app.savedChats().some(c => c.chatId === 'recent-chat')).toBeTrue();
    expect(app.savedChats().length).toBe(2);

    // Click sulla chat precedente nella sidebar per ricaricarla
    const recentItem = Array.from(compiled.querySelectorAll('.chat-history-item'))
      .find(el => el.textContent?.includes('Πρόσφατος')) as HTMLElement;
    expect(recentItem).toBeTruthy();
    recentItem.click();
    fixture.detectChanges();

    expect(app.currentChatId()).toBe('recent-chat');
    expect(app.conversationTurns()[0].turnId).toBe('t-recent');

    // Su viewport stretti (es. Karma Headless < 992px) la sidebar si chiude alla selezione: riaprila
    if (!app.isSidebarOpen()) {
      (compiled.querySelector('.sidebar-toggle-btn') as HTMLButtonElement).click();
      fixture.detectChanges();
    }

    // Eliminazione singola chat dalla sidebar
    const refreshedRecentItem = Array.from(compiled.querySelectorAll('.chat-history-item'))
      .find(el => el.textContent?.includes('Πρόσφατος')) as HTMLElement;
    const deleteBtn = refreshedRecentItem.querySelector('.delete-chat-btn') as HTMLButtonElement;
    deleteBtn.click();
    await app.flushHistoryPersistence();
    fixture.detectChanges();

    expect(app.savedChats().length).toBe(1);
    expect(app.conversationTurns().length).toBe(0);

    // Svuota tutto lo storico tramite il menu impostazioni (tre punti) nella toolbar della sidebar
    if (!app.isSidebarOpen()) {
      (compiled.querySelector('.sidebar-toggle-btn') as HTMLButtonElement).click();
      fixture.detectChanges();
    }
    expect(compiled.querySelector('.sidebar-footer .clear-history-btn')).toBeNull();
    (compiled.querySelector('.sidebar-settings-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(app.isSidebarSettingsOpen()).toBeTrue();
    (compiled.querySelector('.sidebar-settings-menu .clear-history-btn') as HTMLButtonElement).click();
    await app.flushHistoryPersistence();
    fixture.detectChanges();

    expect(app.savedChats().length).toBe(0);
    expect(app.isSidebarSettingsOpen()).toBeFalse();
  });

  it('should render top sidebar toolbar without grey header button, truncate sidebar lacuna text with ellipsis, and manage settings menu', async () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    const api = TestBed.inject(ApiService);

    app.models.set([
      { _id: 'm1', CHECKPOINT: 'Qwen/Qwen2.5-72B-Instruct', BACKEND: 'auto', TOP_K_PASSAGES: 5, TYPE: 'RAG' }
    ]);
    spyOn(api, 'generateSuggestion').and.returnValue(
      of([{ sentence: 'τειχεσιβλῆτα', token_str: 'βλῆτα', score: 0.94, tlg_attested: true }])
    );

    const longLacunaText = 'ὅτε ἤρξω *...* εψιν αὐτοῦ διερωτᾶν, οἴεται δὲ καὶ τόδ’ ὅμοιον ὂν τούτῳ, ὅλως οὐ συνείς, ἀποδεικνύειν ἐπιχειρεῖ αὐτὸς τί δεῖ σοι ἐγγενέσθαι.';
    app.generateSuggestions({
      text: longLacunaText,
      modelID: 'm1',
      num_tokens: 1,
      num_predictions: 5
    });
    await app.flushHistoryPersistence();
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    // Il pulsante grigio "Nuova conversazione" nell'header conversazione non deve più esistere
    expect(compiled.querySelector('.conversation-header .new-conversation-btn')).toBeNull();

    // Apre la sidebar e verifica la toolbar superiore
    (compiled.querySelector('.sidebar-toggle-btn') as HTMLButtonElement).click();
    fixture.detectChanges();

    const toolbar = compiled.querySelector('.sidebar-toolbar') as HTMLElement;
    expect(toolbar).not.toBeNull();
    expect(toolbar.querySelector('.sidebar-new-chat-btn')).not.toBeNull();
    expect(toolbar.querySelector('.sidebar-settings-btn')).not.toBeNull();
    expect(toolbar.querySelector('.sidebar-close-btn')).not.toBeNull();
    // Il footer non contiene più il pulsante "Svuota tutto"
    expect(compiled.querySelector('.sidebar-footer .clear-history-btn')).toBeNull();

    // Verifica il troncamento CSS con ellipsis del testo lacunoso dentro la card della sidebar
    const titleEl = compiled.querySelector('.chat-history-title') as HTMLElement;
    const titleContainer = titleEl.parentElement as HTMLElement;
    expect(titleEl).not.toBeNull();
    expect(titleEl.getAttribute('title')).toContain('ὅτε ἤρξω [...]');
    const titleStyles = window.getComputedStyle(titleEl);
    expect(titleStyles.textOverflow).toBe('ellipsis');
    expect(titleStyles.overflow).toBe('hidden');
    expect(titleStyles.whiteSpace).toBe('nowrap');
    expect(window.getComputedStyle(titleContainer).minWidth).toBe('0px');

    // Click sul pulsante nuova generazione nella toolbar -> reset conversazione
    (toolbar.querySelector('.sidebar-new-chat-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(app.conversationTurns().length).toBe(0);
    expect(compiled.querySelector('.initial-screen-layout')).not.toBeNull();

    // Apertura menu impostazioni, svuotamento chat e verifica stato disabilitato a storico vuoto
    (compiled.querySelector('.sidebar-settings-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    const clearBtn = compiled.querySelector('.sidebar-settings-menu .clear-history-btn') as HTMLButtonElement;
    expect(clearBtn.disabled).toBeFalse();
    clearBtn.click();
    await app.flushHistoryPersistence();
    fixture.detectChanges();

    expect(app.savedChats().length).toBe(0);
    (compiled.querySelector('.sidebar-settings-btn') as HTMLButtonElement).click();
    fixture.detectChanges();
    const disabledClearBtn = compiled.querySelector('.sidebar-settings-menu .clear-history-btn') as HTMLButtonElement;
    expect(disabledClearBtn.disabled).toBeTrue();
  });
});
