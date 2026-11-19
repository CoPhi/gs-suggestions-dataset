import { Component, ElementRef, computed, inject, signal, viewChild } from '@angular/core';
import { ApiService, HitlRefinementOptions, isBertCheckpoint, modelType, SuggestionInterface } from './services/api.service';
import { SuggestsBoxComponent } from './components/suggests-box/suggests-box.component';
import { SuggestionFeedbackEvent } from './components/suggest/suggest.component';
import { ModelsBoxComponent } from "./components/models-box/models-box.component";
import { NotificationService } from './services/notification.service';
import { PromptBoxComponent, PromptRefineSubmitEvent } from './components/prompt-box/prompt-box.component';
import {
  formatNormalizedLacunaForPrompt,
  normalizeLacunaeForApi,
  splitLacunaHighlightSegments
} from './utils/lacuna';

import { catchError, map, Observable, of, Subscription } from 'rxjs';

export interface ModelSuggestionBoxItem {
  modelId: string;
  modelLabel: string;
  modelType: string;
  checkpoint?: string | null;
  use_rag?: boolean | null;
  suggestions: SuggestionInterface[];
  error?: string | null;
  pending?: boolean;
  submittedFeedback?: Partial<Record<string, 'accepted' | 'rejected'>>;
}

export interface ConversationTurn {
  turnId: string;
  sessionId: string;
  rawText: string;
  normalizedText: string;
  numTokens: number;
  numPredictions: number;
  useRag?: boolean;
  isRefinement: boolean;
  parentTurnId?: string | null;
  appliedHumanFeedback?: string;
  appliedRejectedCandidates: string[];
  modelBoxes: ModelSuggestionBoxItem[];
  activeModelId: string;
  userSelectedTab?: boolean;
  isCollapsed?: boolean;
  turnRejectedCandidates: string[];
  turnHumanFeedback: string;
}

@Component({
  selector: 'app-root',
  imports: [SuggestsBoxComponent, ModelsBoxComponent, PromptBoxComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'gs-api';

  api = inject(ApiService);
  notifications = inject(NotificationService);

  promptBox = viewChild(PromptBoxComponent);
  conversationBottomAnchor = viewChild<ElementRef<HTMLElement>>('conversationBottomAnchor');

  // Segnale scrivibile condiviso in two-way binding tra menù a tendina e modale modelli
  models = signal<modelType[]>([]);
  suggestions = signal<SuggestionInterface[] | null>(null);
  // Lista di suggestion box dell'ultimo turno (mantenuta sincronizzata con la cronologia)
  modelSuggestionBoxes = signal<ModelSuggestionBoxItem[] | null>(null);
  curr_id = signal<string | null>(null);
  // Stato confronto condiviso tra PromptBox e modale ModelsBox
  isCompareMode = signal<boolean>(false);
  selectedModelIds = signal<string[]>([]);

  // Cronologia multi-turno della conversazione filologica tra utente e modelli
  conversationTurns = signal<ConversationTurn[]>([]);
  // Turno attualmente collegato alla barra inferiore in modalità raffinamento HITL
  activeRefinementTurnId = signal<string | null>(null);

  activeRefinementTurn = computed<ConversationTurn | null>(() => {
    const id = this.activeRefinementTurnId();
    if (!id) return null;
    return this.conversationTurns().find(t => t.turnId === id) || null;
  });

  // Stato dell'ultima richiesta di generazione per supportare il log di feedback (Fase 1)
  // e il raffinamento guidato in-session (Fase 2) senza reinserire il prompt
  lastGenerationPayload = signal<{
    text: string;
    modelID: string;
    modelIDs?: string[];
    num_tokens: number;
    num_predictions: number;
    use_rag?: boolean;
  } | null>(null);
  sessionHumanFeedback = signal<string>('');
  sessionRejectedCandidates = signal<string[]>([]);
  // Identificativo di sessione univoco per la lacuna corrente: correla confronto multi-modello,
  // raffinamenti HITL, log DSPy/LLM lato backend e feedback persistiti su MongoDB.
  currentSessionId = signal<string | null>(null);

  isGenerating = signal<boolean>(false);
  // Traccia le sottoscrizioni per turno e modello (`${turnId}:${modelId}`) per consentire
  // l'interruzione selettiva di singoli modelli lenti o la rimozione di un intero turno
  private pendingModelSubscriptions = new Map<string, Subscription>();

  constructor() {
    this.api.getModels().subscribe({
      next: (loadedModels) => this.models.set(loadedModels),
      error: () => {}
    });
  }

  selectedModel = computed(() => {
    return this.curr_id() ? this.models().find((model) => model._id === this.curr_id()) || null : null;
  });

  resolveModelCheckpoint(modelId: string): string | null {
    const m = this.models().find(item => item._id === modelId);
    if (!m) return null;
    return 'CHECKPOINT' in m ? m.CHECKPOINT : null;
  }

  resolveModelLabel(modelId: string, useRag?: boolean): string {
    const m = this.models().find(item => item._id === modelId);
    if (!m) return modelId;
    if (m.TYPE === 'RAG' || m.TYPE === 'Ithaca' || m.TYPE === 'BERT') {
      const shortName = (m.CHECKPOINT.includes('/') ? m.CHECKPOINT.split('/').pop() : m.CHECKPOINT) || m.CHECKPOINT;
      const isNeuroLex = m.TYPE === 'RAG' && (m.BACKEND === 'neuro_lexical' || isBertCheckpoint(m.CHECKPOINT));
      if (isNeuroLex || (m.TYPE === 'BERT' && useRag)) {
        return `${shortName} + RAG`;
      }
      return shortName;
    }
    return `N-grammi${m.N ? ` (n=${m.N})` : ''}`;
  }

  resolveModelTypeBadge(modelId: string, useRag?: boolean): string {
    const m = this.models().find(item => item._id === modelId);
    if (!m) return 'Modello';
    if (m.TYPE === 'RAG' && (m.BACKEND === 'neuro_lexical' || isBertCheckpoint(m.CHECKPOINT))) {
      return 'BERT + RAG';
    }
    if (m.TYPE === 'BERT' && useRag) {
      return 'BERT + RAG';
    }
    if (m.TYPE === 'RAG') return 'RAG';
    if (m.TYPE === 'BERT') return 'BERT';
    if (m.TYPE === 'Ithaca') return 'Ithaca';
    return 'N-grammi';
  }

  setCurrentID($event: Event) {
    const target = $event.target as HTMLInputElement;
    this.curr_id.set(target.value);
  }

  showAlert(message: string, type: 'success' | 'danger' | 'warning' | 'info' = 'info') {
    this.notifications.showAlert(message, type);
  }

  getTurnTextSegments(text: string): Array<{ text: string; isLacuna: boolean }> {
    return splitLacunaHighlightSegments(text);
  }

  isTurnWaitingInitialResponse(turn: ConversationTurn): boolean {
    return turn.modelBoxes.length > 0 && turn.modelBoxes.every(b => !!b.pending);
  }

  // Attiva l'intestazione sticky solo quando il turno è espanso e ha almeno un modello
  // completato con suggerimenti pronti (esclude fase di attesa loading... e turni vuoti).
  hasTurnGeneratedSuggestions(turn: ConversationTurn): boolean {
    if (turn.isCollapsed || this.isTurnWaitingInitialResponse(turn)) {
      return false;
    }
    return turn.modelBoxes.some(b => !b.pending && b.suggestions.length > 0);
  }

  getActiveTurnBox(turn: ConversationTurn): ModelSuggestionBoxItem | null {
    if (turn.modelBoxes.length === 0) return null;
    return turn.modelBoxes.find(b => b.modelId === turn.activeModelId) || turn.modelBoxes[0];
  }

  toggleTurnCollapse(turnId: string) {
    this.conversationTurns.update(turns =>
      turns.map(t => (t.turnId === turnId ? { ...t, isCollapsed: !t.isCollapsed } : t))
    );
  }

  getTurnCollapsedSummary(turn: ConversationTurn): string {
    if (this.isTurnWaitingInitialResponse(turn)) {
      return 'Generazione in corso...';
    }
    if (turn.modelBoxes.length === 1) {
      const box = turn.modelBoxes[0];
      const count = box.suggestions.length;
      return `${box.modelLabel} · ${count} ${count === 1 ? 'predizione' : 'predizioni'}`;
    }
    const readyCount = turn.modelBoxes.filter(b => !b.pending).length;
    return `Confronto (${readyCount}/${turn.modelBoxes.length} modelli)`;
  }

  private createUniqueId(prefix: string): string {
    const cryptoApi = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
    if (cryptoApi?.randomUUID) {
      return cryptoApi.randomUUID();
    }
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  }

  private cancelAllRequests() {
    for (const sub of this.pendingModelSubscriptions.values()) {
      sub.unsubscribe();
    }
    this.pendingModelSubscriptions.clear();
  }

  private cancelTurnSubscriptions(turnId: string) {
    const prefix = `${turnId}:`;
    for (const [key, sub] of Array.from(this.pendingModelSubscriptions.entries())) {
      if (key.startsWith(prefix)) {
        sub.unsubscribe();
        this.pendingModelSubscriptions.delete(key);
      }
    }
  }

  private scrollToLatestTurn() {
    queueMicrotask(() => {
      const anchor = this.conversationBottomAnchor()?.nativeElement;
      if (anchor && typeof anchor.scrollIntoView === 'function') {
        anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    });
  }

  /**
   * Riallinea i segnali legacy (`isGenerating`, `modelSuggestionBoxes`, `suggestions`)
   * allo stato corrente dei turni in cronologia.
   */
  private syncStateFromTurns(turns: ConversationTurn[]) {
    const hasPending = turns.some(t => t.modelBoxes.some(b => !!b.pending));
    this.isGenerating.set(hasPending);

    if (turns.length === 0) {
      this.modelSuggestionBoxes.set(null);
      this.suggestions.set(null);
      return;
    }

    const latest = turns[turns.length - 1];
    this.modelSuggestionBoxes.set(latest.modelBoxes);
    const activeBox = this.getActiveTurnBox(latest);
    if (activeBox && !activeBox.pending) {
      this.suggestions.set(activeBox.suggestions);
    } else {
      const firstReady = latest.modelBoxes.find(b => !b.pending);
      this.suggestions.set(firstReady ? firstReady.suggestions : null);
    }
  }

  closeSuggestions() {
    this.cancelAllRequests();
    this.isGenerating.set(false);
    this.conversationTurns.set([]);
    this.activeRefinementTurnId.set(null);
    this.suggestions.set(null);
    this.modelSuggestionBoxes.set(null);
    this.lastGenerationPayload.set(null);
    this.sessionHumanFeedback.set('');
    this.sessionRejectedCandidates.set([]);
    this.currentSessionId.set(null);
    this.promptBox()?.clearForm();
  }

  closeTurn(turnId: string) {
    this.cancelTurnSubscriptions(turnId);
    if (this.activeRefinementTurnId() === turnId) {
      this.activeRefinementTurnId.set(null);
    }
    const remaining = this.conversationTurns().filter(t => t.turnId !== turnId);
    if (remaining.length === 0) {
      this.closeSuggestions();
      return;
    }
    this.conversationTurns.set(remaining);
    const latest = remaining[remaining.length - 1];
    this.currentSessionId.set(latest.sessionId);
    this.lastGenerationPayload.set({
      text: latest.normalizedText,
      modelID: latest.activeModelId,
      ...(latest.modelBoxes.length > 1 ? { modelIDs: latest.modelBoxes.map(b => b.modelId) } : {}),
      num_tokens: latest.numTokens,
      num_predictions: latest.numPredictions,
      ...(latest.useRag ? { use_rag: true } : {})
    });
    this.syncStateFromTurns(remaining);
  }

  closeModelBox(modelId: string, turnId?: string) {
    const turns = this.conversationTurns();
    const targetTurn = turnId
      ? turns.find(t => t.turnId === turnId)
      : turns[turns.length - 1];

    if (!targetTurn) {
      this.closeSuggestions();
      return;
    }

    const subKey = `${targetTurn.turnId}:${modelId}`;
    const pendingSub = this.pendingModelSubscriptions.get(subKey);
    if (pendingSub) {
      pendingSub.unsubscribe();
      this.pendingModelSubscriptions.delete(subKey);
    }

    const remainingBoxes = targetTurn.modelBoxes.filter(b => b.modelId !== modelId);
    if (remainingBoxes.length === 0) {
      this.closeTurn(targetTurn.turnId);
      return;
    }

    let nextActiveId = targetTurn.activeModelId;
    if (targetTurn.activeModelId === modelId) {
      const firstReady = remainingBoxes.find(b => !b.pending);
      nextActiveId = firstReady ? firstReady.modelId : remainingBoxes[0].modelId;
    }

    const updatedTurns = turns.map(t =>
      t.turnId === targetTurn.turnId
        ? { ...t, modelBoxes: remainingBoxes, activeModelId: nextActiveId }
        : t
    );
    this.conversationTurns.set(updatedTurns);
    this.syncStateFromTurns(updatedTurns);

    if (this.activeRefinementTurnId() === targetTurn.turnId) {
      const nextBox = remainingBoxes.find(b => b.modelId === nextActiveId);
      if (nextBox) {
        this.promptBox()?.setSingleModelSelection(nextBox.modelId, nextBox.use_rag ?? targetTurn.useRag);
      }
    }
  }

  selectTurnModelTab(turnId: string, modelId: string) {
    const turns = this.conversationTurns();
    const targetTurn = turns.find(t => t.turnId === turnId);
    if (!targetTurn) return;

    const targetBox = targetTurn.modelBoxes.find(b => b.modelId === modelId);
    if (!targetBox || targetBox.pending) return;

    const updatedTurns = turns.map(t =>
      t.turnId === turnId
        ? { ...t, activeModelId: modelId, userSelectedTab: true }
        : t
    );
    this.conversationTurns.set(updatedTurns);
    this.syncStateFromTurns(updatedTurns);

    // Se il turno è già agganciato alla barra in raffinamento, allinea anche il modello preselezionato
    if (this.activeRefinementTurnId() === turnId) {
      this.promptBox()?.setSingleModelSelection(modelId, targetBox.use_rag ?? targetTurn.useRag);
    }
  }

  toggleTurnRefinement(turnId: string) {
    if (this.activeRefinementTurnId() === turnId) {
      this.activeRefinementTurnId.set(null);
      return;
    }
    const targetTurn = this.conversationTurns().find(t => t.turnId === turnId);
    if (!targetTurn) return;

    this.activeRefinementTurnId.set(turnId);
    this.currentSessionId.set(targetTurn.sessionId);
    this.sessionRejectedCandidates.set([...targetTurn.turnRejectedCandidates]);
    this.lastGenerationPayload.set({
      text: targetTurn.normalizedText,
      modelID: targetTurn.activeModelId,
      ...(targetTurn.modelBoxes.length > 1 ? { modelIDs: targetTurn.modelBoxes.map(b => b.modelId) } : {}),
      num_tokens: targetTurn.numTokens,
      num_predictions: targetTurn.numPredictions,
      ...(targetTurn.useRag ? { use_rag: true } : {})
    });

    const activeBox = this.getActiveTurnBox(targetTurn);
    const activeModelId = activeBox?.modelId || targetTurn.activeModelId;
    this.promptBox()?.setSingleModelSelection(activeModelId, activeBox?.use_rag ?? targetTurn.useRag);
    this.promptBox()?.focusTextarea();
  }

  exitRefinementMode() {
    this.activeRefinementTurnId.set(null);
  }

  removeTurnRejectedCandidate(candidate: string, turnId?: string) {
    const targetId = turnId || this.activeRefinementTurnId();
    if (!targetId) return;

    const updatedTurns = this.conversationTurns().map(t => {
      if (t.turnId !== targetId) return t;
      return {
        ...t,
        turnRejectedCandidates: t.turnRejectedCandidates.filter(c => c !== candidate)
      };
    });
    this.conversationTurns.set(updatedTurns);
    if (this.activeRefinementTurnId() === targetId) {
      const refreshed = updatedTurns.find(t => t.turnId === targetId);
      this.sessionRejectedCandidates.set(refreshed ? [...refreshed.turnRejectedCandidates] : []);
    }
  }

  reuseTurnText(turn: ConversationTurn) {
    this.activeRefinementTurnId.set(null);
    const editableText = formatNormalizedLacunaForPrompt(turn.normalizedText);
    this.promptBox()?.setPromptText(editableText);
  }

  handlePromptRefineSubmit(event: PromptRefineSubmitEvent) {
    const targetTurn = this.activeRefinementTurn();
    if (!targetTurn) {
      return;
    }
    this.sessionHumanFeedback.set(event.human_feedback || '');
    this.sessionRejectedCandidates.set([...event.rejected_candidates]);
    this.generateSuggestions(
      {
        text: targetTurn.normalizedText,
        modelID: event.modelID,
        ...(event.modelIDs ? { modelIDs: event.modelIDs } : {}),
        num_tokens: event.num_tokens,
        num_predictions: event.num_predictions,
        ...(event.use_rag ? { use_rag: true } : {})
      },
      {
        human_feedback: event.human_feedback,
        rejected_candidates: event.rejected_candidates
      },
      targetTurn.turnId
    );
  }

  generateSuggestions(
    payload: {
      text: string;
      modelID: string;
      modelIDs?: string[];
      num_tokens: number;
      num_predictions: number;
      use_rag?: boolean;
    },
    hitlOptions?: HitlRefinementOptions,
    parentTurnId?: string
  ) {
    const parentTurn = parentTurnId
      ? this.conversationTurns().find(t => t.turnId === parentTurnId) || null
      : null;

    // Se è una nuova richiesta dal PromptBox (senza hitlOptions), azzera i vincoli precedenti
    // e genera un nuovo session_id; nei raffinamenti HITL mantiene il session_id della lacuna attiva.
    let sessionId = parentTurn?.sessionId || this.currentSessionId();
    if (!hitlOptions) {
      this.sessionHumanFeedback.set('');
      this.sessionRejectedCandidates.set([]);
      sessionId = this.createUniqueId('sess');
      this.currentSessionId.set(sessionId);
      // Svuota la barra di input all'invio dalla UI come in un workspace conversazionale
      this.promptBox()?.clearForm();
    } else if (!sessionId) {
      sessionId = this.createUniqueId('sess');
      this.currentSessionId.set(sessionId);
    } else {
      this.currentSessionId.set(sessionId);
    }

    this.activeRefinementTurnId.set(null);
    this.lastGenerationPayload.set(payload);

    const { text: rawText, modelID, modelIDs, num_tokens, num_predictions, use_rag } = payload;
    // Pre-processing a livello di coordinamento: garantisce che la lacuna target con asterischi (*...*)
    // sia convertita nel formato atteso dalle API ([...]) e che eventuali lacune contestuali
    // in parentesi quadre non vengano scambiate per il target. L'operazione è idempotente.
    const text = normalizeLacunaeForApi(rawText || '');

    const mergedOptions: HitlRefinementOptions = {
      ...(hitlOptions || {}),
      ...(use_rag ? { use_rag: true } : {}),
      session_id: sessionId,
    };

    const targetIds = modelIDs && modelIDs.length > 1
      ? Array.from(new Set(modelIDs))
      : [modelID];

    const turnId = this.createUniqueId('turn');
    const initialBoxes: ModelSuggestionBoxItem[] = targetIds.map(id => ({
      modelId: id,
      modelLabel: this.resolveModelLabel(id, use_rag),
      modelType: this.resolveModelTypeBadge(id, use_rag),
      checkpoint: this.resolveModelCheckpoint(id),
      use_rag: use_rag,
      suggestions: [],
      error: null,
      pending: true,
      submittedFeedback: {}
    }));

    const newTurn: ConversationTurn = {
      turnId,
      sessionId,
      rawText: rawText || '',
      normalizedText: text,
      numTokens: num_tokens,
      numPredictions: num_predictions,
      useRag: use_rag,
      isRefinement: !!hitlOptions,
      parentTurnId: parentTurn?.turnId ?? parentTurnId ?? null,
      appliedHumanFeedback: hitlOptions?.human_feedback || '',
      appliedRejectedCandidates: [...(hitlOptions?.rejected_candidates || [])],
      modelBoxes: initialBoxes,
      activeModelId: targetIds[0],
      userSelectedTab: false,
      isCollapsed: false,
      turnRejectedCandidates: [],
      turnHumanFeedback: ''
    };

    this.conversationTurns.update(turns => [...turns, newTurn]);
    this.syncStateFromTurns(this.conversationTurns());
    this.scrollToLatestTurn();

    for (const id of targetIds) {
      const subKey = `${turnId}:${id}`;
      const req$ = mergedOptions
        ? this.api.generateSuggestion(id, text, num_tokens, num_predictions, mergedOptions)
        : this.api.generateSuggestion(id, text, num_tokens, num_predictions);

      const sub = req$.pipe(
        map((preds): ModelSuggestionBoxItem => ({
          modelId: id,
          modelLabel: this.resolveModelLabel(id, use_rag),
          modelType: this.resolveModelTypeBadge(id, use_rag),
          checkpoint: this.resolveModelCheckpoint(id),
          use_rag: use_rag,
          suggestions: preds || [],
          error: null,
          pending: false,
          submittedFeedback: {}
        })),
        catchError((err): Observable<ModelSuggestionBoxItem> => {
          const detail = err?.error?.detail || err?.message || 'Errore durante la consultazione del modello';
          if (targetIds.length === 1) {
            this.notifications.showDanger(detail);
          }
          return of({
            modelId: id,
            modelLabel: this.resolveModelLabel(id, use_rag),
            modelType: this.resolveModelTypeBadge(id, use_rag),
            checkpoint: this.resolveModelCheckpoint(id),
            use_rag: use_rag,
            suggestions: [],
            error: detail,
            pending: false,
            submittedFeedback: {}
          });
        })
      ).subscribe({
        next: (resolvedItem) => {
          this.pendingModelSubscriptions.delete(subKey);

          let turnWasCompletedNow = false;
          let resolvedBoxesForNotice: ModelSuggestionBoxItem[] = [];

          this.conversationTurns.update(turns =>
            turns.map(t => {
              if (t.turnId !== turnId) return t;
              const updatedBoxes = t.modelBoxes.map(b => (b.modelId === id ? resolvedItem : b));

              // Sblocca automaticamente la vista selezionando il primo modello che risponde
              // (e preferendo un modello con candidati se quello auto-selezionato aveva 0 risultati),
              // senza sovrascrivere un tab già scelto manualmente dal filologo.
              let nextActiveModelId = t.activeModelId;
              if (!t.userSelectedTab) {
                const currentActive = updatedBoxes.find(b => b.modelId === t.activeModelId);
                if (!currentActive || currentActive.pending) {
                  nextActiveModelId = resolvedItem.modelId;
                } else if (currentActive.suggestions.length === 0 && resolvedItem.suggestions.length > 0) {
                  nextActiveModelId = resolvedItem.modelId;
                }
              }

              if (!updatedBoxes.some(b => !!b.pending)) {
                turnWasCompletedNow = true;
                resolvedBoxesForNotice = updatedBoxes;
              }

              return {
                ...t,
                modelBoxes: updatedBoxes,
                activeModelId: nextActiveModelId
              };
            })
          );

          this.syncStateFromTurns(this.conversationTurns());

          if (turnWasCompletedNow) {
            if (resolvedBoxesForNotice.length > 1) {
              const totalPreds = resolvedBoxesForNotice.reduce((acc, b) => acc + (b.suggestions?.length || 0), 0);
              this.notifications.showSuccess(
                `Confronto pronto! Ricevute ${totalPreds} predizioni complessive da ${resolvedBoxesForNotice.length} modelli.`
              );
              this.notifications.showLocalNotification();
            } else if (resolvedBoxesForNotice.length === 1 && !resolvedBoxesForNotice[0].error) {
              const count = resolvedBoxesForNotice[0].suggestions?.length || 0;
              this.notifications.showSuccess(`Suggerimenti pronti! Generate con successo ${count} predizioni.`);
              this.notifications.showLocalNotification();
            }
          }
        },
        error: () => {
          this.pendingModelSubscriptions.delete(subKey);
        },
        complete: () => {
          this.pendingModelSubscriptions.delete(subKey);
        }
      });

      // Evita di registrare sottoscrizioni già concluse in caso di Observable sincroni
      if (!sub.closed) {
        this.pendingModelSubscriptions.set(subKey, sub);
      }
    }
  }

  /**
   * Invia al backend l'annotazione del filologo (HITL e predisposizione per RLHF / DPO),
   * associando i metadati del modello, i dati intrinseci del suggerimento e le esclusioni del turno.
   */
  recordFeedback(
    event: SuggestionFeedbackEvent,
    boxModelId?: string | null,
    boxModelLabel?: string | null,
    boxModelType?: string | null,
    boxCheckpoint?: string | null,
    boxUseRag?: boolean | null,
    turnId?: string | null
  ) {
    const turns = this.conversationTurns();
    const targetTurn = turnId
      ? turns.find(t => t.turnId === turnId) || null
      : (turns.length > 0 ? turns[turns.length - 1] : null);

    const lastReq = this.lastGenerationPayload();
    // Priorità al model_id specifico della colonna consultata, poi all'evento, poi all'ultima richiesta
    const modelId = boxModelId || event.model_id || targetTurn?.activeModelId || lastReq?.modelID || this.curr_id() || '';
    const context = targetTurn?.normalizedText || lastReq?.text || '';

    // Aggiorna gli scarti accumulati a livello dell'intero turno (condivisi tra tutti i tab del confronto)
    // e memorizza lo stato del feedback sulla singola colonna così da preservarlo al cambio tab.
    if (targetTurn) {
      this.conversationTurns.update(list =>
        list.map(t => {
          if (t.turnId !== targetTurn.turnId) return t;
          const nextRejected = !event.is_accepted
            ? (t.turnRejectedCandidates.includes(event.token_str)
              ? t.turnRejectedCandidates
              : [...t.turnRejectedCandidates, event.token_str])
            : t.turnRejectedCandidates.filter(item => item !== event.token_str);

          const nextBoxes = t.modelBoxes.map(b =>
            b.modelId === modelId
              ? {
                  ...b,
                  submittedFeedback: {
                    ...(b.submittedFeedback || {}),
                    [event.token_str]: (event.is_accepted ? 'accepted' : 'rejected') as 'accepted' | 'rejected'
                  }
                }
              : b
          );

          return {
            ...t,
            turnRejectedCandidates: nextRejected,
            modelBoxes: nextBoxes
          };
        })
      );
      this.syncStateFromTurns(this.conversationTurns());
    }

    if (!event.is_accepted) {
      this.sessionRejectedCandidates.update(list =>
        list.includes(event.token_str) ? list : [...list, event.token_str]
      );
    } else {
      this.sessionRejectedCandidates.update(list =>
        list.filter(item => item !== event.token_str)
      );
    }

    if (!modelId || !context) {
      return;
    }

    const activeBox = targetTurn?.modelBoxes.find(b => b.modelId === modelId) || null;
    const modelLabel = boxModelLabel || event.model_label || activeBox?.modelLabel || this.resolveModelLabel(modelId, lastReq?.use_rag);
    const modelType = boxModelType || event.model_type || activeBox?.modelType || this.resolveModelTypeBadge(modelId, lastReq?.use_rag);
    const checkpoint = boxCheckpoint || event.checkpoint || activeBox?.checkpoint || this.resolveModelCheckpoint(modelId);
    const useRag = boxUseRag !== undefined && boxUseRag !== null
      ? boxUseRag
      : (event.use_rag !== undefined && event.use_rag !== null ? event.use_rag : (activeBox?.use_rag ?? targetTurn?.useRag ?? lastReq?.use_rag ?? null));

    this.api.submitFeedback({
      model_id: modelId,
      context,
      token_str: event.token_str,
      sentence: event.sentence,
      is_accepted: event.is_accepted,
      notes: targetTurn?.appliedHumanFeedback || this.sessionHumanFeedback() || null,
      // Dati intrinseci del suggerimento proposto
      score: event.score,
      rank: event.rank,
      reconstructed_word: event.reconstructed_word,
      tlg_attested: event.tlg_attested,
      candidate_reasoning: event.candidate_reasoning,
      philological_reasoning: event.philological_reasoning,
      attestations: event.attestations,
      // Metadati sul modello generatore
      model_label: modelLabel,
      model_type: modelType,
      checkpoint: checkpoint,
      use_rag: useRag,
      top_k: event.top_k || targetTurn?.numPredictions || lastReq?.num_predictions || null,
      // Supporto per RLHF (coppie di preferenza per DPO / Reward Modeling)
      alternative_candidates: event.alternative_candidates,
      session_id: targetTurn?.sessionId || this.currentSessionId()
    }).subscribe({
      next: () => {
        const label = event.is_accepted ? 'validato' : 'scartato';
        this.notifications.showAlert(`Supplemento "${event.token_str}" registrato come ${label}.`, 'info');
      },
      error: () => {
        this.notifications.showDanger('Impossibile salvare il feedback filologico sul server');
      }
    });
  }

  /**
   * Rigenera i suggerimenti applicando le indicazioni filologiche e le esclusioni (Fase 2 - In-Session HITL).
   */
  refineSuggestions(options: HitlRefinementOptions, turnId?: string) {
    const turns = this.conversationTurns();
    const targetTurn = turnId
      ? turns.find(t => t.turnId === turnId) || null
      : (this.activeRefinementTurn() || (turns.length > 0 ? turns[turns.length - 1] : null));

    const lastReq = this.lastGenerationPayload();
    const basePayload = targetTurn
      ? {
          text: targetTurn.normalizedText,
          modelID: targetTurn.activeModelId,
          ...(targetTurn.modelBoxes.length > 1 ? { modelIDs: targetTurn.modelBoxes.map(b => b.modelId) } : {}),
          num_tokens: targetTurn.numTokens,
          num_predictions: targetTurn.numPredictions,
          ...(targetTurn.useRag ? { use_rag: true } : {})
        }
      : lastReq;

    if (!basePayload) {
      return;
    }
    this.sessionHumanFeedback.set(options.human_feedback || '');
    this.sessionRejectedCandidates.set(options.rejected_candidates || []);
    this.generateSuggestions(basePayload, options, targetTurn?.turnId);
  }
}
