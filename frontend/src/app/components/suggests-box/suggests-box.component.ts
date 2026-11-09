import { Component, computed, effect, inject, input, output, signal } from '@angular/core';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { ApiService, HitlRefinementOptions, SuggestionInterface } from '../../services/api.service';
import {
  HighlightSegment,
  parseHighlightedSegments,
  SuggestComponent,
  SuggestionFeedbackEvent
} from '../suggest/suggest.component';

@Component({
  selector: 'app-suggests-box',
  imports: [ReactiveFormsModule, FormsModule, SuggestComponent],
  templateUrl: './suggests-box.component.html',
  styleUrl: './suggests-box.component.css'
})
export class SuggestsBoxComponent {

  api = inject(ApiService);
  suggestions = input.required<SuggestionInterface[]>();
  // Metadati opzionali del modello da cui provengono i suggerimenti (usati nel confronto multi-modello)
  modelId = input<string | null | undefined>(null);
  modelLabel = input<string | null | undefined>(null);
  modelType = input<string | null | undefined>(null);
  checkpoint = input<string | null | undefined>(null);
  useRag = input<boolean | null | undefined>(null);
  // Stato HITL ereditato dalla sessione padre per preservare i vincoli dopo una rigenerazione
  initialHumanFeedback = input<string>('');
  initialRejectedCandidates = input<string[]>([]);

  onClose = output<void>();
  onFeedback = output<SuggestionFeedbackEvent>();
  onRefine = output<HitlRefinementOptions>();

  // L'analisi filologica RAG (CoT 1° livello + Loci Similes), il menù secondario di approfondimento
  // dei loci similes e il menu di raffinamento HITL sono nascosti di default e apribili su richiesta
  showRagContext = signal<boolean>(false);
  showSecondaryLociMenu = signal<boolean>(false);
  showHitlMenu = signal<boolean>(false);
  humanFeedback = signal<string>('');
  rejectedCandidates = signal<string[]>([]);

  constructor() {
    // Riallinea i segnali locali con i vincoli della sessione corrente quando il componente
    // viene ricreato al termine di una chiamata di raffinamento
    effect(() => {
      const fb = this.initialHumanFeedback();
      const rej = [...this.initialRejectedCandidates()];
      this.humanFeedback.set(fb);
      this.rejectedCandidates.set(rej);
      if (fb.trim().length > 0 || rej.length > 0) {
        this.showHitlMenu.set(true);
      }
    });
  }

  // Il backend allega i metadati filologici RAG (CoT di 1° livello e loci similes) al primo elemento della risposta
  philologicalReasoning = computed(() => {
    const found = this.suggestions().find(s => !!s.philological_reasoning);
    return found?.philological_reasoning || null;
  });

  // Forme greche attestate complessivamente tra i suggerimenti, usate per evidenziare i match in grassetto
  allAttestedForms = computed<string[]>(() => {
    const forms: string[] = [];
    for (const s of this.suggestions()) {
      for (const att of s.attestations || []) {
        if (att.attested_form) {
          forms.push(att.attested_form);
        }
      }
    }
    return forms;
  });

  philologicalReasoningSegments = computed<HighlightSegment[]>(() => {
    return parseHighlightedSegments(this.philologicalReasoning(), this.allAttestedForms());
  });

  retrievedPassages = computed(() => {
    const found = this.suggestions().find(s => s.retrieved_passages && s.retrieved_passages.length > 0);
    return found?.retrieved_passages || [];
  });

  // Lista completa di tutti i loci similes trovati dal retriever per il prompt corrente
  allRetrievedPassages = computed(() => {
    const foundAll = this.suggestions().find(
      s => s.all_retrieved_passages && s.all_retrieved_passages.length > 0
    );
    if (foundAll?.all_retrieved_passages && foundAll.all_retrieved_passages.length > 0) {
      return foundAll.all_retrieved_passages;
    }
    return this.retrievedPassages();
  });

  // Loci similes più rilevanti mostrati nella vista primaria del pannello CoT
  primaryPassages = computed(() => {
    const primary = this.retrievedPassages();
    const all = this.allRetrievedPassages();
    if (all.length > primary.length && primary.length > 0) {
      return primary;
    }
    if (primary.length > 3) {
      return primary.slice(0, 3);
    }
    return primary;
  });

  // Ulteriori loci similes trovati per il prompt, consultabili su richiesta nel menù secondario
  secondaryPassages = computed(() => {
    const all = this.allRetrievedPassages();
    const primaryCount = this.primaryPassages().length;
    return all.slice(primaryCount);
  });

  hasRagDetails = computed(() => {
    return !!this.philologicalReasoning() || this.allRetrievedPassages().length > 0;
  });

  canRefine = computed(() => {
    return this.humanFeedback().trim().length > 0 || this.rejectedCandidates().length > 0;
  });

  toggleRagContext() {
    this.showRagContext.update(v => !v);
    if (!this.showRagContext()) {
      this.showSecondaryLociMenu.set(false);
    }
  }

  toggleSecondaryLociMenu() {
    this.showSecondaryLociMenu.update(v => !v);
  }

  getPassageSegments(passage: string): HighlightSegment[] {
    return parseHighlightedSegments(passage, this.allAttestedForms());
  }

  toggleHitlMenu() {
    this.showHitlMenu.update(v => !v);
  }

  /**
   * Gestisce la validazione o lo scarto di un singolo supplemento:
   * 1. Aggiorna la lista di esclusione in-session (Fase 2) se il candidato è scartato;
   * 2. Calcola il rank 1-based, estrae gli alternative_candidates per RLHF e allega i metadati del modello;
   * 3. Propaga l'evento arricchito al componente padre per la persistenza su MongoDB (Fase 1 e RLHF).
   */
  handleSuggestionFeedback(event: SuggestionFeedbackEvent) {
    if (!event.is_accepted) {
      this.rejectedCandidates.update(list =>
        list.includes(event.token_str) ? list : [...list, event.token_str]
      );
      this.showHitlMenu.set(true);
    } else {
      this.rejectedCandidates.update(list =>
        list.filter(token => token !== event.token_str)
      );
    }

    const all = this.suggestions();
    const index = all.findIndex(s => s.token_str === event.token_str);
    const rank = index >= 0 ? index + 1 : (event.rank || 1);
    const alternativeCandidates = all
      .filter((_, idx) => idx !== index)
      .map((s, idx) => ({
        token_str: s.token_str,
        score: s.score,
        rank: idx >= index ? idx + 2 : idx + 1,
        tlg_attested: s.tlg_attested,
        reconstructed_word: s.reconstructed_word || null
      }));

    const enrichedEvent: SuggestionFeedbackEvent = {
      ...event,
      rank,
      model_id: this.modelId() || event.model_id || null,
      model_label: this.modelLabel() || event.model_label || null,
      model_type: this.modelType() || event.model_type || null,
      checkpoint: this.checkpoint() || event.checkpoint || null,
      use_rag: this.useRag() != null ? this.useRag()! : (event.use_rag ?? null),
      top_k: all.length,
      alternative_candidates: alternativeCandidates
    };

    this.onFeedback.emit(enrichedEvent);
  }

  removeRejectedCandidate(token: string) {
    this.rejectedCandidates.update(list => list.filter(item => item !== token));
  }

  triggerRefinement() {
    if (!this.canRefine()) {
      return;
    }
    this.onRefine.emit({
      human_feedback: this.humanFeedback().trim(),
      rejected_candidates: this.rejectedCandidates()
    });
  }

  close() {
    this.onClose.emit();
  }

}
