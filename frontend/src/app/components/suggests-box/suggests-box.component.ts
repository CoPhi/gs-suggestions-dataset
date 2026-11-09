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
  // Etichetta e tipo opzionali del modello da cui provengono i suggerimenti (usati nel confronto multi-modello)
  modelLabel = input<string | null>(null);
  modelType = input<string | null>(null);
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
   * 2. Propaga l'evento al componente padre per la persistenza su MongoDB (Fase 1).
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
    this.onFeedback.emit(event);
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
