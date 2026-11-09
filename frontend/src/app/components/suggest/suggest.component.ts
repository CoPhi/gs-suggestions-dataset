import { Component, computed, input, output, signal } from '@angular/core';
import { AlternativeCandidateInterface, CorpusAttestationInterface, SuggestionInterface } from '../../services/api.service';
import { Clipboard } from '@angular/cdk/clipboard';

export interface SuggestionFeedbackEvent {
  token_str: string;
  sentence: string;
  is_accepted: boolean;
  score?: number | null;
  rank?: number | null;
  reconstructed_word?: string | null;
  tlg_attested?: boolean | null;
  candidate_reasoning?: string | null;
  philological_reasoning?: string | null;
  attestations?: CorpusAttestationInterface[] | null;
  model_id?: string | null;
  model_label?: string | null;
  model_type?: string | null;
  checkpoint?: string | null;
  use_rag?: boolean | null;
  top_k?: number | null;
  alternative_candidates?: AlternativeCandidateInterface[] | null;
}

export interface HighlightSegment {
  text: string;
  bold: boolean;
}

/**
 * Normalizza una stringa greca rimuovendo spiriti/accenti (NFD) e uniformando il sigma finale,
 * così da individuare le attestazioni nel chunk anche quando variano i diacritici.
 */
export function normalizeGreekForMatch(value: string): string {
  if (!value) {
    return '';
  }
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/ς/g, 'σ')
    .trim();
}

/**
 * Suddivide un testo (sentence, CoT o estratto del corpus) in segmenti normali ed evidenziati (`bold: true`),
 * riconoscendo sia i marcatori Markdown `**...**` restituiti dal backend sia il match diretto
 * (insensibile ai diacritici e al case folding) con le forme greche o le parole composte (`fallbackForms`).
 * Supporta sia forme a singola parola sia espressioni multi-parola.
 */
export function parseHighlightedSegments(
  text: string | null | undefined,
  fallbackForms: string[] = []
): HighlightSegment[] {
  if (!text) {
    return [];
  }

  const normalizedTargets = new Set(
    fallbackForms
      .map(f => normalizeGreekForMatch(f))
      .filter(f => f.length > 0)
  );

  const rawParts = text.split(/(\*\*[^*]+\*\*)/g).filter(p => p.length > 0);
  const segments: HighlightSegment[] = [];

  // Calcola le lunghezze in parole dei target per supportare anche espressioni multi-parola
  const targetWordCounts = Array.from(
    new Set(
      Array.from(normalizedTargets).map(t => t.split(/\s+/).filter(Boolean).length)
    )
  ).sort((a, b) => b - a);

  for (const part of rawParts) {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      segments.push({ text: part.slice(2, -2), bold: true });
      continue;
    }

    if (normalizedTargets.size === 0) {
      segments.push({ text: part, bold: false });
      continue;
    }

    // Scansiona tutti i token alfabetici nel frammento per consentire il matching
    // esatto anche con variazioni di diacritici o con espressioni a più parole.
    const tokenRegex = /([\p{L}\p{M}]+)/gu;
    const tokens: Array<{ word: string; start: number; end: number; norm: string }> = [];
    let match: RegExpExecArray | null;

    while ((match = tokenRegex.exec(part)) !== null) {
      tokens.push({
        word: match[0],
        start: match.index,
        end: match.index + match[0].length,
        norm: normalizeGreekForMatch(match[0])
      });
    }

    if (tokens.length === 0) {
      segments.push({ text: part, bold: false });
      continue;
    }

    let lastIndex = 0;
    for (let i = 0; i < tokens.length; i++) {
      let matchedCount = 0;

      for (const wCount of targetWordCounts) {
        if (i + wCount <= tokens.length) {
          const candidateNorm = tokens.slice(i, i + wCount).map(t => t.norm).join(' ');
          if (normalizedTargets.has(candidateNorm)) {
            matchedCount = wCount;
            break;
          }
        }
      }

      if (matchedCount > 0) {
        const start = tokens[i].start;
        const end = tokens[i + matchedCount - 1].end;

        if (start > lastIndex) {
          segments.push({ text: part.slice(lastIndex, start), bold: false });
        }
        segments.push({ text: part.slice(start, end), bold: true });
        lastIndex = end;
        i += matchedCount - 1;
      }
    }

    if (lastIndex < part.length) {
      segments.push({ text: part.slice(lastIndex), bold: false });
    }
  }

  return segments;
}

@Component({
  selector: 'app-suggest',
  imports: [],
  templateUrl: './suggest.component.html',
  styleUrl: './suggest.component.css'
})
export class SuggestComponent {

  suggest = input.required<SuggestionInterface>();
  // Ragionamento CoT di 1° livello condiviso a livello di risposta RAG (usato come fallback)
  philologicalReasoning = input<string | null>(null);
  // Emette al contenitore padre la scelta del filologo (validazione o scarto del supplemento)
  onFeedback = output<SuggestionFeedbackEvent>();

  score = computed(() => this.suggest().score);
  token = computed(() => this.suggest().token_str);
  sentence = computed(() => this.suggest().sentence);

  // Attestazioni puntuali restituite dal retriever per questo specifico suggerimento nel corpus
  attestations = computed<CorpusAttestationInterface[]>(() => this.suggest().attestations || []);
  // Parola ricostruita tramite boundary reconstruction (prefisso + supplemento + suffisso)
  reconstructedWord = computed<string>(() => this.suggest().reconstructed_word || this.token());

  // Segmentazione della frase per applicare lo stile di evidenziazione (.attestation-highlight)
  // alla parola composta sostituendo la lacuna con il suggerimento proposto (reconstructedWord)
  sentenceSegments = computed<HighlightSegment[]>(() => {
    const sent = this.sentence();
    const word = this.reconstructedWord();
    const tok = this.token();
    const target = word || tok;
    return parseHighlightedSegments(sent, target ? [target] : []);
  });
  // Attestazione lessicale nel TLG (da post-processing post-HCB o intercettata dal retriever RAG)
  isTlgAttested = computed<boolean>(
    () =>
      !!this.suggest().tlg_attested ||
      this.attestations().some(a => (a.corpus_id || '').trim().toLowerCase().startsWith('tlg'))
  );
  tlgTooltip = computed<string>(() => {
    const word = this.reconstructedWord();
    return word
      ? `Forma ricostruita "${word}" attestata nel TLG`
      : 'Forma attestata nel TLG';
  });
  // True se il suggerimento non presenta attestazioni documentarie (né nel TLG né nel corpus),
  // rappresentando un'ipotesi o congettura puramente neurale generata dal modello
  isPureAiHypothesis = computed<boolean>(
    () => !this.isTlgAttested() && this.attestations().length === 0
  );
  aiHypothesisTooltip = computed<string>(() => {
    const word = this.reconstructedWord();
    return word
      ? `Forma ricostruita "${word}" non attestata nei corpora: congettura / ipotesi generata dal modello`
      : 'Forma non attestata nei corpora: congettura / ipotesi generata dal modello';
  });
  // CoT di 2° livello specifica per spiegare perché questa singola integrazione è plausibile
  candidateReasoning = computed<string | null>(() => this.suggest().candidate_reasoning || null);
  // Mantiene retrocompatibilità con la CoT di 1° livello se `candidate_reasoning` non è presente
  effectiveReasoning = computed<string | null>(
    () => this.candidateReasoning() || this.suggest().philological_reasoning || this.philologicalReasoning()
  );
  reasoningSegments = computed<HighlightSegment[]>(() => {
    const reasoning = this.effectiveReasoning();
    const forms = this.attestations().map(a => a.attested_form).filter(Boolean);
    return parseHighlightedSegments(reasoning, forms);
  });
  hasCotOrAttestations = computed<boolean>(() => this.attestations().length > 0 || !!this.effectiveReasoning());

  // Stato locale della card per evidenziare visivamente se il supplemento è stato validato o scartato
  feedbackStatus = signal<'accepted' | 'rejected' | null>(null);
  // Pannello CoT e attestazioni del corpus chiuso di default e consultabile su richiesta dall'utente finale
  showCotDetails = signal<boolean>(false);

  constructor(private clipboard: Clipboard) { }

  toggleCotDetails() {
    this.showCotDetails.update(v => !v);
  }

  getAttestationPassageSegments(att: CorpusAttestationInterface): HighlightSegment[] {
    const rawPassage = att.highlighted_passage || att.passage || '';
    const forms = [att.attested_form, this.token()].filter(Boolean);
    return parseHighlightedSegments(rawPassage, forms);
  }

  toggleCopyToast() {
    // Ensure Bootstrap's JavaScript is loaded and accessible
    console.log(this.score(), this.token(), this.sentence() )
    const toastElement = document.querySelector('#copyToast');
    if (toastElement) {
      const toast = new (window as any).bootstrap.Toast(toastElement);
      toast.show();
    }
  }

  copyToClipboard() {
    this.clipboard.copy(this.token());
    this.toggleCopyToast();
  }

  markFeedback(isAccepted: boolean) {
    this.feedbackStatus.set(isAccepted ? 'accepted' : 'rejected');
    this.onFeedback.emit({
      token_str: this.token(),
      sentence: this.sentence(),
      is_accepted: isAccepted,
      score: this.score(),
      reconstructed_word: this.reconstructedWord(),
      tlg_attested: this.isTlgAttested(),
      candidate_reasoning: this.candidateReasoning(),
      philological_reasoning: this.effectiveReasoning(),
      attestations: this.attestations()
    });
  }

}
