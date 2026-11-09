import { Component, computed, input, output, signal } from '@angular/core';
import { CorpusAttestationInterface, SuggestionInterface } from '../../services/api.service';
import { Clipboard } from '@angular/cdk/clipboard';

export interface SuggestionFeedbackEvent {
  token_str: string;
  sentence: string;
  is_accepted: boolean;
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
 * Suddivide un testo (CoT o estratto del corpus) in segmenti normali ed evidenziati (`bold: true`),
 * riconoscendo sia i marcatori Markdown `**...**` restituiti dal backend sia il match diretto
 * (insensibile ai diacritici) con le forme greche attestate (`fallbackForms`).
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

  for (const part of rawParts) {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      segments.push({ text: part.slice(2, -2), bold: true });
      continue;
    }

    if (normalizedTargets.size === 0) {
      segments.push({ text: part, bold: false });
      continue;
    }

    // Scansiona i token alfabetici greci nel segmento per evidenziare la forma attestata
    // anche qualora il testo non contenga già marcatori **...**
    const tokenRegex = /([\p{L}\p{M}]+)/gu;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = tokenRegex.exec(part)) !== null) {
      const word = match[0];
      const normWord = normalizeGreekForMatch(word);
      if (normalizedTargets.has(normWord)) {
        if (match.index > lastIndex) {
          segments.push({ text: part.slice(lastIndex, match.index), bold: false });
        }
        segments.push({ text: word, bold: true });
        lastIndex = match.index + word.length;
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
      is_accepted: isAccepted
    });
  }

}
