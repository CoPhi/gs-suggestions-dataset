import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { SuggestsBoxComponent } from './suggests-box.component';

describe('SuggestsBoxComponent', () => {
  let component: SuggestsBoxComponent;
  let fixture: ComponentFixture<SuggestsBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SuggestsBoxComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SuggestsBoxComponent);
    fixture.componentRef.setInput('suggestions', []);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should emit onClose when close() is called', () => {
    spyOn(component.onClose, 'emit');
    component.close();
    expect(component.onClose.emit).toHaveBeenCalled();
  });

  it('should emit onClose when close button is clicked in template', () => {
    spyOn(component.onClose, 'emit');
    const closeBtn = fixture.nativeElement.querySelector('.close-btn') as HTMLButtonElement;
    expect(closeBtn).toBeTruthy();
    closeBtn.click();
    expect(component.onClose.emit).toHaveBeenCalled();
  });

  it('should display RAG philological reasoning and retrieved passages when present', () => {
    fixture.componentRef.setInput('suggestions', [
      {
        sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλητα',
        token_str: 'λητα',
        score: 0.92,
        philological_reasoning: 'Epiteto omerico di Ares τειχεσιβλῆτα (4 lettere mancanti).',
        retrieved_passages: ['[Corpus: tlg | Ilias] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα']
      }
    ]);
    fixture.detectChanges();

    expect(component.hasRagDetails()).toBeTrue();
    const compiled = fixture.nativeElement as HTMLElement;
    // Nascosto di default, visibile dopo il click sul bottone rag-toggle-btn nell'intestazione
    expect(compiled.querySelector('.rag-details-panel')).toBeFalsy();
    const ragToggleBtn = compiled.querySelector('.rag-toggle-btn') as HTMLButtonElement;
    expect(ragToggleBtn).toBeTruthy();
    ragToggleBtn.click();
    fixture.detectChanges();

    expect(compiled.querySelector('.rag-details-panel')).toBeTruthy();
    expect(compiled.textContent).toContain('Epiteto omerico di Ares');
    expect(compiled.textContent).toContain('τειχεσιβλῆτα');
  });

  it('should track rejected candidates and emit onRefine with human_feedback and rejected_candidates', () => {
    spyOn(component.onFeedback, 'emit');
    spyOn(component.onRefine, 'emit');

    component.handleSuggestionFeedback({
      token_str: 'πλητα',
      sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβπλητα',
      is_accepted: false
    });
    expect(component.rejectedCandidates()).toEqual(['πλητα']);
    expect(component.onFeedback.emit).toHaveBeenCalled();

    component.humanFeedback.set('Preferisci forma omerica attestata');
    fixture.detectChanges();

    expect(component.canRefine()).toBeTrue();
    component.triggerRefinement();

    expect(component.onRefine.emit).toHaveBeenCalledWith({
      human_feedback: 'Preferisci forma omerica attestata',
      rejected_candidates: ['πλητα']
    });
  });

  it('should display corpus attestation badge and open per-suggestion CoT & Corpus panel on demand', () => {
    fixture.componentRef.setInput('suggestions', [
      {
        sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλητα',
        token_str: 'λητα',
        score: 0.94,
        philological_reasoning: 'Il supplemento λητα ricostruisce τειχεσιβλῆτα, attestato in DCLP | P. Bagnall. 13.',
        retrieved_passages: ['[Corpus: DCLP | P. Bagnall. 13] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα'],
        attestations: [
          {
            corpus_id: 'DCLP',
            source_id: 'P. Bagnall. 13',
            attested_form: 'τειχεσιβλῆτα',
            passage: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα',
            doc_freq: 2,
            source_type: 'retrieved_passage'
          }
        ]
      }
    ]);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.corpus-attested-badge')?.textContent).toContain('Attestato in DCLP (P. Bagnall. 13)');
    // Il pannello CoT & Attestazioni è chiuso di default e si apre su richiesta dell'utente
    expect(compiled.querySelector('.cot-attestations-panel')).toBeFalsy();

    const cotBtn = compiled.querySelector('.cot-toggle-btn') as HTMLButtonElement;
    expect(cotBtn).toBeTruthy();
    cotBtn.click();
    fixture.detectChanges();

    const panel = compiled.querySelector('.cot-attestations-panel') as HTMLElement;
    expect(panel).toBeTruthy();
    expect(panel.textContent).toContain('Catena di Ragionamento (Chain-of-Thought)');
    expect(panel.textContent).toContain('[Corpus: DCLP | P. Bagnall. 13]');
    expect(panel.textContent).toContain('τειχεσιβλῆτα');
    // Verifica che l'attestazione trovata nel chunk e nella CoT sia evidenziata in grassetto
    const highlighted = panel.querySelectorAll('strong.attestation-highlight');
    expect(highlighted.length).toBeGreaterThan(0);
    expect(Array.from(highlighted).some(el => el.textContent?.includes('τειχεσιβλῆτα'))).toBeTrue();
  });

  it('should display 2nd-layer candidate_reasoning and secondary loci similes menu on demand', () => {
    fixture.componentRef.setInput('suggestions', [
      {
        sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλητα',
        token_str: 'λητα',
        score: 0.96,
        philological_reasoning: 'CoT 1° livello: i passi omerici recuperati attestano **τειχεσιβλῆτα**.',
        candidate_reasoning: 'CoT 2° livello: completa il vocativo epico **τειχεσιβλῆτα** (4 lettere) attestato in [DCLP | P. Bagnall. 13].',
        retrieved_passages: [
          '[Corpus: DCLP | P. Bagnall. 13] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα'
        ],
        all_retrieved_passages: [
          '[Corpus: DCLP | P. Bagnall. 13] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα',
          '[Corpus: tlg | Hom. Il. 5.31] Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα',
          '[Corpus: PHerc. 1004 | col. 64] οὐ γὰρ ἦν φανερώτερα τὰ κατ αὐτὰ λείπων'
        ],
        attestations: [
          {
            corpus_id: 'DCLP',
            source_id: 'P. Bagnall. 13',
            attested_form: 'τειχεσιβλῆτα',
            passage: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα',
            highlighted_passage: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε **τειχεσιβλῆτα**',
            doc_freq: 2,
            source_type: 'retrieved_passage'
          }
        ]
      }
    ]);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;

    // 1. Apre il pannello di 1° livello (CoT generale + loci similes primari)
    const ragToggleBtn = compiled.querySelector('.rag-toggle-btn') as HTMLButtonElement;
    ragToggleBtn.click();
    fixture.detectChanges();

    expect(compiled.querySelector('.rag-details-panel')?.textContent).toContain('CoT – 1° Livello');
    expect(compiled.querySelector('.rag-details-panel')?.textContent).toContain('Loci Similes Più Rilevanti (1 di 3)');
    // Il menù secondario con gli ulteriori loci similes è chiuso di default
    expect(compiled.querySelector('.secondary-loci-menu')).toBeFalsy();

    // 2. Apre su richiesta il menù secondario con tutti gli altri loci similes trovati per il prompt
    const secondaryBtn = compiled.querySelector('.secondary-loci-toggle-btn') as HTMLButtonElement;
    expect(secondaryBtn).toBeTruthy();
    secondaryBtn.click();
    fixture.detectChanges();

    const secondaryMenu = compiled.querySelector('.secondary-loci-menu') as HTMLElement;
    expect(secondaryMenu).toBeTruthy();
    expect(secondaryMenu.textContent).toContain('Hom. Il. 5.31');
    expect(secondaryMenu.textContent).toContain('PHerc. 1004');

    // 3. Apre su richiesta la CoT di 2° livello specifica della card suggerimento
    const cotBtn = compiled.querySelector('.cot-toggle-btn') as HTMLButtonElement;
    cotBtn.click();
    fixture.detectChanges();

    const cardPanel = compiled.querySelector('.cot-attestations-panel') as HTMLElement;
    expect(cardPanel.textContent).toContain('Plausibilità del Suggerimento (CoT – 2° Livello)');
    expect(cardPanel.textContent).toContain('completa il vocativo epico');
  });

  it('should enrich feedback event with rank, model metadata and alternative candidates for RLHF', () => {
    spyOn(component.onFeedback, 'emit');
    fixture.componentRef.setInput('suggestions', [
      {
        token_str: 'λῆτα',
        sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα',
        score: 0.95,
        tlg_attested: true
      },
      {
        token_str: 'βλῆτα',
        sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιββλῆτα',
        score: 0.75,
        tlg_attested: false
      }
    ]);
    fixture.componentRef.setInput('modelId', 'mod123');
    fixture.componentRef.setInput('modelLabel', 'gs-GreBerta + RAG');
    fixture.componentRef.setInput('modelType', 'BERT');
    fixture.componentRef.setInput('checkpoint', 'CNR-ILC/gs-GreBerta');
    fixture.componentRef.setInput('useRag', true);
    fixture.detectChanges();

    component.handleSuggestionFeedback({
      token_str: 'λῆτα',
      sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλῆτα',
      is_accepted: true,
      score: 0.95
    });

    expect(component.onFeedback.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      token_str: 'λῆτα',
      is_accepted: true,
      rank: 1,
      model_id: 'mod123',
      model_label: 'gs-GreBerta + RAG',
      model_type: 'BERT',
      checkpoint: 'CNR-ILC/gs-GreBerta',
      use_rag: true,
      top_k: 2,
      alternative_candidates: [
        jasmine.objectContaining({
          token_str: 'βλῆτα',
          score: 0.75,
          rank: 2,
          tlg_attested: false
        })
      ]
    }));
  });
});
