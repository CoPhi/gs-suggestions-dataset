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
});
