import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SuggestComponent } from './suggest.component';

describe('SuggestComponent', () => {
  let component: SuggestComponent;
  let fixture: ComponentFixture<SuggestComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SuggestComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SuggestComponent);
    fixture.componentRef.setInput('suggest', {
      sentence: 'foo [test] bar',
      token_str: 'test',
      score: 0.95
    });
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should emit onFeedback and update feedbackStatus when accept or reject buttons are clicked', () => {
    spyOn(component.onFeedback, 'emit');
    const compiled = fixture.nativeElement as HTMLElement;

    const acceptBtn = compiled.querySelector('.feedback-accept-btn') as HTMLButtonElement;
    const rejectBtn = compiled.querySelector('.feedback-reject-btn') as HTMLButtonElement;
    expect(acceptBtn).toBeTruthy();
    expect(rejectBtn).toBeTruthy();

    acceptBtn.click();
    fixture.detectChanges();
    expect(component.feedbackStatus()).toBe('accepted');
    expect(component.onFeedback.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      token_str: 'test',
      sentence: 'foo [test] bar',
      is_accepted: true,
      score: 0.95,
      reconstructed_word: 'test',
      tlg_attested: false
    }));

    rejectBtn.click();
    fixture.detectChanges();
    expect(component.feedbackStatus()).toBe('rejected');
    expect(component.onFeedback.emit).toHaveBeenCalledWith(jasmine.objectContaining({
      token_str: 'test',
      sentence: 'foo [test] bar',
      is_accepted: false,
      score: 0.95,
      reconstructed_word: 'test',
      tlg_attested: false
    }));
  });

  it('should render the "tlg" badge with reconstructed word tooltip when tlg_attested is true', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.tlg-attested-badge')).toBeNull();

    fixture.componentRef.setInput('suggest', {
      sentence: 'βροτολοιγὲ μιαιφόνε τειχεσιβλητα',
      token_str: 'λητα',
      score: 0.92,
      tlg_attested: true,
      reconstructed_word: 'τειχεσιβλητα'
    });
    fixture.detectChanges();

    const tlgBadge = compiled.querySelector('.tlg-attested-badge') as HTMLElement;
    expect(tlgBadge).toBeTruthy();
    expect(tlgBadge.textContent?.trim()).toBe('tlg');
    expect(tlgBadge.getAttribute('title')).toContain('τειχεσιβλητα');
  });

  it('should highlight the reconstructed word composed in the sentence with attestation-highlight style', () => {
    fixture.componentRef.setInput('suggest', {
      sentence: 'Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλητα',
      token_str: 'λητα',
      score: 0.92,
      reconstructed_word: 'τειχεσιβλητα'
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const highlighted = compiled.querySelector('p strong.attestation-highlight') as HTMLElement;
    expect(highlighted).toBeTruthy();
    expect(highlighted.textContent?.trim()).toBe('τειχεσιβλητα');
    expect(compiled.querySelector('p')?.textContent).toContain('Ἆρες Ἆρες βροτολοιγὲ μιαιφόνε τειχεσιβλητα');
  });

  it('should render the "AI" badge when suggestion is not attested in TLG or corpus', () => {
    fixture.componentRef.setInput('suggest', {
      sentence: 'τὰ κατ\' αὐτὰ λείπων ὑπὲρ',
      token_str: 'ι',
      score: 0.88,
      tlg_attested: false,
      reconstructed_word: 'λείπων',
      attestations: []
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const aiBadge = compiled.querySelector('.ai-hypothesis-badge') as HTMLElement;
    expect(aiBadge).toBeTruthy();
    expect(aiBadge.textContent?.trim()).toContain('AI');
    expect(aiBadge.getAttribute('title')).toContain('non attestata nei corpora');
  });

  it('should not render the "AI" badge when suggestion is attested in TLG or corpus', () => {
    fixture.componentRef.setInput('suggest', {
      sentence: 'βροτολοιγὲ μιαιφόνε τειχεσιβλητα',
      token_str: 'λητα',
      score: 0.92,
      tlg_attested: true,
      reconstructed_word: 'τειχεσιβλητα',
      attestations: []
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.ai-hypothesis-badge')).toBeNull();
    expect(compiled.querySelector('.tlg-attested-badge')).toBeTruthy();
  });
});
