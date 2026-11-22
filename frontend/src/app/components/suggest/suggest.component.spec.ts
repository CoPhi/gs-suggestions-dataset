import { ComponentFixture, fakeAsync, TestBed, tick } from '@angular/core/testing';

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

  it('should emit onFeedback once, hide feedback buttons, and auto-hide thank-you message on accept', fakeAsync(() => {
    const emitSpy = spyOn(component.onFeedback, 'emit');
    const compiled = fixture.nativeElement as HTMLElement;

    const acceptBtn = compiled.querySelector('.feedback-accept-btn') as HTMLButtonElement;
    const rejectBtn = compiled.querySelector('.feedback-reject-btn') as HTMLButtonElement;
    expect(acceptBtn).toBeTruthy();
    expect(rejectBtn).toBeTruthy();

    acceptBtn.click();
    fixture.detectChanges();

    expect(component.feedbackStatus()).toBe('accepted');
    expect(emitSpy).toHaveBeenCalledTimes(1);
    expect(emitSpy).toHaveBeenCalledWith(jasmine.objectContaining({
      token_str: 'test',
      sentence: 'foo [test] bar',
      is_accepted: true,
      score: 0.95,
      reconstructed_word: 'test',
      tlg_attested: false
    }));

    // I pulsanti scompaiono subito e al loro posto appare il ringraziamento minimale
    expect(compiled.querySelector('.feedback-accept-btn')).toBeNull();
    expect(compiled.querySelector('.feedback-reject-btn')).toBeNull();
    expect(compiled.querySelector('.feedback-thankyou-msg')?.textContent).toContain('Grazie, feedback registrato');
    expect(compiled.querySelector('.feedback-status-badge')).toBeTruthy();

    // Eventuali chiamate successive non inviano ulteriori feedback
    component.markFeedback(false);
    expect(component.feedbackStatus()).toBe('accepted');
    expect(emitSpy).toHaveBeenCalledTimes(1);

    // Dopo 3 secondi la scritta temporanea scompare lasciando solo il badge di stato
    tick(3000);
    fixture.detectChanges();
    expect(compiled.querySelector('.feedback-thankyou-msg')).toBeNull();
    expect(compiled.querySelector('.feedback-status-badge')).toBeTruthy();
  }));

  it('should emit onFeedback once and hide feedback buttons on reject', fakeAsync(() => {
    const emitSpy = spyOn(component.onFeedback, 'emit');
    const compiled = fixture.nativeElement as HTMLElement;

    const rejectBtn = compiled.querySelector('.feedback-reject-btn') as HTMLButtonElement;
    expect(rejectBtn).toBeTruthy();

    rejectBtn.click();
    fixture.detectChanges();

    expect(component.feedbackStatus()).toBe('rejected');
    expect(emitSpy).toHaveBeenCalledTimes(1);
    expect(emitSpy).toHaveBeenCalledWith(jasmine.objectContaining({
      token_str: 'test',
      sentence: 'foo [test] bar',
      is_accepted: false,
      score: 0.95,
      reconstructed_word: 'test',
      tlg_attested: false
    }));
    expect(compiled.querySelector('.feedback-accept-btn')).toBeNull();
    expect(compiled.querySelector('.feedback-reject-btn')).toBeNull();
    expect(compiled.querySelector('.feedback-thankyou-msg')).toBeTruthy();

    tick(3000);
    fixture.detectChanges();
    expect(compiled.querySelector('.feedback-thankyou-msg')).toBeNull();
  }));

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

  it('should highlight reconstructed words containing uppercase Greek prefixes or suffixes', () => {
    fixture.componentRef.setInput('suggest', {
      sentence: 'Ἆρες Ἆρες ΤΕΙΧΕΣΙΒλητα',
      token_str: 'λητα',
      score: 0.94,
      reconstructed_word: 'τειχεσιβλητα'
    });
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const highlighted = compiled.querySelectorAll('p strong.attestation-highlight');
    expect(highlighted.length).toBe(1);
    expect(highlighted[0].textContent?.trim()).toBe('ΤΕΙΧΕΣΙΒλητα');

    // Fallback anche quando `reconstructed_word` non è valorizzato nel payload
    fixture.componentRef.setInput('suggest', {
      sentence: 'Καὶ ἈΝΘΕΝΤΙον ἔδοξεν',
      token_str: 'ντι',
      score: 0.85
    });
    fixture.detectChanges();

    const fallbackHighlights = compiled.querySelectorAll('p strong.attestation-highlight');
    expect(fallbackHighlights.length).toBe(1);
    expect(fallbackHighlights[0].textContent?.trim()).toBe('ἈΝΘΕΝΤΙον');
  });
});
