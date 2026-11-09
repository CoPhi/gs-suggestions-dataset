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
    expect(component.onFeedback.emit).toHaveBeenCalledWith({
      token_str: 'test',
      sentence: 'foo [test] bar',
      is_accepted: true
    });

    rejectBtn.click();
    fixture.detectChanges();
    expect(component.feedbackStatus()).toBe('rejected');
    expect(component.onFeedback.emit).toHaveBeenCalledWith({
      token_str: 'test',
      sentence: 'foo [test] bar',
      is_accepted: false
    });
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
});
