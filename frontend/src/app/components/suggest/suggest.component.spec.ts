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
});
