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
});
