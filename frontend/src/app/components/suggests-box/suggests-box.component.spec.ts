import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SuggestsBoxComponent } from './suggests-box.component';

describe('SuggestsBoxComponent', () => {
  let component: SuggestsBoxComponent;
  let fixture: ComponentFixture<SuggestsBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SuggestsBoxComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SuggestsBoxComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
