import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreateModelBoxComponent } from './create-model-box.component';

describe('CreateModelBoxComponent', () => {
  let component: CreateModelBoxComponent;
  let fixture: ComponentFixture<CreateModelBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreateModelBoxComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CreateModelBoxComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
