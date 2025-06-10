import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelsBoxComponent } from './models-box.component';

describe('ModelsBoxComponent', () => {
  let component: ModelsBoxComponent;
  let fixture: ComponentFixture<ModelsBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelsBoxComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelsBoxComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
