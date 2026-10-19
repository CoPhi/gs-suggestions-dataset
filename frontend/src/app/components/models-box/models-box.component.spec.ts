import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { ModelsBoxComponent } from './models-box.component';

describe('ModelsBoxComponent', () => {
  let component: ModelsBoxComponent;
  let fixture: ComponentFixture<ModelsBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelsBoxComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelsBoxComponent);
    fixture.componentRef.setInput('curr_id', null);
    fixture.componentRef.setInput('models', []);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
