import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of } from 'rxjs';

import { ModelsBoxComponent } from './models-box.component';
import { ApiService } from '../../services/api.service';

describe('ModelsBoxComponent', () => {
  let component: ModelsBoxComponent;
  let fixture: ComponentFixture<ModelsBoxComponent>;
  let apiService: ApiService;

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
    apiService = TestBed.inject(ApiService);
    fixture.componentRef.setInput('curr_id', null);
    fixture.componentRef.setInput('models', []);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should toggle RAG creation form and register a new RAG model', () => {
    expect(component.showAddRagForm()).toBeFalse();
    component.toggleAddRagForm();
    fixture.detectChanges();
    expect(component.showAddRagForm()).toBeTrue();

    spyOn(apiService, 'createModel').and.returnValue(of('67066928e4414eb1d8109bfa'));
    component.applyPreset('Qwen/Qwen2.5-3B-Instruct');
    component.submitRagModel();
    fixture.detectChanges();

    expect(apiService.createModel).toHaveBeenCalledWith({
      CHECKPOINT: 'Qwen/Qwen2.5-3B-Instruct',
      BACKEND: 'auto',
      TOP_K_PASSAGES: 5,
      COMPILED_PROGRAM_PATH: null,
      TYPE: 'RAG'
    });
    expect(component.models().length).toBe(1);
    expect(component.curr_id()).toBe('67066928e4414eb1d8109bfa');
    expect(component.showAddRagForm()).toBeFalse();
  });
});
