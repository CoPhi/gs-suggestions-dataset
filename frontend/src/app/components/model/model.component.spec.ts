import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { ModelComponent } from './model.component';
import { IthacaModelInterface, RAGModelInterface } from '../../services/api.service';

describe('ModelComponent', () => {
  let component: ModelComponent;
  let fixture: ComponentFixture<ModelComponent>;

  const mockIthacaModel: IthacaModelInterface = {
    _id: '67066928e4414eb1d8109bf1',
    CHECKPOINT: 'CNR-ILC/gs-ithaca-tlg',
    TYPE: 'Ithaca'
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelComponent);
    fixture.componentRef.setInput('curr_id', null);
    fixture.componentRef.setInput('model', mockIthacaModel);
    fixture.componentRef.setInput('models', [mockIthacaModel]);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create and identify Ithaca model', () => {
    expect(component).toBeTruthy();
    expect(component.isIthaca()).toBeTrue();
    expect(component.isBERT()).toBeFalse();
  });

  it('should render badges without icons', () => {
    const compiled = fixture.nativeElement as HTMLElement;
    const badge = compiled.querySelector('.badge-type');
    expect(badge).toBeTruthy();
    expect(badge?.querySelector('i')).toBeNull();
    expect(badge?.textContent?.trim()).toBe('Ithaca');
  });

  it('should not render infilling metadata for BERT model', () => {
    const mockBertModel = {
      _id: '67066928e4414eb1d8109bf2',
      CHECKPOINT: 'pranaydeeps/Ancient-Greek-BERT',
      TYPE: 'BERT' as const
    };
    fixture.componentRef.setInput('model', mockBertModel);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    const badge = compiled.querySelector('.badge-type');
    expect(badge?.textContent?.trim()).toBe('BERT');
    expect(badge?.querySelector('i')).toBeNull();

    const textContent = compiled.textContent || '';
    expect(textContent).not.toContain('Infilling');
    expect(textContent).not.toContain('Token Masking');
  });

  it('should identify and render RAG model card with backend and loci similes', () => {
    const mockRagModel: RAGModelInterface = {
      _id: '67066928e4414eb1d8109bf9',
      CHECKPOINT: 'Qwen/Qwen2.5-7B-Instruct',
      BACKEND: 'auto',
      TOP_K_PASSAGES: 5,
      COMPILED_PROGRAM_PATH: null,
      TYPE: 'RAG'
    };
    fixture.componentRef.setInput('model', mockRagModel);
    fixture.detectChanges();

    expect(component.isRAG()).toBeTrue();
    expect(component.isIthaca()).toBeFalse();
    expect(component.isBERT()).toBeFalse();

    const compiled = fixture.nativeElement as HTMLElement;
    const badge = compiled.querySelector('.badge-type');
    expect(badge?.textContent?.trim()).toBe('RAG');
    expect(badge?.querySelector('i')).toBeNull();

    const textContent = compiled.textContent || '';
    expect(textContent).toContain('Qwen/Qwen2.5-7B-Instruct');
    expect(textContent).toContain('DSPy Hybrid RAG');
    expect(textContent).toContain('top-k = 5');
  });
});
