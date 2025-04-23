import { Component, effect, inject } from '@angular/core';
import { AbstractControl, FormArray, FormControl, FormGroup, ReactiveFormsModule, ValidationErrors, Validators } from '@angular/forms';
import { ApiService } from '../../services/api.service';

const LM_TYPES: string[] = ["LIDSTONE", "MLE"];
const GAMMAS: number[] = [0.001, 0.01, 0.1];
const K_PREDICTIONS: number[] = [10, 20];
const MIN_FREQS: number[] = [2, 3, 4];
const TEST_SIZES: number[] = [0.05, 0.1];
const DIMENSIONS: number[] = [2, 3];
@Component({
  selector: 'app-create-model-box',
  imports: [ReactiveFormsModule],
  templateUrl: './create-model-box.component.html',
  styleUrl: './create-model-box.component.css'
})

export class CreateModelBoxComponent {

  api = inject(ApiService);

  type_model = new FormControl<string>('BERT')

  IsMinFrequencyValid(c: AbstractControl): ValidationErrors | null {
    if (!c.value) return { notvalid: true };
    return MIN_FREQS.includes(c.value) ? null : { notvalid: true };
  }

  IsKPredValid(c: AbstractControl): ValidationErrors | null {
    if (!c.value) return { notvalid: true };
    return K_PREDICTIONS.includes(c.value) ? null : { notvalid: true };
  }

  IsTestSizeValid(c: AbstractControl): ValidationErrors | null {
    if (!c.value) return { notvalid: true };
    return TEST_SIZES.includes(c.value) ? null : { notvalid: true };
  }

  IsDimensionValid(c: AbstractControl): ValidationErrors | null {
    if (!c.value) return { notvalid: true };
    return DIMENSIONS.includes(c.value) ? null : { notvalid: true };
  }
  IsGammaValid(c: AbstractControl): ValidationErrors | null {
    if (!c.value) return { notvalid: true };
    return GAMMAS.includes(c.value) ? null : { notvalid: true };
  }
  IsLMTypeValid(c: AbstractControl): ValidationErrors | null {
    if (!c.value) return { notvalid: true };
    return LM_TYPES.includes(c.value) ? null : { notvalid: true };
  }


  ngramsFormGroup = new FormGroup({
    LM_SCORE: new FormControl<string>('', {
      validators: [Validators.required, this.IsLMTypeValid],
    }),
    GAMMA: new FormControl<number | null>(null, {
      validators: [this.IsGammaValid],
    }),
    MIN_FREQ: new FormControl<number>(0, {
      validators: [Validators.required, this.IsMinFrequencyValid],
    }),
    K_PRED: new FormControl<number>(0, {
      validators: [Validators.required, this.IsKPredValid],
    }),
    TEST_SIZE: new FormControl<number>(0, {
      validators: [Validators.required, this.IsTestSizeValid],
    }),
    N: new FormControl<number>(0, {
      validators: [Validators.required, this.IsDimensionValid],
    }),
    CORPUS_NAMES: new FormArray<FormControl<string>>([]),
  })


  bertFormGroup = new FormGroup({
    CHECKPOINT: new FormControl<string>('', {
      validators: [Validators.required],
      nonNullable: true
    }),
    K_PRED: new FormControl<number>(0, {
      validators: [Validators.required, this.IsKPredValid],
      nonNullable: true
    }),
  })

  createModel() {
    const modelType = this.type_model.value;
    let data: any;
    if (modelType === 'BERT') {
      data = this.bertFormGroup.getRawValue();
      console.log({ ...data, TYPE: modelType });

      this.api.createModel({ ...data, TYPE: modelType }).subscribe((response) => {
        next: console.log(response);
        error: (error: any) => {
          console.error('Error creating model:', error);
        }
      })


    } else if (modelType === 'Ngrams') {
      data = this.ngramsFormGroup.getRawValue();

      const checkedCorpus = Array.from(document.querySelectorAll<HTMLInputElement>('input[name="corpus"]'))
        .filter(input => input.checked)
        .map(input => input.value)

      this.api.createModel({ ...data, TYPE: modelType, CORPUS_NAMES: checkedCorpus }).subscribe((response) => {
        next: console.log(response);
        error: (error: any) => {
          console.error('Error creating model:', error);
        }
      });
    }
  }
}