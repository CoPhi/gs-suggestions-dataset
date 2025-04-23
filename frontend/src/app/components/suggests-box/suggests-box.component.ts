import { Component, inject } from '@angular/core';
import { setThrowInvalidWriteToSignalError } from '@angular/core/primitives/signals';
import { AbstractControl, FormControl, FormGroup, ReactiveFormsModule, ValidationErrors, Validators } from '@angular/forms';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-suggests-box',
  imports: [ReactiveFormsModule],
  templateUrl: './suggests-box.component.html',
  styleUrl: './suggests-box.component.css'
})
export class SuggestsBoxComponent {

  api = inject(ApiService);

  isContextValid = (c: AbstractControl): ValidationErrors | null => {
    if (!c.value) return { notvalid: true };
    return c.value.search("[MASK]") !== -1 ? null : { notvalid: true };
  }

  form = new FormGroup({
    text: new FormControl<string>('', { validators: [Validators.required, this.isContextValid, Validators.minLength(10)] }),
    modelID: new FormControl<string>('', {
      validators: [Validators.required, Validators.minLength(16),
      Validators.maxLength(16)]
    }),
  })

  generateSuggestions() {
    
  const { text, modelID } = this.form.getRawValue();
  this.api.generateSuggestion(text!, modelID!).subscribe((response) => {
    next: console.log(response);
    error: (error:any) => {
      console.error('Error:', error);
    }
  })
}
}
