import { Component, inject, input } from '@angular/core';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { ApiService, SuggestionInterface } from '../../services/api.service';
import { SuggestComponent } from '../suggest/suggest.component';

@Component({
  selector: 'app-suggests-box',
  imports: [ReactiveFormsModule, FormsModule, SuggestComponent],
  templateUrl: './suggests-box.component.html',
  styleUrl: './suggests-box.component.css'
})
export class SuggestsBoxComponent {

  api = inject(ApiService);
  suggestions = input.required<SuggestionInterface[]>();

}
