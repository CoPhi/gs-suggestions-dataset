import { Component, computed, effect, input, InputSignal } from '@angular/core';
import { SuggestionInterface } from '../../services/api.service';
import { Clipboard } from '@angular/cdk/clipboard';

@Component({
  selector: 'app-suggest',
  imports: [],
  templateUrl: './suggest.component.html',
  styleUrl: './suggest.component.css'
})
export class SuggestComponent {

  suggest = input.required<SuggestionInterface>();

  score = computed(() => this.suggest().score);
  token = computed(() => this.suggest().token_str);
  sentence = computed(() => this.suggest().sentence);

  constructor(private clipboard: Clipboard) { }

  toggleCopyToast() {
    // Ensure Bootstrap's JavaScript is loaded and accessible
    const toastElement = document.querySelector('#copyToast');
    if (toastElement) {
      const toast = new (window as any).bootstrap.Toast(toastElement);
      toast.show();
    }
  }

  copyToClipboard() {
    this.clipboard.copy(this.token());
    this.toggleCopyToast();
  }


}
