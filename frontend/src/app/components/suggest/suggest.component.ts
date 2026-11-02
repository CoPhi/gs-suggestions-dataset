import { Component, computed, input, output, signal } from '@angular/core';
import { SuggestionInterface } from '../../services/api.service';
import { Clipboard } from '@angular/cdk/clipboard';

export interface SuggestionFeedbackEvent {
  token_str: string;
  sentence: string;
  is_accepted: boolean;
}

@Component({
  selector: 'app-suggest',
  imports: [],
  templateUrl: './suggest.component.html',
  styleUrl: './suggest.component.css'
})
export class SuggestComponent {

  suggest = input.required<SuggestionInterface>();
  // Emette al contenitore padre la scelta del filologo (validazione o scarto del supplemento)
  onFeedback = output<SuggestionFeedbackEvent>();

  score = computed(() => this.suggest().score);
  token = computed(() => this.suggest().token_str);
  sentence = computed(() => this.suggest().sentence);

  // Stato locale della card per evidenziare visivamente se il supplemento è stato validato o scartato
  feedbackStatus = signal<'accepted' | 'rejected' | null>(null);

  constructor(private clipboard: Clipboard) { }

  toggleCopyToast() {
    // Ensure Bootstrap's JavaScript is loaded and accessible
    console.log(this.score(), this.token(), this.sentence() )
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

  markFeedback(isAccepted: boolean) {
    this.feedbackStatus.set(isAccepted ? 'accepted' : 'rejected');
    this.onFeedback.emit({
      token_str: this.token(),
      sentence: this.sentence(),
      is_accepted: isAccepted
    });
  }

}
