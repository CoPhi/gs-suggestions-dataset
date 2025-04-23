import { Component, computed, effect, input } from '@angular/core';
import { BERTModelInterface, modelType, NgramsModelInterface } from '../../services/api.service';
import { ClipboardModule } from '@angular/cdk/clipboard';

@Component({
  selector: 'app-model',
  imports: [ClipboardModule],
  templateUrl: './model.component.html',
  styleUrl: './model.component.css'
})
export class ModelComponent {

    model = input.required<modelType>(); // Modello da visualizzare

    id = computed(() => this.model()._id); 
    isBERT = computed(() => this.model().TYPE === 'BERT'); // Verifica se il modello è di tipo BERT
    bertmodel = computed(() => <BERTModelInterface>this.model()); // Modello BERT
    ngramsmodel = computed(() => <NgramsModelInterface>this.model()); // Modello N-grams
}
