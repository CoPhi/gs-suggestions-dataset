import { Component, computed, input } from '@angular/core';
import { BERTModelInterface, modelType, NgramsModelInterface } from '../../services/api.service';

@Component({
  selector: 'app-model',
  imports: [],
  templateUrl: './model.component.html',
  styleUrl: './model.component.css'
})
export class ModelComponent {

    model = input.required<modelType>(); // Modello da visualizzare
    isBERT = computed(() => this.model().TYPE === 'BERT'); // Verifica se il modello è di tipo BERT

    bertmodel = computed(() => <BERTModelInterface>this.model()); // Modello BERT
    ngramsmodel = computed(() => <NgramsModelInterface>this.model()); // Modello N-grams
}
