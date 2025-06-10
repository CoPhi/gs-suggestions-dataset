import { Component, effect, model } from '@angular/core';
import { modelType } from '../../services/api.service';
import { ModelComponent } from "../model/model.component";

@Component({
  selector: 'app-models-box',
  imports: [ModelComponent],
  templateUrl: './models-box.component.html',
  styleUrl: './models-box.component.css'
})
export class ModelsBoxComponent {
  models = model.required<modelType[]>();

  constructor() {
    // Log the models when they are set
    effect(() => {
      console.log("Models available: ", this.models());
    }
    );
  }
}
