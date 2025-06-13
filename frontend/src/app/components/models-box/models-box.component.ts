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
  curr_id = model.required<string | null>();
  
}
