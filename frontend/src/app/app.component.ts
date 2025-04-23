import { Component, computed, effect, inject, input } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Signal } from '@angular/core';
import { ApiService, modelType } from './services/api.service';
import { toSignal } from '@angular/core/rxjs-interop';
import { ModelComponent } from './components/model/model.component';
import { CreateModelBoxComponent } from './components/create-model-box/create-model-box.component';
import { SuggestsBoxComponent } from './components/suggests-box/suggests-box.component';

@Component({
  selector: 'app-root',
  imports: [ModelComponent, CreateModelBoxComponent, SuggestsBoxComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'gs-api';

  api = inject(ApiService)

  models = toSignal(this.api.getModels(), { initialValue: [] }) as Signal<modelType[]>;


  debug = effect(() => {
    console.log(this.models());
  }
  );
}
