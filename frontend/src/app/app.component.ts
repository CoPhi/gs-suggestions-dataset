import { Component, computed, effect, inject, input } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Signal } from '@angular/core';
import { ApiService, modelType } from './services/api.service';
import { toSignal } from '@angular/core/rxjs-interop';
import { ModelComponent } from './components/model/model.component';

@Component({
  selector: 'app-root',
  imports: [ModelComponent],
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
