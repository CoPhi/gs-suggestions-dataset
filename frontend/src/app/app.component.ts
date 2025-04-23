import { Component, effect, inject } from '@angular/core';
import { Signal } from '@angular/core';
import { ApiService, modelType } from './services/api.service';
import { toSignal } from '@angular/core/rxjs-interop';
import { SuggestsBoxComponent } from './components/suggests-box/suggests-box.component';

@Component({
  selector: 'app-root',
  imports: [SuggestsBoxComponent],
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
