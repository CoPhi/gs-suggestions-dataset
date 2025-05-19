import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, Observable } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';


@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl = 'http://127.0.0.1:8000'; // base URL dell'API FastAPI

  constructor(private http: HttpClient) {}

  getModels(): Observable<modelType[]> {
    return this.http.get<{ models: modelType[] }>(`${this.apiUrl}/models`).pipe(
      map(response => response.models)
    );
  }

  createModel(data: modelType): Observable<string> {
    return this.http.post<{ ID: string }>(`${this.apiUrl}/models`, data).pipe(
      map(response => response.ID)
    );
  }

   deleteModel(model_id: string): Observable<string> {
    return this.http.delete<{ description: string }>(`${this.apiUrl}/models/${model_id}`, {
      responseType: 'json'
    }).pipe(
      map(response => response.description)
    )
  }

  generateSuggestion(model_id: string, context: string, num_tokens: number, num_predictions: number ): Observable<SuggestionInterface[]> {
    return this.http.get< {predictions : SuggestionInterface[]}>(`${this.apiUrl}/predictions`, {
      params: { model_id, context, num_tokens, num_predictions },
      responseType: 'json'
    }).pipe(
      map(response => response.predictions)
    );
  }
}

export type modelType = BERTModelInterface | NgramsModelInterface;

export interface BERTModelInterface {
  _id: string; // ID del modello
  CHECKPOINT: string; // Nome del modello BERT da utilizzare
  TYPE: "BERT"; // Tipo di modello
}

export interface NgramsModelInterface {
  _id: string; // ID del modello
  LM_SCORE: string;
  GAMMA: number | null;
  N: number;
  CORPUS_NAMES: string[];
  TYPE: "Ngrams";
}

export interface SuggestionInterface {
  sentence: string;
  token_str: string; // token suggerito
  score: number; // Punteggio del suggerimento
}
