import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, Observable } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl = 'http://localhost:8000'; // base URL dell'API FastAPI

  constructor(private http: HttpClient) {}

  getModels(): Observable<modelType[]> {
    return this.http.get<{ models: modelType[] }>(`${this.apiUrl}/models`).pipe(
      map(response => response.models)
    );
  }

  createModel(data: any): Observable<string> {
    return this.http.post<{ ID: string }>(`${this.apiUrl}/model/`, data).pipe(
      map(response => response.ID)
    );
  }

  generateSuggestion(text: string, model: string): Observable<SuggestionInterface[]> {
    return this.http.post<SuggestionInterface[]>(`${this.apiUrl}/predictions`, {
      text,
      model,
    });
  }
}

export type modelType = BERTModelInterface | NgramsModelInterface;

export interface BERTModelInterface {
  MODEL: string; // Nome del modello BERT da utilizzare
  TOKENIZER: string; // Nome del tokenizer da utilizzare
  K_PRED: number; // Numero di predizioni che restituisce il modello
  TYPE: "BERT"; // Tipo di modello
}

export interface NgramsModelInterface {
  LM_SCORE: string;
  GAMMA: number | null;
  MIN_FREQ: number;
  K_PRED: number;
  TEST_SIZE: number;
  N: number;
  CORPUS_NAMES: string[];
  TYPE: "Ngrams";
}

export interface SuggestionInterface {
  text: string; // Testo suggerito
  score: number; // Punteggio del suggerimento
}
