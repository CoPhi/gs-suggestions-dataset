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

  createModel(data: modelType | CreateRAGModelPayload): Observable<string> {
    return this.http.post<{ ID: string }>(`${this.apiUrl}/models`, data).pipe(
      map(response => response.ID)
    );
  }

   deleteModel(model_id: string): Observable<any> {
    return this.http.delete<any>(`${this.apiUrl}/models/${model_id}`, {
      responseType: 'json'
    });
  }

  generateSuggestion(
    model_id: string,
    context: string,
    num_tokens: number,
    num_predictions: number,
    options?: HitlRefinementOptions
  ): Observable<SuggestionInterface[]> {
    // Costruisce i query params mantenendo invariata la firma base e aggiungendo
    // opzionalmente i vincoli Human-in-the-loop (Fase 2) solo quando valorizzati.
    const params: Record<string, any> = { model_id, context, num_tokens, num_predictions };
    if (options?.human_feedback && options.human_feedback.trim()) {
      params['human_feedback'] = options.human_feedback.trim();
    }
    if (options?.rejected_candidates && options.rejected_candidates.length > 0) {
      params['rejected_candidates'] = options.rejected_candidates;
    }

    return this.http.get<{ predictions: SuggestionInterface[] }>(`${this.apiUrl}/predictions`, {
      params,
      responseType: 'json'
    }).pipe(
      map(response => response.predictions)
    );
  }

  /**
   * Registra su MongoDB la validazione (is_accepted=true) o lo scarto (is_accepted=false)
   * di un supplemento da parte del filologo (Human-in-the-loop Fase 1).
   */
  submitFeedback(payload: FeedbackPayloadInterface): Observable<FeedbackResponseInterface> {
    return this.http.post<FeedbackResponseInterface>(`${this.apiUrl}/feedback`, payload, {
      responseType: 'json'
    });
  }
}

export interface HitlRefinementOptions {
  human_feedback?: string;
  rejected_candidates?: string[];
}

export interface FeedbackPayloadInterface {
  model_id: string;
  context: string;
  token_str: string;
  sentence?: string | null;
  is_accepted: boolean;
  notes?: string | null;
}

export interface FeedbackResponseInterface {
  id: string;
  status: string;
}

export type RagBackendType = "auto" | "hf_api" | "local" | "openai_compatible";

export type modelType = BERTModelInterface | NgramsModelInterface | IthacaModelInterface | RAGModelInterface;

export interface BERTModelInterface {
  _id: string; // ID del modello
  CHECKPOINT: string; // Nome del modello BERT da utilizzare
  TYPE: "BERT"; // Tipo di modello
}

export interface IthacaModelInterface {
  _id: string; // ID del modello
  CHECKPOINT: string; // Nome del checkpoint o repo Hugging Face di Ithaca
  TYPE: "Ithaca"; // Tipo di modello
}

export interface RAGModelInterface {
  _id: string; // ID del modello
  CHECKPOINT: string; // Nome del modello LLM open-weights su Hugging Face
  BACKEND: RagBackendType; // Strategia di esecuzione ('auto', 'hf_api', 'local', 'openai_compatible')
  TOP_K_PASSAGES: number; // Numero di loci similes recuperati da TLG/MAAT
  COMPILED_PROGRAM_PATH?: string | null; // Percorso opzionale al programma DSPy compilato (.json)
  TYPE: "RAG"; // Tipo di modello
}

export interface CreateRAGModelPayload {
  CHECKPOINT: string;
  BACKEND: RagBackendType;
  TOP_K_PASSAGES: number;
  COMPILED_PROGRAM_PATH?: string | null;
  TYPE: "RAG";
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
  philological_reasoning?: string | null; // Ragionamento Chain-of-Thought (modelli RAG)
  retrieved_passages?: string[] | null; // Passi paralleli / loci similes da TLG & MAAT (modelli RAG)
}
