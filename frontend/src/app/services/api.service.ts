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

  getSupportedRagLms(): Observable<SupportedRAGModelInterface[]> {
    return this.http.get<{ models: SupportedRAGModelInterface[] }>(`${this.apiUrl}/models/rag/supported-lms`).pipe(
      map(response => response.models)
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
    if (options?.use_rag) {
      params['use_rag'] = true;
    }

    return this.http.get<{ predictions: SuggestionInterface[] }>(`${this.apiUrl}/predictions`, {
      params,
      responseType: 'json'
    }).pipe(
      map(response => response.predictions)
    );
  }

  /**
   * Esegue in parallelo la consultazione di più modelli sincronizzando le risposte su una barriera.
   */
  generateBatchSuggestions(
    model_ids: string[],
    context: string,
    num_tokens: number,
    num_predictions: number,
    options?: HitlRefinementOptions
  ): Observable<ModelPredictionResultInterface[]> {
    const params: Record<string, any> = { model_ids, context, num_tokens, num_predictions };
    if (options?.human_feedback && options.human_feedback.trim()) {
      params['human_feedback'] = options.human_feedback.trim();
    }
    if (options?.rejected_candidates && options.rejected_candidates.length > 0) {
      params['rejected_candidates'] = options.rejected_candidates;
    }
    if (options?.use_rag) {
      params['use_rag'] = true;
    }

    return this.http.get<{ results: ModelPredictionResultInterface[] }>(`${this.apiUrl}/predictions/batch`, {
      params,
      responseType: 'json'
    }).pipe(
      map(response => response.results)
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
  use_rag?: boolean;
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

export type RagBackendType = "auto" | "hf_api" | "local" | "openai_compatible" | "neuro_lexical";

/**
 * Riconosce se un checkpoint appartiene alla famiglia di modelli Masked Language Model BERT greci
 * (es. gs-GreBerta, gs-aristoBERTo, gs-Logion, Ancient-Greek-BERT).
 */
export function isBertCheckpoint(checkpoint: string | null | undefined): boolean {
  const clean = (checkpoint || '').trim().toLowerCase();
  if (!clean) return false;
  if (['qwen', 'llama', 'gemma', 'deepseek', 'mistral', 'ithaca'].some(kw => clean.includes(kw))) {
    return false;
  }
  return ['greberta', 'aristoberto', 'logion', 'ancient-greek-bert', 'bert'].some(kw => clean.includes(kw));
}

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

export interface SupportedRAGModelInterface {
  id: string; // Identifier Hugging Face repo o tag Ollama
  name: string; // Etichetta human-readable
  category: string; // Qwen, Llama, Gemma, DeepSeek, Ollama
  recommended: boolean; // Flag per evidenziare il modello consigliato di default
  description: string; // Descrizione e dettagli del provider
}

export interface NgramsModelInterface {
  _id: string; // ID del modello
  LM_SCORE: string;
  GAMMA: number | null;
  N: number;
  CORPUS_NAMES: string[];
  TYPE: "Ngrams";
}

export interface CorpusAttestationInterface {
  corpus_id: string; // Collezione del corpus (es. 'tlg', 'DCLP', 'DDbDP', 'PHerc.')
  source_id: string; // Titolo dell'opera, papiro o luogo di attestazione
  attested_form: string; // Forma greca completa attestata nel documento
  passage: string; // Estratto testuale del documento che attesta il suggerimento
  highlighted_passage?: string; // Estratto testuale con la forma attestata evidenziata in grassetto (**...**)
  doc_freq: number; // Numero di documenti nel corpus che attestano la forma
  source_type?: string; // 'retrieved_passage' | 'corpus_index'
}

export interface SuggestionInterface {
  sentence: string;
  token_str: string; // token suggerito
  score: number; // Punteggio del suggerimento
  tlg_attested?: boolean; // True se la parola ricostruita (boundary reconstruction) è attestata nel TLG
  reconstructed_word?: string | null; // Parola greca ricostruita col contesto (prefisso + supplemento + suffisso)
  philological_reasoning?: string | null; // CoT di 1° livello: ragionamento logico/filologico generale (modelli RAG)
  candidate_reasoning?: string | null; // CoT di 2° livello: spiegazione specifica per il singolo suggerimento (modelli RAG)
  retrieved_passages?: string[] | null; // Passi paralleli / loci similes più rilevanti usati nel prompt (modelli RAG)
  all_retrieved_passages?: string[] | null; // Tutti i loci similes trovati dal retriever per il prompt (modelli RAG)
  attestations?: CorpusAttestationInterface[] | null; // Attestazioni puntuali del suggerimento nel corpus (modelli RAG)
}

export interface ModelPredictionResultInterface {
  model_id: string;
  model_label: string;
  model_type: string;
  checkpoint?: string | null;
  use_rag?: boolean;
  predictions: SuggestionInterface[];
  error?: string | null;
}
