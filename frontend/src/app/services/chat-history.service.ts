import { Injectable } from '@angular/core';
import { SuggestionInterface } from './api.service';

export interface ModelSuggestionBoxItem {
  modelId: string;
  modelLabel: string;
  modelType: string;
  checkpoint?: string | null;
  use_rag?: boolean | null;
  suggestions: SuggestionInterface[];
  error?: string | null;
  pending?: boolean;
  submittedFeedback?: Partial<Record<string, 'accepted' | 'rejected'>>;
}

export interface ConversationTurn {
  turnId: string;
  sessionId: string;
  createdAt: string;
  rawText: string;
  normalizedText: string;
  numTokens: number;
  numPredictions: number;
  useRag?: boolean;
  isRefinement: boolean;
  parentTurnId?: string | null;
  appliedHumanFeedback?: string;
  appliedRejectedCandidates: string[];
  modelBoxes: ModelSuggestionBoxItem[];
  activeModelId: string;
  userSelectedTab?: boolean;
  isCollapsed?: boolean;
  turnRejectedCandidates: string[];
  turnHumanFeedback: string;
}

export interface ChatHistorySession {
  chatId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  turns: ConversationTurn[];
}

// Soglia di conservazione dello storico locale (7 giorni)
export const CHAT_HISTORY_TTL_MS = 7 * 24 * 60 * 60 * 1000;

const DB_NAME = 'gs_chat_history_db';
const DB_VERSION = 1;
const STORE_NAME = 'chats';

@Injectable({
  providedIn: 'root',
})
export class ChatHistoryService {
  private dbPromise: Promise<IDBDatabase | null> | null = null;
  // Fallback trasparente in memoria qualora IndexedDB non sia disponibile nell'ambiente corrente
  private memoryFallback = new Map<string, ChatHistorySession>();
  // Coda FIFO per serializzare tutte le operazioni su IndexedDB ed evitare race condition
  private opQueue: Promise<unknown> = Promise.resolve();

  private runSerialized<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.opQueue.then(fn, fn);
    this.opQueue = next.catch(() => {});
    return next;
  }

  flushQueue(): Promise<void> {
    return this.opQueue.then(() => undefined);
  }

  private cloneSession(session: ChatHistorySession): ChatHistorySession {
    return JSON.parse(JSON.stringify(session)) as ChatHistorySession;
  }

  private sortSessionsDesc(sessions: ChatHistorySession[]): ChatHistorySession[] {
    return [...sessions].sort((a, b) => {
      const timeB = Date.parse(b.updatedAt || b.createdAt || '') || 0;
      const timeA = Date.parse(a.updatedAt || a.createdAt || '') || 0;
      return timeB - timeA;
    });
  }

  private openDatabase(): Promise<IDBDatabase | null> {
    if (this.dbPromise) {
      return this.dbPromise;
    }

    const idbFactory = (globalThis as { indexedDB?: IDBFactory }).indexedDB;
    if (!idbFactory || typeof idbFactory.open !== 'function') {
      this.dbPromise = Promise.resolve(null);
      return this.dbPromise;
    }

    this.dbPromise = new Promise<IDBDatabase | null>((resolve) => {
      try {
        const request = idbFactory.open(DB_NAME, DB_VERSION);

        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            const store = db.createObjectStore(STORE_NAME, { keyPath: 'chatId' });
            store.createIndex('updatedAt', 'updatedAt', { unique: false });
          }
        };

        request.onsuccess = () => {
          const db = request.result;
          db.onversionchange = () => {
            db.close();
            this.dbPromise = null;
          };
          db.onclose = () => {
            this.dbPromise = null;
          };
          resolve(db);
        };

        request.onerror = () => {
          this.dbPromise = null;
          resolve(null);
        };

        request.onblocked = () => {
          resolve(null);
        };
      } catch {
        this.dbPromise = null;
        resolve(null);
      }
    });

    return this.dbPromise;
  }

  getAllChats(): Promise<ChatHistorySession[]> {
    return this.runSerialized(() => this._getAllChatsInternal());
  }

  private async _getAllChatsInternal(): Promise<ChatHistorySession[]> {
    const db = await this.openDatabase();
    if (!db) {
      return this.sortSessionsDesc(Array.from(this.memoryFallback.values()).map(s => this.cloneSession(s)));
    }

    return new Promise<ChatHistorySession[]>((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.getAll();
        req.onsuccess = () => {
          const results = Array.isArray(req.result) ? (req.result as ChatHistorySession[]) : [];
          resolve(this.sortSessionsDesc(results));
        };
        req.onerror = () => resolve([]);
      } catch {
        resolve([]);
      }
    });
  }

  getChat(chatId: string): Promise<ChatHistorySession | null> {
    return this.runSerialized(async () => {
      if (!chatId) return null;
      const db = await this.openDatabase();
      if (!db) {
        const item = this.memoryFallback.get(chatId);
        return item ? this.cloneSession(item) : null;
      }

      return new Promise<ChatHistorySession | null>((resolve) => {
        try {
          const tx = db.transaction(STORE_NAME, 'readonly');
          const store = tx.objectStore(STORE_NAME);
          const req = store.get(chatId);
          req.onsuccess = () => {
            resolve((req.result as ChatHistorySession | undefined) ?? null);
          };
          req.onerror = () => resolve(null);
        } catch {
          resolve(null);
        }
      });
    });
  }

  private async _purgeExpiredInternal(
    maxAgeMs: number = CHAT_HISTORY_TTL_MS,
    nowMs: number = Date.now()
  ): Promise<number> {
    const cutoff = nowMs - maxAgeMs;
    const chats = await this._getAllChatsInternal();
    const expiredIds = chats
      .filter((chat) => {
        const ts = Date.parse(chat.updatedAt || chat.createdAt || '');
        return !Number.isFinite(ts) || ts < cutoff;
      })
      .map((chat) => chat.chatId);

    if (expiredIds.length === 0) {
      return 0;
    }

    const db = await this.openDatabase();
    if (!db) {
      for (const id of expiredIds) {
        this.memoryFallback.delete(id);
      }
      return expiredIds.length;
    }

    return new Promise<number>((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        for (const id of expiredIds) {
          store.delete(id);
        }
        tx.oncomplete = () => resolve(expiredIds.length);
        tx.onerror = () => resolve(0);
        tx.onabort = () => resolve(0);
      } catch {
        resolve(0);
      }
    });
  }

  private async _saveChatInternal(session: ChatHistorySession): Promise<void> {
    const cleanSession = this.cloneSession(session);
    const db = await this.openDatabase();
    if (!db) {
      this.memoryFallback.set(cleanSession.chatId, cleanSession);
      return;
    }

    return new Promise<void>((resolve) => {
      try {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        store.put(cleanSession);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  saveChat(session: ChatHistorySession): Promise<void> {
    return this.runSerialized(() => this._saveChatInternal(session));
  }

  deleteChat(chatId: string): Promise<void> {
    return this.runSerialized(async () => {
      if (!chatId) return;
      const db = await this.openDatabase();
      if (!db) {
        this.memoryFallback.delete(chatId);
        return;
      }

      return new Promise<void>((resolve) => {
        try {
          const tx = db.transaction(STORE_NAME, 'readwrite');
          const store = tx.objectStore(STORE_NAME);
          store.delete(chatId);
          tx.oncomplete = () => resolve();
          tx.onerror = () => resolve();
          tx.onabort = () => resolve();
        } catch {
          resolve();
        }
      });
    });
  }

  clearAllChats(): Promise<void> {
    return this.runSerialized(async () => {
      const db = await this.openDatabase();
      if (!db) {
        this.memoryFallback.clear();
        return;
      }

      return new Promise<void>((resolve) => {
        try {
          const tx = db.transaction(STORE_NAME, 'readwrite');
          const store = tx.objectStore(STORE_NAME);
          store.clear();
          tx.oncomplete = () => resolve();
          tx.onerror = () => resolve();
          tx.onabort = () => resolve();
        } catch {
          resolve();
        }
      });
    });
  }

  /**
   * Elimina automaticamente tutte le chat il cui ultimo aggiornamento è antecedente
   * alla soglia `maxAgeMs` (default: 7 giorni) rispetto a `nowMs`.
   */
  purgeExpiredChats(
    maxAgeMs: number = CHAT_HISTORY_TTL_MS,
    nowMs: number = Date.now()
  ): Promise<number> {
    return this.runSerialized(() => this._purgeExpiredInternal(maxAgeMs, nowMs));
  }

  /**
   * Esegue in un unico step serializzato la pulizia delle chat scadute (opzionale),
   * il salvataggio della sessione corrente e il recupero dell'elenco aggiornato.
   */
  syncSessionAndPurge(
    session: ChatHistorySession | null,
    purgeExpired: boolean = false,
    maxAgeMs: number = CHAT_HISTORY_TTL_MS,
    nowMs: number = Date.now()
  ): Promise<ChatHistorySession[]> {
    return this.runSerialized(async () => {
      if (purgeExpired) {
        await this._purgeExpiredInternal(maxAgeMs, nowMs);
      }
      if (session) {
        await this._saveChatInternal(session);
      }
      return this._getAllChatsInternal();
    });
  }
}
