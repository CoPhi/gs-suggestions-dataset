"""
Storage SQLite compatto a sola lettura per modelli linguistici a N-grammi.

Sostituisce a runtime i pickle NLTK (`MLE` / `Lidstone`), che espandono i dizionari
annidati a ~3.9–6.7 GB di RAM, con un database SQLite indicizzato (`WITHOUT ROWID`)
condiviso tra `global_model` (`scope=0`) e `domain_model` (`scope=1`) e tra `MLE` e `LIDSTONE`
(a parità di ordine `n`), mantenendo la stessa semantica di `.vocab`, `.context_counts()` e `.score()`.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence


_SCHEMA_VERSION = "1"
_TOKEN_CACHE_MAX = 8192


class CompactNgramVocab:
    """Interfaccia compatibile con `nltk.lm.Vocabulary` basata su lookup SQLite read-only."""

    def __init__(self, lm: "CompactNgramLM") -> None:
        self._lm = lm
        self.unk_label = lm.unk_label
        self._vocab_size = lm.vocab_size

    def __len__(self) -> int:
        return self._vocab_size

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        if item == self.unk_label:
            return True
        _, in_vocab, _ = self._lm._get_token_info(item)
        return in_vocab

    def lookup(self, words: Any) -> Any:
        """Replica esattamente `nltk.lm.Vocabulary.lookup` su stringhe e sequenze annidate."""
        if isinstance(words, str):
            return words if words in self else self.unk_label
        if isinstance(words, Iterable):
            return tuple(self.lookup(w) for w in words)
        raise TypeError(f"Tipo non supportato per lookup vocabolo: {type(words)!r}")


class CompactNgramLM:
    """
    Modello linguistico N-gram compatto su SQLite read-only, duck-typed rispetto a
    `nltk.lm.MLE` e `nltk.lm.Lidstone` per i metodi usati in inferenza e valutazione.
    """

    def __init__(
        self,
        db_path: str | Path,
        scope: int,
        lm_type: str = "LIDSTONE",
        gamma: float | None = 0.001,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Checkpoint SQLite N-gram non trovato: '{self.db_path}'")

        self.scope = int(scope)
        self.lm_type = lm_type.upper()
        self.gamma = float(gamma) if (self.lm_type != "MLE" and gamma is not None) else 0.0
        self._local = threading.local()
        self._token_cache: OrderedDict[str, tuple[int | None, bool, int]] = OrderedDict()
        self._cache_lock = threading.Lock()

        conn = self._get_conn()
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        self.order = int(meta["n"])
        self.unk_label = meta.get("unk_label", "<UNK>")
        self.vocab_size = int(meta[f"vocab_size_{self.scope}"])
        self.unigram_total = int(meta[f"unigram_total_{self.scope}"])
        self.vocab = CompactNgramVocab(self)

    def _get_conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        uri = f"file:{self.db_path.as_posix()}?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.OperationalError:
            # Fallback se il filesystem non supporta immutable=1
            conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        # Limita la page-cache SQLite per connessione a ~4 MB invece di espandersi
        conn.execute("PRAGMA cache_size = -4000")
        conn.execute("PRAGMA temp_store = MEMORY")
        self._local.conn = conn
        return conn

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _get_token_info(self, word: str) -> tuple[int | None, bool, int]:
        """Restituisce `(token_id, in_vocab, unigram_count)` con cache LRU limitata."""
        with self._cache_lock:
            cached = self._token_cache.get(word)
            if cached is not None:
                self._token_cache.move_to_end(word)
                return cached

        conn = self._get_conn()
        row = conn.execute(
            "SELECT token_id, in_vocab, unigram_count FROM tokens WHERE scope = ? AND word = ?",
            (self.scope, word),
        ).fetchone()
        if row is None:
            info = (None, word == self.unk_label, 0)
        else:
            info = (int(row[0]), bool(row[1]), int(row[2]))

        with self._cache_lock:
            self._token_cache[word] = info
            self._token_cache.move_to_end(word)
            if len(self._token_cache) > _TOKEN_CACHE_MAX:
                self._token_cache.popitem(last=False)
        return info

    def _encode_context(self, context: Sequence[str]) -> bytes | None:
        ids: list[int] = []
        for tok in context:
            tid, _, _ = self._get_token_info(tok)
            if tid is None:
                return None
            ids.append(tid)
        return struct.pack(f">{len(ids)}I", *ids)

    def context_counts(self, context: Sequence[str] | None) -> Counter[str]:
        """Restituisce le frequenze delle parole successive al contesto dato (assumendo lookup già eseguito)."""
        conn = self._get_conn()
        if not context:
            rows = conn.execute(
                "SELECT word, unigram_count FROM tokens WHERE scope = ? AND unigram_count > 0",
                (self.scope,),
            ).fetchall()
            return Counter({str(w): int(c) for w, c in rows})

        ctx_key = self._encode_context(context)
        if ctx_key is None:
            return Counter()

        rows = conn.execute(
            """
            SELECT t.word, n.count
            FROM ngrams n
            JOIN tokens_by_id t
              ON t.scope = n.scope AND t.token_id = n.word_id
            WHERE n.scope = ? AND n.ctx_key = ?
            """,
            (self.scope, ctx_key),
        ).fetchall()
        return Counter({str(w): int(c) for w, c in rows})

    def unmasked_score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Calcola la probabilità `MLE` o `Lidstone` assumendo argomenti già normalizzati via `vocab.lookup`."""
        conn = self._get_conn()

        if not context:
            _, _, word_count = self._get_token_info(word)
            norm_count = self.unigram_total
        else:
            ctx_key = self._encode_context(context)
            if ctx_key is None:
                word_count = 0
                norm_count = 0
            else:
                tot_row = conn.execute(
                    "SELECT total FROM ctx_totals WHERE scope = ? AND ctx_key = ?",
                    (self.scope, ctx_key),
                ).fetchone()
                norm_count = int(tot_row[0]) if tot_row is not None else 0
                if norm_count == 0:
                    word_count = 0
                else:
                    word_id, _, _ = self._get_token_info(word)
                    if word_id is None:
                        word_count = 0
                    else:
                        cnt_row = conn.execute(
                            "SELECT count FROM ngrams WHERE scope = ? AND ctx_key = ? AND word_id = ?",
                            (self.scope, ctx_key, word_id),
                        ).fetchone()
                        word_count = int(cnt_row[0]) if cnt_row is not None else 0

        if self.lm_type == "MLE":
            if norm_count == 0:
                return 0.0
            return float(word_count) / float(norm_count)

        # Smoothing Lidstone: identico a `nltk.lm.Lidstone.unmasked_score`
        gamma = self.gamma
        denom = float(norm_count) + float(self.vocab_size) * gamma
        if denom == 0.0:
            return 0.0
        return (float(word_count) + gamma) / denom

    def score(self, word: str, context: Sequence[str] | None = None) -> float:
        """Maschera parole OOV tramite `vocab.lookup` e calcola lo score MLE/Lidstone."""
        masked_word = self.vocab.lookup(word)
        masked_ctx = self.vocab.lookup(context) if context else None
        return self.unmasked_score(masked_word, masked_ctx)


def load_compact_ngram_pair(
    db_path: str | Path,
    lm_type: str = "LIDSTONE",
    gamma: float | None = 0.001,
) -> tuple[CompactNgramLM, CompactNgramLM]:
    """Carica la coppia `(global_model, domain_model)` dal checkpoint SQLite condiviso."""
    global_model = CompactNgramLM(db_path=db_path, scope=0, lm_type=lm_type, gamma=gamma)
    domain_model = CompactNgramLM(db_path=db_path, scope=1, lm_type=lm_type, gamma=gamma)
    return global_model, domain_model


def export_nltk_models_to_sqlite(
    global_model: Any,
    domain_model: Any,
    output_path: str | Path,
    n: int | None = None,
) -> Path:
    """
    Esporta una coppia `(global_model, domain_model)` addestrata con NLTK in un database
    SQLite compatto condiviso (`ngram_n{n}.sqlite`), utilizzabile sia da MLE che da Lidstone.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    order = int(n or getattr(global_model, "order", 3))
    unk_label = str(getattr(global_model.vocab, "unk_label", "<UNK>"))

    conn = sqlite3.connect(str(tmp_path))
    try:
        conn.execute("PRAGMA page_size = 4096")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")

        conn.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE tokens (
                scope INTEGER NOT NULL,
                word TEXT NOT NULL,
                token_id INTEGER NOT NULL,
                in_vocab INTEGER NOT NULL,
                unigram_count INTEGER NOT NULL,
                PRIMARY KEY (scope, word)
            ) WITHOUT ROWID;

            CREATE TABLE tokens_by_id (
                scope INTEGER NOT NULL,
                token_id INTEGER NOT NULL,
                word TEXT NOT NULL,
                PRIMARY KEY (scope, token_id)
            ) WITHOUT ROWID;

            CREATE TABLE ngrams (
                scope INTEGER NOT NULL,
                ctx_key BLOB NOT NULL,
                word_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (scope, ctx_key, word_id)
            ) WITHOUT ROWID;

            CREATE TABLE ctx_totals (
                scope INTEGER NOT NULL,
                ctx_key BLOB NOT NULL,
                total INTEGER NOT NULL,
                PRIMARY KEY (scope, ctx_key)
            ) WITHOUT ROWID;
            """
        )

        meta_rows: list[tuple[str, str]] = [
            ("schema_version", _SCHEMA_VERSION),
            ("n", str(order)),
            ("unk_label", unk_label),
        ]

        batch_size = 50_000
        for scope, lm in ((0, global_model), (1, domain_model)):
            vocab = lm.vocab
            vocab_size = len(vocab)
            unigrams = lm.counts.unigrams
            unigram_total = int(unigrams.N())
            meta_rows.append((f"vocab_size_{scope}", str(vocab_size)))
            meta_rows.append((f"unigram_total_{scope}", str(unigram_total)))

            # Raccoglie tutti i token conosciuti (vocabolario + eventuali token presenti nei conteggi come <UNK>/<s>/</s>)
            token_to_id: dict[str, int] = {}

            def _ensure_token(tok: str) -> int:
                tid = token_to_id.get(tok)
                if tid is None:
                    tid = len(token_to_id)
                    token_to_id[tok] = tid
                return tid

            for tok in vocab.counts:
                _ensure_token(tok)
            _ensure_token(unk_label)
            for tok in unigrams:
                _ensure_token(tok)

            for k_order in range(2, order + 1):
                cfd = lm.counts[k_order]
                for ctx, fdist in cfd.items():
                    for c_tok in ctx:
                        _ensure_token(c_tok)
                    for w_tok in fdist:
                        _ensure_token(w_tok)

            # Inserisce i token in batch streaming
            tok_rows: list[tuple[int, str, int, int, int]] = []
            tok_id_rows: list[tuple[int, int, str]] = []
            for tok, tid in token_to_id.items():
                in_v = 1 if (tok == unk_label or tok in vocab) else 0
                u_cnt = int(unigrams.get(tok, 0))
                tok_rows.append((scope, tok, tid, in_v, u_cnt))
                tok_id_rows.append((scope, tid, tok))
                if len(tok_rows) >= batch_size:
                    conn.executemany(
                        "INSERT INTO tokens (scope, word, token_id, in_vocab, unigram_count) VALUES (?, ?, ?, ?, ?)",
                        tok_rows,
                    )
                    conn.executemany(
                        "INSERT INTO tokens_by_id (scope, token_id, word) VALUES (?, ?, ?)",
                        tok_id_rows,
                    )
                    tok_rows.clear()
                    tok_id_rows.clear()
            if tok_rows:
                conn.executemany(
                    "INSERT INTO tokens (scope, word, token_id, in_vocab, unigram_count) VALUES (?, ?, ?, ?, ?)",
                    tok_rows,
                )
                conn.executemany(
                    "INSERT INTO tokens_by_id (scope, token_id, word) VALUES (?, ?, ?)",
                    tok_id_rows,
                )

            # Esporta gli ordini >= 2 (contesti e conteggi condizionati)
            ngram_rows: list[tuple[int, bytes, int, int]] = []
            total_rows: list[tuple[int, bytes, int]] = []
            for k_order in range(2, order + 1):
                cfd = lm.counts[k_order]
                packer = struct.Struct(f">{k_order - 1}I")
                for ctx, fdist in cfd.items():
                    if not fdist:
                        continue
                    ctx_bytes = packer.pack(*(token_to_id[c] for c in ctx))
                    ctx_total = 0
                    for w_tok, cnt in fdist.items():
                        c_int = int(cnt)
                        ctx_total += c_int
                        ngram_rows.append((scope, ctx_bytes, token_to_id[w_tok], c_int))
                        if len(ngram_rows) >= batch_size:
                            conn.executemany(
                                "INSERT INTO ngrams (scope, ctx_key, word_id, count) VALUES (?, ?, ?, ?)",
                                ngram_rows,
                            )
                            ngram_rows.clear()
                    total_rows.append((scope, ctx_bytes, ctx_total))
                    if len(total_rows) >= batch_size:
                        conn.executemany(
                            "INSERT INTO ctx_totals (scope, ctx_key, total) VALUES (?, ?, ?)",
                            total_rows,
                        )
                        total_rows.clear()

            if ngram_rows:
                conn.executemany(
                    "INSERT INTO ngrams (scope, ctx_key, word_id, count) VALUES (?, ?, ?, ?)",
                    ngram_rows,
                )
            if total_rows:
                conn.executemany(
                    "INSERT INTO ctx_totals (scope, ctx_key, total) VALUES (?, ?, ?)",
                    total_rows,
                )

        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, out_path)
    return out_path
