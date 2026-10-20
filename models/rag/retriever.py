"""
Modulo di Indicizzazione e Retrieval Ibrido per Documenti in Greco Antico (`data/`).

Combina:
1. Retrieval Semantico Denso tramite `dspy.Embedder` e `dspy.Embeddings` (basato sui modelli
   BERT per il greco antico del progetto, es. `CNR-ILC/gs-GreBerta` o `bowphs/GreBerta`).
2. Retrieval Lessicale e Morfologico tramite BM25 Okapi su word token e character n-grammi
   del testo greco normalizzato (fondamentale per intercettare radici parziali `word_prefix`
   e `word_suffix` adiacenti alla lacuna `[...]`).
3. Fusione dei ranking tramite Reciprocal Rank Fusion (RRF) e filtro anti-data-leakage
   per evitare di recuperare il blocco identico al test case durante i benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import dspy
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from backend.config.settings import (
    DATA_PATH,
    RAG_DEFAULT_EMBEDDER,
    RAG_INDEX_DIR,
    RAG_TOP_K_PASSAGES,
)
from backend.core.cleaner import load_abs
from backend.core.preprocess import (
    normalize_greek,
    process_editorial_marks,
    remove_punctuation,
    transpile,
)


def _has_faiss() -> bool:
    """Verifica se la libreria `faiss` è installata nell'ambiente corrente."""
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


def _strip_passage_header(passage: str) -> str:
    """Rimuove eventuale intestazione `[Corpus: ...]` dal passaggio per il confronto testuale."""
    return re.sub(r"^\[Corpus:[^\]]*\]\s*", "", passage).strip()


def normalize_for_retrieval(text: str) -> str:
    """
    Normalizza una stringa in greco antico (rimuovendo lacune, markup editoriale,
    diacritici e punteggiatura) per il matching lessicale e il controllo di overlap.
    """
    cleaned = _strip_passage_header(text)
    cleaned = re.sub(r"\[(\.+|\-+)\]", " ", cleaned)
    cleaned = re.sub(r"\.{2,}", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = process_editorial_marks(cleaned, preserve_lacunae=False)
    cleaned = normalize_greek(
        cleaned,
        case_folding="fold",
        strip_diacritics_flag=True,
    )
    cleaned = remove_punctuation(cleaned, preserve_lacunae=False)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_lexical_features(text: str, char_ngram_range: tuple[int, int] = (3, 5)) -> list[str]:
    """
    Estrae sia i word token greci sia gli n-grammi di caratteri (per catturare radici,
    prefissi e desinenze parziali attorno alle lacune intra-parola).
    """
    norm = normalize_for_retrieval(text)
    if not norm:
        return []

    words = [w for w in norm.split() if any(c.isalpha() for c in w)]
    features: list[str] = [f"w:{w}" for w in words]

    min_n, max_n = char_ngram_range
    for w in words:
        if len(w) < min_n:
            continue
        padded = f"^{w}$"
        for n in range(min_n, min(max_n + 1, len(padded) + 1)):
            for i in range(len(padded) - n + 1):
                features.append(f"c:{padded[i : i + n]}")

    return features


class GreekLexicalBM25Index:
    """
    Indice invertito BM25 Okapi su word token e character n-grammi per il greco antico.
    Leggero, deterministico e serializzabile su disco senza dipendenze C esterne.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        char_ngram_range: tuple[int, int] = (3, 5),
    ) -> None:
        self.k1 = k1
        self.b = b
        self.char_ngram_range = char_ngram_range
        self.num_docs = len(corpus)
        self.doc_lens: np.ndarray = np.zeros(self.num_docs, dtype=np.float32)
        self.avgdl: float = 1.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        if self.num_docs > 0:
            self._build(corpus)

    def _build(self, corpus: Sequence[str]) -> None:
        df: Counter[str] = Counter()
        total_len = 0

        for doc_idx, passage in enumerate(corpus):
            feats = extract_lexical_features(passage, self.char_ngram_range)
            doc_len = len(feats)
            self.doc_lens[doc_idx] = doc_len
            total_len += doc_len

            tf_counts = Counter(feats)
            for term, freq in tf_counts.items():
                df[term] += 1
                self.postings[term].append((doc_idx, freq))

        self.avgdl = max(1.0, total_len / max(1, self.num_docs))
        for term, doc_freq in df.items():
            self.idf[term] = math.log(
                1.0 + (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5)
            )
        self.postings = dict(self.postings)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Restituisce le coppie `(doc_idx, bm25_score)` ordinate per punteggio decrescente."""
        if self.num_docs == 0:
            return []

        query_feats = extract_lexical_features(query, self.char_ngram_range)
        if not query_feats:
            return []

        scores = np.zeros(self.num_docs, dtype=np.float32)
        query_terms = Counter(query_feats)

        for term, q_weight in query_terms.items():
            idf_val = self.idf.get(term)
            if idf_val is None:
                continue
            # Diamo peso pieno ai token lessicali interi e peso ridotto agli n-grammi di caratteri
            term_multiplier = 1.0 if term.startswith("w:") else 0.35
            weight = idf_val * term_multiplier * min(q_weight, 3)

            for doc_idx, tf in self.postings.get(term, ()):
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_idx] += weight * ((tf * (self.k1 + 1.0)) / max(denom, 1e-9))

        non_zero = np.flatnonzero(scores > 0)
        if len(non_zero) == 0:
            return []

        if len(non_zero) > top_k:
            top_local = np.argpartition(-scores[non_zero], top_k)[:top_k]
            candidate_indices = non_zero[top_local]
        else:
            candidate_indices = non_zero

        sorted_order = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [(int(idx), float(scores[idx])) for idx in sorted_order[:top_k]]

    def to_bytes(self) -> bytes:
        state = {
            "k1": self.k1,
            "b": self.b,
            "char_ngram_range": self.char_ngram_range,
            "num_docs": self.num_docs,
            "doc_lens": self.doc_lens,
            "avgdl": self.avgdl,
            "idf": self.idf,
            "postings": self.postings,
        }
        return zlib.compress(pickle.dumps(state))

    @classmethod
    def from_bytes(cls, raw: bytes) -> "GreekLexicalBM25Index":
        state = pickle.loads(zlib.decompress(raw))
        instance = cls.__new__(cls)
        instance.k1 = state["k1"]
        instance.b = state["b"]
        instance.char_ngram_range = tuple(state["char_ngram_range"])
        instance.num_docs = state["num_docs"]
        instance.doc_lens = state["doc_lens"]
        instance.avgdl = state["avgdl"]
        instance.idf = state["idf"]
        instance.postings = state["postings"]
        return instance


class GreekBertEmbedder:
    """
    Funzione di embedding basata sui modelli BERT greci del progetto (es. `CNR-ILC/gs-GreBerta`,
    `bowphs/GreBerta`, `pranaydeeps/Ancient-Greek-BERT`) compatibile con `dspy.Embedder`.
    Esegue mean-pooling sugli stati nascosti dell'ultimo layer pesati con `attention_mask`.
    """

    def __init__(
        self,
        checkpoint: str = RAG_DEFAULT_EMBEDDER,
        device: str | None = None,
        max_length: int = 256,
        fallback_checkpoint: str = "bowphs/GreBerta",
    ) -> None:
        self.checkpoint = checkpoint
        self.fallback_checkpoint = fallback_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        hf_token = os.getenv("HF_TOKEN")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, token=hf_token)
            self._model = AutoModel.from_pretrained(self.checkpoint, token=hf_token)
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.fallback_checkpoint, token=hf_token)
            self._model = AutoModel.from_pretrained(self.fallback_checkpoint, token=hf_token)

        self._model = self._model.to(self.device)
        self._model.eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        self._ensure_loaded()
        norm_texts = [
            normalize_for_retrieval(t) or "unk"
            for t in texts
        ]

        encoded = self._tokenizer(
            norm_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encoded)
            last_hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled.cpu().numpy().astype(np.float32)


def build_rag_corpus_from_data(
    corpus_set: list[str] | None = None,
    budget: int | None = None,
    max_passages: int | None = None,
    max_chunk_words: int = 100,
    min_chunk_words: int = 4,
    include_metadata: bool = True,
    blocks_override: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    Costruisce la lista di passaggi in greco antico da indicizzare per il RAG
    a partire dai file `tlg_*.json` e `maat_*.json` nella directory `data/`.

    Args:
        corpus_set: Lista opzionale di corpus_id da includere (es. `["tlg", "DCLP", "DDbDP"]`).
        budget: Percentuale (1..100) di blocchi anonimi da campionare da `data/`.
        max_passages: Tetto massimo opzionale al numero di passaggi indicizzati.
        max_chunk_words: Numero massimo di parole per ciascun passaggio indicizzato.
        min_chunk_words: Numero minimo di parole richieste per conservare un passaggio.
        include_metadata: Se True, antepone `[Corpus: <id> | <title>]` al testo del passaggio.
        blocks_override: Lista opzionale di blocchi anonimi già caricati (utile per i test).
    """
    raw_blocks = (
        blocks_override
        if blocks_override is not None
        else load_abs(corpus_set=corpus_set, budget=budget)
    )

    passages: list[str] = []
    seen_normalized: set[str] = set()

    for ab in raw_blocks:
        raw_text = ab.get("training_text") or ab.get("text") or ""
        if not raw_text:
            continue

        clean_text = transpile(
            raw_text,
            case_folding="none",
            normalize=True,
            strip_diacritics=False,
        )
        clean_text = re.sub(r"<UNK>", " ", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if not clean_text:
            continue

        words = clean_text.split()
        if len(words) < min_chunk_words:
            continue

        corpus_id = ab.get("corpus_id") or "unknown"
        title = ab.get("title") or ab.get("file_id") or ""

        for start in range(0, len(words), max_chunk_words):
            chunk_words = words[start : start + max_chunk_words]
            if len(chunk_words) < min_chunk_words:
                continue
            chunk_str = " ".join(chunk_words)
            norm_key = normalize_for_retrieval(chunk_str)
            if len(norm_key.split()) < min_chunk_words or norm_key in seen_normalized:
                continue
            seen_normalized.add(norm_key)

            if include_metadata:
                header = f"[Corpus: {corpus_id}" + (f" | {title}] " if title else "] ")
                passages.append(header + chunk_str)
            else:
                passages.append(chunk_str)

            if max_passages is not None and len(passages) >= max_passages:
                return passages

    return passages


def _is_overlapping_leakage(query_norm: str, passage_norm: str, threshold: float = 0.85) -> bool:
    """
    Rileva se un passaggio recuperato coincide sostanzialmente con il contesto di test
    (data leakage), calcolando la frazione di token del contesto presenti nel passaggio.
    """
    if not query_norm or not passage_norm:
        return False
    if query_norm in passage_norm or passage_norm in query_norm:
        if min(len(query_norm), len(passage_norm)) >= 25:
            return True

    q_tokens = [t for t in query_norm.split() if len(t) > 2]
    p_tokens = set(t for t in passage_norm.split() if len(t) > 2)
    if len(q_tokens) < 4 or not p_tokens:
        return False

    overlap = sum(1 for t in q_tokens if t in p_tokens) / len(q_tokens)
    return overlap >= threshold


class GreekHybridRetriever:
    """
    Retriever Ibrido per il Greco Antico che combina:
    - `dspy.Embeddings` (Dense Semantic Search tramite `dspy.Embedder`)
    - `GreekLexicalBM25Index` (BM25 su word token e character n-grammi)
    tramite Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        corpus: list[str],
        embedder: dspy.Embedder | Callable[[list[str]], np.ndarray] | None = None,
        embedder_checkpoint: str = RAG_DEFAULT_EMBEDDER,
        k: int = RAG_TOP_K_PASSAGES,
        lexical_weight: float = 0.55,
        semantic_weight: float = 0.45,
        rrf_k: int = 60,
        enable_dense: bool = True,
        dspy_embeddings: dspy.Embeddings | None = None,
        lexical_index: GreekLexicalBM25Index | None = None,
    ) -> None:
        if not corpus:
            raise ValueError("Il corpus per GreekHybridRetriever non può essere vuoto.")

        self.corpus = list(corpus)
        self.k = k
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k
        self.enable_dense = enable_dense
        self.embedder_checkpoint = embedder_checkpoint

        self.lexical_index = (
            lexical_index
            if lexical_index is not None
            else GreekLexicalBM25Index(self.corpus)
        )

        self.dspy_embedder: dspy.Embedder | None = None
        self.dspy_embeddings: dspy.Embeddings | None = None

        if self.enable_dense:
            if isinstance(embedder, dspy.Embedder):
                self.dspy_embedder = embedder
            elif callable(embedder):
                self.dspy_embedder = dspy.Embedder(embedder, batch_size=64, caching=False)
            else:
                bert_fn = GreekBertEmbedder(checkpoint=embedder_checkpoint)
                self.dspy_embedder = dspy.Embedder(bert_fn, batch_size=64, caching=False)

            if dspy_embeddings is not None:
                self.dspy_embeddings = dspy_embeddings
            else:
                # Se faiss non è installato, alziamo brute_force_threshold per usare np.einsum
                bf_threshold = 20_000 if _has_faiss() else len(self.corpus) + 100
                self.dspy_embeddings = dspy.Embeddings(
                    corpus=self.corpus,
                    embedder=self.dspy_embedder,
                    k=max(self.k * 4, 20),
                    brute_force_threshold=bf_threshold,
                    normalize=True,
                )

    def __call__(
        self,
        query: str,
        k: int | None = None,
        word_prefix: str = "",
        word_suffix: str = "",
        exclude_query_overlap: bool = False,
    ) -> dspy.Prediction:
        return self.forward(
            query=query,
            k=k,
            word_prefix=word_prefix,
            word_suffix=word_suffix,
            exclude_query_overlap=exclude_query_overlap,
        )

    def forward(
        self,
        query: str,
        k: int | None = None,
        word_prefix: str = "",
        word_suffix: str = "",
        exclude_query_overlap: bool = False,
    ) -> dspy.Prediction:
        """
        Esegue la ricerca ibrida sul corpus greco e restituisce un `dspy.Prediction`
        con campi `passages`, `indices` e `scores`.
        """
        top_k = k or self.k
        candidate_pool_size = min(len(self.corpus), max(top_k * 5, 25))

        # Costruiamo la query arricchita con prefisso e/o suffisso della parola lacunosa
        affix_hint = " ".join(part for part in (word_prefix, word_suffix) if part)
        enriched_query = f"{query} {affix_hint}".strip()
        query_norm = normalize_for_retrieval(query)

        rrf_scores: dict[int, float] = defaultdict(float)

        # 1. Ranking Lessicale BM25 + Character N-grams
        if self.lexical_weight > 0:
            lex_results = self.lexical_index.search(enriched_query, top_k=candidate_pool_size)
            for rank_idx, (doc_idx, _score) in enumerate(lex_results):
                rrf_scores[doc_idx] += self.lexical_weight / (self.rrf_k + rank_idx + 1)

        # 2. Ranking Semantico Denso tramite dspy.Embeddings
        if self.enable_dense and self.dspy_embeddings is not None and self.semantic_weight > 0:
            self.dspy_embeddings.k = candidate_pool_size
            dense_pred = self.dspy_embeddings(query_norm or enriched_query)
            for rank_idx, doc_idx in enumerate(dense_pred.indices):
                rrf_scores[int(doc_idx)] += self.semantic_weight / (self.rrf_k + rank_idx + 1)

        # Fallback nel caso in cui nessun termine lessicale corrisponda e dense sia disabilitato
        if not rrf_scores:
            for idx in range(min(top_k, len(self.corpus))):
                rrf_scores[idx] = 1.0 / (self.rrf_k + idx + 1)

        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        selected_passages: list[str] = []
        selected_indices: list[int] = []
        selected_scores: list[float] = []

        for doc_idx, score in sorted_candidates:
            passage = self.corpus[doc_idx]
            if exclude_query_overlap:
                passage_norm = normalize_for_retrieval(passage)
                if _is_overlapping_leakage(query_norm, passage_norm):
                    continue

            selected_passages.append(passage)
            selected_indices.append(doc_idx)
            selected_scores.append(float(score))
            if len(selected_passages) >= top_k:
                break

        return dspy.Prediction(
            passages=selected_passages,
            indices=selected_indices,
            scores=selected_scores,
        )

    def save(self, path: str | Path) -> None:
        """Salva su disco l'indice ibrido (`dspy.Embeddings` + `GreekLexicalBM25Index`)."""
        index_dir = Path(path)
        index_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "k": self.k,
            "lexical_weight": self.lexical_weight,
            "semantic_weight": self.semantic_weight,
            "rrf_k": self.rrf_k,
            "enable_dense": self.enable_dense,
            "embedder_checkpoint": self.embedder_checkpoint,
            "corpus_size": len(self.corpus),
        }
        (index_dir / "hybrid_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (index_dir / "lexical_bm25.pkl.zlib").write_bytes(self.lexical_index.to_bytes())

        if self.enable_dense and self.dspy_embeddings is not None:
            self.dspy_embeddings.save(str(index_dir / "dense_dspy"))
        else:
            (index_dir / "corpus.json").write_text(
                json.dumps(self.corpus, ensure_ascii=False), encoding="utf-8"
            )

    @classmethod
    def from_saved(
        cls,
        path: str | Path,
        embedder: dspy.Embedder | Callable[[list[str]], np.ndarray] | None = None,
    ) -> "GreekHybridRetriever":
        """Carica un `GreekHybridRetriever` precedentemente salvato su disco."""
        index_dir = Path(path)
        meta_path = index_dir / "hybrid_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Indice ibrido RAG non trovato in '{index_dir}'.")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lexical_index = GreekLexicalBM25Index.from_bytes(
            (index_dir / "lexical_bm25.pkl.zlib").read_bytes()
        )

        enable_dense = bool(meta.get("enable_dense", True))
        embedder_checkpoint = meta.get("embedder_checkpoint", RAG_DEFAULT_EMBEDDER)

        dspy_embedder: dspy.Embedder | None = None
        dspy_embeddings: dspy.Embeddings | None = None

        if enable_dense and (index_dir / "dense_dspy").exists():
            if isinstance(embedder, dspy.Embedder):
                dspy_embedder = embedder
            elif callable(embedder):
                dspy_embedder = dspy.Embedder(embedder, batch_size=64, caching=False)
            else:
                bert_fn = GreekBertEmbedder(checkpoint=embedder_checkpoint)
                dspy_embedder = dspy.Embedder(bert_fn, batch_size=64, caching=False)

            dspy_embeddings = dspy.Embeddings.from_saved(
                str(index_dir / "dense_dspy"), dspy_embedder
            )
            corpus = list(dspy_embeddings.corpus)
        else:
            corpus = json.loads((index_dir / "corpus.json").read_text(encoding="utf-8"))

        return cls(
            corpus=corpus,
            embedder=dspy_embedder,
            embedder_checkpoint=embedder_checkpoint,
            k=int(meta.get("k", RAG_TOP_K_PASSAGES)),
            lexical_weight=float(meta.get("lexical_weight", 0.55)),
            semantic_weight=float(meta.get("semantic_weight", 0.45)),
            rrf_k=int(meta.get("rrf_k", 60)),
            enable_dense=enable_dense,
            dspy_embeddings=dspy_embeddings,
            lexical_index=lexical_index,
        )


_DEFAULT_RETRIEVER_CACHE: dict[str, GreekHybridRetriever] = {}


def get_or_build_default_retriever(
    index_dir: str | Path = RAG_INDEX_DIR,
    embedder_checkpoint: str = RAG_DEFAULT_EMBEDDER,
    enable_dense: bool = True,
    max_passages: int = 5000,
    corpus_set: list[str] | None = None,
) -> GreekHybridRetriever | None:
    """
    Restituisce il `GreekHybridRetriever` dalla cache in memoria o da disco (`index_dir`).
    Se l'indice su disco non esiste ma `data/` contiene i JSON dei corpora, costruisce
    un indice rapido sui blocchi disponibili.
    """
    cache_key = f"{Path(index_dir).resolve()}::{embedder_checkpoint}::{enable_dense}"
    if cache_key in _DEFAULT_RETRIEVER_CACHE:
        return _DEFAULT_RETRIEVER_CACHE[cache_key]

    idx_path = Path(index_dir)
    if (idx_path / "hybrid_meta.json").exists():
        retriever = GreekHybridRetriever.from_saved(idx_path)
        _DEFAULT_RETRIEVER_CACHE[cache_key] = retriever
        return retriever

    if DATA_PATH.exists() and any(DATA_PATH.glob("*.json")):
        passages = build_rag_corpus_from_data(
            corpus_set=corpus_set,
            max_passages=max_passages,
        )
        if passages:
            retriever = GreekHybridRetriever(
                corpus=passages,
                embedder_checkpoint=embedder_checkpoint,
                enable_dense=enable_dense,
            )
            _DEFAULT_RETRIEVER_CACHE[cache_key] = retriever
            return retriever

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Costruzione e salvataggio dell'indice ibrido DSPy RAG sui corpora in data/ (TLG e MAAT)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAG_INDEX_DIR),
        help=f"Directory di destinazione dell'indice (default: {RAG_INDEX_DIR})",
    )
    parser.add_argument(
        "--corpora",
         nargs="*",
        default=None,
        help="Lista opzionale di corpus_id da includere (es. tlg DCLP DDbDP EDH). Default: tutti.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Percentuale opzionale (1-100) di blocchi da campionare da data/.",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=10000,
        help="Numero massimo di passaggi da indicizzare (default: 10000).",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=RAG_DEFAULT_EMBEDDER,
        help=f"Modello Hugging Face per gli embedding densi (default: {RAG_DEFAULT_EMBEDDER}).",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Costruisce solo l'indice lessicale BM25 + char n-grams (senza calcolare embedding BERT).",
    )
    args = parser.parse_args()

    print(f"[RAG Indexer] Estrazione passaggi da '{DATA_PATH}'...")
    passages = build_rag_corpus_from_data(
        corpus_set=args.corpora,
        budget=args.budget,
        max_passages=args.max_passages,
    )
    if not passages:
        raise RuntimeError(f"Nessun passaggio valido trovato in '{DATA_PATH}'.")

    print(
        f"[RAG Indexer] Indicizzazione di {len(passages)} passaggi "
        f"(dense={not args.lexical_only}, embedder={args.embedder})..."
    )
    retriever = GreekHybridRetriever(
        corpus=passages,
        embedder_checkpoint=args.embedder,
        enable_dense=not args.lexical_only,
    )
    retriever.save(args.output_dir)
    print(f"[RAG Indexer] Indice salvato con successo in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
