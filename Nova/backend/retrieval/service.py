"""Hybrid retrieval for the desktop document-grounded tutor.

The service ranks only relevant chunks from the active document.  It combines
MiniLM semantic similarity with lexical coverage, filters weak matches, and
uses a small MMR-style diversity pass so adjacent overlapping chunks do not
consume the entire context window.
"""

from __future__ import annotations

import logging
import math
import re
import struct
from functools import lru_cache
from typing import Optional

from .embeddings import EmbeddingProvider, MiniLMEmbedding, cosine_similarity
from ..config import Settings
from ..storage.database import Storage

try:  # numpy ships with sentence-transformers; the fallback keeps Nova pure-stdlib safe
    import numpy as _np
except Exception:  # pragma: no cover - depends on the installation
    _np = None

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
@lru_cache(maxsize=4096)
def _cached_terms(normalized: str) -> frozenset[str]:
    """Tokenize a chunk once and reuse the term set across queries."""
    return frozenset(_TOKEN_RE.findall(normalized))


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "between",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "in", "into", "is", "it", "its", "may", "might",
    "of", "on", "or", "should", "that", "the", "their", "these", "this",
    "those", "through", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "list", "lists",
    "show", "give", "tell", "name", "names", "all",
}


class RetrievalService:
    """Retrieve a small, relevant evidence set from the active document."""

    def __init__(
        self,
        storage: Storage,
        embedding_provider: Optional[EmbeddingProvider] = None,
        settings: Optional[Settings] = None,
        embedding_manager=None,
    ):
        self.storage = storage
        self.settings = settings or Settings.load()
        self.embedding_provider = embedding_provider
        self.embedding_manager = embedding_manager
        self._embedding_failed = False

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token for token in _cached_terms(text.lower())
            if (len(token) > 1 or token.isdigit()) and token not in _STOP_WORDS
        }

    def _ensure_embeddings(self) -> None:
        """Lazy-load the local embedding provider only when vectors exist."""
        if self.embedding_provider is not None:
            return
        if self._embedding_failed:
            # Recovery path: if the embedding model was installed after the
            # first failure, retry instead of staying lexical-only until the
            # app is restarted.
            available = False
            if self.embedding_manager is not None:
                try:
                    available = bool(self.embedding_manager.status()["available"])
                except Exception:
                    available = False
            if not available:
                return
            self._embedding_failed = False
        try:
            self.embedding_provider = MiniLMEmbedding(
                model_name=self.settings.embedding_model,
                cache_dir=getattr(self.settings, "embedding_cache_dir", "") or None,
            )
        except Exception as exc:
            logger.warning("Embedding model unavailable; using lexical retrieval: %s", exc)
            self.embedding_provider = None
            self._embedding_failed = True

    def _embeddings_available(self) -> bool:
        """Whether the embedding model is installed (no model load involved)."""
        if self.embedding_manager is None:
            return False
        try:
            return bool(self.embedding_manager.status()["available"])
        except Exception:
            return False

    def embedding_status(self, probe: bool = False) -> dict:
        """Report whether semantic retrieval can run (used by /health)."""
        if self.embedding_manager is not None:
            return self.embedding_manager.status(probe=probe)
        return {
            "state": "ready" if self.embedding_provider is not None else "unknown",
            "available": self.embedding_provider is not None,
            "model": getattr(self.settings, "embedding_model", ""),
            "cache_dir": getattr(self.settings, "embedding_cache_dir", ""),
            "error": "",
        }

    def refresh_embedding_provider(self) -> None:
        """Forget the current provider so the next query re-resolves it."""
        self.embedding_provider = None
        self._embedding_failed = False

    def retrieve(
        self, query: str, top_k: int = 5, document_id: Optional[str] = None
    ) -> list[dict]:
        """Return only the strongest, diverse chunks for ``query``.

        ``top_k`` is an upper bound, not a promise: weak or unrelated chunks
        are omitted.  When a document is selected, chunks from other PDFs are
        never considered.
        """
        top_k = max(1, min(int(top_k), 8))
        # Load only the active document's rows: querying a single PDF must not
        # read every chunk and vector in the library.
        if document_id:
            chunks = self.storage.get_chunks_for_document(document_id)
            stored_embeddings = dict(self.storage.get_embeddings_for_document(document_id))
        else:
            chunks = self.storage.get_all_chunks()
            stored_embeddings = dict(self.storage.get_all_embeddings())
        if not chunks or not query.strip():
            return []

        lexical = self._lexical_scores(query, chunks)
        semantic: dict[str, float] = {}
        # A document indexed while the embedding model was unavailable has no
        # stored vectors.  That used to skip the semantic pass entirely, so
        # those documents could never be repaired.  Now the pass runs whenever
        # there is something to score or the model is known to be installed.
        if (stored_embeddings or self.embedding_provider is not None
                or self._embeddings_available()):
            try:
                semantic = self._semantic_scores(query, chunks, stored_embeddings)
            except Exception as exc:
                logger.warning("Semantic retrieval failed; using lexical scores: %s", exc)
                self.embedding_provider = None
                self._embedding_failed = True

        ranked = []
        for chunk in chunks:
            chunk_id = chunk["id"]
            keyword_score = lexical.get(chunk_id, 0.0)
            semantic_score = max(0.0, semantic.get(chunk_id, 0.0))
            if semantic:
                # Semantic similarity handles paraphrases; lexical coverage
                # keeps exact lesson names and chapter numbers precise.
                score = 0.72 * semantic_score + 0.28 * keyword_score
            else:
                score = keyword_score
            ranked.append((score, semantic_score, keyword_score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] <= 0.0:
            if document_id:
                sorted_chunks = sorted(chunks, key=lambda c: c.get("position", 0))[:top_k]
                return [
                    dict(c, score=0.1, semantic_score=0.0, keyword_score=0.0)
                    for c in sorted_chunks
                ]
            return []

        # Do not pass low-confidence neighbors just because top_k is large.
        # The relative threshold adapts to short and long documents.
        top_score = ranked[0][0]
        minimum = max(0.08, top_score * 0.42)
        candidates = [item for item in ranked if item[0] >= minimum]
        if semantic and not any(value > 0.0 for value in lexical.values()):
            candidates = [item for item in candidates if item[1] >= 0.18]
        # Exact section terms (for example, "appendix" or "chapter 1") are
        # stronger than a loose semantic neighbor.  When one is present,
        # keep only chunks that contain that lexical evidence.
        if lexical and max(lexical.values(), default=0.0) >= 0.7:
            candidates = [item for item in candidates if item[2] > 0.0]
        if not candidates:
            if document_id:
                sorted_chunks = sorted(chunks, key=lambda c: c.get("position", 0))[:top_k]
                return [
                    dict(c, score=0.1, semantic_score=0.0, keyword_score=0.0)
                    for c in sorted_chunks
                ]
            return []

        selected = self._select_diverse(candidates, top_k)
        results = []
        for score, semantic_score, keyword_score, chunk in selected:
            result = dict(chunk)
            result["score"] = round(score, 4)
            result["semantic_score"] = round(semantic_score, 4)
            result["keyword_score"] = round(keyword_score, 4)
            results.append(result)
        return results

    def _semantic_scores(
        self,
        query: str,
        chunks: list[dict],
        stored_embeddings: dict[str, bytes],
    ) -> dict[str, float]:
        self._ensure_embeddings()
        if self.embedding_provider is None:
            return {}
        query_embedding = self.embedding_provider.embed(query)
        dimension = len(query_embedding)

        scores: dict[str, float] = {}
        stored_pairs: list[tuple[str, bytes]] = []
        missing: list[dict] = []
        for chunk in chunks:
            blob = stored_embeddings.get(chunk["id"])
            # A blob from another embedding model has a different dimension;
            # treating it as missing keeps old and new vectors from mixing.
            if blob and len(blob) == dimension * 4:
                stored_pairs.append((chunk["id"], blob))
            else:
                missing.append(chunk)

        if stored_pairs:
            scores.update(self._score_stored_vectors(query_embedding, stored_pairs))
        if missing:
            scores.update(self._score_missing_chunks(query_embedding, missing))
        return scores

    def _score_stored_vectors(
        self,
        query_embedding: list[float],
        pairs: list[tuple[str, bytes]],
    ) -> dict[str, float]:
        """Score persisted float32 vectors, vectorized when numpy is available."""
        dimension = len(query_embedding)
        if _np is None:
            return {
                chunk_id: cosine_similarity(
                    query_embedding,
                    list(struct.unpack(f"{len(blob) // 4}f", blob)),
                )
                for chunk_id, blob in pairs
            }

        matrix = _np.frombuffer(
            b"".join(blob for _, blob in pairs), dtype=_np.float32
        ).reshape(len(pairs), dimension)
        query = _np.asarray(query_embedding, dtype=_np.float32)
        query_norm = float(_np.linalg.norm(query)) or 1.0
        norms = _np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        similarities = (matrix @ query) / (norms * query_norm)
        return {
            chunk_id: float(similarity)
            for (chunk_id, _), similarity in zip(pairs, similarities)
        }

    def _score_missing_chunks(
        self,
        query_embedding: list[float],
        chunks: list[dict],
    ) -> dict[str, float]:
        """Embed and score chunks that have no usable stored vector.

        Documents indexed while the embedding model was unavailable still
        participate in semantic retrieval, and the vectors are persisted so the
        next query is a pure lookup.
        """
        if self.embedding_provider is None:
            return {}
        vectors = self.embedding_provider.embed_batch(
            [chunk.get("content", "") for chunk in chunks]
        )
        dimension = len(query_embedding)
        pairs = [
            (chunk["id"], struct.pack(f"{len(vector)}f", *vector))
            for chunk, vector in zip(chunks, vectors)
            if vector and len(vector) == dimension
        ]
        if not pairs:
            return {}
        scores = self._score_stored_vectors(query_embedding, pairs)
        self._persist_backfilled_vectors(pairs, dimension)
        return scores

    def _persist_backfilled_vectors(
        self,
        pairs: list[tuple[str, bytes]],
        dimension: int,
    ) -> None:
        """Store vectors computed on the fly so the next query is cheap."""
        valid = [
            (chunk_id, blob) for chunk_id, blob in pairs
            if len(blob) == dimension * 4
        ]
        if not valid:
            return
        try:
            self.storage.add_embeddings(valid)
        except Exception as exc:  # pragma: no cover - storage failure is non-fatal
            logger.debug("Could not persist backfilled embeddings: %s", exc)

    def _lexical_scores(self, query: str, chunks: list[dict]) -> dict[str, float]:
        query_terms = self._terms(query)
        if not query_terms:
            return {}
        query_lower = " ".join(query.lower().split())
        scores: dict[str, float] = {}
        for chunk in chunks:
            content = chunk.get("content", "")
            content_terms = self._terms(content)
            overlap = query_terms & content_terms
            if not overlap:
                scores[chunk["id"]] = 0.0
                continue
            coverage = len(overlap) / len(query_terms)
            phrase_bonus = 0.12 if query_lower in " ".join(content.lower().split()) else 0.0
            heading = str(chunk.get("metadata", {}).get("heading", ""))
            heading_overlap = overlap & self._terms(heading)
            heading_bonus = 0.22 * len(heading_overlap) / len(query_terms)
            # A matching section heading is strong evidence even when the
            # section body is short (for example, an Appendix URL list).
            heading_floor = min(1.0, 0.82 + heading_bonus) if heading_overlap else 0.0
            # Mild length normalization prevents a giant chunk from winning
            # merely because it contains many repeated words.
            length_penalty = 1.0 / max(1.0, math.log(len(content_terms) + 2, 2) / 8)
            scores[chunk["id"]] = min(
                1.0,
                max(heading_floor, coverage * length_penalty + phrase_bonus + heading_bonus),
            )
        return scores

    @staticmethod
    def _chunk_overlap(left: dict, right: dict) -> float:
        left_terms = RetrievalService._terms(left.get("content", ""))
        right_terms = RetrievalService._terms(right.get("content", ""))
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / len(left_terms | right_terms)

    def _select_diverse(
        self,
        candidates: list[tuple[float, float, float, dict]],
        top_k: int,
    ) -> list[tuple[float, float, float, dict]]:
        """Select high-scoring chunks while avoiding near-duplicate overlap."""
        remaining = list(candidates)
        selected: list[tuple[float, float, float, dict]] = []
        while remaining and len(selected) < top_k:
            best_index = 0
            best_value = float("-inf")
            for index, item in enumerate(remaining):
                relevance = item[0]
                redundancy = max(
                    (self._chunk_overlap(item[3], chosen[3]) for chosen in selected),
                    default=0.0,
                )
                mmr_value = 0.78 * relevance - 0.22 * redundancy
                if mmr_value > best_value:
                    best_value = mmr_value
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected