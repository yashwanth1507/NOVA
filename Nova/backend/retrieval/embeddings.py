"""Embedding provider interface, the desktop MiniLM implementation, and the
embedding-model installer.

Nova stays offline-first: retrieval never blocks on the network.  The model is
loaded from a local cache only.  When the cache is empty the UI can offer an
explicit, user-initiated download (``EmbeddingModelManager.start_download``)
after which hybrid retrieval works offline again.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Loaded models are shared so a status check and the retrieval service do not
# each pay for (or hold) a separate copy.
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _candidate_cache_folders(cache_dir: Optional[str]) -> list[Optional[str]]:
    candidates: list[Optional[str]] = []
    if cache_dir:
        candidates.append(str(cache_dir))
    candidates.append(None)  # the user's default Hugging Face cache
    return candidates


def _load_local_model(model_name: str, cache_dir: Optional[str]):
    """Load a cached SentenceTransformer, trying the app cache then the user cache."""
    from sentence_transformers import SentenceTransformer

    last_error: Optional[Exception] = None
    for folder in _candidate_cache_folders(cache_dir):
        key = f"{model_name}::{folder or 'default'}"
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            model = SentenceTransformer(
                model_name,
                cache_folder=folder,
                local_files_only=True,
            )
        except Exception as exc:  # not cached here (or not loadable)
            last_error = exc
            continue
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = model
        logger.info("Loaded local embedding model %s (cache=%s)", model_name, folder or "default")
        return model
    raise RuntimeError(
        f"Embedding model '{model_name}' is not available in the local cache"
    ) from last_error


def load_embedding_model(model_name: str, cache_dir: Optional[str] = None):
    """Return the shared local embedding model, or raise when it is not cached."""
    return _load_local_model(model_name, cache_dir)


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class MiniLMEmbedding(EmbeddingProvider):
    """Desktop embedding provider using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._dimension = 384  # all-MiniLM-L6-v2
        self._load_lock = threading.Lock()

    def _lazy_load(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                # Nova is offline-first.  Never block a tutor request with
                # Hugging Face network retries; a missing local model falls
                # back immediately to deterministic lexical retrieval.
                self._model = load_embedding_model(self.model_name, self.cache_dir)
            except ImportError as exc:
                logger.error(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
            except Exception as exc:
                logger.warning("Embedding model unavailable: %s", exc)
                raise

    def embed(self, text: str) -> list[float]:
        self._lazy_load()
        return self._model.encode(text, show_progress_bar=False).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._lazy_load()
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
        ).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingModelManager:
    """Report and install the local embedding model state."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = ""):
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else ""
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._state = "unknown"
        self._error = ""

    def _probe(self) -> bool:
        try:
            load_embedding_model(self.model_name, self.cache_dir or None)
            return True
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            return False

    def _download_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self, probe: bool = False) -> dict:
        """Return install state.  ``probe=True`` attempts a local load."""
        if probe and not self._download_running():
            if self._probe():
                with self._lock:
                    self._state = "ready"
                    self._error = ""
            else:
                with self._lock:
                    self._state = "missing"
        with self._lock:
            return {
                "state": self._state,
                "available": self._state == "ready",
                "model": self.model_name,
                "cache_dir": self.cache_dir,
                "error": self._error,
            }

    def start_download(self) -> dict:
        """Download the embedding model in the background (explicit user action)."""
        if self._state == "ready" and self._probe():
            return self.status()
        with self._lock:
            if not self._download_running():
                self._state = "downloading"
                self._error = ""
                self._thread = threading.Thread(
                    target=self._download,
                    name="nova-embeddings-download",
                    daemon=True,
                )
                self._thread.start()
        return self.status()

    def download_blocking(self) -> dict:
        self.start_download()
        thread = self._thread
        if thread:
            thread.join()
        return self.status()

    def _download(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            target = self.cache_dir or None
            if target:
                Path(target).mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(self.model_name, cache_folder=target)
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[f"{self.model_name}::{target or 'default'}"] = model
            with self._lock:
                self._state = "ready"
                self._error = ""
            logger.info("Embedding model %s installed", self.model_name)
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc)
            logger.warning("Embedding model download failed: %s", exc)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
