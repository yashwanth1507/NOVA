"""Desktop local inference through llama-cpp-python and a GGUF model."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Prevent native threading oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .base import Engine
from ..models.manager import ModelManager

logger = logging.getLogger(__name__)


class LocalEngine(Engine):
    """Engine that runs a local model via llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        context_size: int = 2048,
        temperature: float = 0.7,
        model_manager: Optional[ModelManager] = None,
        n_threads: int = 0,
    ):
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            if getattr(sys, "frozen", False):
                project_path = Path(sys.executable).resolve().parent / path
            else:
                project_path = Path(__file__).resolve().parents[3] / path
            if project_path.exists():
                path = project_path
        self.model_path = str(path)
        self.context_size = context_size
        # 0 keeps the llama.cpp default (cpu_count() // 2).  Setting this to
        # cpu_count() measured ~190x slower generation on a 4P+4E host.
        self.n_threads = int(n_threads or 0)
        self._temperature = temperature
        self.model_manager = model_manager
        self._model = None
        self._load_lock = threading.Lock()
        self._loaded_at: Optional[float] = None

    # ── Lifecycle ──────────────────────────────────────────────

    def _lazy_load(self):
        """Load the GGUF once, even when several requests race the first call."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # another thread finished the load
                return
            self._load_model()

    def _load_model(self) -> None:
        if not self.model_path:
            raise FileNotFoundError("Local model path is not configured")
        if not Path(self.model_path).is_file() and self.model_manager is not None:
            logger.info("Local model is missing; starting in-app installation")
            self.model_manager.download_blocking()
            self.model_path = str(self.model_manager.path)
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"Local model not found: {self.model_path}")
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
            raise
        logger.info("Loading local model from %s", self.model_path)
        options = {
            "model_path": self.model_path,
            "n_ctx": self.context_size,
            "verbose": False,
        }
        if self.n_threads > 0:
            options["n_threads"] = self.n_threads
            options["n_threads_batch"] = self.n_threads
        self._model = Llama(**options)
        self._loaded_at = time.time()

    def warm_up(self, timeout: Optional[float] = None) -> bool:
        """Load the model now instead of on the first user request.

        The startup hook used to call ``EngineManager._get_local()``, which only
        constructs this object, so the first question paid the full cold-load.
        """
        self._lazy_load()
        return self._model is not None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Prompt handling ────────────────────────────────────────

    def _build_prompt_text(self, messages: list[dict]) -> str:
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            else:
                prompt_parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
        prompt_parts.append("<start_of_turn>model\n")
        return "\n".join(prompt_parts)

    def _encode(self, text: str) -> list[int]:
        """Tokenize exactly like ``create_completion`` does for a string prompt.

        ``special=True`` collapses ``<start_of_turn>`` into one token, which is
        what the library's own string path does.  The previous estimate used
        ``special=False`` and over-counted the prompt by roughly 2x, which made
        the output budget far smaller than the model could actually produce.
        """
        return self._model.tokenize(text.encode("utf-8"), add_bos=True, special=True)

    def _fit_generation_tokens(self, prompt_tokens: int, requested: int) -> int:
        """Clamp output tokens so prompt plus output always fits ``n_ctx``."""
        available = max(1, self.context_size - prompt_tokens)
        fitted = max(1, min(int(requested), available))
        if fitted < requested:
            logger.warning(
                "Compacted generation budget from %d to %d tokens "
                "(prompt=%d, context=%d)",
                requested,
                fitted,
                prompt_tokens,
                self.context_size,
            )
        return fitted

    def _complete(
        self,
        prompt_tokens: list[int],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool = False,
    ):
        requested_tokens = max_tokens or 1024
        fitted = self._fit_generation_tokens(len(prompt_tokens), requested_tokens)
        return self._model.create_completion(
            # Passing tokens re-uses the tokenization done for budgeting instead
            # of tokenizing the same prompt a second time inside llama.cpp.
            prompt=prompt_tokens,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=fitted,
            stop=["<end_of_turn>", "<start_of_turn>", "</s>", "<|im_end|>"],
            stream=stream,
        )

    # ── Generation ─────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._lazy_load()
        prompt_tokens = self._encode(self._build_prompt_text(messages))
        start = time.time()
        response = self._complete(prompt_tokens, temperature, max_tokens)
        text = response.get("choices", [{}])[0].get("text", "").strip()
        logger.info(
            "Local generation: %d chars in %.2fs (prompt=%d tokens)",
            len(text),
            time.time() - start,
            len(prompt_tokens),
        )
        return text

    def generate_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._lazy_load()
        prompt_tokens = self._encode(self._build_prompt_text(messages))
        start = time.time()
        stream_iter = self._complete(prompt_tokens, temperature, max_tokens, stream=True)

        first_token = True
        total_tokens = 0
        for chunk in stream_iter:
            delta = chunk.get("choices", [{}])[0].get("text", "")
            if first_token:
                delta = delta.lstrip()
                if delta:
                    first_token = False
            if delta:
                total_tokens += 1
                yield delta

        logger.info(
            "Local stream generation: %d tokens in %.2fs (prompt=%d tokens)",
            total_tokens,
            time.time() - start,
            len(prompt_tokens),
        )

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._lazy_load()
        prompt_tokens = self._encode(prompt)
        start = time.time()
        response = self._model.create_completion(
            prompt=prompt_tokens,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=self._fit_generation_tokens(
                len(prompt_tokens), max_tokens or 1024
            ),
        )
        text = response.get("choices", [{}])[0].get("text", "")
        logger.info(
            "Local text generation: %d chars in %.2fs", len(text), time.time() - start
        )
        return text

    # ── Metadata ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"local ({self.model_path})"

    @property
    def is_online(self) -> bool:
        return False

    def health_check(self) -> bool:
        try:
            self._lazy_load()
            return self._model is not None
        except Exception:
            return False
