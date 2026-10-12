"""OnlineEngine — uses the OpenAI API for cloud inference.

Used by desktop when configured as 'online' engine.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .base import Engine

logger = logging.getLogger(__name__)


class OnlineEngine(Engine):
    """Engine that calls the OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
    ):
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url
        self._temperature = temperature
        self._client = None

    def _lazy_load(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            logger.error(
                "openai not installed. Install with: pip install openai"
            )
            raise

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self._lazy_load()
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens or 2048

        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            elapsed = time.time() - start
            text = response.choices[0].message.content or ""
            logger.info(
                "Online generation (%s): %d chars in %.2fs (%.1f tokens/s)",
                self.model_name,
                len(text),
                elapsed,
                response.usage.completion_tokens / elapsed if elapsed > 0 else 0,
            )
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            raise

    def generate_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._lazy_load()
        temp = temperature if temperature is not None else self._temperature
        tokens = max_tokens or 2048

        start = time.time()
        try:
            stream_resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stream=True,
            )
            total_tokens = 0
            for chunk in stream_resp:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        total_tokens += 1
                        yield delta
            elapsed = time.time() - start
            logger.info("Online stream generation: %d chunks in %.2fs", total_tokens, elapsed)
        except Exception as e:
            logger.error("OpenAI streaming call failed: %s", e)
            raise

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """For text completion, wrap in a chat message."""
        return self.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def name(self) -> str:
        return f"online ({self.model_name})"

    @property
    def is_online(self) -> bool:
        return True

    def health_check(self) -> bool:
        try:
            self._lazy_load()
            # Just check the API key is set and client initialized
            return self._client is not None and bool(self.api_key)
        except Exception:
            return False
