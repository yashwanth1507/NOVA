"""Engine interface — all engines implement this.

The rule: the API/Service layer never calls the model directly.
It always goes through the engine interface, keeping the model swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Engine(ABC):
    """Abstract engine interface for Nova."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the model.

        Args:
            messages: List of dicts with 'role' and 'content'.
            temperature: Override the default temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text response.
        """
        ...

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a text completion from a raw string prompt.

        Used for quiz generation and other non-chat tasks.
        """
        ...

    def generate_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Generate a streaming response (token by token iterator)."""
        # Default fallback if engine does not implement streaming natively
        yield self.generate(messages=messages, temperature=temperature, max_tokens=max_tokens)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @property
    @abstractmethod
    def is_online(self) -> bool:
        """Whether this engine requires network connectivity."""
        ...

    def health_check(self) -> bool:
        """Check if the engine is available and working."""
        return True
