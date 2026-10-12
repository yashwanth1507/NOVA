"""Configuration for Nova backend.

Supports JSON file configuration and environment variable overrides.
Provides a single source of truth for all settings.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

EngineType = Literal["local", "online"]

# Available local models with metadata
AVAILABLE_MODELS: dict[str, dict] = {
    "gemma-3-1b": {
        "id": "gemma-3-1b",
        "name": "Gemma 3 1B IT",
        "size": "1B",
        "path": "models/gemma-3-1b-it-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf?download=true",
        "parameters": "1B",
        "description": "Best quality, ~20s response time",
    },
    "gemma-3-270m": {
        "id": "gemma-3-270m",
        "name": "Gemma 3 270M IT",
        "size": "270M",
        "path": "models/gemma-3-270m-it-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf?download=true",
        "parameters": "270M",
        "description": "Faster, lightweight, ~5s response time",
    },
}


def runtime_root() -> Path:
    """Return the directory used for writable desktop runtime data."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """All configuration values for Nova."""

    # Engine
    engine: EngineType = "local"

    # Model selection
    active_model_id: str = "gemma-3-1b"
    model_path: str = ""
    model_url: str = ""
    context_size: int = 2048
    temperature: float = 0.7
    # 0 means "let llama.cpp decide" (cpu_count() // 2).  Raising this to
    # cpu_count() measured ~190x SLOWER generation on a 4P+4E host because of
    # thread oversubscription, so it must stay an explicit opt-in.
    n_threads: int = 0

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_dir: str = "models/embeddings"

    # Storage
    sqlite_path: str = "nova_data/nova.db"

    # API
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"

    # Server
    host: str = "127.0.0.1"
    port: int = 8765

    # Internal
    _config_path: str = ""

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        """Load configuration from a JSON file and environment overrides.

        Priority (highest wins):
          1. Environment variables (NOVA_*)
          2. Config file
          3. Defaults
        """
        project_config = runtime_root() / "config.json"
        cfg_path = config_path or os.environ.get("NOVA_CONFIG") or str(project_config)

        # Start with defaults
        settings = cls()

        # Try loading from config file
        if cfg_path and Path(cfg_path).exists():
            try:
                with open(cfg_path) as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)
                settings._config_path = cfg_path
            except (json.JSONDecodeError, IOError) as e:
                import logging
                logging.warning("Failed to load config from %s: %s", cfg_path, e)

        # Environment variable overrides (NOVA_<KEY>)
        env_prefix = "NOVA_"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                attr_name = env_key[len(env_prefix):].lower()
                if hasattr(settings, attr_name):
                    # Cast to the right type
                    current = getattr(settings, attr_name)
                    if isinstance(current, bool):
                        setattr(settings, attr_name, env_value.lower() in ("1", "true", "yes"))
                    elif isinstance(current, int):
                        setattr(settings, attr_name, int(env_value))
                    elif isinstance(current, float):
                        setattr(settings, attr_name, float(env_value))
                    else:
                        setattr(settings, attr_name, env_value)

        # If engine is "online", require API key from env
        if settings.engine == "online":
            settings.openai_api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

        settings._resolve_paths()
        return settings

    def _resolve_paths(self) -> None:
        """Anchor relative runtime paths to the application root, not the CWD.

        Without this, launching Nova from another working directory silently
        creates a second, empty database and the user's documents look lost.
        """
        for attr_name in ("model_path", "sqlite_path", "embedding_cache_dir"):
            value = str(getattr(self, attr_name, "") or "")
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = runtime_root() / path
            setattr(self, attr_name, str(path))

    @staticmethod
    def _portable_path(value: str) -> str:
        """Store paths relative to the runtime root so config.json stays portable."""
        if not value:
            return value
        path = Path(value).expanduser()
        try:
            return str(path.relative_to(runtime_root()))
        except ValueError:
            return str(path)

    def resolve_model(self) -> None:
        """Fill model_path/model_url from the active model when they are unset.

        A path the user configured explicitly is preserved: the previous
        implementation overwrote it with the pinned default on every save.
        """
        if self.active_model_id in AVAILABLE_MODELS:
            mdl = AVAILABLE_MODELS[self.active_model_id]
            if not self.model_path:
                self.model_path = mdl["path"]
            if not self.model_url:
                self.model_url = mdl["url"]

    def save(self, path: Optional[str] = None) -> None:
        """Save current settings to a JSON file."""
        self.resolve_model()
        save_path = path or self._config_path or "config.json"
        data = {
            "engine": self.engine,
            "active_model_id": self.active_model_id,
            "model_path": self._portable_path(self.model_path),
            "model_url": self.model_url,
            "context_size": self.context_size,
            "temperature": self.temperature,
            "n_threads": self.n_threads,
            "embedding_model": self.embedding_model,
            "embedding_cache_dir": self._portable_path(self.embedding_cache_dir),
            "sqlite_path": self._portable_path(self.sqlite_path),
            "openai_model": self.openai_model,
            "openai_base_url": self.openai_base_url,
            "host": self.host,
            "port": self.port,
        }
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
