"""EngineManager — selects and provides access to the active engine.

Per spec: no automatic fallback or capability detection for the demo.
Engine choice is fixed per device via settings.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import Engine
from .local import LocalEngine
from .online import OnlineEngine
from ..config import AVAILABLE_MODELS, Settings
from ..models.manager import ModelManager

logger = logging.getLogger(__name__)


class EngineManager:
    """Manages engine selection and lifecycle."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.load()
        # Resolve model path/url from active_model_id
        self.settings.resolve_model()
        self._local: Optional[LocalEngine] = None
        self._online: Optional[OnlineEngine] = None
        self.model_manager = ModelManager(
            model_path=self.settings.model_path,
            model_url=self.settings.model_url,
        )

    def get_engine(self) -> Engine:
        """Get the currently configured engine based on settings."""
        if self.settings.engine == "local":
            return self._get_local()
        else:
            return self._get_online()

    def _get_local(self) -> LocalEngine:
        if self._local is None:
            self._local = LocalEngine(
                model_path=self.settings.model_path,
                context_size=self.settings.context_size,
                temperature=self.settings.temperature,
                model_manager=self.model_manager,
                n_threads=getattr(self.settings, "n_threads", 0),
            )
        return self._local

    def warm_up_local_model(self) -> bool:
        """Load the local GGUF now so the first question is not delayed."""
        if self.settings.engine != "local":
            return False
        return self._get_local().warm_up()

    @property
    def local_model_loaded(self) -> bool:
        return bool(self._local is not None and self._local.is_loaded)

    def _get_online(self) -> OnlineEngine:
        if self._online is None:
            self._online = OnlineEngine(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
                base_url=self.settings.openai_base_url,
                temperature=self.settings.temperature,
            )
        return self._online

    def switch_engine(self, engine_type: str) -> Engine:
        """Switch the active engine type and return it."""
        if engine_type not in ("local", "online"):
            raise ValueError(f"Unknown engine type: {engine_type}")
        self.settings.engine = engine_type  # type: ignore
        return self.get_engine()

    def update_setting(self, key: str, value: str) -> None:
        """Apply a persisted UI setting to the active engine configuration.

        Rebuilding the cached engine after a model-affecting change keeps the
        settings screen and the next chat request in sync.
        """
        if key == "engine":
            if value not in ("local", "online"):
                raise ValueError("engine must be 'local' or 'online'")
            self.settings.engine = value  # type: ignore
        elif key == "temperature":
            temperature = float(value)
            if not 0 <= temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
            self.settings.temperature = temperature
        elif key == "context_size":
            context_size = int(value)
            if context_size not in (1024, 2048, 4096, 8192):
                raise ValueError("context_size must be one of 1024, 2048, 4096, or 8192")
            self.settings.context_size = context_size
        elif key == "n_threads":
            n_threads = int(value)
            if n_threads < 0:
                raise ValueError("n_threads must be 0 (auto) or a positive thread count")
            self.settings.n_threads = n_threads
        elif key == "model_path":
            self.settings.model_path = value
            self.model_manager = ModelManager(
                model_path=self.settings.model_path,
                model_url=self.settings.model_url,
            )
        elif key == "active_model":
            if value not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {value}. Available: {list(AVAILABLE_MODELS.keys())}")
            self.set_active_model(value)
            return  # set_active_model already saves
        elif key == "openai_api_key":
            self.settings.openai_api_key = value
        elif key == "openai_model":
            self.settings.openai_model = value
        else:
            return

        self._local = None
        self._online = None
        self.settings.save()

    def get_available_models(self) -> list[dict]:
        """Return the list of available local models with metadata."""
        models = []
        for mid, info in AVAILABLE_MODELS.items():
            models.append({
                "id": mid,
                "name": info["name"],
                "size": info["size"],
                "path": info["path"],
                "parameters": info["parameters"],
                "description": info["description"],
                "active": mid == self.settings.active_model_id,
            })
        return models

    def get_active_model(self) -> dict:
        """Return the currently active model metadata."""
        mid = self.settings.active_model_id
        if mid in AVAILABLE_MODELS:
            info = AVAILABLE_MODELS[mid]
            return {
                "id": mid,
                "name": info["name"],
                "size": info["size"],
                "path": info["path"],
                "parameters": info["parameters"],
                "description": info["description"],
                "active": True,
            }
        return {"id": mid, "name": mid, "active": True}

    def set_active_model(self, model_id: str) -> None:
        """Switch the active local model. Resets the local engine cache."""
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}")
        mdl = AVAILABLE_MODELS[model_id]
        self.settings.active_model_id = model_id
        self.settings.model_path = mdl["path"]
        self.settings.model_url = mdl["url"]
        self._local = None  # Force engine reload
        self.model_manager = ModelManager(
            model_path=self.settings.model_path,
            model_url=self.settings.model_url,
        )
        self.settings.save()
        logger.info("Switched to local model: %s (%s)", mdl["name"], model_id)

    @property
    def engine_name(self) -> str:
        """Name of the currently selected engine."""
        try:
            return self.get_engine().name
        except Exception:
            return "unavailable"
