"""Tests for runtime configuration: path resolution and persistence."""

import json
import os
import tempfile
from pathlib import Path

from Nova.backend.config import AVAILABLE_MODELS, Settings, runtime_root


def _write_config(data: dict) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(data, handle)
    handle.close()
    return handle.name


def test_relative_paths_resolve_against_the_app_root():
    """Data must be found no matter which directory Nova is launched from."""
    config_path = _write_config({
        "engine": "local",
        "sqlite_path": "nova_data/nova.db",
        "model_path": "models/lecture-model.gguf",
    })
    try:
        settings = Settings.load(config_path=config_path)
        assert Path(settings.sqlite_path).is_absolute()
        assert Path(settings.sqlite_path) == runtime_root() / "nova_data" / "nova.db"
        assert Path(settings.model_path) == runtime_root() / "models" / "lecture-model.gguf"
    finally:
        os.unlink(config_path)


def test_custom_model_path_survives_save():
    """A user-provided GGUF path must not be replaced by the pinned default."""
    config_path = _write_config({"engine": "local", "active_model_id": "gemma-3-1b"})
    try:
        settings = Settings.load(config_path=config_path)
        settings.model_path = "models/my-custom-model.gguf"
        settings.save(path=config_path)
        saved = json.loads(Path(config_path).read_text())
        assert saved["model_path"] == "models/my-custom-model.gguf"
    finally:
        os.unlink(config_path)


def test_paths_are_saved_relative_to_the_runtime_root():
    config_path = _write_config({"engine": "local", "sqlite_path": "nova_data/nova.db"})
    try:
        settings = Settings.load(config_path=config_path)
        settings.temperature = 0.5
        settings.save(path=config_path)
        saved = json.loads(Path(config_path).read_text())
        assert saved["sqlite_path"] == "nova_data/nova.db"
        assert not Path(saved["sqlite_path"]).is_absolute()
        assert saved["temperature"] == 0.5
    finally:
        os.unlink(config_path)


def test_model_defaults_fill_in_when_unset():
    config_path = _write_config({"engine": "local", "active_model_id": "gemma-3-270m"})
    try:
        settings = Settings.load(config_path=config_path)
        assert settings.model_path == ""  # nothing pinned yet
        settings.resolve_model()
        expected = AVAILABLE_MODELS["gemma-3-270m"]
        assert settings.model_path.endswith(expected["path"].split("/")[-1])
        assert settings.model_url == expected["url"]
    finally:
        os.unlink(config_path)


def test_thread_override_defaults_to_library_auto():
    settings = Settings()
    assert settings.n_threads == 0  # 0 keeps llama.cpp's own thread choice
