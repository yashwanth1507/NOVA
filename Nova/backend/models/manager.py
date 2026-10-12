"""Local model installation for the desktop release."""

from __future__ import annotations

import os
import ssl
import threading
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl.create_default_context()


from ..config import runtime_root

DEFAULT_MODEL_URL = (
    "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/"
    "gemma-3-1b-it-Q4_K_M.gguf?download=true"
)


class ModelManager:
    """Download and validate the local GGUF model on demand.

    Downloads are resumable and run in a daemon thread for the UI endpoint.
    The engine can call ``download_blocking`` when a chat arrives before the
    user has installed the model. A valid GGUF is kept beside the release so
    later sessions run without network access.
    """

    def __init__(self, model_path: str, model_url: Optional[str] = None):
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = runtime_root() / path
        self.path = path
        self.url = model_url or DEFAULT_MODEL_URL
        self._part_path = self.path.with_name(self.path.name + ".part")
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._state = "ready" if self._is_valid_model() else "missing"
        self._downloaded_bytes = self.path.stat().st_size if self.path.exists() else 0
        self._total_bytes = self._downloaded_bytes
        self._error = ""

    def _is_valid_model(self) -> bool:
        try:
            if not self.path.is_file() or self.path.stat().st_size < 4:
                return False
            with self.path.open("rb") as model_file:
                return model_file.read(4) == b"GGUF"
        except OSError:
            return False

    def status(self) -> dict:
        with self._lock:
            downloaded = self._downloaded_bytes
            total = self._total_bytes
            state = self._state
            error = self._error
        progress = round(downloaded / total * 100, 1) if total else 0.0
        return {
            "state": state,
            "available": state == "ready" and self._is_valid_model(),
            "filename": self.path.name,
            "path": str(self.path),
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "progress": progress,
            "error": error,
        }

    def start_download(self) -> dict:
        """Start a background download and return its current status."""
        with self._lock:
            if self._is_valid_model():
                self._state = "ready"
            elif self._thread and self._thread.is_alive():
                pass
            else:
                self._state = "downloading"
                self._error = ""
                self._thread = threading.Thread(
                    target=self._download,
                    name="nova-model-download",
                    daemon=True,
                )
                self._thread.start()
        return self.status()

    def download_blocking(self) -> None:
        """Install the model synchronously for a first local inference call."""
        self.start_download()
        thread = self._thread
        if thread:
            thread.join()
        current = self.status()
        if not current["available"]:
            raise RuntimeError(current["error"] or "The local model could not be installed")

    def _download(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._part_path.stat().st_size if self._part_path.exists() else 0
            headers = {"User-Agent": "Nova/1.0"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(self.url, headers=headers)
            with urllib.request.urlopen(request, timeout=60, context=_SSL_CONTEXT) as response:
                status_code = getattr(response, "status", response.getcode())
                resumed = existing > 0 and status_code == 206
                if not resumed:
                    existing = 0
                content_length = int(response.headers.get("Content-Length", "0") or 0)
                total = existing + content_length if content_length else 0
                mode = "ab" if resumed else "wb"
                with self._part_path.open(mode) as output:
                    with self._lock:
                        self._downloaded_bytes = existing
                        self._total_bytes = total
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        with self._lock:
                            self._downloaded_bytes += len(block)
                with self._lock:
                    self._total_bytes = self._downloaded_bytes

            with self._part_path.open("rb") as model_file:
                if model_file.read(4) != b"GGUF":
                    raise RuntimeError("Downloaded file is not a valid GGUF model")
            os.replace(self._part_path, self.path)
            with self._lock:
                self._state = "ready"
                self._error = ""
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc)