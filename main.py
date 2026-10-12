"""PyInstaller entry point for the Nova desktop release."""

from __future__ import annotations

import faulthandler
import logging
import os
import sys

# Reduce native threading oversubscription that can trigger segfaults
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

faulthandler.enable(all_threads=True)

from Nova.backend.config import Settings
from Nova.backend.main import run_desktop, run_server


def main() -> None:
    """Launch the desktop UI, or the local server for diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        settings = Settings.load()
        if "--server" in sys.argv or "--headless" in sys.argv:
            run_server(settings)
        else:
            run_desktop(settings)
    except Exception:
        logging.exception("Nova failed to start")
        raise


if __name__ == "__main__":
    main()
