"""Main entry point for the Nova desktop backend.

Serves the FastAPI API and the desktop frontend (HTML/JS/CSS) for pywebview.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import threading
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Reduce native threading oversubscription that can trigger segfaults
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

faulthandler.enable(all_threads=True)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from .api.routes import create_router
from .config import Settings
from .engines.manager import EngineManager
from .prompting.builder import PromptBuilder
from .retrieval.embeddings import EmbeddingModelManager
from .retrieval.service import RetrievalService
from .services.conversation import ConversationService
from .services.quiz import QuizService
from .storage.database import Storage

logger = logging.getLogger(__name__)

# Path to the desktop frontend directory
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "desktop"
STATIC_DIR = FRONTEND_DIR / "static"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the Nova FastAPI application."""
    settings = settings or Settings.load()

    # Initialize storage
    storage = Storage(db_path=settings.sqlite_path)

    # Initialize engine manager
    engine_manager = EngineManager(settings=settings)

    # Initialize the embedding model installer/reporter.  Retrieval itself
    # stays offline-first; this only powers status and an explicit download.
    embedding_manager = EmbeddingModelManager(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )

    # Initialize retrieval service
    retrieval = RetrievalService(
        storage=storage,
        settings=settings,
        embedding_manager=embedding_manager,
    )

    # Initialize prompt builder
    prompt_builder = PromptBuilder()

    # Initialize services
    conversation_service = ConversationService(
        storage=storage,
        engine_manager=engine_manager,
        retrieval=retrieval,
        prompt_builder=prompt_builder,
    )
    quiz_service = QuizService(
        storage=storage,
        engine_manager=engine_manager,
        prompt_builder=prompt_builder,
    )

    lifespan = _build_lifespan(settings, engine_manager)

    # Create FastAPI app
    app = FastAPI(title="Nova", version="1.0.0", lifespan=lifespan)

    # CORS: the desktop UI is served from this same local server, so only local
    # origins are allowed and credentials stay off.  A wildcard origin combined
    # with credentials lets any visited web page drive the local API.
    local_origins = [
        f"http://{settings.host}:{settings.port}",
        f"http://localhost:{settings.port}",
        f"http://127.0.0.1:{settings.port}",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=local_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    router = create_router(
        storage=storage,
        conversation_service=conversation_service,
        quiz_service=quiz_service,
        engine_manager=engine_manager,
        embedding_manager=embedding_manager,
    )
    app.include_router(router, prefix="/api")

    # Serve static files (JS, CSS) with no-cache headers
    os.makedirs(str(STATIC_DIR), exist_ok=True)

    class NoCacheStaticFiles(StaticFiles):
        """StaticFiles that adds no-cache headers to prevent stale JS/CSS."""
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            if path.endswith(".js") or path.endswith(".css"):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")

    # Serve index.html for the root and all non-API paths (SPA-like)
    @app.get("/")
    async def serve_index():
        response = FileResponse(str(FRONTEND_DIR / "index.html"))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Avoid noisy browser 404s when no custom favicon is bundled."""
        return Response(status_code=204)

    return app


def _build_lifespan(settings: Settings, engine_manager: EngineManager):
    """Startup checks: validate the GGUF and genuinely pre-load it."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.engine == "local":
            model_status = engine_manager.model_manager.status()
            if not model_status["available"]:
                logger.warning(
                    "⚠️  Local model NOT ready — state=%s path=%s. "
                    "Open Settings → Install model to download it.",
                    model_status["state"],
                    model_status["path"],
                )
            else:
                def _warm_up() -> None:
                    try:
                        logger.info("Pre-loading local model in background…")
                        # ``warm_up`` actually loads the weights; merely
                        # constructing the engine leaves the first user request
                        # paying the full cold-load.
                        if engine_manager.warm_up_local_model():
                            logger.info("✅ Local model loaded and ready.")
                        else:
                            logger.warning("Local model pre-load returned no model")
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Model pre-load skipped: %s", exc)

                threading.Thread(
                    target=_warm_up, name="nova-model-warmup", daemon=True
                ).start()
        yield

    return lifespan


def run_server(settings: Optional[Settings] = None) -> None:
    """Run the FastAPI server."""
    try:
        settings = settings or Settings.load()
        app = create_app(settings)
        logger.info(
            "Starting Nova server on http://%s:%d",
            settings.host,
            settings.port,
        )
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level="info",
        )
    except Exception:
        logger.exception("Server failed to start/run")
        raise


def run_desktop(settings: Optional[Settings] = None) -> None:
    """Run the Nova desktop application with pywebview UI."""
    try:
        settings = settings or Settings.load()

        # Start server in background thread
        server_thread = threading.Thread(
            target=run_server,
            args=(settings,),
            daemon=True,
        )
        server_thread.start()

        # Give the server a moment to start
        import time
        time.sleep(2.0)

        # Launch pywebview
        try:
            import webview
            window = webview.create_window(
                title="Nova — Your On-Device Tutor",
                url=f"http://{settings.host}:{settings.port}",
                width=1200,
                height=800,
                resizable=True,
                min_size=(800, 600),
            )
            webview.start()
        except ImportError:
            logger.warning(
                "pywebview not installed. Server is running at http://%s:%s",
                settings.host,
                settings.port,
            )
            # Keep the server running
            server_thread.join()
    except Exception:
        logger.exception("Desktop app failed to start")
        raise


if __name__ == "__main__":
    import sys

    faulthandler.enable(all_threads=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        settings = Settings.load()
        if "--headless" in sys.argv or "--server" in sys.argv:
            run_server(settings)
        else:
            run_desktop(settings)
    except Exception:
        logging.exception("Nova failed to start")
        sys.exit(1)
