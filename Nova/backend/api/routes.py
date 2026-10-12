"""FastAPI routes for the Nova backend API.

Provides endpoints for:
  - Documents: upload, list, delete, chunk previews
  - Sessions: create, list, get
  - Chat: send message, get history, stream tokens
  - Quizzes: generate, answer, get results
  - Models: GGUF install, embedding-model status/install

Blocking work (model inference, SQLite, PDF parsing, embeddings) is dispatched
off the event loop: handlers that do not need ``await`` are declared ``def`` so
FastAPI runs them in its threadpool, and the upload handler defers its heavy
steps with ``run_in_threadpool``.  Otherwise one local generation freezes every
other request for seconds.
"""

from __future__ import annotations

import logging
import os
import struct
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..prompting.builder import PromptBuilder
from ..services.conversation import ConversationService
from ..services.quiz import QuizService
from ..storage.database import Storage
from ..retrieval.chunker import process_document

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 200
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# ── Request / Response Models (module-level for FastAPI inspection) ──

class ChatRequest(BaseModel):
    session_id: str
    query: str = Field(min_length=1, max_length=12000)
    document_id: Optional[str] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class CreateSessionRequest(BaseModel):
    engine_used: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    engine_used: str
    created_at: float
    last_activity_at: float


class QuizGenerateRequest(BaseModel):
    session_id: str
    chunk_id: str
    num_questions: int = Field(default=5, ge=1, le=10)


class QuizAnswerRequest(BaseModel):
    quiz_id: str
    question_index: int = Field(ge=0)
    student_answer: str
    # Accepted for backwards compatibility; the stored quiz is authoritative.
    correct_index: Optional[int] = Field(default=None, ge=0, le=3)


class SettingUpdate(BaseModel):
    value: str


class DocumentResponse(BaseModel):
    id: str
    title: str
    source_filename: str
    created_at: float
    chunk_count: int = 0
    chunks: list[dict] = Field(default_factory=list)


def _sample_chunks(chunks: list[dict], max_chunks: int) -> list[dict]:
    """Pick chunks spread across the document instead of its first pages."""
    if len(chunks) <= max_chunks:
        return chunks
    step = len(chunks) / max_chunks
    return [chunks[min(len(chunks) - 1, int(index * step))] for index in range(max_chunks)]


def create_router(
    storage: Storage,
    conversation_service: ConversationService,
    quiz_service: QuizService,
    engine_manager=None,
    model_manager=None,
    embedding_manager=None,
) -> APIRouter:
    """Create the FastAPI router with all endpoints."""
    router = APIRouter()
    model_manager = model_manager or getattr(engine_manager, "model_manager", None)
    retrieval = getattr(conversation_service, "retrieval", None)

    def _document_response(doc: dict, summaries: dict[str, list[dict]]) -> DocumentResponse:
        chunks = summaries.get(doc["id"], [])
        return DocumentResponse(
            id=doc["id"],
            title=doc["title"],
            source_filename=doc["source_filename"],
            created_at=doc["created_at"],
            chunk_count=len(chunks),
            chunks=chunks,
        )

    # ── Health ─────────────────────────────────────────────────

    @router.get("/health")
    def health():
        result = {"status": "ok"}
        if engine_manager is not None:
            get_active = getattr(engine_manager, "get_active_model", None)
            active = get_active() if callable(get_active) else {}
            result.update({
                "engine": engine_manager.settings.engine,
                "active_model_id": engine_manager.settings.active_model_id,
                "active_model_name": active.get("name", ""),
                "model_path": engine_manager.settings.model_path,
                "model_configured": bool(engine_manager.settings.model_path),
                "model_loaded": bool(getattr(engine_manager, "local_model_loaded", False)),
            })
        if embedding_manager is not None:
            embedding_state = embedding_manager.status()
            result["embeddings"] = {
                "state": embedding_state["state"],
                "available": embedding_state["available"],
                "model": embedding_state["model"],
            }
        return result

    @router.get("/models/available")
    def list_available_models():
        if engine_manager is None:
            raise HTTPException(status_code=503, detail="Engine manager is unavailable")
        return {
            "models": engine_manager.get_available_models(),
            "active_model_id": engine_manager.settings.active_model_id,
        }

    @router.get("/models/status")
    def model_status():
        if model_manager is None:
            raise HTTPException(status_code=503, detail="Model installer is unavailable")
        return model_manager.status()

    @router.post("/models/download", status_code=202)
    def download_model():
        if model_manager is None:
            raise HTTPException(status_code=503, detail="Model installer is unavailable")
        return model_manager.start_download()

    # ── Embedding model (optional hybrid retrieval) ────────────

    def _require_embedding_manager():
        if embedding_manager is None:
            raise HTTPException(status_code=503, detail="Embedding installer is unavailable")
        return embedding_manager

    @router.get("/models/embeddings/status")
    def embedding_model_status(probe: bool = False):
        """Report whether semantic retrieval is available.

        ``probe=true`` performs a real local load check (a few seconds); it is
        used by the Settings view so a missing model is visible to the user
        instead of silently degrading to keyword search.
        """
        manager = _require_embedding_manager()
        return manager.status(probe=probe)

    @router.post("/models/embeddings/download", status_code=202)
    def download_embedding_model():
        manager = _require_embedding_manager()
        result = manager.start_download()
        if retrieval is not None:
            retrieval.refresh_embedding_provider()
        return result

    # ── Documents ─────────────────────────────────────────────

    @router.post("/documents/upload", response_model=DocumentResponse)
    async def upload_document(file: UploadFile = File(...)):
        """Upload a PDF, extract text, chunk it, embed it, and store."""
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.close()
        try:
            content = await file.read()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="PDF must be smaller than 25 MB")
            with open(tmp_path, "wb") as output:
                output.write(content)

            title = os.path.splitext(file.filename)[0]
            doc_id = storage.add_document(title=title, source_filename=file.filename)

            try:
                # PDF parsing and chunking are CPU-bound: keep them off the loop.
                chunks = await run_in_threadpool(process_document, tmp_path, doc_id)
            except Exception as exc:
                storage.delete_document(doc_id)
                logger.warning("PDF processing failed for %s: %s", file.filename, exc)
                raise HTTPException(status_code=422, detail="Could not extract readable text from this PDF") from exc
            if not chunks:
                storage.delete_document(doc_id)
                raise HTTPException(status_code=422, detail="The PDF did not contain extractable text")
            storage.add_chunks(chunks)

            try:
                embedder = _build_embedding_provider(retrieval)
                texts = [chunk["content"] for chunk in chunks]
                embeddings = await run_in_threadpool(embedder.embed_batch, texts)
                blobs = [
                    (chunk["id"], struct.pack(f"{len(vector)}f", *vector))
                    for chunk, vector in zip(chunks, embeddings)
                    if vector
                ]
                if blobs:
                    storage.add_embeddings(blobs)
                if retrieval is not None:
                    # Reuse the loaded embedder for the next query instead of
                    # loading MiniLM a second time.
                    retrieval.embedding_provider = embedder
                    retrieval._embedding_failed = False
                logger.info("Stored %d embeddings", len(blobs))
            except Exception as e:
                logger.warning("Embedding generation skipped: %s", e)

            doc = storage.get_document(doc_id)
            if not doc:
                raise HTTPException(status_code=500, detail="Document was not persisted")
            summaries = storage.get_chunk_summaries([doc_id])
            return _document_response(doc, summaries)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @router.get("/documents", response_model=list[DocumentResponse])
    def list_documents():
        docs = storage.list_documents()
        if not docs:
            return []
        # One SQL pass for every document's chunk previews; nothing loads full
        # chunk bodies just to display a count and a preview.
        summaries = storage.get_chunk_summaries([doc["id"] for doc in docs])
        return [_document_response(doc, summaries) for doc in docs]

    @router.get("/documents/{doc_id}", response_model=DocumentResponse)
    def get_document(doc_id: str):
        doc = storage.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return _document_response(doc, storage.get_chunk_summaries([doc_id]))

    @router.get("/documents/{doc_id}/chunks")
    def get_document_chunks(doc_id: str):
        """Chunk list used by the quiz chunk picker."""
        doc = storage.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = storage.get_chunk_summaries([doc_id]).get(doc_id, [])
        return {"document_id": doc_id, "chunk_count": len(chunks), "chunks": chunks}

    @router.delete("/documents/{doc_id}")
    def delete_document(doc_id: str):
        doc = storage.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        storage.delete_document(doc_id)
        return {"status": "deleted", "id": doc_id}

    @router.get("/documents/{doc_id}/summary")
    def summarize_document(doc_id: str):
        """Generate an AI summary of an uploaded document."""
        doc = storage.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = storage.get_chunks_for_document(doc_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document contains no text chunks to summarize")

        context_window = 2048
        if engine_manager is not None:
            context_window = int(getattr(engine_manager.settings, "context_size", 2048))
        # Sample across the whole document so summaries cover more than the
        # first few pages of a long PDF.
        sampled = _sample_chunks(chunks, max_chunks=10)
        combined_text = "\n\n".join(chunk["content"] for chunk in sampled)

        engine = engine_manager.get_engine()
        prompt_builder = PromptBuilder()
        messages = prompt_builder.build_summary_messages(
            combined_text,
            context_window=context_window,
            max_response_tokens=600,
        )
        try:
            summary_text = engine.generate(messages=messages, max_tokens=600)
            summary_text = PromptBuilder.strip_special_tokens(summary_text)
            return {"summary": summary_text, "document_id": doc_id, "title": doc["title"]}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {exc}")

    # ── Sessions ──────────────────────────────────────────────

    @router.post("/sessions", response_model=SessionResponse)
    def create_session(body: CreateSessionRequest):
        engine_used = body.engine_used
        if not engine_used:
            engine_used = engine_manager.settings.engine if engine_manager is not None else "local"
        session_id = storage.create_session(engine_used=engine_used)
        session = storage.get_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create session")
        return SessionResponse(
            id=session["id"],
            engine_used=session["engine_used"],
            created_at=session["created_at"],
            last_activity_at=session["last_activity_at"],
        )

    @router.get("/sessions", response_model=list[SessionResponse])
    def list_sessions():
        sessions = storage.list_sessions()
        return [
            SessionResponse(
                id=s["id"],
                engine_used=s["engine_used"],
                created_at=s["created_at"],
                last_activity_at=s["last_activity_at"],
            )
            for s in sessions
        ]

    @router.get("/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str):
        session = storage.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse(
            id=session["id"],
            engine_used=session["engine_used"],
            created_at=session["created_at"],
            last_activity_at=session["last_activity_at"],
        )

    @router.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        session = storage.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        storage.delete_session(session_id)
        return {"status": "deleted", "id": session_id}

    # ── Chat ──────────────────────────────────────────────────

    @router.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest):
        """Send a message in a session and get the assistant's response."""
        if not storage.get_session(body.session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            response = conversation_service.ask(
                session_id=body.session_id,
                query=body.query,
                document_id=body.document_id,
                temperature=body.temperature,
            )
            return ChatResponse(response=response, session_id=body.session_id)
        except Exception as e:
            logger.error("Chat error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/chat/stream")
    async def chat_stream(body: ChatRequest):
        """Send a message in a session and stream the assistant's response tokens."""
        if not storage.get_session(body.session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        def event_stream():
            import json
            try:
                for token in conversation_service.ask_stream(
                    session_id=body.session_id,
                    query=body.query,
                    document_id=body.document_id,
                    temperature=body.temperature,
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as exc:
                logger.error("Chat stream error: %s", exc)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/sessions/{session_id}/messages")
    def get_messages(session_id: str):
        """Get all messages in a session."""
        if not storage.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return storage.get_messages(session_id)

    # ── Quizzes ───────────────────────────────────────────────

    @router.post("/quizzes/generate")
    def generate_quiz(body: QuizGenerateRequest):
        """Generate a quiz from a document chunk."""
        if not storage.get_session(body.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        if not storage.get_chunk(body.chunk_id):
            raise HTTPException(status_code=404, detail="Chunk not found")
        quiz = quiz_service.generate_quiz(
            session_id=body.session_id,
            chunk_id=body.chunk_id,
            num_questions=body.num_questions,
        )
        if not quiz:
            raise HTTPException(status_code=400, detail="Quiz generation failed - model could not produce valid questions. Try again or use a different chunk.")
        return quiz

    @router.post("/quizzes/answer")
    def answer_quiz(body: QuizAnswerRequest):
        """Submit an answer to a quiz question."""
        try:
            result = quiz_service.answer_question(
                quiz_id=body.quiz_id,
                question_index=body.question_index,
                student_answer=body.student_answer,
                correct_index=body.correct_index,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    @router.get("/sessions/{session_id}/quiz-results")
    def get_quiz_results(session_id: str):
        """Get all quiz results for a session."""
        if not storage.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return quiz_service.get_quiz_results(session_id)

    # ── Chunks ────────────────────────────────────────────────

    @router.get("/chunks/{chunk_id}")
    def get_chunk(chunk_id: str):
        chunk = storage.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return chunk

    # ── Settings ──────────────────────────────────────────────

    @router.get("/settings/{key}")
    def get_setting(key: str):
        # Runtime settings are owned by EngineManager.  The database is kept
        # as a compatibility fallback for older installations that persisted
        # a setting before the runtime configuration was introduced.
        public_runtime_settings = {"engine", "temperature", "context_size", "model_path", "n_threads"}
        if engine_manager is not None and key in public_runtime_settings:
            settings = engine_manager.settings
            return {"key": key, "value": getattr(settings, key)}
        value = storage.get_setting(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Setting not found")
        return {"key": key, "value": value}

    @router.put("/settings/{key}")
    def set_setting(key: str, body: SettingUpdate):
        if engine_manager is not None:
            try:
                engine_manager.update_setting(key, body.value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid setting {key}: {exc}") from exc
        storage.set_setting(key, body.value)
        return {"key": key, "value": body.value}

    return router


def _build_embedding_provider(retrieval):
    """Create an embedding provider from the retrieval settings."""
    from ..retrieval.embeddings import MiniLMEmbedding

    settings = getattr(retrieval, "settings", None)
    return MiniLMEmbedding(
        model_name=getattr(settings, "embedding_model", "all-MiniLM-L6-v2"),
        cache_dir=getattr(settings, "embedding_cache_dir", "") or None,
    )
