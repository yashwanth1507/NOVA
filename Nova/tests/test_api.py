"""API-level tests for the desktop learning loop."""

import json
import os
import tempfile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from Nova.backend.api.routes import create_router
from Nova.backend.config import Settings
from Nova.backend.prompting.builder import PromptBuilder
from Nova.backend.retrieval.service import RetrievalService
from Nova.backend.services.conversation import ConversationService
from Nova.backend.services.quiz import QuizService
from Nova.backend.storage.database import Storage


class FakeEngine:
    name = "fake-local"
    is_online = False

    def generate(self, messages, temperature=None, max_tokens=None):
        if any("Quiz Generator" in m.get("content", "") for m in messages):
            return json.dumps({"questions": [{
                "question": "Which subject is being studied?",
                "options": ["A) Science", "B) Music", "C) Art", "D) Sport"],
                "correct_index": 0,
            }]})
        return "The study material explains the topic clearly."

    def generate_stream(self, messages, temperature=None, max_tokens=None):
        yield "The study "
        yield "material explains "
        yield "the topic clearly."


class FakeEngineManager:
    def __init__(self):
        self.settings = Settings(engine="local")
        self.engine = FakeEngine()

    def get_engine(self):
        return self.engine

    def update_setting(self, key, value):
        if key == "engine" and value not in ("local", "online"):
            raise ValueError("invalid engine")


def build_client(db_path):
    storage = Storage(db_path)
    manager = FakeEngineManager()
    retrieval = RetrievalService(storage=storage, settings=manager.settings)
    conversation = ConversationService(storage, manager, retrieval, PromptBuilder())
    quizzes = QuizService(storage, manager, PromptBuilder())
    app = FastAPI()
    app.include_router(create_router(storage, conversation, quizzes, manager), prefix="/api")
    return TestClient(app), storage


def test_desktop_chat_quiz_and_persistence_flow():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, storage = build_client(db_path)
        document_id = storage.add_document("Lesson", "lesson.pdf")
        storage.add_chunk(
            "chunk-1",
            document_id,
            0,
            "Science explains the study topic.",
            {"heading": "Introduction"},
        )

        session = client.post("/api/sessions", json={})
        assert session.status_code == 200
        session_id = session.json()["id"]
        assert session.json()["engine_used"] == "local"

        chat = client.post("/api/chat", json={
            "session_id": session_id,
            "query": "What does the lesson explain?",
            "document_id": document_id,
        })
        assert chat.status_code == 200
        assert "study material" in chat.json()["response"]

        generated = client.post("/api/quizzes/generate", json={
            "session_id": session_id,
            "chunk_id": "chunk-1",
            "num_questions": 1,
        })
        assert generated.status_code == 200
        quiz_id = generated.json()["id"]

        answer = client.post("/api/quizzes/answer", json={
            "quiz_id": quiz_id,
            "question_index": 0,
            "student_answer": "0",
            "correct_index": 0,
        })
        assert answer.status_code == 200
        assert answer.json()["is_correct"] is True

        results = client.get(f"/api/sessions/{session_id}/quiz-results")
        assert results.status_code == 200
        assert results.json()[0]["score"] == "1/1"

        restored = Storage(db_path)
        assert len(restored.get_messages(session_id)) == 2
        assert restored.get_quiz(quiz_id) is not None
    finally:
        os.unlink(db_path)


def test_desktop_chat_streaming_flow():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, storage = build_client(db_path)
        document_id = storage.add_document("Lesson", "lesson.pdf")
        storage.add_chunk(
            "chunk-1",
            document_id,
            0,
            "Science explains the study topic.",
            {"heading": "Introduction"},
        )

        session = client.post("/api/sessions", json={})
        assert session.status_code == 200
        session_id = session.json()["id"]

        chat_stream = client.post("/api/chat/stream", json={
            "session_id": session_id,
            "query": "What does the lesson explain?",
            "document_id": document_id,
        })
        assert chat_stream.status_code == 200
        assert "text/event-stream" in chat_stream.headers["content-type"]
        body_text = chat_stream.text
        assert "The study " in body_text
        assert "material explains " in body_text

        # Verify assistant message was persisted after streaming
        messages = storage.get_messages(session_id)
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert "The study material explains the topic clearly." in messages[1]["content"]
    finally:
        os.unlink(db_path)


def test_invalid_runtime_setting_is_rejected():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, _ = build_client(db_path)
        response = client.put("/api/settings/engine", json={"value": "unsupported"})
        assert response.status_code == 400
    finally:
        os.unlink(db_path)


def test_runtime_settings_are_available_to_the_desktop_ui():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, _ = build_client(db_path)
        for key in ("engine", "temperature", "context_size", "model_path"):
            response = client.get(f"/api/settings/{key}")
            assert response.status_code == 200
            assert response.json()["key"] == key
    finally:
        os.unlink(db_path)


# ── Regressions for the document/chunk/summary endpoints ──

def _seed_document(client, storage, chunks=3):
    document_id = storage.add_document("Lecture", "lecture.pdf")
    storage.add_chunks([
        {
            "id": f"chunk-{i}",
            "document_id": document_id,
            "position": i,
            "content": f"CHAPTER {i + 1} TOPIC\n\nBody text for section {i + 1}. " * 12,
            "metadata": {"heading": f"CHAPTER {i + 1} TOPIC"},
        }
        for i in range(chunks)
    ])
    return document_id


def test_document_summary_endpoint_succeeds():
    """This endpoint used to raise NameError (PromptBuilder was never imported)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, storage = build_client(db_path)
        document_id = _seed_document(client, storage)
        response = client.get(f"/api/documents/{document_id}/summary")
        assert response.status_code == 200
        assert response.json()["summary"]
        assert response.json()["document_id"] == document_id
    finally:
        os.unlink(db_path)


def test_document_list_returns_chunk_counts_and_previews():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, storage = build_client(db_path)
        _seed_document(client, storage, chunks=4)

        listed = client.get("/api/documents").json()[0]
        assert listed["chunk_count"] == 4
        assert len(listed["chunks"]) == 4
        preview = listed["chunks"][0]
        assert len(preview["content_preview"]) <= 200
        assert preview["content_length"] > len(preview["content_preview"])
        assert preview["heading"] == "CHAPTER 1 TOPIC"
    finally:
        os.unlink(db_path)


def test_document_chunks_endpoint_powers_the_quiz_picker():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, storage = build_client(db_path)
        document_id = _seed_document(client, storage)
        response = client.get(f"/api/documents/{document_id}/chunks")
        assert response.status_code == 200
        body = response.json()
        assert body["chunk_count"] == 3
        assert [c["heading"] for c in body["chunks"]] == [
            "CHAPTER 1 TOPIC", "CHAPTER 2 TOPIC", "CHAPTER 3 TOPIC",
        ]
    finally:
        os.unlink(db_path)


def test_quiz_can_be_generated_from_any_chunk():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, storage = build_client(db_path)
        document_id = _seed_document(client, storage)
        session_id = client.post("/api/sessions", json={}).json()["id"]

        third = client.get(f"/api/documents/{document_id}/chunks").json()["chunks"][2]
        generated = client.post("/api/quizzes/generate", json={
            "session_id": session_id,
            "chunk_id": third["id"],
            "num_questions": 2,
        })
        assert generated.status_code == 200
        assert generated.json()["chunk_id"] == third["id"]
    finally:
        os.unlink(db_path)


def test_answer_does_not_require_a_client_supplied_correct_index():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, storage = build_client(db_path)
        document_id = _seed_document(client, storage, chunks=1)
        session_id = client.post("/api/sessions", json={}).json()["id"]
        quiz = client.post("/api/quizzes/generate", json={
            "session_id": session_id,
            "chunk_id": "chunk-0",
            "num_questions": 1,
        }).json()

        response = client.post("/api/quizzes/answer", json={
            "quiz_id": quiz["id"],
            "question_index": 0,
            "student_answer": "1",
        })
        assert response.status_code == 200
        assert isinstance(response.json()["is_correct"], bool)
    finally:
        os.unlink(db_path)


def test_health_reports_embedding_and_model_state():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name
    try:
        client, _ = build_client(db_path)
        health = client.get("/api/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
    finally:
        os.unlink(db_path)
