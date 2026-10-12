"""Tests for the SQLite storage layer."""

import os
import tempfile

from Nova.backend.storage.database import Storage


def test_storage_creates_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        # Verify tables exist by running a query
        docs = storage.list_documents()
        assert docs == []
    finally:
        os.unlink(db_path)


def test_document_crud():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)

        # Create
        doc_id = storage.add_document(title="Test Doc", source_filename="test.pdf")
        assert doc_id is not None

        # Read
        doc = storage.get_document(doc_id)
        assert doc is not None
        assert doc["title"] == "Test Doc"
        assert doc["source_filename"] == "test.pdf"

        # List
        docs = storage.list_documents()
        assert len(docs) == 1

        # Delete
        storage.delete_document(doc_id)
        assert storage.get_document(doc_id) is None
    finally:
        os.unlink(db_path)


def test_chunk_storage():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("Test", "test.pdf")

        chunks = [
            {"id": "chunk1", "document_id": doc_id, "position": 0, "content": "Content A", "metadata": {"heading": "Intro"}},
            {"id": "chunk2", "document_id": doc_id, "position": 1, "content": "Content B", "metadata": {"heading": "Body"}},
        ]
        storage.add_chunks(chunks)

        retrieved = storage.get_chunks_for_document(doc_id)
        assert len(retrieved) == 2
        assert retrieved[0]["content"] == "Content A"
        assert retrieved[1]["content"] == "Content B"
    finally:
        os.unlink(db_path)


def test_session_and_messages():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)

        session_id = storage.create_session(engine_used="local")
        assert session_id is not None

        msg_id = storage.add_message(session_id, "user", "Hello")
        assert msg_id is not None

        storage.add_message(session_id, "assistant", "Hi there!")

        messages = storage.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[0]["sequence_number"] == 1
        assert messages[1]["sequence_number"] == 2

        # Session activity updated
        session = storage.get_session(session_id)
        assert session is not None
        assert session["last_activity_at"] > 0
    finally:
        os.unlink(db_path)


def test_quiz_and_answers():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("Test", "test.pdf")
        storage.add_chunk("chunk1", doc_id, 0, "Content", {"heading": "Intro"})

        session_id = storage.create_session()
        questions = [
            {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_index": 0},
        ]

        quiz_id = storage.create_quiz(session_id, "chunk1", questions)
        assert quiz_id is not None

        quiz = storage.get_quiz(quiz_id)
        assert quiz is not None
        assert len(quiz["questions"]) == 1

        answer_id = storage.add_quiz_answer(quiz_id, 0, True, "0")
        assert answer_id is not None

        answers = storage.get_answers_for_quiz(quiz_id)
        assert len(answers) == 1
        assert answers[0]["is_correct"] is True
    finally:
        os.unlink(db_path)


def test_settings():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)

        assert storage.get_setting("nonexistent", "default") == "default"

        storage.set_setting("engine", "online")
        assert storage.get_setting("engine") == "online"

        storage.set_setting("temperature", 0.7)
        assert storage.get_setting("temperature") == 0.7
    finally:
        os.unlink(db_path)


# ── Scoped reads, previews and concurrency ───────────────

def test_scoped_reads_ignore_other_documents():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_a = storage.add_document("A", "a.pdf")
        doc_b = storage.add_document("B", "b.pdf")
        storage.add_chunks([
            {"id": "a1", "document_id": doc_a, "position": 0, "content": "alpha", "metadata": {}},
            {"id": "b1", "document_id": doc_b, "position": 0, "content": "beta", "metadata": {}},
        ])
        storage.add_embeddings([("a1", b"\x00\x00\x80\x3f"), ("b1", b"\x00\x00\x00\x40")])

        assert [c["id"] for c in storage.get_chunks_for_documents([doc_b])] == ["b1"]
        assert [cid for cid, _ in storage.get_embeddings_for_document(doc_b)] == ["b1"]
        assert [cid for cid, _ in storage.get_embeddings_for_chunks(["a1"])] == ["a1"]
        assert storage.get_chunks_for_documents([]) == []
        assert storage.get_embeddings_for_chunks([]) == []
    finally:
        os.unlink(db_path)


def test_chunk_summaries_are_previews_not_full_bodies():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("Long", "long.pdf")
        storage.add_chunks([{
            "id": "c1", "document_id": doc_id, "position": 0,
            "content": "x" * 5000, "metadata": {"heading": "Chapter 1"},
        }])

        grouped = storage.get_chunk_summaries([doc_id])
        summary = grouped[doc_id][0]
        assert summary["content_preview"] == "x" * 200
        assert summary["content_length"] == 5000
        assert summary["heading"] == "Chapter 1"
        assert "content" not in summary
        assert storage.get_chunk_summaries([]) == {}
    finally:
        os.unlink(db_path)


def test_deleting_a_document_removes_its_embeddings():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("A", "a.pdf")
        storage.add_chunks([{"id": "c1", "document_id": doc_id, "position": 0, "content": "text", "metadata": {}}])
        storage.add_embeddings([("c1", b"\x00\x00\x80\x3f")])

        storage.delete_document(doc_id)
        assert storage.get_chunks_for_documents([doc_id]) == []
        assert storage.get_embeddings_for_document(doc_id) == []
    finally:
        os.unlink(db_path)


def test_concurrent_message_inserts_get_unique_sequence_numbers():
    import threading

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        session_id = storage.create_session()

        def writer(index):
            storage.add_message(session_id, "user", f"message {index}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        messages = storage.get_messages(session_id)
        sequences = [m["sequence_number"] for m in messages]
        assert len(messages) == 8
        assert sorted(sequences) == list(range(1, 9))
    finally:
        os.unlink(db_path)
