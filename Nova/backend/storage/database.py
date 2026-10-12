"""Storage layer — SQLite implementation for desktop.

Schema mirrors the spec exactly:
  documents, chunks, embeddings, sessions, messages, quizzes, quiz_answers, settings
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Storage:
    """SQLite-backed storage for Nova."""

    def __init__(self, db_path: str = "nova_data/nova.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        # ``timeout`` + ``busy_timeout`` allow concurrent readers/writers
        # (the API threadpool can serve a chat request while the UI polls
        # status) instead of failing immediately with "database is locked".
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 30000")
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            except sqlite3.OperationalError:
                # Read-only or non-WAL capable filesystems still work.
                pass
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    engine_used TEXT NOT NULL DEFAULT 'local',
                    created_at REAL NOT NULL,
                    last_activity_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS quizzes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                );

                CREATE TABLE IF NOT EXISTS quiz_answers (
                    id TEXT PRIMARY KEY,
                    quiz_id TEXT NOT NULL,
                    question_index INTEGER NOT NULL,
                    is_correct INTEGER NOT NULL,
                    student_answer TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_quizzes_session ON quizzes(session_id);
                CREATE INDEX IF NOT EXISTS idx_quizzes_chunk ON quizzes(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_quiz_answers_quiz ON quiz_answers(quiz_id);
            """)
        logger.info("Database schema initialized at %s", self.db_path)

    # ── Documents ──────────────────────────────────────────────

    def add_document(self, title: str, source_filename: str) -> str:
        doc_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO documents (id, title, source_filename, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, title, source_filename, now),
            )
        return doc_id

    def get_document(self, doc_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as conn:
            # Remove dependent records before the chunks they reference. This
            # keeps deletion correct for both fresh databases and databases
            # opened with foreign-key enforcement enabled.
            conn.execute(
                """DELETE FROM quiz_answers
                   WHERE quiz_id IN (
                       SELECT id FROM quizzes
                       WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
                   )""",
                (doc_id,),
            )
            conn.execute(
                "DELETE FROM quizzes WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (doc_id,),
            )
            conn.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (doc_id,),
            )
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # ── Chunks ─────────────────────────────────────────────────

    def add_chunk(self, chunk_id: str, document_id: str, position: int, content: str, metadata: Optional[dict] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chunks (id, document_id, position, content, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, document_id, position, content, json.dumps(metadata or {})),
            )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Bulk insert chunks."""
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks (id, document_id, position, content, metadata_json) VALUES (:id, :document_id, :position, :content, :metadata_json)",
                [
                    {
                        "id": c["id"],
                        "document_id": c["document_id"],
                        "position": c["position"],
                        "content": c["content"],
                        "metadata_json": json.dumps(c.get("metadata", {})),
                    }
                    for c in chunks
                ],
            )

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return self._chunk_row(row) if row else None

    def get_chunks_for_document(self, document_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY position", (document_id,)
            ).fetchall()
        return [self._chunk_row(r) for r in rows]

    def get_all_chunks(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY document_id, position").fetchall()
        return [self._chunk_row(r) for r in rows]

    def get_chunks_for_documents(self, document_ids: list[str]) -> list[dict]:
        """Return chunks for a set of documents without touching the rest."""
        ids = [document_id for document_id in (document_ids or []) if document_id]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT * FROM chunks WHERE document_id IN ({placeholders})
                    ORDER BY document_id, position""",
                ids,
            ).fetchall()
        return [self._chunk_row(r) for r in rows]

    def get_chunk_summaries(
        self,
        document_ids: Optional[list[str]] = None,
        preview_chars: int = 200,
    ) -> dict[str, list[dict]]:
        """Return chunk previews grouped by document, read entirely in SQL.

        Only the preview slice and the content length cross the SQLite → Python
        boundary, so listing documents never materializes every chunk body.
        """
        if document_ids is not None and not document_ids:
            return {}
        sql = (
            "SELECT document_id, id, position, substr(content, 1, ?) AS content_preview, "
            "length(content) AS content_length, metadata_json FROM chunks"
        )
        params: list[Any] = [max(1, int(preview_chars))]
        if document_ids is not None:
            placeholders = ",".join("?" for _ in document_ids)
            sql += f" WHERE document_id IN ({placeholders})"
            params.extend(document_ids)
        sql += " ORDER BY document_id, position"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        grouped: dict[str, list[dict]] = {}
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            grouped.setdefault(row["document_id"], []).append({
                "id": row["id"],
                "position": row["position"],
                "content_preview": row["content_preview"] or "",
                "content_length": row["content_length"] or 0,
                "heading": str(metadata.get("heading", "") or ""),
            })
        return grouped

    @staticmethod
    def _chunk_row(row: sqlite3.Row) -> dict:
        chunk = dict(row)
        metadata_json = chunk.pop("metadata_json", None)
        try:
            chunk["metadata"] = json.loads(metadata_json) if metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            chunk["metadata"] = {}
        return chunk

    # ── Embeddings ─────────────────────────────────────────────

    def add_embedding(self, chunk_id: str, vector_blob: bytes) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, vector_blob) VALUES (?, ?)",
                (chunk_id, vector_blob),
            )

    def add_embeddings(self, embeddings: list[tuple[str, bytes]]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (chunk_id, vector_blob) VALUES (?, ?)",
                embeddings,
            )

    def get_embedding(self, chunk_id: str) -> Optional[bytes]:
        with self._conn() as conn:
            row = conn.execute("SELECT vector_blob FROM embeddings WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return row["vector_blob"] if row else None

    def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT chunk_id, vector_blob FROM embeddings").fetchall()
        return [(r["chunk_id"], r["vector_blob"]) for r in rows]

    def get_embeddings_for_document(self, document_id: str) -> list[tuple[str, bytes]]:
        """Return only the vectors belonging to one document."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT embeddings.chunk_id, embeddings.vector_blob
                   FROM embeddings
                   JOIN chunks ON chunks.id = embeddings.chunk_id
                   WHERE chunks.document_id = ?""",
                (document_id,),
            ).fetchall()
        return [(r["chunk_id"], r["vector_blob"]) for r in rows]

    def get_embeddings_for_chunks(self, chunk_ids: list[str]) -> list[tuple[str, bytes]]:
        ids = [chunk_id for chunk_id in (chunk_ids or []) if chunk_id]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, vector_blob FROM embeddings WHERE chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
        return [(r["chunk_id"], r["vector_blob"]) for r in rows]

    # ── Sessions ───────────────────────────────────────────────

    def create_session(self, engine_used: str = "local") -> str:
        session_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, engine_used, created_at, last_activity_at) VALUES (?, ?, ?, ?)",
                (session_id, engine_used, now, now),
            )
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY last_activity_at DESC").fetchall()
        return [dict(r) for r in rows]

    def update_session_activity(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
                (time.time(), session_id),
            )

    # ── Messages ───────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str) -> str:
        msg_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            # Sequence allocation and insert are one statement, so two
            # concurrent requests can never pick the same sequence number.
            conn.execute(
                """INSERT INTO messages (id, session_id, sequence_number, role, content, timestamp)
                   SELECT ?, ?, COALESCE(MAX(sequence_number), 0) + 1, ?, ?, ?
                   FROM messages WHERE session_id = ?""",
                (msg_id, session_id, role, content, now, session_id),
            )
            conn.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
                (now, session_id),
            )
        return msg_id

    def get_messages(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY sequence_number",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY sequence_number DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Quizzes ────────────────────────────────────────────────

    def create_quiz(self, session_id: str, chunk_id: str, questions: list[dict]) -> str:
        quiz_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO quizzes (id, session_id, chunk_id, questions_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (quiz_id, session_id, chunk_id, json.dumps(questions), now),
            )
        return quiz_id

    def get_quiz(self, quiz_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
        if row:
            d = dict(row)
            d["questions"] = json.loads(d.pop("questions_json"))
            return d
        return None

    def get_quizzes_for_session(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM quizzes WHERE session_id = ? ORDER BY created_at", (session_id,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["questions"] = json.loads(d.pop("questions_json"))
            result.append(d)
        return result

    def get_quizzes_for_chunk(self, chunk_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM quizzes WHERE chunk_id = ? ORDER BY created_at", (chunk_id,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["questions"] = json.loads(d.pop("questions_json"))
            result.append(d)
        return result

    # ── Quiz Answers ───────────────────────────────────────────

    def add_quiz_answer(self, quiz_id: str, question_index: int, is_correct: bool, student_answer: str) -> str:
        answer_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO quiz_answers (id, quiz_id, question_index, is_correct, student_answer, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (answer_id, quiz_id, question_index, int(is_correct), student_answer, now),
            )
        return answer_id

    def get_answers_for_quiz(self, quiz_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM quiz_answers WHERE quiz_id = ? ORDER BY question_index",
                (quiz_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["is_correct"] = bool(d["is_correct"])
            result.append(d)
        return result

    # ── Settings ────────────────────────────────────────────────

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )

    # ── Cleanup ────────────────────────────────────────────────

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM quiz_answers WHERE quiz_id IN (SELECT id FROM quizzes WHERE session_id = ?)", (session_id,))
            conn.execute("DELETE FROM quizzes WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
