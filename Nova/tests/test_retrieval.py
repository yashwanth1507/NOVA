"""Tests for the retrieval service."""

import os
import tempfile

from Nova.backend.retrieval.service import RetrievalService
from Nova.backend.storage.database import Storage


def test_keyword_retrieval_returns_relevant_chunks():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("Test", "test.pdf")

        chunks = [
            {"id": "c1", "document_id": doc_id, "position": 0, "content": "Photosynthesis is the process plants use to convert sunlight into energy.", "metadata": {}},
            {"id": "c2", "document_id": doc_id, "position": 1, "content": "Mitosis is cell division that produces two identical daughter cells.", "metadata": {}},
            {"id": "c3", "document_id": doc_id, "position": 2, "content": "Gravity is a force that attracts objects with mass toward each other.", "metadata": {}},
        ]
        storage.add_chunks(chunks)

        retrieval = RetrievalService(storage=storage)

        # Search for photosynthesis content
        results = retrieval.retrieve("sunlight energy plants", top_k=2)
        assert len(results) > 0
        assert results[0]["id"] == "c1"  # Most relevant

        # Search for cell division
        results = retrieval.retrieve("cell division identical", top_k=2)
        assert len(results) > 0
        assert results[0]["id"] == "c2"

        # Search for gravity
        results = retrieval.retrieve("force mass attraction", top_k=2)
        assert len(results) > 0
        assert results[0]["id"] == "c3"
    finally:
        os.unlink(db_path)


def test_retrieval_with_document_filter():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc1 = storage.add_document("Doc1", "d1.pdf")
        doc2 = storage.add_document("Doc2", "d2.pdf")

        chunks = [
            {"id": "c1", "document_id": doc1, "position": 0, "content": "Python is a programming language.", "metadata": {}},
            {"id": "c2", "document_id": doc2, "position": 0, "content": "Python is also a type of snake.", "metadata": {}},
        ]
        storage.add_chunks(chunks)

        retrieval = RetrievalService(storage=storage)

        # Should only return from doc1
        results = retrieval.retrieve("programming language", top_k=5, document_id=doc1)
        assert len(results) == 1
        assert results[0]["id"] == "c1"

        # Should only return from doc2
        results = retrieval.retrieve("snake", top_k=5, document_id=doc2)
        assert len(results) == 1
        assert results[0]["id"] == "c2"
    finally:
        os.unlink(db_path)


def test_empty_retrieval():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        retrieval = RetrievalService(storage=storage)
        results = retrieval.retrieve("anything", top_k=5)
        assert results == []
    finally:
        os.unlink(db_path)


def test_retrieval_omits_unrelated_chunks():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("Test", "test.pdf")
        storage.add_chunks([
            {"id": "relevant", "document_id": doc_id, "position": 0,
             "content": "The water cycle includes evaporation and condensation.", "metadata": {}},
            {"id": "unrelated", "document_id": doc_id, "position": 1,
             "content": "The Roman empire used roads and aqueducts.", "metadata": {}},
        ])

        results = RetrievalService(storage=storage).retrieve(
            "how does evaporation work", top_k=5, document_id=doc_id
        )
        assert [chunk["id"] for chunk in results] == ["relevant"]
    finally:
        os.unlink(db_path)


# ── Scoping, backfill and vector scoring ─────────────────

class _FakeProvider:
    """Deterministic embedding provider for offline tests."""

    def __init__(self, dimension: int = 4, vector=None):
        self._dimension = dimension
        self._vector = list(vector or [1.0, 0.0, 0.0, 0.0])
        self.batch_calls = 0

    def embed(self, text):
        return list(self._vector)

    def embed_batch(self, texts):
        self.batch_calls += 1
        return [list(self._vector) for _ in texts]

    @property
    def dimension(self):
        return self._dimension


def test_scoped_query_never_reads_the_whole_corpus():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        wanted = storage.add_document("Wanted", "w.pdf")
        other = storage.add_document("Other", "o.pdf")
        storage.add_chunks([fake_chunk(f"o{i}", other, i, "unrelated filler" * 5) for i in range(50)])
        storage.add_chunks([fake_chunk("w1", wanted, 0, "photosynthesis converts sunlight")])

        def forbidden():
            raise AssertionError("retrieval must not load every document")

        storage.get_all_chunks = forbidden
        storage.get_all_embeddings = forbidden

        results = RetrievalService(storage=storage).retrieve(
            "sunlight", top_k=3, document_id=wanted
        )
        assert [chunk["id"] for chunk in results] == ["w1"]
    finally:
        os.unlink(db_path)


def test_chunks_without_vectors_get_backfilled_and_persisted():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("Mixed", "m.pdf")
        storage.add_chunks([
            fake_chunk("has-vector", doc_id, 0, "mitosis divides cells"),
            fake_chunk("missing-vector", doc_id, 1, "cell division produces daughter cells"),
        ])
        import struct
        storage.add_embeddings([("has-vector", struct.pack("4f", 1.0, 0.0, 0.0, 0.0))])

        service = RetrievalService(storage=storage, embedding_provider=_FakeProvider())
        results = service.retrieve("cell division", top_k=2, document_id=doc_id)
        assert results, "partially indexed documents must still rank"

        stored = dict(storage.get_embeddings_for_document(doc_id))
        assert "missing-vector" in stored, "the missing vector should be persisted"
        assert service.embedding_provider.batch_calls == 1
    finally:
        os.unlink(db_path)


def test_vector_scores_match_pure_python_cosine():
    import math
    import struct

    from Nova.backend.retrieval.embeddings import cosine_similarity

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        storage = Storage(db_path=db_path)
        doc_id = storage.add_document("Vectors", "v.pdf")
        vectors = {
            "c1": [0.2, 0.9, 0.1, 0.0],
            "c2": [0.8, 0.1, 0.3, 0.4],
            "c3": [0.0, 0.0, 1.0, 0.0],
        }
        storage.add_chunks([fake_chunk(cid, doc_id, i, f"chunk {cid}") for i, cid in enumerate(vectors)])
        storage.add_embeddings([(cid, struct.pack("4f", *vec)) for cid, vec in vectors.items()])

        service = RetrievalService(storage=storage)
        query = [0.5, 0.5, 0.25, 0.0]
        pairs = [(cid, struct.pack("4f", *vec)) for cid, vec in vectors.items()]
        scored = service._score_stored_vectors(query, pairs)
        for cid, vec in vectors.items():
            expected = cosine_similarity(query, vec)
            assert math.isclose(scored[cid], expected, abs_tol=1e-5), cid
    finally:
        os.unlink(db_path)


def fake_chunk(chunk_id: str, document_id: str, position: int, content: str) -> dict:
    return {
        "id": chunk_id,
        "document_id": document_id,
        "position": position,
        "content": content,
        "metadata": {},
    }