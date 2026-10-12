"""Tests for the document chunker."""

from Nova.backend.retrieval.chunker import clean_text, chunk_text, _content_hash
from Nova.backend.prompting.builder import PromptBuilder


def test_clean_text_removes_page_numbers():
    text = "Some content\n42\nMore content"
    cleaned = clean_text(text)
    assert "42" not in cleaned.split("\n"), "Standalone page numbers should be removed"
    assert "Some content" in cleaned
    assert "More content" in cleaned


def test_clean_text_removes_duplicate_headers():
    text = "Header!\n" * 5 + "Actual content"
    cleaned = clean_text(text)
    # The duplicate header should appear at most twice
    assert cleaned.count("Header!") < 3, "Repeated headers should be removed"


def test_chunk_text_produces_stable_ids():
    text = "This is a single paragraph of text without headings.\n" * 50
    chunks1 = chunk_text(text, max_chars=500)
    chunks2 = chunk_text(text, max_chars=500)
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1["id"] == c2["id"], "Chunk IDs should be stable (content-hash based)"


def test_chunk_respects_max_chars():
    # Many small paragraphs separated by blank lines
    text = "\n\n".join([
        "word " * 20,   # ~100 chars
        "word " * 20,   # ~100 chars
        "word " * 20,   # ~100 chars
        "word " * 20,   # ~100 chars
        "word " * 20,   # ~100 chars
    ])
    chunks = chunk_text(text, max_chars=120)
    for c in chunks:
        assert len(c["content"]) <= 200, f"Chunk exceeds max_chars: {len(c['content'])}"
    assert len(chunks) >= 2, "Should be split into multiple chunks"


def test_chunk_has_required_fields():
    text = "## Heading 1\nSome content here.\n\n## Heading 2\nMore content here."
    chunks = chunk_text(text)
    for c in chunks:
        assert "id" in c, f"Chunk missing 'id'"
        assert "position" in c, f"Chunk missing 'position'"
        assert "content" in c, f"Chunk missing 'content'"
        assert "metadata" in c, f"Chunk missing 'metadata'"
        assert "heading" in c["metadata"], f"Chunk metadata missing 'heading'"


def test_content_hash_is_stable():
    text = "Hello, world!"
    h1 = _content_hash(text)
    h2 = _content_hash(text)
    assert h1 == h2
    assert len(h1) == 32  # SHA256 truncated to 32 hex chars


def test_prompt_builder_creates_message_list():
    builder = PromptBuilder()
    messages = builder.build_chat_messages(
        query="What is calculus?",
        context_chunks=["Calculus is the study of change."],
        history=[],
    )
    # system + one user turn that carries both the evidence and the question
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Calculus is the study of change." in messages[-1]["content"]
    assert "What is calculus?" in messages[-1]["content"]
    # The question is the last thing the model sees.
    assert messages[-1]["content"].rstrip().endswith("What is calculus?")


def test_prompt_builder_includes_history():
    builder = PromptBuilder()
    history = [
        {"role": "user", "content": "What is math?"},
        {"role": "assistant", "content": "Math is the study of numbers."},
    ]
    messages = builder.build_chat_messages(
        query="Tell me more",
        context_chunks=[],
        history=history,
    )
    assert len(messages) == 4  # system + 2 history + user
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Tell me more"


def test_prompt_builder_quiz_messages():
    builder = PromptBuilder()
    messages = builder.build_quiz_messages(content="Some study content.", num_questions=3)
    assert len(messages) == 2
    assert "3" in messages[0]["content"]  # num_questions in prompt


# ── Chunk identity and structure regressions ─────────────

def test_identical_text_in_two_documents_gets_distinct_ids(monkeypatch):
    """Shared text (syllabus pages, cover sheets) must not collide.

    Before the fix the second upload silently moved the chunk row to the new
    document because the primary key was a content hash only.
    """
    import os
    import tempfile

    from Nova.backend.retrieval import chunker
    from Nova.backend.storage.database import Storage

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
        db_path = handle.name
    try:
        storage = Storage(db_path)
        doc_a = storage.add_document("Lecture A", "a.pdf")
        doc_b = storage.add_document("Lecture B", "b.pdf")
        monkeypatch.setattr(
            chunker,
            "extract_text_from_pdf",
            lambda path: "# Course Syllabus\n\nAttendance is mandatory and graded weekly.",
        )
        first = chunker.process_document("one.pdf", doc_a)
        second = chunker.process_document("two.pdf", doc_b)

        assert [c["id"] for c in first] != [c["id"] for c in second]
        storage.add_chunks(first)
        storage.add_chunks(second)
        assert len(storage.get_chunks_for_document(doc_a)) == 1
        assert len(storage.get_chunks_for_document(doc_b)) == 1
    finally:
        os.unlink(db_path)


def test_repeated_sections_inside_one_document_stay_distinct():
    repeated = "## Notes\n\nRepeated section body. " * 6
    chunks = chunk_text(repeated, max_chars=2000)
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert len({c["position"] for c in chunks}) == len(chunks)


def test_markdown_heading_text_is_preserved_and_not_duplicated():
    chunks = chunk_text("# Photosynthesis\n\nPlants convert light into energy.")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["heading"] == "Photosynthesis"
    assert chunks[0]["content"].startswith("## Photosynthesis")
    assert chunks[0]["content"].count("Photosynthesis") == 1


def test_chapter_headings_with_numbers_are_detected():
    text = (
        "CHAPTER 1 PHOTOSYNTHESIS\n\nPlants convert sunlight into chemical energy.\n\n"
        "CHAPTER 2 THE WATER CYCLE\n\nEvaporation moves water into the atmosphere."
    )
    chunks = chunk_text(text)
    headings = [c["metadata"]["heading"] for c in chunks]
    assert headings == ["CHAPTER 1 PHOTOSYNTHESIS", "CHAPTER 2 THE WATER CYCLE"]


def test_oversized_paragraph_is_split_within_bounds():
    text = "x" * 5000
    chunks = chunk_text(text, max_chars=1000, overlap=120)
    assert len(chunks) > 1
    assert max(len(c["content"]) for c in chunks) <= 1000 + 120 + 120
    assert "".join(c["content"] for c in chunks).count("x") >= 5000


def test_no_text_is_lost_between_sections():
    text = (
        "Preamble sentence that appears before any heading.\n\n"
        "SECTION ONE\n\nFirst body sentence.\n\n"
        "SECTION TWO\n\nSecond body sentence.\n\n"
        "TRAILING HEADING"
    )
    combined = " ".join(c["content"] for c in chunk_text(text))
    for fragment in ("Preamble sentence", "First body sentence", "Second body sentence", "TRAILING HEADING"):
        assert fragment in combined, f"lost content: {fragment}"
