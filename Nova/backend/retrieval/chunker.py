"""Document chunker — structure-based chunking with stable content-hash IDs.

Pipeline: extract text → clean artifacts → split by heading → window long
sections → assign stable, document-scoped chunk IDs.

Chunk identity is ``sha256(document_id:position:content)``.  It is stable for a
given document and position, and *never* shared between documents: the previous
content-only hash made two PDFs that share a cover page collapse into a single
row because storage upserts on the primary key.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Markdown heading: "# Title", "### Title ###"
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Legacy/decorated heading used by some exports: "___Section___"
_UNDERLINE_HEADING_RE = re.compile(r"^___(.+?)___$")
# Uppercase heading line: "INTRODUCTION", "CHAPTER 2 THE WATER CYCLE", "UNIT 1: BASICS".
# Digits and separators are required for real lecture PDFs, which are usually
# titled "CHAPTER 1 ..." rather than with a bare word.
_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s\-–—:/&,.()'\"]+$")
# Numbered section heading: "3 Cell Structure", "3.2 Membranes", "4) Summary".
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+){0,3}[.)]?\s+\S.{0,70}$")
# Headings are short lines; a long all-caps paragraph is body text.
_MAX_HEADING_CHARS = 80
_MAX_HEADING_WORDS = 14
_HEADING_PREFIX = "## "


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file.

    Tries pypdf first, falls back to pdfplumber.
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            logger.info("Extracted text via pypdf (%d chars)", len(text))
            return text
    except Exception as e:
        logger.warning("pypdf failed: %s", e)

    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        logger.info("Extracted text via pdfplumber (%d chars)", len(text))
        return text
    except Exception as e:
        logger.warning("pdfplumber also failed: %s", e)
        raise ValueError(f"Could not extract text from {path}")


def clean_text(text: str) -> str:
    """Clean extraction artifacts: page numbers and repeated headers/footers."""
    # Remove standalone page numbers (a line that only contains a number,
    # surrounded by line breaks so real answers like "42" inside a sentence
    # and the first/last line of the document are preserved).
    text = re.sub(r'\n\d+\n', '\n', text)

    lines = text.split('\n')
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1

    # A line repeated three or more times in a document is a running
    # header/footer.  Only short, letter-bearing lines qualify so repeated
    # prose or table rows are not destroyed.
    repeated = {
        line for line, count in counts.items()
        if count >= 3
        and len(line) <= _MAX_HEADING_CHARS
        and any(char.isalpha() for char in line)
    }

    emitted: set[str] = set()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in repeated:
            if stripped in emitted:
                continue
            emitted.add(stripped)
        cleaned.append(line)
    return '\n'.join(cleaned)


def _heading_title(line: str) -> Optional[str]:
    """Return the heading text for a heading line, or ``None``.

    The heading *text* is what matters: the previous implementation split on
    the ``#`` marker alone, so the title became body text and the stored
    heading was ``'#'``.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return None

    markdown = _MARKDOWN_HEADING_RE.match(stripped)
    if markdown:
        return markdown.group(2).strip()

    underlined = _UNDERLINE_HEADING_RE.match(stripped)
    if underlined:
        return underlined.group(1).strip()

    words = stripped.split()
    if len(words) > _MAX_HEADING_WORDS:
        return None

    if _CAPS_HEADING_RE.match(stripped) and any(char.isalpha() for char in stripped):
        return stripped

    # A numbered section line is a heading; a numbered *sentence* is not.
    if _NUMBERED_HEADING_RE.match(stripped) and stripped.rstrip().endswith((".", "?", "!")) is False:
        return stripped
    return None


def _split_sections(text: str) -> list[dict]:
    """Split cleaned text into heading-anchored sections, dropping nothing."""
    sections: list[dict] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            sections.append({"heading": heading, "content": content})
        elif heading:
            # A heading with no body (for example the last line of a file)
            # must still reach the index instead of being dropped.
            sections.append({"heading": "", "content": heading})

    for line in text.split("\n"):
        title = _heading_title(line)
        if title is not None:
            flush()
            heading = title
            buffer.clear()
        else:
            buffer.append(line)
    flush()
    return sections


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph into pieces of at most ``max_chars``."""
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = -1
        for pattern in ("\n", ". ", "; ", ", "):
            index = window.rfind(pattern)
            if index > max_chars * 0.6:
                cut = index + len(pattern)
                break
        if cut <= 0:
            index = window.rfind(" ")
            cut = index + 1 if index > max_chars * 0.6 else max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append(remaining)
    return [piece for piece in pieces if piece]


def _iter_paragraphs(content: str, max_chars: int) -> Iterator[str]:
    """Yield paragraphs, hard-splitting any paragraph larger than a chunk."""
    for paragraph in content.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            yield paragraph
        else:
            yield from _hard_split(paragraph, max_chars)


def _window_section(content: str, max_chars: int, overlap: int) -> list[str]:
    """Window a long section body into overlap-joined pieces.

    The overlap carries the *tail of the already-flushed window* into the next
    window.  Overlap is only carried when the flushed window is longer than the
    overlap itself, which keeps short-paragraph documents from ballooning.
    """
    windows: list[str] = []
    current = ""
    for paragraph in _iter_paragraphs(content, max_chars):
        if current and len(current) + len(paragraph) + 2 > max_chars:
            windows.append(current.strip())
            current = current[-overlap:].strip() if len(current) > overlap else ""
        current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current.strip():
        windows.append(current.strip())
    return windows


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 120) -> list[dict]:
    """Split text into chunks based on structure (headings/paragraphs).

    Args:
        text: The cleaned text to chunk.
        max_chars: Maximum characters per chunk body.
        overlap: Character overlap between consecutive windows of one section.

    Returns:
        List of dicts with keys: id, position, content, metadata.
        ``id`` is a content hash; :func:`process_document` additionally scopes
        it to the document and position before storage.
    """
    max_chars = max(160, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))

    sections = _split_sections(text)
    if not sections and text.strip():
        sections.append({"heading": "", "content": text.strip()})

    chunks: list[dict] = []
    position = 0

    for section in sections:
        heading = section["heading"]
        prefix = f"{_HEADING_PREFIX}{heading}\n\n" if heading else ""
        bodies = (
            [section["content"]]
            if len(section["content"]) <= max_chars
            else _window_section(section["content"], max_chars, overlap)
        )
        for body in bodies:
            content = f"{prefix}{body}".strip()
            if not content:
                continue
            chunks.append({
                "id": _content_hash(content),
                "position": position,
                "content": content,
                "metadata": {
                    "heading": heading,
                    "section": len(bodies) == 1,
                    "title": heading,
                },
            })
            position += 1

    logger.info("Text chunked into %d chunks", len(chunks))
    return chunks


def process_document(
    path: str,
    document_id: str,
    max_chars: int = 1400,
    overlap: int = 120,
) -> list[dict]:
    """Full document processing pipeline: extract → clean → chunk → scope IDs.

    Returns chunks ready for storage with ``document_id`` and a
    document-scoped unique ``id`` set.  Two documents may contain identical
    text without their chunks colliding.
    """
    text = extract_text_from_pdf(path)
    text = clean_text(text)
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

    for chunk in chunks:
        chunk["document_id"] = document_id
        chunk["id"] = _chunk_id(document_id, chunk["position"], chunk["content"])

    return chunks


def _content_hash(text: str) -> str:
    """Generate a stable content-hash ID for a chunk body."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _chunk_id(document_id: str, position: int, content: str) -> str:
    """Generate a document-scoped, position-stable chunk identity."""
    payload = f"{document_id}:{int(position)}:{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
