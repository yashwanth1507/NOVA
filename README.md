# Nova

Nova is an offline-first AI study companion for students who need useful
learning support even when connectivity is unreliable. The target is a **Windows desktop application**: a local Gemma 3 1B tutor that
reads a PDF, retrieves only the relevant evidence, answers questions, creates
quizzes, and persists the learner session in SQLite.

## Overview

- **Production target:** Desktop
- **Default mode:** Offline local inference with Gemma 3 1B
- **Optional mode:** Connected OpenAI-compatible inference when explicitly configured by the user

## What the demo proves

1. Launch Nova and create a study session.
2. Upload a text-based PDF.
3. Extract and chunk the document locally (headings become sections).
4. Build a local retrieval index with MiniLM embeddings when available.
5. Ask a grounded question and receive an answer from the selected evidence.
6. Pick any section of a document and generate a quiz from it.
7. Answer quiz questions and persist the result.
8. Restart Nova and restore the session, messages, and quiz history.

Nova never sends the whole PDF to the tutor prompt. Retrieval combines exact
keyword coverage with semantic similarity, filters weak matches, removes
near-duplicate chunks, and forwards only the strongest evidence that fits the
2,048-token local context window.

## Features

- Offline Gemma 3 1B GGUF inference through `llama-cpp-python`
- PDF text extraction with deterministic structure-aware chunking
- Hybrid retrieval: local MiniLM embeddings plus lexical fallback
- Relevance filtering and diversity selection for grounded context
- Automatic prompt compaction for small context windows
- Persistent sessions, messages, documents, chunks, embeddings, quizzes, and
  answers in SQLite
- Text-first desktop UI with loading, error, and recovery states
- Optional OpenAI-compatible online engine behind the same tutor interface

## Architecture

```text
Desktop Web UI (pywebview)
          |
       FastAPI
          |
   ConversationService
      /          \
 RetrievalService  EngineManager
      |             |
 SQLite index   Local Gemma 3 1B
      |
 MiniLM vectors + keyword fallback
```

The provider boundary is `Engine`; the learning workflow lives in services;
and persistence is handled by `Storage`.

## Installation

Requirements:

- Windows 10/11
- Python 3.10+
- `llama-cpp-python` compatible with the local Python installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The repository `config.json` selects the local model path and stores SQLite data
in `nova_data/nova.db`. If the GGUF is missing, open Settings and click
**Install model**. Nova downloads the pinned Gemma 3 1B Q4_K_M file into the
local `models/` directory, validates the GGUF header, and reuses it offline on
future launches.

The optional `all-MiniLM-L6-v2` embedding model is loaded from the local
Hugging Face cache only, so retrieval never blocks a lesson on the network.
When it is missing Nova uses its keyword retriever, reports that state in
**Settings → Semantic Search**, and offers an explicit **Install embeddings**
action. After that install, sections indexed earlier get their vectors
backfilled on the next question — no re-upload needed.

## Run locally

From the repository root:

```powershell
python main.py
```

For browser-based development instead of the pywebview window:

```powershell
python main.py --server
```

Then open <http://127.0.0.1:8765/>.

## Project structure

```text
Nova/
  backend/                 FastAPI app, engines, retrieval, services, storage
  frontend/desktop/       Text-first desktop HTML/CSS/JavaScript UI
  tests/                  Unit and API-level tests
models/                   Local GGUF model files (ignored by Git)
nova_data/                Runtime SQLite data (ignored by Git)
config.json               Local runtime configuration
```

## Verification

```powershell
python -m pytest -q
python -m compileall -q Nova main.py
node --check Nova/frontend/desktop/static/app.js
```

The test suite covers chunking and chunk identity, hybrid retrieval, scoped
storage reads, vector scoring, 2,048-token prompt compaction (chat, quiz and
summary), configuration path handling, local-engine warm-up and token
budgeting, SQLite persistence, and the API chat/quiz/summary flow. The local
model is intentionally not loaded during unit tests.

## Performance notes (measured on an 8-core CPU laptop)

| Change | Before | After |
|---|---|---|
| Query scoped to one document (3,000-chunk library) | 16.6 ms | 0.9 ms |
| Query across the whole library (3,000 vectors) | 547 ms | 147 ms |
| Cold-start first token | model loaded on first question | pre-loaded at startup |
| Thread override `n_threads=cpu_count()` | 134.9 tok/s (default) | 0.7 tok/s — do not force it |

The last row is why `n_threads` defaults to `0` (let llama.cpp decide) and is
only changed explicitly through settings or `NOVA_N_THREADS`.

## Known limitations

- Chat grounding uses the active document; clear the document to search across
  every uploaded PDF.
- Scanned or image-only PDFs need OCR, which is not included.
- First-time local model loading depends on CPU and available RAM (Nova
  pre-loads it at startup so the first question is not delayed).
- MiniLM embeddings require a local model cache; keyword retrieval remains
  available without it (install them from Settings for semantic search).

