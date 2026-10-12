# Nova — Audit & Fixing Report

> **Scope:** chunking, data loading/persistence, model optimization, retrieval performance, API and frontend defects.
> **Method:** static audit of every backend module + measured reproductions (Python 3.14, `llama-cpp-python` 0.3.35, `sentence-transformers` 6.0.0, 8-core macOS host).
> **Baseline:** `python -m pytest -q` → **24 passed**. Every defect below is *outside* the current test suite, which is why they were not caught.

Legend: 🔴 critical (data loss / broken feature) · 🟠 high (wrong behaviour or large cost) · 🟡 medium (robustness/UX) · 🔵 docs/build

---

## 1. Findings (measured)

### CHUNKS

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| 🔴 **C1** | **Chunk IDs are content-only, so identical text collapses across documents.** `_content_hash(body)` ignores `document_id` and `position`; storage uses `INSERT OR REPLACE`. Uploading a second PDF that shares a syllabus/cover page **moves** the chunk row to the new document (PK delete+insert). | Repro: doc1 has 1 chunk → after uploading the same text as doc2, `get_chunks_for_document(doc1)` returns **0**, doc2 returns 1. Repeated sections inside one PDF collapse too (2 chunks produced → 1 unique id). | Silent chunk loss = RAG answers lose evidence. Data corruption. |
| 🔴 **C2** | **Markdown headings are mangled and swallowed into the body.** `re.split(heading_pattern, text)` captures only the `#` marker, so the heading *text* becomes body content and the prefix is doubled. | Repro with `"# Photosynthesis\n\n…"` → `metadata.heading='#'`, `content` starts `'## #\n\nPhotosynthesis\n\n…'`. | Section-aware retrieval/heading bonus can never match; chunks read wrong to the model. |
| 🟠 **C3** | **Long-section overlap is applied to the wrong side of the seam and is not de-duplicated.** The tail of the flushed chunk is prepended to the next chunk, but the flushed chunk keeps the *old* tail as well (tail text is duplicated and never trimmed from the previous chunk). | Code path in `chunk_text()` long-section branch. | Duplicate evidence inflates the prompt and triggers MMR rejection of legitimate chunks. |
| 🟡 **C4** | `clean_text()` removes any standalone `\n<number>\n` (also legitimate data lines) and drops repeated lines globally after 3 occurrences (drops real repeated content, e.g. a table header repeated on purpose). | Code path in `clean_text()`. | Minor over-cleaning; documented behaviour, fix is conservative. |

### LOADING / DATA

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| 🔴 **L1** | **`GET /api/documents/{id}/summary` returns HTTP 500 for every document.** `routes.py` uses `PromptBuilder()` but never imports it. | Repro against a router built exactly like the test suite does: `status: 500`. | The "Document summarization" feature advertised at 100% is completely broken. |
| 🟠 **L2** | **Relative `sqlite_path` is resolved against the process CWD, not the app root.** Only frozen builds are resolved. | Running `Settings.load()` from another directory prints `resolved sqlite_path: nova_data/nova.db` while `runtime_root()` is the repo root. | Launching from another folder creates a *second empty database* — the user's documents/sessions appear lost. |
| 🟠 **L3** | **Retrieval loads the entire corpus on every query.** `get_all_chunks()` (full content) + `get_all_embeddings()` (all vectors) are fetched, then filtered in Python even when `document_id` is set. | 3000-chunk DB, query scoped to a **1-chunk** document: **16.6 ms** and 3000 chunks JSON-decoded per query. | Latency and memory grow with the whole library, not with the active document. |
| 🟠 **L4** | **`/api/documents` loads every chunk of every document** (N+1 queries, full text) only to return 200-char previews. | `routes.list_documents()` loop. | Large JSON + slow list view with many documents. |
| 🟠 **L5** | **Blocking work runs on the async event loop.** `/chat`, `/quizzes/generate`, summary, upload (PDF parse + embedding) are `async def` but call blocking code. | Code path. | The whole API (status polling, settings, health) freezes for the 5–25 s of a local generation. |
| 🟡 **L6** | SQLite opens in default journal mode with no `busy_timeout`; message sequence numbers are computed with `MAX()+1` in a separate statement. | Code path. | `database is locked` under concurrent requests; duplicate sequence numbers possible. |
| 🟡 **L7** | Summary uses a hard-coded 2048 window and only the **first 6 chunks**; `build_summary_messages` is called without `context_size`. | Code path + 1511-token prompt measured. | Summaries miss the rest of the document and ignore the user's context setting. |

### MODEL / INFERENCE

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| 🟠 **M1** | **Startup "pre-load" never loads the model.** `create_app()` startup calls `engine_manager._get_local()`, which only *constructs* `LocalEngine`; `_lazy_load()` is never triggered. | `inspect.getsource` check → `True`; log line "Local model loaded and ready" is unconditional. | First chat pays the full cold-load (0.95 s warm-disk, 11.7 s cold-disk for the 270M/1B here) — the exact delay the feature claims to remove. |
| 🟠 **M2** | **Local model load is not thread-safe.** `_lazy_load()` has no lock; two concurrent requests can both build a `Llama` object. | Code path. | Two models (≈1 GB + 2× KV) resident; RAM blow-up / crash on low-memory machines. |
| 🟠 **M3** | **Quiz prompt budget ignores the output reservation**, so JSON is truncated. | `build_quiz_messages(..., num_questions=5, context_window=2048)` → prompt **≈1686 tokens**; `max_tokens=1536` requested → `_fit_generation_tokens` clamps to **362** tokens of output. | Quiz generation frequently hits the early `_parse_text_quiz` fallback or fails. |
| 🟡 **M4** | Prompt is tokenized once for budgeting and again inside `create_completion`. | Code path (`_fit_generation_tokens` → `create_completion(prompt=str)`). | One redundant full tokenization per request (~ms, free win). |
| 🟡 **M5** | No configurable thread count; library default `cpu_count()//2` applies silently. | See §2 measurement — tuning must be opt-in, **not** a hardcoded `cpu_count()`. | Preserves the good default while allowing tuning on other hardware. |

### RETRIEVAL

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| 🟠 **R1** | Cosine scoring is pure Python over every stored vector. | 3000 embedded chunks, one query: **547 ms**. | 0.5 s added to every local answer before the model even starts. |
| 🟠 **R2** | Chunks whose embedding row is missing get **no** semantic score (upload happened while the embedder was unavailable), so they can only win through the 0.28 lexical term. | Code path `_semantic_scores` (`continue` on missing blob). | Partially indexed documents silently degrade. |
| 🟠 **R3** | **Hybrid retrieval silently degrades to lexical-only and the UI never says so.** MiniLM is loaded `local_files_only=True`; if it is not in the HF cache, semantic retrieval is disabled forever with only a log line. | Measured warning during audit: *"couldn't find them in the cached files"*. | The headline "hybrid RAG" feature is inactive on a clean offline machine, invisibly. |

### API / FRONTEND / OTHER

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| 🟠 **F1** | **Quizzes are always generated from the first chunk of the document** (`doc.chunks[0]`). No chunk picker exists although REPORT.md claims a "chunk viewer". | `grep activeChunkId` → set only from `doc.chunks?.[0]?.id`. | Cover pages / short first chunks produce useless quizzes. |
| 🟠 **F2** | SSE server errors are swallowed: `throw new Error(data.error)` sits inside the same `try/catch` that logs parse errors. | `app.js` stream loop. | Server-side failures show as "I could not find an answer…" with no reason. |
| 🟡 **F3** | Every streamed token re-renders full Markdown into the DOM. | `updateMessageContent()` per token. | O(n²) DOM work, visible jank on long answers. |
| 🟡 **F4** | `POST /api/quizzes/answer` requires a client-supplied `correct_index` that the server then ignores (frontend sends a dummy `0`). | `QuizAnswerRequest` + `answer_question()`. | Confusing API; a client omitting it gets 422. |
| 🟡 **F5** | CORS is `allow_origins=["*"]` **with** `allow_credentials=True`. | `main.py`. | Any web page can drive the local API; credentials must not be combined with `*`. |
| 🟡 **F6** | Deprecated `@app.on_event("startup")`. | `main.py`. | Deprecation warnings; removal in future FastAPI. |
| 🔵 **D1** | `.env.template` documents `LUMAS_*` variables; the code reads `NOVA_*` (and the file has a `[TEMPLATE]` header). | `.env.template` vs `config.py`. | Environment overrides silently never apply. |
| 🔵 **D2** | `REPORT.md` / `README.md` claim a chunk viewer, always-on hybrid retrieval, and 100% feature completeness that the code does not have. | Compare claims to code. | Misleading status → this audit exists because of it. |
| 🔵 **D3** | `nova.spec` never uses `collect_data_files`/`collect_submodules` it imports and omits embedding-stack hidden imports (`sentence_transformers`, `transformers`, `tokenizers`, `numpy`). | `nova.spec`. | Frozen build may fall back to lexical retrieval / fail import. |
| 🔵 **D4** | **The CI workflow cannot pass from the repository root**, where the project's files do not exist: `pip install -r requirements.txt`, `compileall Nova main.py` and `pytest` all target missing paths. | `.github/workflows/ci.yml` vs the `Nova/` layout. | A red or silently wrong CI signal on every push. |

---

## 2. Measurement: thread tuning is a trap (do not hardcode `cpu_count()`)

`n_ctx=2048`, Gemma 3 270M Q4_K_M, 3 warm generations, median:

| `n_threads` / `n_threads_batch` | prefill tok/s | generation tok/s |
|---|---|---|
| library default (`cpu_count()//2` = 4) | 51 189 | **134.9** |
| `8` (= `cpu_count()`) | 304 | **0.7** |
| `4` | 49 284 | **137.4** |

Hardcoding `n_threads = cpu_count()` would have made generation **~190× slower** on this host (oversubscription of 4P+4E cores). Fix = keep the library default and expose an *optional* override. (Prefill figures above reuse the KV prefix, which is exactly why the current "system prompt first" ordering is good and must be preserved.)

---

## 3. Fix plan (order of execution)

| # | Work item | Findings | Verification |
|---|---|---|---|
| 1 | Chunk identity: hash `document_id + position + heading + body`; clean heading capture; trim overlap seam; keep `_content_hash` stable API | C1, C2, C3 | New tests: cross-document uniqueness, heading text preserved, no duplicated tail, existing 6 chunker tests |
| 2 | Storage: WAL + `busy_timeout` + `synchronous=NORMAL`; atomic sequence insert; scoped `get_chunks_for_documents` / `get_embeddings_for_documents`; SQL-side chunk previews (`length(content)`, 200-char preview) | L3, L4, L6 | New tests: scoped reads, no full-table access, preview shapes, existing storage tests |
| 3 | Retrieval: scoped queries, numpy cosine fast path with pure-Python fallback, dimension guard, backfill missing vectors for the scoped set | L3, R1, R2 | Benchmark 3000 chunks before/after; new test that a partially embedded doc still ranks semantically |
| 4 | Model: real warm-up (`warm_up()` → `_lazy_load()`), thread-safe load lock, optional `n_threads` setting, reuse of tokenized prompt, `is_loaded` health | M1, M2, M4, M5 | New test proving warm-up loads before the first request; guarded `Llama` load test |
| 5 | Prompting/quiz: reserve output tokens per question in `build_quiz_messages`, derive quiz `max_tokens`, pass `context_size` into summary, sample chunks across the document | M3, L7 | Test: quiz prompt + requested output ≤ context window for 1/3/5/10 questions |
| 6 | API: import `PromptBuilder` (fixes summary 500), sync handlers for blocking work, `run_in_threadpool` for upload, embedding status/download endpoints, optional `correct_index`, CORS hardening, lifespan | L1, L5, R3, F4, F5, F6 | New tests: summary 200, health exposes embedding state, answer without `correct_index` works, existing API tests |
| 7 | Frontend: chunk picker (chunk list endpoint + default = longest chunk), SSE error propagation, rAF render throttle, embedding status hint | F1, F2, F3, R3 | `node --check` + manual DOM/flow verification through the API |
| 8 | Docs/build: `.env.template` → `NOVA_*`, README/REPORT corrections, `nova.spec` hidden imports, CI working directory, this report's status table | D1, D2, D3, D4 | `python -m compileall`, `node --check`, CI commands re-run from `Nova/` |
| 9 | Regression suite + full verification | all | `pytest -q`, `compileall`, `node --check`, benchmark re-run |

**Non-goals (explicitly unchanged):** chunk-size defaults, MMR weights, prompt wording, model URLs, UI redesign. Thread count stays at the library default unless configured.

---

## 4. Status

| # | Item | Status |
|---|---|---|
| 1 | Chunk pipeline | ✅ fixed |
| 2 | Storage / loading | ✅ fixed |
| 3 | Retrieval performance | ✅ fixed |
| 4 | Local model optimization | ✅ fixed |
| 5 | Prompting / quiz budget | ✅ fixed |
| 6 | API layer | ✅ fixed |
| 7 | Frontend | ✅ fixed |
| 8 | Docs / build | ✅ fixed |
| 9 | Tests & verification | ✅ done — 58 tests pass |

Final measured results are recorded in §5.

---

## 5. Verification log (after fixes)

### 5.1 Automated suite

```
python -m pytest -q            -> 58 passed (baseline: 24 passed)
python -m compileall -q Nova main.py  -> OK
node --check Nova/frontend/desktop/static/app.js -> OK
```

New coverage (34 tests) targets each finding: chunk identity across documents, heading
preservation, oversized-paragraph bounds, text-loss checks, scoped storage reads, SQL
previews, cascading deletes, concurrent message sequencing, scoped retrieval without a
full-corpus read, vector backfill, vectorized-vs-python cosine parity, quiz/summary token
budgets, config path resolution and persistence, engine warm-up/threading/token clamp, and
the document/summary/quiz/health endpoints.

### 5.2 Measured performance (8-core CPU host)

| Workload | Before | After |
|---|---|---|
| Query scoped to a 1-chunk document inside a 3,000-chunk library | 16.6 ms | **0.9 ms** |
| Query over the whole library (3,000 chunks) | 547 ms | **139.8 ms** |
| Cosine scoring of 3,000 × 384 floats | 107.1 ms (pure Python) | **3.8 ms** (vectorized, max delta 1.2e-7) |
| Starting the app with the model present | first question paid the cold load | pre-loaded at startup (`model_loaded: true` in `/health`) |
| `n_threads=cpu_count()` (rejected) | 134.9 tok/s default | 0.7 tok/s — regression, kept opt-in |

### 5.3 End-to-end run against a live server (real Gemma 3 GGUF, real PDF)

```
1. upload PDF (4-page lecture notes)      -> 200  chunks=4   [was 2 chunks with mangled headings]
2. list documents                         -> 200  chunk_count=4  preview=200 chars  heading='CHAPTER 1 PHOTOSYNTHESIS'
3. chunk picker                           -> 200  4 labelled sections with char counts
4. create session                         -> 200
5. document summary                       -> 200  [was HTTP 500]
6. goal-grounded chat (Gemma 3 1B)        -> 200  "Condensation forms clouds when rising vapour cools and returns to liquid."
7. quiz from CHAPTER 2 (not chunk 0)      -> 200  4 questions, correct indices spread 0/1/2/3
8. answer without legacy correct_index    -> 200  correct=True
9. quiz results                           -> 200  score persisted
10. embeddings probe                      -> missing  available=false  (honest state, not a silent skip)
11. install embeddings                    -> 202 downloading -> ready in 15 s
12. chat after install                    -> 200  quoted the source section
13. vectors deleted to emulate a legacy   -> next query backfilled 4 vectors and answered from them
    document
14. streaming chat                        -> 200  SSE tokens delivered
15. persisted messages                    -> user/assistant pairs stored for every turn
```

### 5.4 Remaining, explicitly out of scope

- **OCR for scanned pages** — still backlog (unchanged).
- **Embedding model is not bundled** — by design (offline-first, explicit install).
- **`n_threads`** — left at the llama.cpp default; forcing more threads measured a large
  regression on this host.
- Chunk size (1400 chars), overlap (120), MMR weights and prompt wording were not tuned.
