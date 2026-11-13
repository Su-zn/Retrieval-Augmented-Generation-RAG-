---

### 6. `Memory.md` (Agent State Tracker for OpenCode)

```markdown
# Project Memory & Progress Log

## Current Status
- **Active Phase:** Phase 5 (LLM Tool Calling for Interview Booking)
- **Completed Phases:** Phase 1 (Scaffolding & Infrastructure), Phase 2
  (Database Models & Pydantic Schemas), Phase 3 (Document Ingestion Pipeline),
  Phase 4 (Redis Chat Memory + Custom Multi-Turn RAG Service)
- **Blockers:** None. Phases 3 and 4 are verified end-to-end with **real local
  inference** (Ollama `nomic-embed-text` embeddings + `llama3.2:3b` chat), so
  ingestion and multi-turn chat run without a paid API key. Google Gemini was
  evaluated first, but the available Google Cloud project was not eligible for
  the free tier (HTTP 403), so the project is configured for local Ollama
  through its OpenAI-compatible endpoint.

## Key Architectural Decisions
- **Vector Store:** Qdrant (`AsyncQdrantClient`) using cosine distance and dynamic vector dimension matching the configured embedding model (default `1536` for `text-embedding-3-small`, configurable via `EMBEDDING_DIMENSION` env var so local/free models also work).
- **Relational DB:** SQLAlchemy 2.0 async configured via `DATABASE_URL` (defaults to `sqlite+aiosqlite:///./app.db` for instant local run, and `postgresql+asyncpg://...` inside `docker-compose.yml`).
- **LLM Client:** OpenAI-compatible `AsyncOpenAI` client reading `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`, and `EMBEDDING_DIMENSION` from `.env`, so OpenAI, Gemini, Mistral, Ollama, or any OpenAI-compatible provider works without code changes.
- **Local provider (Ollama):** `.env` points to `http://host.docker.internal:11434/v1` with `LLM_MODEL=llama3.2:3b`, `EMBEDDING_MODEL=nomic-embed-text`, `EMBEDDING_DIMENSION=768`, and a dummy `LLM_API_KEY=ollama`. `host.docker.internal` is required because the API runs inside Docker while Ollama runs on the Windows host; the endpoint was verified reachable from inside the container.
- **Configuration:** one cached `Settings` object (`app/core/config.py`, `pydantic-settings`) shared process-wide; `ENVIRONMENT` accepts `local | development | docker | staging | production | test`. Blank optional env values (`QDRANT_API_KEY`, `LLM_BASE_URL`) are normalised to `None`.
- **Startup resilience:** database failures are fatal (metadata/booking writes are critical); Redis and Qdrant *transport* failures are logged as warnings so `/docs` remains reachable, with capability endpoints responsible for returning structured 503s. A Qdrant collection dimension mismatch still fails fast with an actionable `RuntimeError`.
- **Vector collection bootstrap:** the collection is created automatically on startup with cosine distance and `EMBEDDING_DIMENSION`; existing collections are reused and their dimension validated.
- **Session management:** `async_sessionmaker(expire_on_commit=False)`; `get_db()` yields a session, rolls back on exception, and services own commits. SQLite uses `check_same_thread=False`, plus `StaticPool` for in-memory URLs.
- **Upload limit:** `MAX_UPLOAD_BYTES` (default 10 MiB) lives in settings and is enforced per request in Phase 3 (413).
- **Ingestion pipeline (Phase 3):** extract → chunk (in a worker thread) → batched embeddings → Qdrant upsert → metadata commit. Point IDs are deterministic (`uuid5(NAMESPACE_URL, "rag://documents/{document_id}/{chunk_index}")`), so retries cannot silently duplicate chunks. Payload carries `document_id`, `chunk_index`, `text`, `filename`, and `chunking_strategy` for Phase 4 source attribution. If the DB commit fails after the upsert, the written points are deleted on a best-effort basis.
- **Docker:** `api` waits on healthy `qdrant`, `redis`, and `postgres`; the container runs as a non-root user and exposes a `/health` liveness probe used by the compose healthcheck.
- **ORM modelling (Phase 2):** SQLAlchemy 2.0 `DeclarativeBase` with `Mapped`/`mapped_column`. Primary keys use the portable `sqlalchemy.Uuid` type (`default=uuid.uuid4`) so SQLite and PostgreSQL share one schema. `created_at` is timezone-aware `DateTime(timezone=True)` with `server_default=func.now()`. `InterviewBooking.session_id` is indexed for per-session lookups.
- **API schemas (Phase 2):** Pydantic v2 only, with a shared `ChunkingStrategy = Literal["fixed", "semantic"]`. `ChatRequest` trims whitespace and rejects blank inputs; `InterviewBookingToolArgs` keeps calendar date/time as strings (LLM JSON payloads) but enforces `YYYY-MM-DD` / `HH:MM` patterns plus real-calendar validation; `InterviewBookingRead` uses `from_attributes=True`; `ChatResponse` carries optional `sources` and `booking`.
- **Chat memory (Phase 4):** each session is a Redis list at `chat:history:{session_id}` containing JSON `{"role", "content"}` entries. `MemoryService.append_exchange()` writes the user/assistant pair in one transaction and atomically applies `LTRIM -CHAT_HISTORY_MAX_MESSAGES -1` plus `EXPIRE CHAT_HISTORY_TTL_SECONDS` (defaults 20 messages / 24 h), so history is bounded and self-expiring. Malformed entries are logged and skipped. Redis failures raise `MemoryStoreError`, which the endpoint maps to 503.
- **RAG service (Phase 4):** `RagService.answer()` loads history, rewrites follow-ups into a standalone query when history exists (small local models are unreliable at this, so a reformulation failure logs a warning and falls back to the original question), embeds that query with `EmbeddingService`, and calls Qdrant `query_points(limit=RAG_TOP_K, with_payload=True)`. Hits are mapped to `SourceChunk` with clamped cosine scores and rendered as `[n] filename=..., chunk=..., score=...` labels in the grounded prompt. Empty retrieval returns a deterministic "no relevant information" answer without calling the LLM; the exchange is still stored so the conversation stays coherent. Embedding/LLM failures map to 502, Redis/Qdrant failures to 503.
- **Retrieval configuration:** `RAG_TOP_K` (default 5, max 50) lives in `Settings`; compose forwards `RAG_TOP_K`, `CHAT_HISTORY_TTL_SECONDS`, `CHAT_HISTORY_MAX_MESSAGES`, and `EMBEDDING_BATCH_SIZE` from `.env`.

## Completed Checklist
- [x] Phase 1: Scaffolding, Docker Compose, Core Clients (Qdrant, Redis, SQLAlchemy)
- [x] Phase 2: ORM Models (`DocumentMetadata`, `InterviewBooking`) & Pydantic Schemas
- [x] Phase 3: Document Ingestion API (`/api/v1/ingest`) + Fixed & Semantic Chunkers
- [x] Phase 4: Redis Chat Memory + Custom Multi-Turn RAG Service
- [ ] Phase 5: LLM Tool Calling (`book_interview`) + Booking Persistence
- [ ] Phase 6: End-to-End Verification, Pytest Suite & `README.md`

## Phase 1 — Delivered Artifacts
- `pyproject.toml` (project metadata, pinned deps, ruff/mypy/pytest config),
  `requirements.txt` (runtime, fully pinned), `requirements-dev.txt`.
- `Dockerfile` (python:3.12-slim, non-root `appuser`, uvicorn entrypoint),
  `docker-compose.yml` (qdrant `6333`/`6334`, redis `6379`, postgres `5432`,
  api `8000`, named volumes + healthchecks), `.env.example` (every setting
  documented), `.gitignore`, `.dockerignore`.
- `app/core/config.py`, `app/core/database.py`, `app/core/redis_client.py`,
  `app/core/vector_store.py`.
- `app/main.py` with async `lifespan` startup/shutdown hooks, app factory, and
  `/health`; `app/api/v1/router.py` ready for endpoint registration.
- Package skeleton: `app/`, `core/`, `models/`, `services/`, `api/v1/endpoints/`
  (all `__init__.py` files present, `app.services`/`app.models` documented).
- Verification performed: `ruff check app` clean, `mypy app` clean (13 files),
  TestClient lifespan test starts/stops cleanly and returns HTTP 200 from
  `/health`; settings override test passed (`ENVIRONMENT=docker`, blank
  `QDRANT_API_KEY` -> `None`); compose YAML parses with all four services.
- Extended verification (29 behavioral + 35 infrastructure checks, all passed):
  settings validators, SQLite engine/session commit-rollback paths, Redis
  unreachable-server handling, Qdrant collection create/reuse/dimension-mismatch/
  outage paths via a fake client, `/health` + `/docs` + OpenAPI, compose
  topology/healthchecks/dependency ordering, pinned dependency parity,
  `.env.example` coverage, and a forbidden-library scan (no faiss/chromadb/
  langchain/RetrievalQA references). A live uvicorn smoke test served `/health`
  and `/docs` with HTTP 200.
- Docker verification (completed): `docker compose build` succeeded and
  `docker compose up -d` brought up qdrant, redis, postgres, and api with all
  four reporting **healthy**. Startup logs confirm Postgres connectivity, Redis
  ping, and automatic creation of the `documents` Qdrant collection
  (dimension 1536, cosine). Live checks: `/health` 200 (env=docker), `/docs`
  200, Qdrant `/collections` lists `documents`, `redis-cli ping` -> PONG,
  Postgres 16.16 reachable. The stack is now verified end-to-end; no outstanding
  Docker issues.
- Phase 1 was committed locally as `eaf3e15 "phase 1"` and pushed to
  `origin/main` on GitHub (`Su-zn/Retrieval-Augmented-Generation-RAG-`).
  `app/models/domain.py` and `app/models/schemas.py` were intentionally left
  untracked (Phase 2 work not yet committed).
- Dependency pin correction: `openai==1.59.0` does not exist on PyPI; pinned to
  `openai==1.59.2` (the docs only require the OpenAI-compatible SDK).

## Phase 2 — Delivered Artifacts
- `app/models/domain.py`: `DocumentMetadata` (`id`, `filename`, `content_type`,
  `chunking_strategy`, `chunk_count`, `created_at`) and `InterviewBooking`
  (`id`, `session_id`, `candidate_name`, `candidate_email`, `interview_date`,
  `interview_time`, `created_at`), both on the shared `Base`, with portable
  `Uuid` primary keys, indexed `session_id`, and `created_at` server defaults.
- `app/models/schemas.py`: `ChunkingStrategy`, `IngestionResponse`,
  `ChatRequest`, `SourceChunk`, `InterviewBookingRead`, `ChatResponse`, and
  `InterviewBookingToolArgs` (with `as_booking_values()` for Phase 5).
- `app/core/database.py`: `init_db()` now imports `app.models.domain` so both
  tables are registered on `Base.metadata` before `create_all`.
- Verification performed: `ruff check app` clean; `mypy app` clean (15 files);
  in-memory SQLite round-trip inserted and read back a document + booking
  (UUID parsing, `created_at` defaults, date/time fidelity); Pydantic positive
  and negative cases for email, impossible dates (`2026-02-30`), invalid times
  (`25:00`), blank names, and blank chat fields; TestClient lifespan startup
  still returns HTTP 200 from `/health` with the new models registered.
- Extended verification (25 checks, all passed): table registration, SQLite
  persistence round-trips, UUID defaults, `created_at` population, query by
  `session_id`, `InterviewBookingRead.model_validate` from ORM attributes,
  JSON date/time/email serialisation, `as_booking_values()` parsing, and
  negative cases for bad email/date/time/name plus blank chat inputs.
  In-memory SQLite (`StaticPool`) startup was verified separately.
- PostgreSQL verification (Docker): rebuilt the API image and restarted the
  stack; startup created both tables in Postgres with `uuid` primary keys,
  `varchar` columns, `date`/`time` fields, timezone-aware `created_at`
  defaulting to `now()`, and `ix_interview_bookings_session_id`. The API
  reported healthy and `/health` returned 200.
- `schemas.py` exposes `BookingValues` (`TypedDict`) so
  `as_booking_values()` can be splatted into `InterviewBooking(...)` under
  strict mypy in Phase 5.

## Phase 3 — Delivered Artifacts
- `app/services/extractor.py`: `ExtractedDocument`, `validate_upload()` (suffix
  + MIME allow-list, normalises `charset` parameters, accepts
  `application/octet-stream`), `extract_document()` (UTF-8 BOM-aware TXT
  decoding, `pypdf` parsing offloaded with `asyncio.to_thread`),
  `collapse_internal_whitespace()`, and `ExtractionError` /
  `UnsupportedFileTypeError` / `EmptyDocumentError`.
- `app/services/chunker.py`: `BaseChunker` ABC, `FixedSizeChunker(500, 50)`
  sliding window (validates size/overlap, stops at exact boundaries),
  `SemanticRecursiveChunker(800)` (paragraph → sentence → word-boundary hard
  split, never exceeds `max_chunk_size`), and `build_chunker()` factory.
- `app/services/embedder.py`: `EmbeddingService` issues
  `AsyncOpenAI.embeddings.create` in `EMBEDDING_BATCH_SIZE` (64) batches,
  re-sorts by `index` to preserve order, and rejects vectors whose length is
  not `EMBEDDING_DIMENSION`. The client is created lazily so a missing
  `LLM_API_KEY` becomes a per-request `EmbeddingError` (502) instead of a
  startup crash.
- `app/services/ingestion_service.py`: `ingest_document()` orchestrates
  extract → chunk → embed → upsert → commit and returns `IngestionResult`.
  Sanitises upload names with `Path(...).name` (max 512 chars), raises
  `VectorStoreError` / `MetadataPersistenceError`, and deletes already-upserted
  points if the metadata commit fails.
- `app/api/v1/endpoints/ingest.py`: `POST /api/v1/ingest` multipart upload
  (`file` + `chunking_strategy` form field defaulting to `semantic`), streams
  at most `MAX_UPLOAD_BYTES + 1` bytes (413 on overflow), maps extraction
  failures to 400, embedding failures to 502, Qdrant/DB failures to 503, and
  returns 201 with `IngestionResponse`.
- Wiring: `app/api/v1/router.py` includes the ingest router;
  `app/core/config.py` and `.env.example` add `EMBEDDING_BATCH_SIZE=64`.
- Verification performed: `ruff check app` clean; `mypy app` clean (20 files);
  47 unit checks + 19 HTTP checks (66 total) passed, covering chunker window
  sizes/overlap/config validation, a real `reportlab` PDF, corrupt PDFs,
  BOM/CRLF handling, non-UTF-8 TXT, batching, dimension mismatch, provider
  failure, missing API key, `../../` filename sanitisation, Qdrant-failure
  rollback (no metadata row), 201/400/413/422/502/503 responses, and the
  OpenAPI multipart schema.
- Live Docker end-to-end verification (offline OpenAI-compatible stub serving
  1536-dimensional vectors on the compose network): TXT (`semantic`) and PDF
  (`fixed`) uploads both returned 201; Postgres `document_metadata` and Qdrant
  payloads agreed (2 documents / 2 points); a 4-chunk TXT produced
  `chunk_index` 0–3 and point IDs byte-identical to the `uuid5` formula; the
  `documents` collection reports `size=1536`, `distance=Cosine`. Error paths:
  `.md` 400, MIME mismatch 400, empty upload 400, corrupt PDF 400, invalid
  strategy 422, missing file 422. Test rows/points and the stub container were
  removed (later superseded by the real Ollama run below).
- Local provider verification (Ollama, real inference): `nomic-embed-text`
  returns 768-dimensional vectors and accepts batched `input` lists;
  `llama3.2:3b` answers chat completions through the same OpenAI-compatible
  endpoint. Rebuilt the stack with
  `LLM_BASE_URL=http://host.docker.internal:11434/v1`, recreated the Qdrant
  collection at 768 dimensions, and ingested `sample.txt` (`semantic`, 1 chunk)
  and a 10-section TXT (`fixed`, 3 chunks) — both HTTP 201, Postgres rows and
  Qdrant payloads consistent (4 points), and a query embedding for “How many
  days of annual leave…?” ranked the leave-policy chunk first (score 0.7436).
  Test documents were removed afterwards; Postgres and Qdrant are back to zero.
- Google Gemini was also attempted: the `AQ.`-prefixed AI Studio key
  authenticated (`models.list` succeeded), but inference returned
  `403 PERMISSION_DENIED: The Gemini API free tier is not available for this
  project`. Per Google's troubleshooting docs this is project-level
  eligibility; a new key in the same project does not help, and the project
  needs billing or a new eligible project. No application code changes were
  needed for either provider.

## Phase 4 — Delivered Artifacts
- `app/services/memory_service.py`: `ChatRole`, frozen `ChatMessage`,
  `history_key()` (`chat:history:{session_id}`), and `MemoryService` with
  `get_history()`, `append_exchange()`, and `clear()`. Each append uses a
  transactional pipeline (`RPUSH` pair → `LTRIM` window → `EXPIRE` TTL), so the
  history is atomically bounded and refreshed. Corrupt JSON and invalid
  role/content pairs are skipped with a warning instead of failing the turn.
  Every Redis failure becomes `MemoryStoreError`.
- `app/services/rag_service.py`: `RagService`, `RagResult`, `LLMError`, and
  `RetrievalError`. `answer()` loads history → reformulates follow-ups (when
  history exists) → embeds the standalone query → searches Qdrant with
  `limit=RAG_TOP_K` and payloads → returns a grounded answer plus attributed
  `SourceChunk`s. `NO_CONTEXT_ANSWER` short-circuits empty retrieval, and a
  dedicated `_clean_query()` tolerates small-model output (quotes, prefixes,
  extra lines). Source scores are clamped to the cosine range and malformed
  payloads are skipped defensively.
- `app/api/v1/endpoints/chat.py`: `POST /api/v1/chat` accepts `ChatRequest`,
  resolves the shared Redis/Qdrant/embedding dependencies, builds the service,
  and maps `MemoryStoreError`/`RetrievalError` → 503 and
  `EmbeddingError`/`LLMError` → 502. Registered in `app/api/v1/router.py`;
  OpenAPI now lists `/api/v1/ingest`, `/api/v1/chat`, and `/health`.
- Wiring: `RAG_TOP_K` added to `Settings` and `.env.example`; compose now
  forwards `RAG_TOP_K`, `CHAT_HISTORY_TTL_SECONDS`,
  `CHAT_HISTORY_MAX_MESSAGES`, and `EMBEDDING_BATCH_SIZE`. The Postman
  collection is renamed to Phases 1–4 and adds chat turn 1, a same-session
  follow-up, and a blank-message 422 case.
- Verification performed: `ruff check app` clean; `mypy app` clean (23 files);
  36 checks passed (18 memory/helper + 18 stubbed service branches). Covered:
  key format, chronological order, window trimming (both logical and physical
  Redis list), TTL bounds, malformed-entry skipping, clear/delete, query
  cleaning, context labels, score clamping, invalid payloads, empty retrieval
  (no LLM call, exchange still stored), `top_k`, two-call reformulation flow,
  prompt contents, source mapping, Qdrant failure → `RetrievalError`, empty
  completion → `LLMError`, and reformulation failure fallback.
- Live Docker verification with real Ollama inference: rebuilt the API
  (`RAG_TOP_K=5`, max messages 20, TTL 86400), ingested `sample.txt` (HTTP 201),
  and ran two turns in one session. Turn 1 answered “25 days … two weeks in
  advance” with `sample.txt` chunk 0 (score 0.7406). Turn 2 answered “three days
  per week” (score 0.6525) and the logs show the follow-up was rewritten to
  “How many days per week are allowed for remote work” before retrieval. Redis
  `chat:history:phase4-live` held exactly 4 messages with TTL 86400. An
  out-of-domain question produced an explicit refusal based on the handbook
  rather than a hallucinated answer. A whitespace-only message returned 422.
  Stopping Redis made chat return
  `503 {"detail":"Chat memory is temporarily unavailable."}`; after restarting
  Redis, chat recovered and history was written. All test documents, points, and
  session keys were removed afterwards (Postgres 0 rows, Qdrant 0 points, 0
  chat keys).

## Local Provider Setup (Ollama)
- Host install: `ollama pull nomic-embed-text` and `ollama pull llama3.2:3b`.
- The API container reaches the host via `host.docker.internal:11434`; this was
  verified with a container-side probe (model list, 2×768 embedding batch, chat
  reply), not just from the host.
- `.env` (gitignored) currently uses:
  `LLM_API_KEY=ollama`,
  `LLM_BASE_URL=http://host.docker.internal:11434/v1`,
  `LLM_MODEL=llama3.2:3b`,
  `EMBEDDING_MODEL=nomic-embed-text`,
  `EMBEDDING_DIMENSION=768`,
  `EMBEDDING_BATCH_SIZE=16`.
- When switching embedding providers, delete the `documents` collection before
  restarting so startup recreates it at the new dimension (768 for Ollama, 1536
  for OpenAI `text-embedding-3-small`, 3072 for Gemini embeddings).

## Notes for Next Step
- Phase 5: add `app/services/booking_service.py` with an OpenAI-compatible
  `book_interview` tool schema requiring `name`, `email`, `date`, and `time`.
  Reuse `InterviewBookingToolArgs` for validation so malformed or impossible
  values are rejected before any database write.
- Integrate the tool loop into `RagService`: pass `tools=[BOOK_INTERVIEW_TOOL]`
  to the chat completion, inspect `tool_calls`, validate the JSON arguments
  defensively (`llama3.2:3b` can emit malformed JSON), insert an
  `InterviewBooking` row through an `AsyncSession`, append the `tool` role
  message, and run a second completion for the natural-language confirmation.
  The `RagService` and `/api/v1/chat` signature will need the DB session added;
  `ChatResponse.booking` already exists and currently serialises as `null`.
- Persist only validated bookings. On invalid tool arguments, ask the user for
  the missing/corrected fields instead of raising; map database failures to 503
  and provider failures to 502. Keep the final user/assistant text exchange in
  Redis like Phase 4 so follow-ups remain coherent.
- Do not add `RetrievalQAChain`/LangChain chains, FAISS, or ChromaDB at any
  point; retrieval, prompting, and tool execution stay explicit and hand-written.
- Phase 6 will own the pytest suite (`tests/`) and the final `README.md`; keep
  ad-hoc verification scripts outside the repository.
```
