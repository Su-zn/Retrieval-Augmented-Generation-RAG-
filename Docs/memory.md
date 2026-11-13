---

### 6. `Memory.md` (Agent State Tracker for OpenCode)

```markdown
# Project Memory & Progress Log

## Current Status
- **Active Phase:** Phase 4 (Redis Chat Memory + Custom RAG Service)
- **Completed Phases:** Phase 1 (Scaffolding & Infrastructure), Phase 2
  (Database Models & Pydantic Schemas), Phase 3 (Document Ingestion Pipeline)
- **Blockers:** None. The stack is healthy and Phase 3 was verified end-to-end
  with **real local embeddings** (Ollama `nomic-embed-text`, 768 dims), so the
  full HTTP → Postgres + Qdrant path runs without a paid API key. Google Gemini
  was evaluated first, but the available Google Cloud project was not eligible
  for the free tier (HTTP 403), so the project is configured for local Ollama
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

## Completed Checklist
- [x] Phase 1: Scaffolding, Docker Compose, Core Clients (Qdrant, Redis, SQLAlchemy)
- [x] Phase 2: ORM Models (`DocumentMetadata`, `InterviewBooking`) & Pydantic Schemas
- [x] Phase 3: Document Ingestion API (`/api/v1/ingest`) + Fixed & Semantic Chunkers
- [ ] Phase 4: Redis Chat Memory + Custom Multi-Turn RAG Service
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
- Phase 4: add Redis-backed chat memory keyed by `session_id` (append
  user/assistant messages, trim to a bounded window, set a TTL) using the
  shared client from `app/core/redis_client.py`.
- Build `app/services/chat_service.py`: embed the query with
  `EmbeddingService.embed_texts([question])`, call Qdrant `query_points`
  (`with_payload=True`) against `QDRANT_COLLECTION`, and map hits to
  `SourceChunk` using the Phase 3 payload (`document_id`, `filename`,
  `chunk_index`, `text`, `score`).
- Compose a grounded system/user prompt from retrieved chunks plus recent
  history, then call `AsyncOpenAI.chat.completions.create` with `LLM_MODEL` and
  `LLM_TEMPERATURE`; expose `POST /api/v1/chat` returning `ChatResponse` and
  keep endpoints thin.
- Handle empty retrieval explicitly, translate Redis/Qdrant/provider outages to
  structured 503/502 errors, and make sure Redis degrades gracefully (a memory
  outage should not take down `/docs` or ingestion).
- Ollama note for Phase 4/5: `nomic-embed-text` is the query/document embedding
  model, and `llama3.2:3b` handles chat plus basic tool calling. Validate tool
  arguments defensively in Phase 5, because small-model JSON can be malformed.
- Do not add `RetrievalQAChain`/LangChain chains, FAISS, or ChromaDB at any
  point; retrieval and completion stay explicit and hand-written.
```
