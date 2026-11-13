---

### 6. `Memory.md` (Agent State Tracker for OpenCode)

```markdown
# Project Memory & Progress Log

## Current Status
- **Status:** All six phases complete and verified.
- **Completed Phases:** Phase 1 (Scaffolding & Infrastructure), Phase 2
  (Database Models & Pydantic Schemas), Phase 3 (Document Ingestion Pipeline),
  Phase 4 (Redis Chat Memory + Custom Multi-Turn RAG Service), Phase 5
  (LLM Tool Calling + Interview Booking), Phase 6 (End-to-End Verification,
  Pytest Suite & `README.md`)
- **Blockers:** None. Phases 3–6 are verified end-to-end with **real local
  inference** (Ollama `nomic-embed-text` embeddings + `llama3.2:3b` chat and
  tool calling), so ingestion, multi-turn chat, and interview booking run
  without a paid API key. Google Gemini was evaluated first, but the available
  Google Cloud project was not eligible for the free tier (HTTP 403), so the
  project is configured for local Ollama through its OpenAI-compatible endpoint.

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
- **RAG service (Phase 4):** `RagService.answer()` loads history, rewrites follow-ups into a standalone query when history exists (small local models are unreliable at this, so a reformulation failure logs a warning and falls back to the original question), embeds that query with `EmbeddingService`, and calls Qdrant `query_points(limit=RAG_TOP_K, with_payload=True)`. Hits are mapped to `SourceChunk` with clamped cosine scores and rendered as `[n] filename=..., chunk=..., score=...` labels in the grounded prompt. With tool calling disabled, empty retrieval returns a deterministic "no relevant information" answer without calling the LLM; with tool calling enabled (the production path), the turn runs a tool-enabled completion so a booking can still proceed, and the same deterministic refusal is returned when no booking occurs. The exchange is stored either way. Embedding/LLM failures map to 502, Redis/Qdrant failures to 503.
- **Retrieval configuration:** `RAG_TOP_K` (default 5, max 50) lives in `Settings`; compose forwards `RAG_TOP_K`, `CHAT_HISTORY_TTL_SECONDS`, `CHAT_HISTORY_MAX_MESSAGES`, and `EMBEDDING_BATCH_SIZE` from `.env`.
- **Interview booking (Phase 5):** `BOOK_INTERVIEW_TOOL` mirrors `InterviewBookingToolArgs` (name, email, date `YYYY-MM-DD`, time `HH:MM`). `BookingService` validates JSON with Pydantic, rejects schema/calendar errors and past dates, and requires the email to appear in the user's turns so a small model cannot invent placeholder contact details; only then is an `InterviewBooking` row committed and refreshed. `RagService` passes `tools=[BOOK_INTERVIEW_TOOL]` on the first completion, replays the assistant tool call plus a `role=tool` result, and runs a second completion without tools. Unknown tools, extra bookings in one turn, and validation failures are returned to the model for clarification. If the follow-up completion fails or is empty after a success, a deterministic confirmation is returned so the persisted booking is never hidden; failed attempts use a deterministic clarification. Database failures map to 503.
- **Testing strategy (Phase 6):** endpoint services are composed through `app/api/v1/dependencies.py`, and tests override those dependencies rather than patching endpoint bodies. The suite never starts the app lifespan: `get_db`, Redis, Qdrant, and embeddings are replaced with in-memory fakes, while the LLM client is injected by monkeypatching `RagService._get_client`. This keeps all 88 tests deterministic, offline, and fast (~1 s) while still exercising the real routers, services, validation, and error mapping.

## Completed Checklist
- [x] Phase 1: Scaffolding, Docker Compose, Core Clients (Qdrant, Redis, SQLAlchemy)
- [x] Phase 2: ORM Models (`DocumentMetadata`, `InterviewBooking`) & Pydantic Schemas
- [x] Phase 3: Document Ingestion API (`/api/v1/ingest`) + Fixed & Semantic Chunkers
- [x] Phase 4: Redis Chat Memory + Custom Multi-Turn RAG Service
- [x] Phase 5: LLM Tool Calling (`book_interview`) + Booking Persistence
- [x] Phase 6: End-to-End Verification, Pytest Suite & `README.md`

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
  `SourceChunk`s. `NO_CONTEXT_ANSWER` short-circuits empty retrieval when tool
  calling is not configured; Phase 5 adds a tool-enabled path that still refuses
  deterministically if no booking results. A dedicated `_clean_query()` tolerates
  small-model output (quotes, prefixes, extra lines). Source scores are clamped to
  the cosine range and malformed
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

## Phase 5 — Delivered Artifacts
- `app/services/booking_service.py`: `BOOK_INTERVIEW_TOOL_NAME`, typed
  `BOOK_INTERVIEW_TOOL` (name/email/date/time, all required), plus
  `BookingValidationError`, `BookingPersistenceError`, and `BookingService` with
  `parse_arguments()` and `book()`. Validation runs in layers: raw JSON → Pydantic
  schema (email format, real calendar date, `HH:MM` time) → date not in the past
  → email must appear in the user's conversation. Only then is an
  `InterviewBooking` row committed and refreshed into `InterviewBookingRead`.
  Pydantic errors are rendered as compact `field: message` text the model can act
  on, and the conversation-grounding check stops small models from inventing
  placeholder contact details.
- `app/services/rag_service.py`: `RagResult.booking`, internal `_Completion`,
  tool-enabled `_complete()`, assistant/tool message replay, one follow-up
  completion, a per-turn duplicate-booking guard, deterministic
  confirmation/clarification fallbacks, and rejection logging. Unknown tools and
  invalid arguments are fed back to the model as tool results rather than raised.
  Retrieval-empty non-booking turns still resolve to `NO_CONTEXT_ANSWER`.
- `app/api/v1/endpoints/chat.py`: injects `AsyncSession`, builds
  `BookingService(session)`, maps `BookingPersistenceError` to 503, and returns
  `result.booking` in `ChatResponse`.
- Postman: collection renamed to Phases 1–5 and extended with a complete booking
  request (asserts the persisted fields) and an incomplete booking request
  (asserts `booking == null`).
- Verification performed: `ruff check app` clean; `mypy app` clean (24 files);
  37 checks passed. Covered: valid/missing/malformed arguments, invalid email,
  impossible date, bad time, past date, SQLite persistence (exactly one row,
  candidate fields, `created_at`), invented-email rejection with zero inserts,
  tool schema passed on the first call, tool result replayed on the second call,
  follow-up completion not tool-enabled, booking returned and stored in Redis,
  deterministic confirmation/clarification on follow-up failure, persistence
  failure propagation, booking without retrieved context, and grounded RAG
  regression.
- Live Docker verification with real Ollama tool calling: a complete request
  created booking `b81342c2-39da-4eca-aab5-eca65be0b19c` for Jane Doe
  (`jane.doe@example.com`, 2026-12-01 14:30). The response carried the full
  `booking` object, PostgreSQL held the matching row, and Redis stored the
  two-message exchange. “I'd like to schedule an interview.” did **not** create a
  booking: the model invented an email, the grounding check rejected it, and the
  assistant asked for details. An explicit 2020-01-01 request was rejected by the
  past-date rule. A leave-policy question still answered from `sample.txt`
  (chunk 0, 0.7343) with `booking: null`. Test rows, points, and session keys were
  removed; a pre-existing resume PDF dataset in Qdrant and its chat session were
  identified and deliberately left untouched.

## Phase 6 — Delivered Artifacts
- `app/api/v1/dependencies.py`: composable dependencies
  (`get_memory_service`, `get_booking_service`, `get_rag_service`) so the chat
  endpoint stays thin and tests can override infrastructure at the boundary.
  `app/api/v1/endpoints/chat.py` now depends only on `RagService` and maps each
  service error to its HTTP status.
- `tests/`: 88 hermetic tests across `test_chunkers.py`, `test_schemas.py`,
  `test_memory.py`, `test_ingestion.py`, `test_booking.py`, `test_rag.py`, and
  `test_endpoints.py`. Shared fixtures in `tests/conftest.py` provide
  `ApiContext` (ASGITransport client), `FakeEmbedder`, `FakeQdrant`,
  `FakeLLMClient`, `scripted_handler`, `make_point`, `session_factory`
  (in-memory SQLite with `StaticPool`), `fake_redis` (`fakeredis`), and
  `install_llm` (monkeypatched `RagService._get_client`). The app lifespan is
  intentionally not run, so no live Redis/Qdrant/Postgres connections occur.
- `requirements-dev.txt` / `pyproject.toml`: pinned `fakeredis==2.26.2` and set
  `asyncio_default_fixture_loop_scope = "function"` to remove the
  pytest-asyncio deprecation warning.
- `README.md`: architecture diagram, ingestion/chat/booking request flows,
  project layout, Docker and local setup, full environment-variable table,
  curl examples with real response shapes, booking validation rules, error
  mapping, testing instructions, design notes, and known limitations.
- Verification performed: `pytest` 88 passed; `ruff check app tests` clean;
  `mypy app tests` clean (34 files). Coverage includes chunker boundaries,
  schema validation, Redis window/TTL/isolation/failures, ingestion success and
  every error path, booking validation/persistence/tool loop, RAG grounding,
  empty retrieval, malformed Qdrant payloads, reformulation fallback, and
  502/503 mappings.
- Full Docker E2E with `docker compose up --build` and real Ollama:
  `/health` returned `ok`; `/docs` returned 200; OpenAPI listed `/health`,
  `/api/v1/ingest`, and `/api/v1/chat`. A TXT (`semantic`) and a generated PDF
  (`fixed`) both ingested with HTTP 201. A two-turn session answered the leave
  question from `sample.txt` (score 0.7343) and the follow-up from the PDF
  (score 0.8843), with logs showing the follow-up rewritten to “How many days
  per week can employees work remotely” and Redis holding 4 messages. A booking
  for `jane.doe@example.com` on 2030-06-15 15:30 returned
  `3c9cb860-325c-4d62-9c3e-4f1799e36648` and a matching PostgreSQL row; an
  incomplete request created no booking and asked for the missing fields. All
  Phase 6 test rows, points, and session keys were removed. Pre-existing user
  data was preserved: 6 resume points, `chat:history:abc678d5-...`, and the
  user's `swagger-booking` row/session.

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
- All planned phases are complete. Future work is maintenance/enhancement rather
  than a numbered phase.
- Possible enhancements: slot conflict detection and availability calendars,
  authentication and per-tenant isolation, streaming chat responses, reranking
  or hybrid (keyword + vector) retrieval, structured evaluation of answer
  groundedness, and a CI workflow that runs `pytest`, `ruff`, and `mypy`.
- Keep the hermetic test strategy: deterministic fakes for Redis/Qdrant/LLM and
  dependency overrides, with any live-Ollama check opt-in and skipped when the
  endpoint is absent.
- Do not add `RetrievalQAChain`/LangChain chains, FAISS, or ChromaDB at any
  point; retrieval, prompting, and tool execution stay explicit and hand-written.
- Keep ad-hoc verification scripts outside the repository; the committed test
  suite is self-contained and reproducible.
```
