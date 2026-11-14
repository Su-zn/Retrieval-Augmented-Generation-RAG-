# Conversational RAG & Interview Booking API

A production-style FastAPI backend for **document ingestion**, **multi-turn retrieval-augmented chat**, and **interview booking through LLM tool calling**. Retrieval, prompting, memory, and tool execution are implemented directly with `qdrant-client`, `redis.asyncio`, SQLAlchemy 2.0, and the OpenAI-compatible SDK — no LangChain chains, FAISS, or ChromaDB.

- **Framework:** FastAPI + Uvicorn, fully async, Python 3.11+
- **Vector store:** Qdrant (cosine distance, async client)
- **Chat memory:** Redis lists with a sliding window and TTL
- **Relational store:** PostgreSQL in Docker; SQLite for zero-friction local runs
- **LLM / embeddings:** any OpenAI-compatible endpoint (OpenAI, Gemini, Mistral, Ollama, ...)
- **Validation:** Pydantic v2, SQLAlchemy 2.0 typed ORM, strict `ruff` + `mypy`
- **Tests:** 88 hermetic pytest tests (no live Redis, Qdrant, database, or model required)

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quick start with Docker](#quick-start-with-docker)
- [Local development](#local-development)
- [Configuration](#configuration)
- [API walkthrough](#api-walkthrough)
- [Interview booking](#interview-booking)
- [Error responses](#error-responses)
- [Testing and quality gates](#testing-and-quality-gates)
- [Design notes](#design-notes)

---

## Features

| Area | Capability |
|---|---|
| Ingestion | `POST /api/v1/ingest` accepts PDF and UTF-8 TXT uploads, validates extension/MIME/size, extracts text, chunks it, embeds it, and stores vectors plus metadata. |
| Chunking | Strategy pattern: fixed-size sliding windows (`fixed`) and structure-aware paragraph/sentence splitting (`semantic`). |
| Retrieval | Async Qdrant similarity search with `RAG_TOP_K`; hits are returned as attributed source chunks with clamped cosine scores. |
| Multi-turn chat | Redis-backed history, follow-up rewriting into standalone search queries, and grounded answers that cite `[n]` sources. |
| Memory hygiene | Per-session key `chat:history:{session_id}`, bounded by `CHAT_HISTORY_MAX_MESSAGES` and expiring after `CHAT_HISTORY_TTL_SECONDS`. |
| Booking | `book_interview` tool calling with layered validation and PostgreSQL persistence. Returns the committed booking in the chat response. |
| Resilience | Provider failures map to 502; Redis/Qdrant/database failures map to 503; empty retrieval returns a deterministic refusal instead of hallucinating. |
| Observability | Structured request/answer logs, `/health` liveness probe, Docker healthchecks, OpenAPI docs at `/docs`. |

---

## Architecture

```text
                              ┌──────────────────────────────────────────┐
                              │            FastAPI (port 8000)           │
                              │  GET /health   GET /docs                 │
                              │  POST /api/v1/ingest                     │
                              │  POST /api/v1/chat                       │
                              └───────────┬───────────────┬──────────────┘
                                          │               │
                     ingest path          │               │        chat path
                                          ▼               ▼
                              ┌──────────────────┐   ┌───────────────────────┐
                              │   extractor      │   │  Redis chat memory    │
                              │   ↓              │   │  chat:history:{id}    │
                              │   chunker        │   └───────────┬───────────┘
                              │   ↓              │               │ history
                              │   embedder       │               ▼
                              └───────┬──────────┘   ┌───────────────────────┐
                                      │              │ reformulate follow-up │
                                      ▼              │ (standalone query)    │
                              ┌──────────────┐       └───────────┬───────────┘
                              │    Qdrant    │◀──────────────────┘ embed + search
                              │  documents   │
                              └──────┬───────┘
                                     │ vectors + payload
                                     ▼
                     ┌────────────────────────────────────┐
                     │ OpenAI-compatible LLM (chat/tools) │
                     └───────────────┬────────────────────┘
                                     │ book_interview tool call
                                     ▼
                     ┌────────────────────────────────────┐
                     │ PostgreSQL / SQLite                │
                     │ document_metadata                  │
                     │ interview_bookings                 │
                     └────────────────────────────────────┘
```

### Chat request flow

1. `ChatRequest` strips whitespace and rejects blank `session_id`/`message` (422).
2. `RagService` loads recent history from Redis.
3. If history exists, a short completion rewrites the follow-up into a standalone search query. A provider failure or blank rewrite falls back to the original question.
4. The query is embedded and used to search Qdrant (`RAG_TOP_K`, `with_payload=True`).
5. If no usable points match and no booking is attempted, the API returns a deterministic `NO_CONTEXT_ANSWER` without inviting a hallucination.
6. Otherwise the grounded prompt is sent with the `book_interview` tool available. A validated tool call is persisted; the tool result is replayed and a second completion produces the natural-language confirmation.
7. The user/assistant exchange is appended to Redis and the answer, sources, and optional booking are returned.

### Ingestion flow

1. Enforce `MAX_UPLOAD_BYTES` (413) and validate `.pdf` / `.txt` extension plus MIME type (400).
2. Extract PDF text in a worker thread with `pypdf`, or decode UTF-8 TXT.
3. Split text with the selected chunker.
4. Embed chunks in batches, preserving order and enforcing `EMBEDDING_DIMENSION`.
5. Upsert points with deterministic IDs (`uuid5(NAMESPACE_URL, "rag://documents/{document_id}/{chunk_index}")`).
6. Commit `document_metadata`. If the commit fails, best-effort delete of the written points keeps the stores consistent.

---

## Project structure

```text
.
├── app/
│   ├── main.py                  # App factory, lifespan, /health
│   ├── api/v1/
│   │   ├── dependencies.py      # Composable service dependencies
│   │   ├── router.py
│   │   └── endpoints/
│   │       ├── ingest.py        # POST /api/v1/ingest
│   │       └── chat.py          # POST /api/v1/chat
│   ├── core/
│   │   ├── config.py            # Pydantic settings
│   │   ├── database.py          # Async engine/session lifecycle
│   │   ├── redis_client.py      # Redis lifecycle/dependency
│   │   └── vector_store.py      # Qdrant lifecycle + collection bootstrap
│   ├── models/
│   │   ├── domain.py            # DocumentMetadata, InterviewBooking
│   │   └── schemas.py           # Pydantic request/response/tool schemas
│   └── services/
│       ├── extractor.py         # PDF/TXT validation and extraction
│       ├── chunker.py           # Fixed + semantic chunkers
│       ├── embedder.py          # Batched async embeddings
│       ├── ingestion_service.py # Ingestion orchestration
│       ├── memory_service.py    # Redis sliding-window history
│       ├── rag_service.py       # Retrieval, prompting, tool loop
│       └── booking_service.py   # Booking validation + persistence
├── tests/                       # Hermetic pytest suite
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Quick start with Docker

### Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- An OpenAI-compatible model endpoint. The project was verified with **local Ollama**; no paid key is required.

### 1. Configure the environment

```bash
cp .env.example .env
```

For local Ollama, edit `.env`:

```dotenv
LLM_API_KEY=ollama
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

`host.docker.internal` is required because the API runs in a container while Ollama runs on the host. Pull the models first:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

> To use OpenAI instead, set `LLM_API_KEY`, `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_MODEL=gpt-4o-mini`, `EMBEDDING_MODEL=text-embedding-3-small`, and `EMBEDDING_DIMENSION=1536`. The Qdrant collection dimension is fixed at startup — recreate the collection (or use a new `QDRANT_COLLECTION`) if you switch embedding dimensions.

### 2. Start the stack

```bash
docker compose up --build
```

This starts:

| Service | Port | Purpose |
|---|---|---|
| `api` | `8000` | FastAPI application |
| `qdrant` | `6333`, `6334` | Vector database |
| `redis` | `6379` | Chat memory |
| `postgres` | `5432` | Document metadata + bookings |

The API waits for all dependencies to become healthy and creates database tables plus the Qdrant collection on startup.

### 3. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","app":"Conversational RAG & Interview Booking API","version":"0.1.0","environment":"docker"}
```

Open Swagger UI: **http://localhost:8000/docs**

---

## Local development

Run the Python API directly (SQLite by default) while using Docker only for Redis/Qdrant:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

docker compose up -d qdrant redis
cp .env.example .env
# For host-run Ollama use: LLM_BASE_URL=http://127.0.0.1:11434/v1

uvicorn app.main:app --reload
```

Without a `DATABASE_URL` override, metadata and bookings are stored in `./app.db` (SQLite). Tables are created automatically at startup.

---

## Configuration

All settings are read from environment variables (and `.env`), case-insensitively, via `pydantic-settings`.

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Conversational RAG & Interview Booking API` | OpenAPI title |
| `APP_VERSION` | `0.1.0` | API version |
| `ENVIRONMENT` | `local` | One of `local`, `development`, `docker`, `staging`, `production`, `test` |
| `DEBUG` | `false` | FastAPI debug mode |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `API_V1_PREFIX` | `/api/v1` | Versioned route prefix |
| `DATABASE_URL` | `sqlite+aiosqlite:///./app.db` | SQLAlchemy async URL (Compose uses PostgreSQL) |
| `DATABASE_ECHO` | `false` | Log SQL statements |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CHAT_HISTORY_TTL_SECONDS` | `86400` | Session expiry (24 h) |
| `CHAT_HISTORY_MAX_MESSAGES` | `20` | Sliding-window size (messages, not turns) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `QDRANT_API_KEY` | unset | Optional Qdrant Cloud key |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `EMBEDDING_DIMENSION` | `1536` | Must match the embedding model |
| `LLM_API_KEY` | empty | Provider API key; use a dummy value such as `ollama` for local models |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Chat/tool-calling model |
| `LLM_TEMPERATURE` | `0.2` | 0.0–2.0 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks embedded per provider request |
| `RAG_TOP_K` | `5` | Retrieved chunks per question (max 50) |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload limit (10 MiB) |

`docker-compose.yml` passes the model, embedding, retrieval, memory, and upload settings from `.env` into the API container. Database, Redis, and Qdrant URLs are set to the Compose service names.

---

## API walkthrough

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@./handbook.txt;type=text/plain" \
  -F "chunking_strategy=semantic"
```

```json
{
  "document_id": "42955151-f94d-4cf0-8688-43a5c5b001d0",
  "filename": "sample.txt",
  "chunking_strategy": "semantic",
  "chunks_indexed": 1,
  "message": "Document ingested and indexed successfully."
}
```

- `chunking_strategy` defaults to `semantic`; `fixed` is also accepted.
- PDF uploads work the same way with `type=application/pdf`.
- PowerShell: use `curl.exe` instead of the `curl` alias.

### Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-session","message":"How many days of paid annual leave do employees get?"}'
```

```json
{
  "session_id": "demo-session",
  "answer": "According to the context, employees receive 25 days of paid annual leave.",
  "sources": [
    {
      "document_id": "42955151-f94d-4cf0-8688-43a5c5b001d0",
      "filename": "sample.txt",
      "chunk_index": 0,
      "text": "Leave Policy ... Employees receive 25 days of paid annual leave ...",
      "score": 0.7343
    }
  ],
  "booking": null
}
```

Send a follow-up with the **same `session_id`** to exercise multi-turn memory. The follow-up is rewritten into a standalone query before retrieval and the assistant sees the prior exchange:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-session","message":"And how many days per week can they work remotely?"}'
```

If no document matches, the response is deterministic:

```json
{
  "session_id": "demo-session",
  "answer": "I could not find any relevant information in the indexed documents to answer that question. Try ingesting a document that covers this topic, or rephrase the question.",
  "sources": [],
  "booking": null
}
```

### Health and docs

```bash
curl http://localhost:8000/health
open http://localhost:8000/docs      # macOS
# Windows: start http://localhost:8000/docs
```

---

## Interview booking

Booking is exposed through the same `POST /api/v1/chat` endpoint. The model must emit a `book_interview` tool call with **all four** fields; the API validates and persists it.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-booking",
    "message": "Book an interview for Jane Doe, jane.doe@example.com, on 2030-06-15 at 15:30."
  }'
```

```json
{
  "session_id": "demo-booking",
  "answer": "The interview has been booked for Jane Doe on 2030-06-15 at 15:30 ...",
  "sources": [],
  "booking": {
    "id": "3c9cb860-325c-4d62-9c3e-4f1799e36648",
    "session_id": "demo-booking",
    "candidate_name": "Jane Doe",
    "candidate_email": "jane.doe@example.com",
    "interview_date": "2030-06-15",
    "interview_time": "15:30:00",
    "created_at": "2026-11-13T23:09:33.785449Z"
  }
}
```

Validation is layered so a small model cannot fabricate a booking:

1. Tool arguments are parsed as JSON and validated by `InterviewBookingToolArgs`.
2. Email format, real calendar date, and `HH:MM` time are enforced.
3. The date must be today or later.
4. **The email must appear in the user's conversation.** An invented placeholder address is rejected even if it is syntactically valid.
5. Only then is an `InterviewBooking` row committed and refreshed into the response.

If details are missing or invalid, no row is written; the validation error is fed back to the model so it asks for the missing fields.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-booking","message":"I would like to schedule an interview."}'
```

```json
{
  "session_id": "demo-booking",
  "answer": "To schedule an interview, I need your full name, email address, date (YYYY-MM-DD), and time (HH:MM) ...",
  "sources": [],
  "booking": null
}
```

If the follow-up completion fails after a successful insert, the API returns a deterministic confirmation so the persisted booking is never hidden. If it fails after a rejected attempt, it returns a deterministic clarification.

---

## Error responses

All errors use FastAPI's `{"detail": ...}` shape.

| Status | When |
|---|---|
| `400` | Unsupported extension/MIME type, corrupt or password-protected PDF, non-UTF-8 TXT, empty extracted text |
| `413` | Upload larger than `MAX_UPLOAD_BYTES` |
| `422` | Blank/oversized `session_id` or `message`, invalid `chunking_strategy`, missing file |
| `502` | Embedding or chat provider request failed (network, auth, malformed provider response, empty completion) |
| `503` | Redis unavailable, Qdrant unavailable, booking persistence failure |

Example provider failure:

```json
{"detail": "Embedding provider request failed: Connection error."}
```

---

## Testing and quality gates

The suite is hermetic: Redis is replaced with `fakeredis`, Qdrant and embeddings with deterministic fakes, PostgreSQL with in-memory SQLite (`StaticPool`), and the LLM client is monkeypatched. The app lifespan is not started during tests.

```bash
pytest                 # 88 tests
ruff check app tests
mypy app tests
```

Coverage highlights:

| Test file | Focus |
|---|---|
| `tests/test_chunkers.py` | Fixed-window boundaries/overlap, semantic paragraph/sentence splitting, configuration errors |
| `tests/test_schemas.py` | Chat input trimming/blanks, booking tool date/time/email validation, source score bounds |
| `tests/test_memory.py` | Redis ordering, sliding window, TTL, isolation, malformed-entry skipping, failure mapping |
| `tests/test_ingestion.py` | Endpoint success, vector payloads, metadata persistence, 400/413/422/502/503 paths |
| `tests/test_booking.py` | Validation layers, SQLite persistence, invented-email rejection, commit failure, tool loop |
| `tests/test_rag.py` | Grounded answers, empty retrieval, malformed points, reformulation and fallback, 502/503 mapping |
| `tests/test_endpoints.py` | Health/docs/OpenAPI, request validation, chat turns, unknown tools, persistence failure |

A live end-to-end run was also performed with `docker compose up --build` and local Ollama: both TXT and PDF ingestion, a two-turn conversation (the follow-up was rewritten to a standalone query), a successful booking persisted to PostgreSQL, and an incomplete booking that correctly wrote no row.

---

## Design notes

- **No chain abstractions.** Retrieval, prompt assembly, source attribution, memory, and tool execution are explicit service code. This keeps failure modes visible and testable.
- **Deterministic refusal over hallucination.** When retrieval is empty and no booking is attempted, the model's text is discarded in favour of `NO_CONTEXT_ANSWER`.
- **Grounded citations.** Context is rendered as `[n] filename=..., chunk=..., score=...`, and the system prompt instructs the model to answer only from that context and cite labels.
- **Deterministic point IDs.** Re-ingesting the same document/chunk cannot silently duplicate vectors.
- **Bounded memory.** History is trimmed atomically with `LTRIM` and always given a TTL, so Redis does not grow without bound.
- **Typed boundaries.** SQLAlchemy 2.0 `Mapped` models, Pydantic v2 schemas, `ruff`, and strict `mypy` are part of the normal workflow.
- **Small-model hardening.** `llama3.2:3b` can invent tool arguments. The email-grounding rule and deterministic fallbacks are deliberate safeguards; larger models generally produce cleaner tool calls without changing application code.

### Known limitations

- No authentication/authorisation; the API is intended as a backend exercise rather than a public multi-tenant service.
- Booking does not check slot availability or double-booking; it persists a requested slot.
- The in-process service singletons (`EmbeddingService`) assume a single Uvicorn worker unless replaced; scale out with shared connection pools and an external task queue as needed.
- The Qdrant collection dimension is fixed at startup. Changing `EMBEDDING_DIMENSION` requires recreating the collection.
