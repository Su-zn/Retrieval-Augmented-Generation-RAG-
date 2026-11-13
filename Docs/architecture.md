# System Architecture

## 1. Tech Stack
- **Framework:** FastAPI (Python 3.11+, async/await throughout)
- **Package Manager:** `uv` (with `pyproject.toml` and `requirements.txt` exported for compatibility)
- **Vector Database:** Qdrant (`qdrant-client` async)
- **In-Memory Store (Chat Memory):** Redis (`redis.asyncio`)
- **Relational Database (Metadata & Bookings):** PostgreSQL (`asyncpg` + SQLAlchemy 2.0 async) or SQLite (`aiosqlite`) as fallback
- **LLM & Embeddings Provider:** OpenAI-compatible SDK (`openai.AsyncOpenAI`) configurable via `.env` (supports OpenAI, OpenRouter, or Groq endpoints seamlessly)
- **PDF Parsing:** `pypdf` or `pymupdf` (`fitz`)

## 2. Directory Structure
```text
├── docker-compose.yml          # Spins up Qdrant, Redis, and FastAPI app
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── .env.example
├── README.md
└── app/
    ├── __init__.py
    ├── main.py                 # FastAPI app initialization, lifespan events, routers
    ├── core/
    │   ├── config.py           # Pydantic BaseSettings for env vars
    │   ├── database.py         # SQLAlchemy async engine & session dependency
    │   ├── redis_client.py     # Redis connection pool & lifecycle
    │   └── vector_store.py     # Qdrant client initialization & collection setup
    ├── models/
    │   ├── domain.py           # SQLAlchemy ORM models (DocumentMetadata, InterviewBooking)
    │   └── schemas.py          # Pydantic v2 request/response/tool schemas
    ├── services/
    │   ├── extractor.py        # PDF and TXT text extraction
    │   ├── chunker.py          # Strategy pattern: FixedChunker & RecursiveSemanticChunker
    │   ├── embedder.py         # Direct async embedding generation
    │   ├── rag_service.py      # Query contextualization, Qdrant search, prompt assembly
    │   ├── memory_service.py   # Redis chat history CRUD
    │   └── booking_service.py  # DB persistence for interview bookings & LLM tool handler
    └── api/
        └── v1/
            ├── endpoints/
            │   ├── ingest.py   # POST /ingest
            │   └── chat.py     # POST /chat
            └── router.py       # Aggregates v1 routes