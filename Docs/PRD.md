# Product Requirements Document (PRD): Conversational RAG & Interview Booking Backend

## 1. Objective
Build a production-grade, modular FastAPI backend featuring two primary REST APIs:
1. **Document Ingestion API** for parsing, chunking, embedding, and indexing `.pdf` and `.txt` documents.
2. **Conversational RAG API** supporting multi-turn dialogue via Redis memory, custom retrieval-augmented generation, and automated interview scheduling via LLM tool-calling.

## 2. Core Functional Requirements

### API 1: Document Ingestion (`POST /api/v1/ingest`)
- **Input:** Multipart form-data accepting `.pdf` or `.txt` files, plus a `chunking_strategy` enum parameter (`"fixed"` | `"semantic"`).
- **Text Extraction:** Extract raw text cleanly from uploaded files (using `pypdf` or `pymupdf` for PDFs; UTF-8 decoding for TXT).
- **Chunking Strategies:**
  - `fixed`: Fixed-size character/token chunking with configurable overlap.
  - `semantic` (or recursive paragraph/sentence-aware): Structure-aware splitting preserving sentence/paragraph boundaries.
- **Embedding & Vector Indexing:** Generate dense vector embeddings and upsert them into **Qdrant** (via Docker or Qdrant Cloud) with payload metadata (`document_id`, `chunk_index`, `text`).
- **Metadata Persistence:** Store document metadata (`id`, `filename`, `file_type`, `chunking_strategy`, `chunk_count`, `created_at`) in a relational database (**PostgreSQL** via SQLAlchemy async or **SQLite** via `aiosqlite` for zero-friction local evaluation).

### API 2: Conversational RAG (`POST /api/v1/chat`)
- **Input:** JSON payload containing `session_id` (UUID/string) and `message` (string).
- **Chat Memory:** Retrieve and append conversation turns (`user` and `assistant` messages) in **Redis** keyed by `session_id` with an appropriate TTL.
- **Multi-Turn Query Contextualization:** Reformulate follow-up queries using recent Redis chat history so standalone retrieval queries capture full conversational context.
- **Custom RAG Pipeline:**
  - Embed the contextualized query.
  - Perform similarity search directly against Qdrant's client SDK.
  - Construct a grounded system/user prompt containing retrieved chunks and chat history.
  - Generate response directly via LLM SDK (OpenAI / OpenRouter / Groq compatible client).
- **Interview Booking Tool-Calling:**
  - Equip the LLM with a structured function/tool definition (`book_interview`).
  - Extract required fields: `name` (str), `email` (EmailStr), `date` (YYYY-MM-DD), and `time` (HH:MM).
  - If any required field is missing, the LLM must politely ask the user for the missing detail(s) across turns before invoking the tool.
  - Once all fields are collected, execute the tool to persist the booking into the SQL database and confirm the booking to the user.

## 3. Hard Constraints
- **NO** FAISS or ChromaDB. (Use Qdrant).
- **NO** `RetrievalQAChain` or high-level LangChain chain abstractions. Write raw SDK calls for embeddings, vector search, and LLM completions.
- **NO** frontend/UI code. Rely on FastAPI's auto-generated OpenAPI/Swagger UI (`/docs`).
- **48-Hour Delivery Scope:** Must include a `docker-compose.yml` (Qdrant + Redis + API) and a thorough `README.md` so reviewers can spin up and test everything in under 2 minutes.