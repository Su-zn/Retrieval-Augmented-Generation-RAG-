---

### 3. `Rules.md` (Coding Standards & Guardrails for OpenCode)

```markdown
# Engineering Rules & Coding Standards

Follow these strict rules for every file generated or modified in this repository:

1. **Strict Type Annotations:**
   - Every function, method, and fixture MUST have complete Python 3.11+ type annotations for all parameters and return types (e.g., `async def chunk_text(self, text: str) -> list[str]:`).
   - Use built-in generic types (`list[str]`, `dict[str, Any]`) rather than deprecated `typing.List` or `typing.Dict`.

2. **Pydantic v2 Everywhere:**
   - Use `pydantic.BaseModel` (v2 syntax: `model_config = ConfigDict(...)`, `Field(...)`) for all API request/response schemas, configuration (`pydantic_settings.BaseSettings`), and LLM tool argument validation.

3. **Forbidden Libraries & Abstractions:**
   - NEVER import `faiss`, `chromadb`, or `langchain.chains.RetrievalQA` / `RetrievalQAChain`.
   - Write explicit, readable Python functions for prompt formatting, vector querying, and tool execution loops.

4. **Async-First I/O:**
   - All FastAPI endpoints, database queries (`AsyncSession`), Redis operations (`redis.asyncio`), Qdrant client calls (`AsyncQdrantClient`), and LLM calls (`AsyncOpenAI`) must be asynchronous.
   - Wrap CPU-bound PDF parsing in `asyncio.to_thread()` if needed so the event loop is never blocked.

5. **Error Handling & HTTP Semantics:**
   - Validate file extensions and MIME types before processing; raise `HTTPException(status_code=400, ...)` for unsupported formats or empty files.
   - Catch upstream LLM/VectorDB connection errors gracefully and return structured HTTP 502/503 errors with clear diagnostic messages.

6. **Modularity & Single Responsibility:**
   - Keep route handlers (`api/v1/endpoints/*.py`) thin: validate input, call a service method, return a Pydantic response schema. Business logic belongs strictly in `app/services/`.