# Detailed API & Prompt Design

## 1. API Contracts

### `POST /api/v1/ingest`
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `file`: `UploadFile` (`.pdf` or `.txt`)
  - `chunking_strategy`: `Literal["fixed", "semantic"] = "semantic"`
- **Response (`201 Created`):**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "company_handbook.pdf",
  "chunking_strategy": "semantic",
  "chunks_indexed": 14,
  "message": "Document ingested and indexed successfully."
}