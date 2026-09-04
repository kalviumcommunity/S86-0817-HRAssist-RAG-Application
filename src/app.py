"""FastAPI application — HRAssist RAG Service.

HRS3.45 — Document Upload & Indexing Endpoint

Exposes:
  POST /documents  — upload, ingest, chunk, and index a document at runtime
  GET  /health     — service health check

After a successful upload the new document's chunks are immediately
searchable via the in-memory vector store without restarting the app.

Running locally
---------------
    uvicorn src.app:app --reload

Sample upload
-------------
    curl -X POST http://localhost:8000/documents \\
         -F "file=@new-policy.md"
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse

from src.document_processor import (
    store_upload,
    process_uploaded_document,
    UploadValidationError,
    VECTOR_STORE,
    UPLOAD_DIR,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)

load_dotenv()

# ── Optional real embed function ──────────────────────────────────────────
# When OPENAI_API_KEY and EMBED_MODEL are set the app will embed uploaded
# documents for real.  When they are absent the app indexes chunks without
# vectors (safe for local dev without an API key).

_embed_fn = None

try:
    from openai import OpenAI

    _api_key  = os.getenv("OPENAI_API_KEY")
    _base_url = os.getenv("OPENAI_BASE_URL")
    _model    = os.getenv("EMBED_MODEL")

    if _api_key and _model:
        _client = OpenAI(api_key=_api_key, base_url=_base_url)

        def _embed_fn(texts):
            response = _client.embeddings.create(model=_model, input=texts)
            return [item.embedding for item in response.data]

except Exception:
    _embed_fn = None


# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(
    title="HRAssist RAG API",
    description=(
        "Upload HR policy documents and query them via semantic search. "
        "Uploaded documents are indexed immediately and become searchable "
        "without restarting the service."
    ),
    version="1.0.0",
)


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/health", summary="Service health check")
def health() -> Dict[str, Any]:
    """Return service status and current vector store size."""
    return {
        "status": "ok",
        "indexed_chunks": len(VECTOR_STORE),
        "upload_dir": str(UPLOAD_DIR),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
        "embed_enabled": _embed_fn is not None,
    }


# ── Document upload & indexing ────────────────────────────────────────────

@app.post(
    "/documents",
    status_code=status.HTTP_200_OK,
    summary="Upload and index a document",
    response_description="Indexing summary for the uploaded document",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (.txt, .md, or .pdf)"),
) -> Dict[str, Any]:
    """Accept a document upload, run the full ingestion pipeline, and index it.

    The pipeline stages are:
      1. Validate extension (.txt / .md / .pdf) and file size (max 10 MB).
      2. Store the file under the ``uploads/`` directory.
      3. Load, clean, chunk, tag, and embed the document.
      4. Append embedded records to the in-memory vector store.

    After indexing the new content is immediately available for retrieval
    by any subsequent ``/query`` request.

    Returns a structured summary::

        {
            "status": "indexed",
            "filename": "new-policy.md",
            "summary": {
                "document": "uploads/new-policy.md",
                "chunks": 12,
                "indexed": 12
            }
        }

    Error responses:
      - **415** — unsupported file type
      - **400** — empty file or file too large
      - **500** — unexpected error during indexing
    """
    content = await file.read()

    try:
        path = store_upload(
            filename=file.filename,
            content=content,
        )
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        )

    try:
        summary = process_uploaded_document(
            path=path,
            embed_fn=_embed_fn,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {exc}",
        )

    if summary.get("status") == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=summary.get("error", "Document indexing failed"),
        )

    return {
        "status": "indexed",
        "filename": file.filename,
        "summary": summary,
    }
