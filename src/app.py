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
import json
from typing import AsyncIterator
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.document_processor import (
    store_upload,
    process_uploaded_document,
    UploadValidationError,
    VECTOR_STORE,
    UPLOAD_DIR,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)
from src.rag_pipeline import answer_query
from src.rag_pipeline import embed_query, retrieve_context, assemble_retrieved_context
from src.citations import build_citation_map

load_dotenv()

# ── Optional real embed function ──────────────────────────────────────────
# When OPENAI_API_KEY and EMBED_MODEL are set the app will embed uploaded
# documents for real.  When they are absent the app indexes chunks without
# vectors (safe for local dev without an API key).

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("EMBED_MODEL", ""))
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("MODEL_NAME", ""))
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")

_embed_fn = None
_generate_fn = None
_generate_stream_fn = None

try:
    from openai import OpenAI

    _api_key  = os.getenv("OPENAI_API_KEY")
    _base_url = os.getenv("OPENAI_BASE_URL")
    _model    = EMBEDDING_MODEL

    if _api_key and _model:
        _client = OpenAI(api_key=_api_key, base_url=_base_url)

        def _embed_fn(texts):
            response = _client.embeddings.create(model=_model, input=texts)
            return [item.embedding for item in response.data]

        if LLM_MODEL:
            def _generate_fn(question, context):
                response = _client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer only from the provided context. "
                                "If it is insufficient, say so."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}",
                        },
                    ],
                )
                return response.choices[0].message.content

            def _generate_stream_fn(question, context):
                response = _client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer only from the provided context. If it is insufficient, say so.",
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}",
                        },
                    ],
                    stream=True,
                )
                for event in response:
                    text = event.choices[0].delta.content
                    if text:
                        yield text

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


class QueryRequest(BaseModel):
    """Validated user question accepted by the RAG endpoint."""

    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    """A source reference returned with a grounded answer."""

    source: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Stable JSON response contract for RAG clients."""

    answer: str
    sources: List[Source]
    status: str


def _sse_event(event: Dict[str, Any]) -> str:
    """Serialize one server-sent event with JSON payload."""

    return f"data: {json.dumps(event)}\n\n"


def _answer_chunks(answer: str, chunk_size: int = 24) -> List[str]:
    """Split a completed answer into small UI-friendly stream chunks."""

    return [answer[start : start + chunk_size] for start in range(0, len(answer), chunk_size)]


async def _stream_query_events(question: str) -> AsyncIterator[str]:
    """Yield citation, answer, completion, or error SSE events."""

    try:
        if _embed_fn is None or (_generate_fn is None and _generate_stream_fn is None):
            raise RuntimeError("RAG query service is not configured")

        query_vector = embed_query(question, _embed_fn)
        chunks = retrieve_context(query_vector, VECTOR_STORE)
        assembled = assemble_retrieved_context(chunks)
        citation_map = build_citation_map(chunks[:assembled["chunks_included"]])
        sources = []
        for label, citation in citation_map.items():
            sources.append({
                "id": citation.get("chunk_id") or label,
                "label": label,
                "document": citation.get("source"),
                "chunk_id": citation.get("chunk_id"),
                "section": citation.get("section"),
                "text": citation.get("text", ""),
            })
        yield _sse_event({"type": "citations", "sources": sources})

        if not chunks:
            yield _sse_event({"type": "token", "text": "I could not find relevant context for that question."})
        elif _generate_stream_fn is not None:
            for text in _generate_stream_fn(question, assembled["context"]):
                yield _sse_event({"type": "token", "text": text})
        else:
            result = answer_query(
                query=question,
                chunk_records=VECTOR_STORE,
                embed_fn=_embed_fn,
                generate_fn=_generate_fn,
            )
            for text in _answer_chunks(result["answer"]):
                yield _sse_event({"type": "token", "text": text})

        yield _sse_event({"type": "done"})
    except Exception:
        yield _sse_event({
            "type": "error",
            "message": "The answer stopped streaming. Please retry.",
        })


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
        "query_enabled": _embed_fn is not None and (
            _generate_fn is not None or _generate_stream_fn is not None
        ),
        "collection_name": COLLECTION_NAME,
        "vector_db_configured": bool(VECTOR_DB_URL),
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


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question using retrieved policy context",
)
def query_rag(request: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline and return an answer with source metadata."""

    if _embed_fn is None or _generate_fn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG query service is not configured",
        )

    try:
        result = answer_query(
            query=request.question,
            chunk_records=VECTOR_STORE,
            embed_fn=_embed_fn,
            generate_fn=_generate_fn,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[
                Source(
                    source=source.get("source", "unknown"),
                    chunk_id=source.get("chunk_id"),
                    score=source.get("score"),
                )
                for source in result.get("sources", [])
            ],
            status=result.get("status", "answered"),
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service failed",
        )


@app.post(
    "/query/stream",
    summary="Stream a grounded answer with citations",
    response_class=StreamingResponse,
)
async def stream_query(request: QueryRequest) -> StreamingResponse:
    """Stream citations and answer chunks as server-sent events."""

    return StreamingResponse(
        _stream_query_events(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
