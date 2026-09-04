"""Document upload validation, storage, and indexing pipeline.

HRS3.45 — Document Upload & Indexing Endpoint

This module contains the pure pipeline logic for the upload endpoint,
separated from the FastAPI layer so it can be tested independently and
reused across different transport layers (HTTP, CLI, background jobs).

Pipeline stages for each uploaded document
-------------------------------------------
  1. Validate   — check file extension against the supported set; reject
                  unsupported formats and empty files early.
  2. Store      — write the raw bytes to a safe path under UPLOAD_DIR,
                  sanitising the filename to prevent path traversal.
  3. Load       — extract plain text using the same DocumentLoader used
                  for the original corpus so all formats are handled.
  4. Clean      — apply the same text cleaning pipeline used at corpus
                  ingestion time for consistency.
  5. Chunk      — split cleaned text into token-aware overlapping chunks.
  6. Tag        — attach source, chunk_index, and region metadata so
                  retrieved chunks can be cited back to their origin.
  7. Index      — store the embedded records in the in-memory vector
                  store so the new content is searchable immediately
                  without restarting the application.

The in-memory vector store
--------------------------
``VECTOR_STORE`` is a module-level list that holds all embedded chunk
records. For this assignment it uses a simple list as backing storage.
In production this would be replaced by Chroma, Qdrant, Pinecone, etc.

The ``embed_fn`` parameter is injected into ``process_uploaded_document``
so the embedding model can be swapped or mocked in tests without touching
the pipeline logic.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.document_loader import DocumentLoader
from src.text_cleaner import clean_text
from src.token_chunker import token_chunks
from src.document_chunker import tag_chunks


# ── Configuration ─────────────────────────────────────────────────────────

UPLOAD_DIR: Path = Path("uploads")

SUPPORTED_EXTENSIONS: frozenset = frozenset({".txt", ".md", ".pdf"})

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024   # 10 MB

# Default chunking parameters — match the corpus ingestion pipeline
DEFAULT_CHUNK_SIZE: int = 100
DEFAULT_CHUNK_OVERLAP: int = 20


# ── In-memory vector store (module-level singleton) ───────────────────────

# Each entry is a dict with "text", "metadata", and "embedding" keys.
# New records are appended here after every successful upload so the
# content becomes immediately searchable via retrieve().
VECTOR_STORE: List[Dict[str, Any]] = []


# ── Validation ────────────────────────────────────────────────────────────

class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation checks.

    Carries an HTTP-friendly ``status_code`` (415 for unsupported type,
    400 for empty file or size exceeded) so the API layer can translate it
    into the correct HTTP response without knowing the details.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_upload(
    filename: str,
    content: bytes,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> str:
    """Validate an uploaded file before storing or processing it.

    Checks:
      - File extension is in ``SUPPORTED_EXTENSIONS``.
      - File content is not empty.
      - File size does not exceed ``max_size_bytes``.

    Args:
        filename: Original filename from the upload (used for extension check).
        content: Raw file bytes.
        max_size_bytes: Maximum allowed file size in bytes.

    Returns:
        The lower-case file extension (e.g. ``".txt"``).

    Raises:
        UploadValidationError: With ``status_code=415`` for unsupported
            extension, ``status_code=400`` for empty file or size exceeded.

    Example::

        ext = validate_upload("policy.txt", file_bytes)
    """
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
            status_code=415,
        )

    if not content:
        raise UploadValidationError(
            "Uploaded file is empty.",
            status_code=400,
        )

    if len(content) > max_size_bytes:
        limit_mb = max_size_bytes / (1024 * 1024)
        raise UploadValidationError(
            f"File size {len(content)} bytes exceeds limit of {limit_mb:.0f} MB.",
            status_code=400,
        )

    return suffix


def safe_filename(filename: str) -> str:
    """Return just the final name component to prevent path traversal.

    Strips any directory parts from the filename so a malicious upload
    like ``../../etc/passwd`` cannot write outside UPLOAD_DIR.

    Args:
        filename: The raw filename from the upload.

    Returns:
        The sanitised filename (basename only).
    """
    return Path(filename).name


# ── Storage ───────────────────────────────────────────────────────────────

def store_upload(
    filename: str,
    content: bytes,
    upload_dir: Path = UPLOAD_DIR,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Path:
    """Validate and persist an uploaded file to disk.

    Validates the file first; if validation passes, creates the upload
    directory (if it does not exist) and writes the bytes to a safe path.

    Args:
        filename: Original filename from the upload.
        content: Raw file bytes.
        upload_dir: Directory to store uploaded files. Created if absent.
        max_size_bytes: Maximum allowed file size.

    Returns:
        The ``Path`` where the file was written.

    Raises:
        UploadValidationError: If validation fails (see ``validate_upload``).

    Example::

        path = store_upload("policy.txt", raw_bytes)
        print(f"Saved to {path}")
    """
    validate_upload(filename, content, max_size_bytes)

    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(filename)
    dest = upload_dir / safe_name
    dest.write_bytes(content)
    return dest


# ── Processing pipeline ───────────────────────────────────────────────────

def process_uploaded_document(
    path: Path,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    region: str = "Global",
    vector_store: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Load, clean, chunk, tag, embed, and index an uploaded document.

    Runs the same pipeline stages used for the original corpus ingestion
    so uploaded documents behave identically to pre-indexed content.

    Args:
        path: Path to the stored uploaded file.
        embed_fn: Callable ``(texts: List[str]) -> List[List[float]]``.
                  When ``None``, embeddings are skipped and chunks are
                  stored without vectors (useful for testing without an
                  API key).
        chunk_size: Tokens per chunk (default 100).
        chunk_overlap: Overlap tokens between chunks (default 20).
        region: Region tag attached to every chunk's metadata.
        vector_store: The list to append indexed records into. Defaults
                      to the module-level ``VECTOR_STORE`` singleton.

    Returns:
        Dict with keys:
          - ``"document"``  : str path of the processed file
          - ``"chunks"``    : number of chunks produced
          - ``"indexed"``   : number of records added to the vector store
          - ``"status"``    : ``"indexed"`` on success, ``"failed"`` on error
          - ``"error"``     : error message string (only present on failure)

    Example::

        summary = process_uploaded_document(
            Path("uploads/policy.txt"),
            embed_fn=embed,
        )
        print(summary["chunks"], "chunks indexed")
    """
    store = vector_store if vector_store is not None else VECTOR_STORE

    try:
        # ── Stage 1: load ────────────────────────────────────────────────
        loader = DocumentLoader(default_region=region)
        document = loader.load_single_file(path, region=region)

        if document.status == "FAILED":
            return {
                "document": str(path),
                "chunks": 0,
                "indexed": 0,
                "status": "failed",
                "error": document.error or "Document loading failed",
            }

        # ── Stage 2: clean ───────────────────────────────────────────────
        cleaned = clean_text(document.text)

        if not cleaned.strip():
            return {
                "document": str(path),
                "chunks": 0,
                "indexed": 0,
                "status": "failed",
                "error": "Document produced no text after cleaning",
            }

        # ── Stage 3: chunk ───────────────────────────────────────────────
        raw_chunks = token_chunks(
            cleaned,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        # ── Stage 4: tag ─────────────────────────────────────────────────
        tagged = tag_chunks(
            source=document.source,
            chunks=raw_chunks,
            region=region,
        )

        # ── Stage 5 & 6: embed and index ─────────────────────────────────
        indexed_count = 0

        if embed_fn is not None:
            texts = [chunk["text"] for chunk in tagged]
            embeddings = embed_fn(texts)

            for chunk, embedding in zip(tagged, embeddings):
                record = {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": embedding,
                }
                store.append(record)
                indexed_count += 1
        else:
            # Store without embeddings (index-only mode, useful for testing)
            for chunk in tagged:
                record = {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": [],
                }
                store.append(record)
                indexed_count += 1

        return {
            "document": str(path),
            "chunks": len(tagged),
            "indexed": indexed_count,
            "status": "indexed",
        }

    except Exception as exc:
        return {
            "document": str(path),
            "chunks": 0,
            "indexed": 0,
            "status": "failed",
            "error": str(exc),
        }
