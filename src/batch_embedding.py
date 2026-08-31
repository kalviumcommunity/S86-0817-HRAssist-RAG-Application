"""Batch embedding pipeline with retry, cost tracking, and resume support.

HRS3.28 — Batch Embedding & Rate/Cost Management

This module turns a flat list of chunks into embedded records efficiently:
  - Batching    : sends many chunks per API request instead of one at a time
  - Skip-existing: chunks that already have an embedding are left untouched,
                   making every run safely resumable after a crash or pause
  - Retry/backoff: transient rate-limit and network errors are retried with
                   exponential backoff so the pipeline self-heals at scale
  - Run summary  : tracks skipped, embedded, failed, token usage, and
                   approximate cost so every run is auditable
"""

import time
from typing import Any, Dict, Generator, List, Optional, Set

from openai import RateLimitError, APIError

# ── Embedding cost constant ────────────────────────────────────────────────
# Replace this value with the price listed for your chosen embedding model.
# The default matches the gemini-embedding-001 approximate rate.
PRICE_PER_1K_TOKENS: float = 0.00002

# Default batch size: 64 chunks per API request is a safe starting point
# that stays within most providers' per-request size limits.
DEFAULT_BATCH_SIZE: int = 64


def batches(
    items: List[Any],
    size: int = DEFAULT_BATCH_SIZE,
) -> Generator[List[Any], None, None]:
    """Yield successive fixed-size slices of *items*.

    Splitting a large chunk list into batches reduces the number of items
    in each API request to a manageable size, avoids per-request token limits,
    and gives the pipeline natural checkpointing opportunities.

    Args:
        items: Any list to split — typically a list of chunk dicts.
        size: Maximum number of items per batch. Must be >= 1.

    Yields:
        Consecutive, non-overlapping sub-lists of *items*.

    Raises:
        ValueError: If *size* is less than 1.

    Example::

        for batch in batches(all_chunks, size=64):
            embed_and_store(batch)
    """
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def estimate_tokens(texts: List[str]) -> int:
    """Approximate token count for a list of texts.

    Uses a simple heuristic (words ÷ 0.75) when tiktoken is unavailable,
    or delegates to tiktoken when it is installed. The estimate is used for
    cost forecasting only — it does not need to be exact.

    Args:
        texts: Plain text strings to measure.

    Returns:
        Integer token estimate for the combined input.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts)
    except ImportError:
        # Fallback heuristic: ~0.75 tokens per word on average
        total_words = sum(len(t.split()) for t in texts)
        return max(1, int(total_words / 0.75))


def embed_with_retry(
    client: Any,
    model: str,
    texts: List[str],
    max_attempts: int = 5,
) -> Any:
    """Call the embeddings API with exponential backoff on transient errors.

    Transient failures — rate limits, network timeouts, temporary server
    errors — are expected at scale. This function retries up to
    *max_attempts* times, doubling the wait after every failure
    (1 s, 2 s, 4 s, 8 s, …). Permanent errors (authentication, invalid
    input) propagate immediately on the final attempt.

    Args:
        client: An OpenAI-compatible client instance.
        model: The embedding model identifier string.
        texts: A batch of text strings to embed.
        max_attempts: Maximum total tries before re-raising the last error.

    Returns:
        The raw API response object (``response.data`` holds the embeddings).

    Raises:
        Exception: Re-raises the last exception when all attempts are
            exhausted.

    Example::

        response = embed_with_retry(client, "gemini-embedding-001", texts)
        vectors = [item.embedding for item in response.data]
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return client.embeddings.create(model=model, input=texts)

        except (RateLimitError, APIError) as error:
            last_error = error
            if attempt == max_attempts - 1:
                raise
            wait_seconds = 2 ** attempt          # 1, 2, 4, 8, 16 …
            print(
                f"  [retry {attempt + 1}/{max_attempts - 1}] "
                f"error: {error} | waiting {wait_seconds}s"
            )
            time.sleep(wait_seconds)

        except Exception as error:
            # Non-transient errors (auth failures, bad input) — fail fast
            last_error = error
            if attempt == max_attempts - 1:
                raise
            wait_seconds = 2 ** attempt
            print(
                f"  [retry {attempt + 1}/{max_attempts - 1}] "
                f"unexpected error: {error} | waiting {wait_seconds}s"
            )
            time.sleep(wait_seconds)

    # Should not be reached, but satisfies type checkers
    if last_error:
        raise last_error
    raise RuntimeError("embed_with_retry exhausted all attempts")


def run_batch_embedding(
    client: Any,
    model: str,
    chunks: List[Dict[str, Any]],
    existing_ids: Optional[Set[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_attempts: int = 5,
    price_per_1k_tokens: float = PRICE_PER_1K_TOKENS,
) -> Dict[str, Any]:
    """Embed a corpus of chunks in batches and return a run summary.

    Pipeline steps
    --------------
    1. **Skip existing** — chunks whose ``id`` key is already in
       *existing_ids* are excluded from the API run. This makes every
       execution safely restartable: re-running after a crash embeds only
       the remaining chunks without wasting API credits on work already done.
    2. **Batch** — pending chunks are split into groups of *batch_size* so
       each API call stays within provider limits.
    3. **Embed with retry** — each batch is sent with exponential backoff to
       handle transient rate-limit and network errors.
    4. **Collect records** — successful embeddings are merged back with their
       source chunk text and metadata.
    5. **Track summary** — total, skipped, embedded, failed, token usage, and
       estimated cost are returned for auditing.

    Args:
        client: An OpenAI-compatible client instance.
        model: Embedding model identifier (e.g. ``"gemini-embedding-001"``).
        chunks: List of chunk dicts. Each must have at least a ``"text"`` key.
                An optional ``"id"`` key enables skip-existing logic.
        existing_ids: Set of chunk IDs that already have embeddings and should
                      be skipped. Pass ``None`` or an empty set to embed all.
        batch_size: Number of chunks per API request.
        max_attempts: Retry attempts per batch before marking it as failed.
        price_per_1k_tokens: Embedding cost per 1 000 input tokens in USD.

    Returns:
        A dict with keys:
          - ``"records"``          : list of embedded chunk records
          - ``"summary"``          : run statistics dict
          - ``"estimated_cost_usd"``: float rounded to 6 decimal places

    Example::

        result = run_batch_embedding(client, model, chunks,
                                     existing_ids={"chunk_0", "chunk_1"})
        print(result["summary"])
        print("cost:", result["estimated_cost_usd"])
    """
    if existing_ids is None:
        existing_ids = set()

    # ── Step 1: filter out already-embedded chunks ───────────────────────
    pending_chunks = [
        chunk for chunk in chunks
        if chunk.get("id") not in existing_ids
    ]

    summary: Dict[str, Any] = {
        "total_chunks": len(chunks),
        "skipped_existing": len(chunks) - len(pending_chunks),
        "embedded": 0,
        "failed": 0,
        "input_tokens": 0,
        "batches_processed": 0,
        "batches_failed": 0,
    }

    records: List[Dict[str, Any]] = []

    # ── Step 2 & 3: iterate batches, embed with retry ────────────────────
    for batch in batches(pending_chunks, size=batch_size):
        texts = [chunk["text"] for chunk in batch]
        token_count = estimate_tokens(texts)
        summary["input_tokens"] += token_count

        try:
            response = embed_with_retry(
                client=client,
                model=model,
                texts=texts,
                max_attempts=max_attempts,
            )

            # ── Step 4: merge embeddings back with chunk metadata ─────────
            for chunk, item in zip(batch, response.data):
                record: Dict[str, Any] = {
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "embedding": item.embedding,
                }
                if "id" in chunk:
                    record["id"] = chunk["id"]
                records.append(record)

            summary["embedded"] += len(response.data)
            summary["batches_processed"] += 1

        except Exception as error:
            # Keep failed batches visible in the summary instead of silently
            # dropping them — the caller can decide how to handle them.
            summary["failed"] += len(batch)
            summary["batches_failed"] += 1
            print(f"  [batch FAILED] {len(batch)} chunks | error: {error}")

    # ── Step 5: compute estimated cost ───────────────────────────────────
    estimated_cost = round(
        summary["input_tokens"] / 1000 * price_per_1k_tokens,
        6,
    )

    return {
        "records": records,
        "summary": summary,
        "estimated_cost_usd": estimated_cost,
    }


def print_run_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable run summary from a ``run_batch_embedding`` result.

    Args:
        result: The dict returned by :func:`run_batch_embedding`.
    """
    summary = result["summary"]
    cost = result["estimated_cost_usd"]

    print("\n" + "=" * 70)
    print("BATCH EMBEDDING RUN SUMMARY")
    print("=" * 70)
    print(f"  Total chunks        : {summary['total_chunks']}")
    print(f"  Skipped (existing)  : {summary['skipped_existing']}")
    print(f"  Embedded            : {summary['embedded']}")
    print(f"  Failed              : {summary['failed']}")
    print(f"  Input tokens        : {summary['input_tokens']}")
    print(f"  Batches processed   : {summary['batches_processed']}")
    print(f"  Batches failed      : {summary['batches_failed']}")
    print(f"  Estimated cost (USD): ${cost}")
    print("=" * 70)
