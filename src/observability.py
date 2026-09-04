"""Caching, logging, and usage monitoring for the HRAssist RAG pipeline.

HRS3.48 — Caching, Logging & Usage Monitoring

Observability turns a black-box RAG system into something you can trust,
debug, and improve. This module provides three layers:

  1. Query cache  — SHA-256 keyed in-memory cache with TTL expiry. Repeated
                    identical questions are served without re-embedding,
                    re-retrieving, or re-generating, saving latency and cost.

  2. Structured logging — every request is logged as a JSON line with enough
                    context to trace: question → retrieved chunks → answer →
                    token usage → latency → cache hit / miss.

  3. Usage monitoring — per-request cost estimation and an aggregated report
                    that surfaces total cost, cache hit rate, average latency,
                    and the most-expensive or most-frequent queries.

Design principles
-----------------
- The cache, logger, and usage store are module-level singletons so they
  accumulate state across the lifetime of the process. Tests inject fresh
  instances through function arguments to stay isolated.
- Cost rates are constants that can be overridden per-call; they are not
  hardcoded into the data so switching models does not require changing logs.
- Logs never contain embedding vectors, raw file bytes, or API keys — only
  the data needed for debugging and monitoring.
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Cost constants ────────────────────────────────────────────────────────
# Default approximate rates (USD per 1 000 tokens).
# Override at call-time via the ``input_cost_per_1k`` / ``output_cost_per_1k``
# parameters; these defaults match common low-cost chat model pricing.

MODEL_INPUT_COST_PER_1K: float  = 0.00015
MODEL_OUTPUT_COST_PER_1K: float = 0.00060

# Default cache TTL: 15 minutes
CACHE_TTL_SECONDS: int = 15 * 60


# ── Module-level singletons ───────────────────────────────────────────────

# In-memory query cache: { cache_key: {"created_at": float, "response": Any} }
_query_cache: Dict[str, Dict[str, Any]] = {}

# Append-only log of every RAG request this process has handled
_usage_log: List[Dict[str, Any]] = []

# Standard Python logger for the application
logger = logging.getLogger("rag_app")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1 — Query cache
# ─────────────────────────────────────────────────────────────────────────

def cache_key(question: str, filters: Optional[Dict[str, Any]] = None) -> str:
    """Compute a deterministic SHA-256 cache key for a question + filters pair.

    The question is normalised (stripped and lower-cased) before hashing so
    "How do I apply?" and "how do i apply?" share the same cache entry.

    Args:
        question: The user's question string.
        filters: Optional dict of metadata filters applied during retrieval
                 (e.g. ``{"region": "India"}``). These are part of the key
                 because the same question with different filters may produce
                 a different answer.

    Returns:
        A 64-character hex string.

    Example::

        key = cache_key("What is sick leave?", {"region": "India"})
    """
    raw = {
        "question": question.strip().lower(),
        "filters": filters or {},
    }
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()


def get_cached_answer(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    cache: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Return a cached response, or None if the entry is absent or expired.

    Entries older than ``ttl_seconds`` are evicted on read so stale answers
    are never returned to users.

    Args:
        question: The user's question.
        filters: Metadata filters that were part of the original request.
        ttl_seconds: Cache time-to-live in seconds.
        cache: Override the module-level cache (useful in tests).

    Returns:
        The cached response dict, or ``None`` on a cache miss / expiry.
    """
    store = cache if cache is not None else _query_cache
    key = cache_key(question, filters)
    entry = store.get(key)

    if entry is None:
        return None

    if time.time() - entry["created_at"] > ttl_seconds:
        store.pop(key, None)
        return None

    return entry["response"]


def save_cached_answer(
    question: str,
    response: Any,
    filters: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> str:
    """Store a response in the cache under its question + filters key.

    Args:
        question: The user's question.
        response: The full response dict to cache.
        filters: Metadata filters that were part of the request.
        cache: Override the module-level cache (useful in tests).

    Returns:
        The cache key string (useful for logging or debugging).
    """
    store = cache if cache is not None else _query_cache
    key = cache_key(question, filters)
    store[key] = {
        "created_at": time.time(),
        "response": response,
    }
    return key


def invalidate_cache(
    question: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> int:
    """Remove one or all entries from the cache.

    When ``question`` is provided, only that entry is removed.
    When ``question`` is None, the entire cache is cleared.

    Args:
        question: The question whose entry should be removed, or None to
                  clear everything.
        filters: Filters used to compute the key (only used when question
                 is provided).
        cache: Override the module-level cache.

    Returns:
        Number of entries removed.
    """
    store = cache if cache is not None else _query_cache
    if question is None:
        count = len(store)
        store.clear()
        return count
    key = cache_key(question, filters)
    if key in store:
        del store[key]
        return 1
    return 0


def cache_size(cache: Optional[Dict[str, Any]] = None) -> int:
    """Return the number of entries currently in the cache."""
    store = cache if cache is not None else _query_cache
    return len(store)


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2 — Structured logging
# ─────────────────────────────────────────────────────────────────────────

def new_request_id() -> str:
    """Generate a unique request identifier (UUID4 hex string)."""
    return uuid.uuid4().hex


def log_rag_request(
    record: Dict[str, Any],
    app_logger: Optional[logging.Logger] = None,
    usage_log: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Log a completed RAG request as a structured JSON line.

    Appends the record to the in-memory usage log AND emits a JSON line via
    the Python logging framework. Both sinks are used so the data is
    accessible programmatically (for ``summarize_usage``) and in log files
    or log aggregation systems.

    Expected keys in ``record``
    ---------------------------
    Required:
      - ``request_id``     : str — unique identifier for this request
      - ``question``       : str — original user question
      - ``answer``         : str — final answer text
      - ``sources``        : list — metadata dicts of retrieved chunks
      - ``cache_hit``      : bool — True when the answer came from cache
      - ``input_tokens``   : int — tokens in the prompt
      - ``output_tokens``  : int — tokens in the answer
      - ``estimated_cost`` : float — USD cost estimate
      - ``latency_ms``     : float — end-to-end request latency
    Optional:
      - ``status``         : str — "answered" / "refused_*" / "cached"
      - ``model``          : str — model identifier used for generation
      - ``retrieval_score``: float — top chunk similarity score

    Args:
        record: Dict containing the fields above.
        app_logger: Override the module-level logger.
        usage_log: Override the module-level usage log list.
    """
    log = app_logger if app_logger is not None else logger
    store = usage_log if usage_log is not None else _usage_log

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": record.get("request_id", new_request_id()),
        "question": record.get("question", ""),
        "answer_preview": str(record.get("answer", ""))[:180],
        "sources": record.get("sources", []),
        "cache_hit": record.get("cache_hit", False),
        "input_tokens": record.get("input_tokens", 0),
        "output_tokens": record.get("output_tokens", 0),
        "estimated_cost": record.get("estimated_cost", 0.0),
        "latency_ms": record.get("latency_ms", 0.0),
        "status": record.get("status", "answered"),
        "model": record.get("model", ""),
        "retrieval_score": record.get("retrieval_score"),
    }

    store.append(entry)
    log.info(json.dumps(entry))


def get_usage_log(
    usage_log: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return a copy of the accumulated usage log entries."""
    store = usage_log if usage_log is not None else _usage_log
    return list(store)


def clear_usage_log(usage_log: Optional[List[Dict[str, Any]]] = None) -> int:
    """Clear all usage log entries and return the count removed."""
    store = usage_log if usage_log is not None else _usage_log
    count = len(store)
    store.clear()
    return count


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3 — Cost estimation
# ─────────────────────────────────────────────────────────────────────────

def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float = MODEL_INPUT_COST_PER_1K,
    output_cost_per_1k: float = MODEL_OUTPUT_COST_PER_1K,
) -> float:
    """Estimate the USD cost of a single LLM call from token counts.

    Args:
        input_tokens: Number of tokens in the prompt (context + question).
        output_tokens: Number of tokens in the generated answer.
        input_cost_per_1k: Cost per 1 000 input tokens.
        output_cost_per_1k: Cost per 1 000 output tokens.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places.

    Example::

        cost = estimate_cost(800, 150)
        # → 0.000210 USD at default rates
    """
    input_cost  = (input_tokens  / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return round(input_cost + output_cost, 6)


def build_usage_metadata(
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool = False,
    input_cost_per_1k: float = MODEL_INPUT_COST_PER_1K,
    output_cost_per_1k: float = MODEL_OUTPUT_COST_PER_1K,
) -> Dict[str, Any]:
    """Return a usage metadata dict for embedding in a RAG response.

    Attaches token counts, cost estimate, and cache status directly to the
    response so callers can inspect usage without consulting separate logs.

    Args:
        input_tokens: Prompt token count.
        output_tokens: Answer token count.
        cache_hit: Whether the answer was served from cache.
        input_cost_per_1k: Override the default input token rate.
        output_cost_per_1k: Override the default output token rate.

    Returns:
        Dict with ``input_tokens``, ``output_tokens``, ``estimated_cost``,
        and ``cache_hit`` keys.
    """
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimate_cost(
            input_tokens, output_tokens,
            input_cost_per_1k, output_cost_per_1k,
        ),
        "cache_hit": cache_hit,
    }


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4 — Usage report
# ─────────────────────────────────────────────────────────────────────────

def summarize_usage(
    log_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate log records into a usage summary report.

    Useful for debugging, cost monitoring, and identifying patterns like
    frequent repeated questions (cache candidates), expensive queries, or
    high refusal rates.

    Args:
        log_records: List of log entry dicts. Defaults to the module-level
                     usage log when None.

    Returns:
        Dict with keys:
          - ``total_requests``        : int
          - ``cache_hits``            : int
          - ``cache_hit_rate``        : float in [0.0, 1.0]
          - ``total_input_tokens``    : int
          - ``total_output_tokens``   : int
          - ``total_estimated_cost``  : float (USD)
          - ``average_latency_ms``    : float
          - ``status_breakdown``      : dict mapping status → count
          - ``top_questions``         : list of (question, count) pairs,
                                        most frequent first, up to 5

    Example::

        report = summarize_usage()
        print(f"Total cost: ${report['total_estimated_cost']:.6f}")
        print(f"Cache hit rate: {report['cache_hit_rate']:.0%}")
    """
    records = log_records if log_records is not None else list(_usage_log)

    total = len(records)

    if total == 0:
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_estimated_cost": 0.0,
            "average_latency_ms": 0.0,
            "status_breakdown": {},
            "top_questions": [],
        }

    cache_hits        = sum(1 for r in records if r.get("cache_hit", False))
    total_input       = sum(r.get("input_tokens",    0) for r in records)
    total_output      = sum(r.get("output_tokens",   0) for r in records)
    total_cost        = sum(r.get("estimated_cost",  0.0) for r in records)
    total_latency     = sum(r.get("latency_ms",      0.0) for r in records)

    # Status breakdown
    status_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        status_counts[r.get("status", "answered")] += 1

    # Top-5 most frequent questions
    question_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        q = r.get("question", "").strip()
        if q:
            question_counts[q] += 1
    top_questions = sorted(
        question_counts.items(), key=lambda kv: kv[1], reverse=True
    )[:5]

    return {
        "total_requests": total,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_estimated_cost": round(total_cost, 6),
        "average_latency_ms": round(total_latency / total, 2),
        "status_breakdown": dict(status_counts),
        "top_questions": top_questions,
    }


def print_usage_report(
    log_records: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print a human-readable usage summary to stdout.

    Args:
        log_records: Optional override for the module-level usage log.
    """
    report = summarize_usage(log_records)

    print("\n" + "=" * 70)
    print("RAG USAGE REPORT")
    print("=" * 70)
    print(f"  Total requests        : {report['total_requests']}")
    print(f"  Cache hits            : {report['cache_hits']} "
          f"({report['cache_hit_rate']:.0%})")
    print(f"  Total input tokens    : {report['total_input_tokens']}")
    print(f"  Total output tokens   : {report['total_output_tokens']}")
    print(f"  Total estimated cost  : ${report['total_estimated_cost']:.6f}")
    print(f"  Average latency       : {report['average_latency_ms']:.1f} ms")

    if report["status_breakdown"]:
        print("\n  Status breakdown:")
        for status, count in sorted(report["status_breakdown"].items()):
            print(f"    {status:<30} {count}")

    if report["top_questions"]:
        print("\n  Top questions:")
        for question, count in report["top_questions"]:
            print(f"    [{count:>3}x]  {question[:60]}")

    print("=" * 70)
