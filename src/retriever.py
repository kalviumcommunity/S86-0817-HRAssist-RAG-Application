"""Top-k similarity retrieval for RAG pipelines.

HRS3.32 — Similarity Search & Top-K Retrieval

Retrieval is the bridge between a user query and the language model. This
module implements the four-step retrieval path:

  1. Embed the query with the same model used for corpus chunks.
  2. Score every chunk embedding against the query vector with cosine similarity.
  3. Return the k highest-scoring chunks — the "top-k" results.
  4. Attach rank, score, text, and metadata to each result so the caller
     can cite, filter, or inspect what was retrieved.

The query and corpus MUST use the same embedding model. Mixing models
produces vectors in different spaces and makes similarity scores unreliable,
even though cosine similarity will still return a number.
"""

from typing import Any, Callable, Dict, List, Optional

from src.similarity import cosine_similarity


# ── Core retrieval function ───────────────────────────────────────────────

def retrieve(
    query: str,
    chunk_records: List[Dict[str, Any]],
    embed_fn: Callable[[List[str]], List[List[float]]],
    k: int = 3,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Embed a query and return the top-k most similar chunks.

    This is the standard RAG retrieval step. The query is embedded with the
    same model used to build the corpus, then every chunk is scored by
    cosine similarity and the k highest-scoring chunks are returned.

    Args:
        query: The user's natural-language question or search string.
        chunk_records: Embedded corpus records. Each must have at minimum:
            - ``"embedding"`` : the chunk's vector
            - ``"text"``      : the raw chunk text
            - ``"metadata"``  : a dict that should contain at least ``"source"``
        embed_fn: A callable that accepts a list of strings and returns a
            list of float vectors — one vector per input string. Must use
            the *same model* that was used to embed the corpus chunks.
        k: Number of top results to return. Must be >= 1.
        score_threshold: When provided, only chunks whose cosine similarity
            score is >= this value are returned (applied after top-k slicing).
            Useful for dropping low-confidence matches from small corpora.

    Returns:
        List of result dicts sorted by descending score, length <= k. Each
        dict contains:
          - ``"rank"``     : 1-based position in the ranked list
          - ``"score"``    : cosine similarity rounded to 4 decimal places
          - ``"text"``     : the chunk text
          - ``"metadata"`` : the chunk metadata dict (source, chunk_index, …)

    Raises:
        ValueError: If ``k`` < 1 or ``chunk_records`` is empty.

    Example::

        results = retrieve(
            "How do I reset my password?",
            corpus_records,
            embed_fn=embed,   # same model used for corpus
            k=3,
        )
        for r in results:
            print(r["rank"], r["score"], r["metadata"]["source"])
            print(r["text"][:120])
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not chunk_records:
        raise ValueError("chunk_records must not be empty")

    # ── Step 1: embed the query ───────────────────────────────────────────
    # The embed function must use the same model as the corpus. If models
    # differ, scores are still numbers but the ranking cannot be trusted.
    query_vector = embed_fn([query])[0]

    # ── Step 2: score every chunk ─────────────────────────────────────────
    scored: List[Dict[str, Any]] = []
    for record in chunk_records:
        score = cosine_similarity(query_vector, record["embedding"])
        scored.append(
            {
                "score": score,
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
            }
        )

    # ── Step 3: sort descending and take top-k ────────────────────────────
    scored.sort(key=lambda item: item["score"], reverse=True)
    top_k = scored[:k]

    # ── Step 4: optionally filter by score threshold ──────────────────────
    if score_threshold is not None:
        top_k = [item for item in top_k if item["score"] >= score_threshold]

    # ── Step 5: attach rank and round score ───────────────────────────────
    results = []
    for rank, item in enumerate(top_k, start=1):
        results.append(
            {
                "rank": rank,
                "score": round(item["score"], 4),
                "text": item["text"],
                "metadata": item["metadata"],
            }
        )

    return results


def retrieve_at_k_values(
    query: str,
    chunk_records: List[Dict[str, Any]],
    embed_fn: Callable[[List[str]], List[List[float]]],
    k_values: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    """Run retrieval at multiple k values to show how context changes with k.

    A small k is cheaper and focused on the best match. A larger k gives
    the model more grounding context but risks adding loosely-related filler
    that consumes context-window budget. This helper lets you inspect the
    trade-off with real queries before committing to a fixed k.

    The query is embedded only once; scoring is shared across all k values
    so this is efficient even for large k_values lists.

    Args:
        query: The user's natural-language question.
        chunk_records: Embedded corpus records (same format as ``retrieve``).
        embed_fn: Callable that returns a float vector per input string,
            using the same model as the corpus.
        k_values: List of k values to evaluate. Must all be >= 1. Duplicates
            are allowed (they produce identical results).

    Returns:
        Dict mapping each k value to its retrieval result list, in the same
        format as ``retrieve``.

    Raises:
        ValueError: If any k value is < 1 or ``chunk_records`` is empty.

    Example::

        comparison = retrieve_at_k_values(
            "How do I apply for sick leave?",
            corpus_records,
            embed_fn=embed,
            k_values=[1, 3, 5],
        )
        for k, results in comparison.items():
            print(f"k={k}: {[r['metadata']['source'] for r in results]}")
    """
    if not chunk_records:
        raise ValueError("chunk_records must not be empty")
    if any(k < 1 for k in k_values):
        raise ValueError("all k values must be >= 1")

    # Embed once, share across all k values
    query_vector = embed_fn([query])[0]

    # Score all chunks once
    scored: List[Dict[str, Any]] = []
    for record in chunk_records:
        score = cosine_similarity(query_vector, record["embedding"])
        scored.append(
            {
                "score": score,
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)

    # Slice to each k, attach rank
    results_by_k: Dict[int, List[Dict[str, Any]]] = {}
    for k in k_values:
        top_k = scored[:k]
        results_by_k[k] = [
            {
                "rank": rank,
                "score": round(item["score"], 4),
                "text": item["text"],
                "metadata": item["metadata"],
            }
            for rank, item in enumerate(top_k, start=1)
        ]

    return results_by_k


# ── Display helpers ───────────────────────────────────────────────────────

def print_retrieval_results(
    query: str,
    results: List[Dict[str, Any]],
    text_preview_length: int = 120,
) -> None:
    """Print a formatted retrieval result list to stdout.

    This output is the first retrieval proof: the system can turn a user
    question into ranked, citable source chunks.

    Args:
        query: The original query string, shown as a header.
        results: The list returned by ``retrieve``.
        text_preview_length: How many characters of each chunk's text to show.
    """
    print("\n" + "=" * 70)
    print(f'RETRIEVAL RESULTS  k={len(results)}')
    print("=" * 70)
    print(f'query: "{query}"')
    print("-" * 70)

    if not results:
        print("  (no results returned)")
        return

    for result in results:
        metadata = result["metadata"]
        print(f"\nrank        : {result['rank']}")
        print(f"score       : {result['score']}")
        print(f"source      : {metadata.get('source', 'unknown')}")
        print(f"chunk_index : {metadata.get('chunk_index', 'n/a')}")
        print(f"text        : {result['text'][:text_preview_length]}")

    print("=" * 70)


def print_k_comparison(
    query: str,
    results_by_k: Dict[int, List[Dict[str, Any]]],
) -> None:
    """Print a side-by-side comparison of retrieval results at multiple k values.

    Use this to inspect whether extra chunks added by larger k values are
    genuinely helpful context or loosely-related filler.

    Args:
        query: The original query string.
        results_by_k: The dict returned by ``retrieve_at_k_values``.
    """
    print("\n" + "=" * 70)
    print("TOP-K COMPARISON")
    print("=" * 70)
    print(f'query: "{query}"')

    for k in sorted(results_by_k):
        results = results_by_k[k]
        print(f"\n  k={k}:")
        for r in results:
            source = r["metadata"].get("source", "unknown")
            preview = r["text"][:80].replace("\n", " ")
            print(f"    [{r['rank']}] score={r['score']}  {source}  \"{preview}\"")

    print("=" * 70)
    print(
        "\nWhat to observe:\n"
        "  - Does rank 1 stay consistent as k grows?\n"
        "  - Are the extra chunks (ranks 2+) genuinely related or just filler?\n"
        "  - How much score drops between rank 1 and the last rank?\n"
        "  The best k depends on chunk size, corpus quality, answer complexity,\n"
        "  and the language model's available context window."
    )
