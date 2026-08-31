"""Chunk re-ranking for precision in RAG retrieval pipelines.

HRS3.35 — Chunk Re-Ranking for Precision

Vector retrieval is fast and casts a wide net, but the first cosine-similarity
ordering is not always the best final ordering. Re-ranking adds a focused
second pass over a small candidate set so the most relevant chunks float to
the top before they are sent to the language model.

The two-stage pattern
---------------------
  Stage 1 — retrieve:  fast vector search over the full corpus, returns k
                        candidates (k is larger than the final context size).
  Stage 2 — re-rank:   score each candidate more carefully against the query,
                        then keep only the final top-n chunks.

This module provides three re-ranking strategies:

  1. Keyword overlap   — lightweight, zero-latency, no API calls required.
                         Counts how many query terms appear in the chunk text.
                         Useful as a baseline or tie-breaker.

  2. Pluggable scorer  — ``rerank()`` accepts any callable that takes
                         (query, chunk) and returns a float. Swap in a
                         cross-encoder, a semantic model, or a custom heuristic
                         without changing the pipeline code.

  3. LLM-based scorer  — ``rerank_with_llm()`` sends a structured prompt to
                         an LLM asking it to score each candidate from 0–10.
                         This is the most accurate strategy but adds one LLM
                         call per candidate, so keep the candidate set small.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Scoring strategies ────────────────────────────────────────────────────

def keyword_overlap_score(query: str, chunk: Dict[str, Any]) -> float:
    """Score a chunk by the fraction of unique query terms it contains.

    Converts both query and chunk text to lower-case, tokenises on whitespace,
    and returns the proportion of unique query tokens that appear at least once
    in the chunk. The result is in [0.0, 1.0].

    This is the simplest possible re-ranker: no model, no API call, zero
    latency. Use it as a baseline to compare against heavier approaches, or
    as a cheap tie-breaker after vector retrieval.

    Args:
        query: The user's question or search string.
        chunk: A retrieval result dict that must contain a ``"text"`` key.

    Returns:
        Float in [0.0, 1.0]. 1.0 means every unique query word appears in
        the chunk. 0.0 means no query word is present, or the query is empty.

    Example::

        score = keyword_overlap_score(
            "sick leave policy",
            {"text": "Employees may apply for sick leave under company policy."}
        )
        # → 0.666... (2 of 3 unique query words found: "sick", "leave")
    """
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return 0.0
    chunk_text = chunk.get("text", "").lower()
    matches = sum(1 for token in query_tokens if token in chunk_text)
    return matches / len(query_tokens)


# ── Core re-ranking function ──────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    score_fn: Callable[[str, Dict[str, Any]], float],
    final_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-rank a candidate list using a pluggable scoring function.

    Takes the output of vector retrieval (``candidates``), applies
    ``score_fn`` to each candidate, re-sorts by the new score, and returns
    the top ``final_k`` chunks as the final context alongside the full
    re-ranked list for comparison.

    The scoring function is injected so you can swap strategies — keyword
    overlap, a cross-encoder model, an LLM call — without changing this
    function.

    Args:
        query: The user's question, passed verbatim to ``score_fn``.
        candidates: List of retrieval result dicts. Each must contain at
            minimum ``"text"`` and ``"metadata"`` keys. The ``"score"`` key
            (vector similarity) is preserved alongside the new re-rank score.
        score_fn: Callable ``(query, chunk) -> float``. Higher means more
            relevant. The function may be deterministic (keyword overlap) or
            call an external model.
        final_k: Number of top chunks to keep as the final context. Must be
            >= 1 and <= len(candidates).

    Returns:
        A tuple of ``(final_context, reranked_all)`` where:
          - ``final_context``: top ``final_k`` chunks after re-ranking, each
            with a ``"rerank_score"`` and ``"rerank_rank"`` key added.
          - ``reranked_all``: all candidates sorted by re-rank score, also
            with ``"rerank_score"`` and ``"rerank_rank"`` added. Useful for
            the before/after comparison.

    Raises:
        ValueError: If ``final_k`` < 1 or ``final_k`` > len(candidates),
            or if ``candidates`` is empty.

    Example::

        candidates = retrieve(query, corpus, embed_fn, k=10)
        final_context, reranked_all = rerank(
            query, candidates, keyword_overlap_score, final_k=3
        )
        compare_reranking(query, candidates[:3], final_context)
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if final_k < 1:
        raise ValueError(f"final_k must be >= 1, got {final_k}")
    if final_k > len(candidates):
        raise ValueError(
            f"final_k ({final_k}) cannot exceed number of candidates "
            f"({len(candidates)})"
        )

    # Score each candidate with the injected function
    scored = []
    for chunk in candidates:
        rerank_score = score_fn(query, chunk)
        scored.append({**chunk, "rerank_score": rerank_score})

    # Sort descending by rerank score (ties broken by original vector score)
    scored.sort(
        key=lambda item: (item["rerank_score"], item.get("score", 0.0)),
        reverse=True,
    )

    # Attach re-rank positions
    for position, item in enumerate(scored, start=1):
        item["rerank_rank"] = position

    reranked_all = scored
    final_context = scored[:final_k]

    return final_context, reranked_all


def rerank_with_llm(
    query: str,
    candidates: List[Dict[str, Any]],
    llm_client: Any,
    model: str,
    final_k: int,
    max_score: float = 10.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-rank candidates by asking an LLM to score each one against the query.

    Sends a structured prompt asking the model to rate relevance on a 0–10
    scale. This is the most accurate re-ranking strategy but adds one LLM
    call per candidate, so keep ``candidates`` small (typically 10–20 chunks).

    The prompt follows the assignment pattern::

        Score how relevant this chunk is to the query from 0 to 10.
        Query: <query>
        Chunk: <chunk text>
        Return only the number.

    Unparseable responses default to 0.0 so a single bad response does not
    break the pipeline — it just pushes that chunk to the bottom.

    Args:
        query: The user's question.
        candidates: Retrieval result dicts with ``"text"`` and ``"metadata"``.
        llm_client: An OpenAI-compatible client with a
            ``chat.completions.create`` method.
        model: Chat model identifier (e.g. ``"gemini-3.1-flash-lite"``).
        final_k: Number of top chunks to return as final context.
        max_score: Upper bound for score normalisation (default 10.0).

    Returns:
        Same ``(final_context, reranked_all)`` tuple as ``rerank()``, with
        ``"rerank_score"`` containing the raw LLM score (0–10) and
        ``"rerank_rank"`` the 1-based position after LLM re-ranking.

    Raises:
        ValueError: If ``final_k`` < 1, ``final_k`` > len(candidates), or
            ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if final_k < 1:
        raise ValueError(f"final_k must be >= 1, got {final_k}")
    if final_k > len(candidates):
        raise ValueError(
            f"final_k ({final_k}) cannot exceed number of candidates "
            f"({len(candidates)})"
        )

    def _llm_score(q: str, chunk: Dict[str, Any]) -> float:
        prompt = (
            f"Score how relevant this chunk is to the query from 0 to {int(max_score)}.\n\n"
            f"Query: {q}\n"
            f"Chunk: {chunk.get('text', '')}\n\n"
            f"Return only the number."
        )
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            return float(raw)
        except (ValueError, AttributeError, IndexError):
            # Unparseable or API error → lowest priority
            return 0.0

    return rerank(query, candidates, _llm_score, final_k)


# ── Comparison helpers ────────────────────────────────────────────────────

def compare_reranking(
    query: str,
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
    text_preview_length: int = 120,
) -> None:
    """Print a before/after comparison of retrieval and re-ranking results.

    Shows the original vector-retrieval order alongside the re-ranked order
    for the same query so you can see whether re-ranking improves precision.

    Args:
        query: The user's question string, shown as header.
        before: The first ``final_k`` candidates in original vector order
                (i.e. ``candidates[:final_k]``).
        after: The final context returned by ``rerank()`` or
               ``rerank_with_llm()``.
        text_preview_length: Characters of chunk text to display per entry.
    """
    def _show(label: str, rows: List[Dict[str, Any]]) -> None:
        print(f"\n{label}")
        print("-" * 60)
        for rank, item in enumerate(rows, start=1):
            print(f"  rank          : {rank}")
            print(f"  vector_score  : {round(item.get('score', 0.0), 4)}")
            rerank_score = item.get("rerank_score")
            if rerank_score is not None:
                print(f"  rerank_score  : {round(rerank_score, 4)}")
            print(f"  source        : {item.get('metadata', {}).get('source', 'unknown')}")
            print(f"  text          : {item.get('text', '')[:text_preview_length]}")

    print("\n" + "=" * 70)
    print("RE-RANKING COMPARISON")
    print("=" * 70)
    print(f'query: "{query}"')

    _show("BEFORE RE-RANKING  (vector order, top-k slice)", before)
    _show("AFTER RE-RANKING   (re-ranked order, final context)", after)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(
        "  If re-ranking is working, the AFTER list should contain chunks\n"
        "  that more directly answer the query — not just broadly related ones.\n"
        "\n"
        "  Cost / latency trade-off:\n"
        "    - Keyword overlap : zero cost, zero latency, rough precision.\n"
        "    - LLM re-ranker   : one LLM call per candidate — keep the\n"
        "      candidate set small (10–20 chunks) and measure whether answer\n"
        "      quality justifies the extra latency before deploying."
    )


def build_reranking_report(
    query: str,
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a structured dict summarising the before/after reranking result.

    Useful for programmatic inspection or logging without relying on stdout.

    Args:
        query: The user's question.
        before: Candidates in original vector order (top final_k slice).
        after: Final context returned by ``rerank()``.

    Returns:
        Dict with keys:
          - ``"query"``             : the query string
          - ``"before_sources"``    : ordered list of sources before re-ranking
          - ``"after_sources"``     : ordered list of sources after re-ranking
          - ``"order_changed"``     : True when before and after differ
          - ``"top_source_changed"``: True when rank-1 source is different
    """
    before_sources = [r.get("metadata", {}).get("source", "") for r in before]
    after_sources  = [r.get("metadata", {}).get("source", "") for r in after]

    return {
        "query": query,
        "before_sources": before_sources,
        "after_sources": after_sources,
        "order_changed": before_sources != after_sources,
        "top_source_changed": (
            bool(before_sources) and bool(after_sources)
            and before_sources[0] != after_sources[0]
        ),
    }
