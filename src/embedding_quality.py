"""Embedding quality checks and sanity tests for retrieval pipelines.

HRS3.29 — Embedding Quality Checks & Sanity Tests

A broken embedding pipeline can still produce vectors — they may just come
from the wrong model, be attached to the wrong chunk, or be ranked with the
wrong metric. This module provides fast smoke tests that catch these problems
by checking that *known-related* texts rank above *known-unrelated* ones.

Three categories of check are provided:
  1. Retrieval relevance  — does the expected source appear at rank 1?
  2. Dimension consistency— do all vectors share the same length?
  3. Model mismatch risk  — are query and corpus vectors from the same space?

These are not full evaluation benchmarks; they are the minimum sanity layer
every RAG pipeline should run before serving users.
"""

from typing import Any, Dict, List, Optional, Sequence

from src.similarity import cosine_similarity, rank_chunks


# ── Default HR test cases ─────────────────────────────────────────────────
# Each case pairs a realistic HR query with the metadata source that should
# rank first. Build your own cases whenever you add a new document source.
DEFAULT_TEST_CASES: List[Dict[str, Any]] = [
    {
        "query": "How can a learner reset their password?",
        "expected_source": "account-guide.md",
        "note": "Password/login recovery should rank above unrelated content.",
    },
    {
        "query": "When does the cafeteria menu change?",
        "expected_source": "campus-guide.md",
        "note": "Campus facilities content should rank above HR policy chunks.",
    },
    {
        "query": "How do I apply for annual leave?",
        "expected_source": "employee_leave_policy.txt",
        "note": "Leave application query should retrieve the policy document.",
    },
    {
        "query": "What are the steps for sick leave submission?",
        "expected_source": "employee_leave_policy.txt",
        "note": (
            "Sick leave is a specific sub-topic; a generic leave chunk could "
            "rank above it — this is a known edge case to watch."
        ),
    },
]


def run_sanity_checks(
    test_cases: List[Dict[str, Any]],
    chunk_records: List[Dict[str, Any]],
    query_embeddings: List[List[float]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Run retrieval relevance checks for a list of known query-chunk pairs.

    For each test case the query embedding is ranked against all chunk records.
    The check passes when the top-ranked chunk's metadata source matches the
    expected source. The result also captures whether the expected source
    appears *anywhere* in the top-k results, which is useful when the corpus
    contains multiple equally-valid chunks for the same query.

    Args:
        test_cases: List of dicts with at minimum ``"query"`` and
            ``"expected_source"`` keys. An optional ``"note"`` key can carry
            a human-readable description of the expected behaviour or risk.
        chunk_records: Embedded corpus records. Each must have ``"embedding"``
            and ``"metadata"`` keys; ``metadata`` must contain ``"source"``.
        query_embeddings: Pre-computed embedding vectors for each test case
            query, in the same order as *test_cases*.
        top_k: How many top-ranked chunks to inspect when checking whether
            the expected source appears in the neighbourhood (not just rank 1).

    Returns:
        List of result dicts, one per test case, with keys:
          - ``"query"``           : the query string
          - ``"expected_source"`` : the source that should rank first
          - ``"top_source"``      : the actual source of the top-ranked chunk
          - ``"top_score"``       : cosine similarity of the top result
          - ``"top_k_sources"``   : sources of the top-k ranked chunks
          - ``"passed"``          : True when top source == expected source
          - ``"in_top_k"``        : True when expected source appears in top-k
          - ``"note"``            : forwarded from the test case if present

    Raises:
        ValueError: If *test_cases* and *query_embeddings* differ in length.
    """
    if len(test_cases) != len(query_embeddings):
        raise ValueError(
            f"test_cases ({len(test_cases)}) and query_embeddings "
            f"({len(query_embeddings)}) must have the same length"
        )

    if not chunk_records:
        raise ValueError("chunk_records must not be empty")

    results = []

    for case, query_embedding in zip(test_cases, query_embeddings):
        ranked = rank_chunks(query_embedding, chunk_records, top_k=top_k)

        top = ranked[0]
        top_source = top["metadata"].get("source", "")
        top_score = round(top["score"], 4)
        top_k_sources = [r["metadata"].get("source", "") for r in ranked]
        expected = case["expected_source"]

        results.append(
            {
                "query": case["query"],
                "expected_source": expected,
                "top_source": top_source,
                "top_score": top_score,
                "top_k_sources": top_k_sources,
                "passed": top_source == expected,
                "in_top_k": expected in top_k_sources,
                "note": case.get("note", ""),
            }
        )

    return results


def build_sanity_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate individual check results into a structured sanity report.

    Args:
        results: The list returned by :func:`run_sanity_checks`.

    Returns:
        A dict with keys:
          - ``"total"``   : number of test cases
          - ``"passed"``  : number where top result matched expected source
          - ``"failed"``  : number where top result did not match
          - ``"in_top_k_only"`` : passed in_top_k but not at rank 1
          - ``"rows"``    : the original results list (for detailed printing)
    """
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    in_top_k_only = sum(
        1 for r in results if r["in_top_k"] and not r["passed"]
    )

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "in_top_k_only": in_top_k_only,
        "rows": results,
    }


def check_dimension_consistency(
    chunk_records: List[Dict[str, Any]],
    query_embeddings: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """Verify that all embedding vectors share the same dimension.

    Mixing vectors from different models or truncating embeddings produces
    vectors of different lengths. Cosine similarity between vectors of
    different lengths raises a ``ValueError``, so this check should always
    run before retrieval.

    Args:
        chunk_records: Embedded corpus records with an ``"embedding"`` key.
        query_embeddings: Optional list of query vectors to check alongside
            the corpus. When provided, their dimensions are validated against
            the corpus dimension.

    Returns:
        Dict with keys:
          - ``"consistent"``       : True when all dimensions match
          - ``"corpus_dimension"`` : dimension found in the first chunk
          - ``"mismatched_chunks"``: indices of chunks with a different size
          - ``"query_dimension_ok"``: True when query vectors match (or None
                                      when no query vectors were provided)
    """
    if not chunk_records:
        raise ValueError("chunk_records must not be empty")

    corpus_dim = len(chunk_records[0]["embedding"])
    mismatched = [
        idx
        for idx, rec in enumerate(chunk_records)
        if len(rec["embedding"]) != corpus_dim
    ]

    query_ok: Optional[bool] = None
    if query_embeddings:
        query_ok = all(
            len(vec) == corpus_dim for vec in query_embeddings
        )

    return {
        "consistent": len(mismatched) == 0 and (query_ok is not False),
        "corpus_dimension": corpus_dim,
        "mismatched_chunks": mismatched,
        "query_dimension_ok": query_ok,
    }


def detect_near_duplicate_chunks(
    chunk_records: List[Dict[str, Any]],
    threshold: float = 0.98,
) -> List[Dict[str, Any]]:
    """Find pairs of chunks whose embeddings are near-identical.

    Near-duplicates inflate retrieval precision metrics and may cause the
    same information to appear multiple times in a generated answer. A
    cosine similarity above *threshold* (default 0.98) is a strong signal
    that two chunks are either exact duplicates or copies with trivial edits.

    Args:
        chunk_records: Embedded corpus records.
        threshold: Cosine similarity above which two chunks are flagged.

    Returns:
        List of dicts, each with keys ``"index_a"``, ``"index_b"``,
        ``"score"``, ``"text_a"``, and ``"text_b"``.
    """
    duplicates = []
    n = len(chunk_records)
    for i in range(n):
        for j in range(i + 1, n):
            score = cosine_similarity(
                chunk_records[i]["embedding"],
                chunk_records[j]["embedding"],
            )
            if score >= threshold:
                duplicates.append(
                    {
                        "index_a": i,
                        "index_b": j,
                        "score": round(score, 4),
                        "text_a": chunk_records[i].get("text", "")[:80],
                        "text_b": chunk_records[j].get("text", "")[:80],
                    }
                )
    return duplicates


def print_sanity_report(report: Dict[str, Any]) -> None:
    """Print a human-readable sanity report to stdout.

    Args:
        report: The dict returned by :func:`build_sanity_report`.
    """
    rows = report["rows"]

    print("\n" + "=" * 70)
    print("EMBEDDING QUALITY SANITY REPORT")
    print("=" * 70)
    print(
        f"  tests: {report['total']}  "
        f"passed: {report['passed']}  "
        f"failed: {report['failed']}  "
        f"in_top_k_only: {report['in_top_k_only']}"
    )
    print("-" * 70)

    for row in rows:
        status = "PASS" if row["passed"] else ("NEAR" if row["in_top_k"] else "FAIL")
        print(f"\n[{status}] {row['query']}")
        print(f"  expected : {row['expected_source']}")
        print(f"  top hit  : {row['top_source']}  (score={row['top_score']})")
        print(f"  top-k    : {row['top_k_sources']}")
        if row["note"]:
            print(f"  note     : {row['note']}")

    print("\n" + "=" * 70)
    print("WHAT TO INSPECT ON FAILURE")
    print("=" * 70)
    print(
        "  1. Model mismatch  — were query and corpus embedded with the same model?\n"
        "     Different models produce vectors in different spaces; scores become\n"
        "     meaningless even though cosine similarity still returns a number.\n"
        "\n"
        "  2. Dimension mismatch — do query and corpus vectors have the same length?\n"
        "     Run check_dimension_consistency() to detect this quickly.\n"
        "\n"
        "  3. Chunk-to-vector alignment — is each embedding stored next to the\n"
        "     correct chunk? A shifted index silently attaches the wrong vector.\n"
        "\n"
        "  4. Generic chunk dominance — a broad policy header can score higher than\n"
        "     a specific sub-section. Consider splitting or re-weighting such chunks.\n"
        "\n"
        "  5. Near-duplicates — run detect_near_duplicate_chunks() to find chunks\n"
        "     that are so similar they compete with each other in retrieval."
    )
