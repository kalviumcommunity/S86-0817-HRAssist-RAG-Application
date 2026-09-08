"""Offline retrieval relevance evaluation and settings comparison."""

from typing import Any, Callable, Dict, List, Optional

from src.retriever import retrieve


def evaluate_retrieval_settings(
    test_queries: List[Dict[str, str]],
    settings: List[Dict[str, Any]],
    chunk_records: List[Dict[str, Any]],
    embed_fn: Callable[[List[str]], List[List[float]]],
) -> List[Dict[str, Any]]:
    """Evaluate settings using source hit rate for a labelled query set."""

    if not test_queries:
        raise ValueError("test_queries must not be empty")
    if not settings:
        raise ValueError("settings must not be empty")

    summary = []
    for setting in settings:
        rows = []
        for item in test_queries:
            results = retrieve(
                item["query"],
                chunk_records,
                embed_fn,
                k=setting.get("k", 3),
                score_threshold=setting.get("min_score"),
                metadata_filter=setting.get("filter"),
            )
            sources = [result["metadata"].get("source") for result in results]
            rows.append({
                "query": item["query"],
                "expected_source": item["expected_source"],
                "returned_sources": sources,
                "hit": item["expected_source"] in sources,
            })

        hits = sum(row["hit"] for row in rows)
        summary.append({
            "setting": setting.get("name", "unnamed"),
            "hit_rate": hits / len(rows),
            "hits": hits,
            "details": rows,
        })
    return summary


def choose_best_setting(summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the first setting with the highest measured hit rate."""

    if not summary:
        raise ValueError("summary must not be empty")
    return max(summary, key=lambda row: row["hit_rate"])