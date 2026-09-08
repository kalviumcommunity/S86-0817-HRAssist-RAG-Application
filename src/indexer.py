"""Batch indexing helpers for embedded corpus chunks."""

from typing import Any, Dict, Iterable, List, Mapping


def to_vector_record(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an embedded chunk into a vector-database record."""

    metadata = chunk.get("metadata", {})
    return {
        "id": chunk["id"],
        "vector": chunk["embedding"],
        "text": chunk["text"],
        "metadata": {
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "section": metadata.get("section"),
        },
    }


def _batches(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    if size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def index_embeddings(
    collection: Any,
    embedded_chunks: Iterable[Mapping[str, Any]],
    batch_size: int = 100,
) -> Dict[str, Any]:
    """Upsert embedded chunks into a Chroma-compatible collection.

    Returns counts and batch failures so callers can verify that indexing was
    complete instead of treating a successful method call as sufficient.
    """

    records = [to_vector_record(chunk) for chunk in embedded_chunks]
    failures: List[Dict[str, Any]] = []
    inserted = 0

    for batch in _batches(records, batch_size):
        try:
            collection.upsert(
                ids=[record["id"] for record in batch],
                embeddings=[record["vector"] for record in batch],
                documents=[record["text"] for record in batch],
                metadatas=[record["metadata"] for record in batch],
            )
            inserted += len(batch)
        except Exception as error:
            failures.append({"batch_start_id": batch[0]["id"], "error": str(error)})

    indexed_count = collection.count()
    return {
        "expected_count": len(records),
        "inserted": inserted,
        "indexed_count": indexed_count,
        "failures": failures,
        "count_matches": indexed_count == len(records),
    }


def spot_check(collection: Any, chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """Read one indexed chunk and verify text, source, and vector dimension."""

    expected = to_vector_record(chunk)
    stored = collection.get(
        ids=[expected["id"]],
        include=["embeddings", "documents", "metadatas"],
    )
    if not stored.get("ids"):
        raise AssertionError(f"indexed record not found: {expected['id']}")

    actual_metadata = stored["metadatas"][0]
    checks = {
        "text_matches": stored["documents"][0] == expected["text"],
        "source_matches": actual_metadata.get("source") == expected["metadata"]["source"],
        "vector_dimension_matches": len(stored["embeddings"][0]) == len(expected["vector"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"indexed record failed spot check: {expected['id']}")
    return {"id": expected["id"], **checks}