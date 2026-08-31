"""Vector similarity helpers for embedding-based retrieval."""

import math
from numbers import Real
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def cosine_similarity(a: Sequence[Real], b: Sequence[Real]) -> float:
    """Return cosine similarity between two vectors.

    Scores range from -1 to 1 for non-zero vectors. Both vectors must have
    the same dimensionality and neither may be a zero vector.

    Cosine similarity compares the *direction* of two vectors in high-dimensional
    space, ignoring their magnitude. This makes it well-suited for comparing
    embedding vectors where the scale of individual values is not meaningful.

    Args:
        a: First embedding vector.
        b: Second embedding vector, must have the same length as ``a``.

    Returns:
        Float in [-1, 1]. Values near 1.0 indicate high semantic similarity;
        values near 0.0 indicate unrelated content.

    Raises:
        ValueError: If vectors differ in length, are empty, or either is a
            zero vector (similarity undefined).
    """

    if len(a) != len(b):
        raise ValueError("vectors must have the same dimensionality")
    if not a:
        raise ValueError("vectors must not be empty")

    dot_product = sum(left * right for left, right in zip(a, b))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    return dot_product / (norm_a * norm_b)


def compare_embeddings(
    query_embedding: Sequence[Real],
    candidate_embeddings: Sequence[Sequence[Real]],
    labels: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Compare a query embedding against a list of candidate embeddings.

    Useful for demonstrating how well an embedding model separates similar
    from dissimilar texts without building a full retrieval pipeline.

    Args:
        query_embedding: The reference vector to compare against.
        candidate_embeddings: Vectors to score against the query.
        labels: Optional human-readable labels for each candidate. When
            provided, must have the same length as ``candidate_embeddings``.

    Returns:
        List of dicts with 'label', 'score', and 'rank' keys, sorted by
        descending similarity score.

    Example::

        embeddings = embed(["reset password", "account recovery", "cafeteria"])
        results = compare_embeddings(embeddings[0], embeddings[1:],
                                     labels=["account recovery", "cafeteria"])
        for r in results:
            print(r["rank"], r["label"], f"{r['score']:.4f}")
    """
    if labels is not None and len(labels) != len(candidate_embeddings):
        raise ValueError(
            "labels and candidate_embeddings must have the same length"
        )

    results = []
    for index, candidate in enumerate(candidate_embeddings):
        label = labels[index] if labels is not None else f"candidate_{index}"
        score = cosine_similarity(query_embedding, candidate)
        results.append({"label": label, "score": score})

    results.sort(key=lambda item: item["score"], reverse=True)

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results


def rank_chunks(
    query_embedding: Sequence[Real],
    chunk_records: Iterable[Mapping[str, Any]],
    top_k: int | None = None,
) -> List[dict[str, Any]]:
    """Rank embedding records by cosine similarity to a query vector.

    Each returned record is a shallow copy of the input record with a
    numeric ``score`` field. The input records are not modified.
    """

    if top_k is not None and top_k < 0:
        raise ValueError("top_k cannot be negative")

    ranked = []
    for record in chunk_records:
        scored_record = dict(record)
        scored_record["score"] = cosine_similarity(
            query_embedding, record["embedding"]
        )
        ranked.append(scored_record)

    ranked.sort(key=lambda record: record["score"], reverse=True)
    return ranked if top_k is None else ranked[:top_k]