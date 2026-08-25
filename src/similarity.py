"""Vector similarity helpers for embedding-based retrieval."""

import math
from numbers import Real
from typing import Any, Iterable, List, Mapping, Sequence


def cosine_similarity(a: Sequence[Real], b: Sequence[Real]) -> float:
    """Return cosine similarity between two vectors.

    Scores range from -1 to 1 for non-zero vectors. Both vectors must have
    the same dimensionality and neither may be a zero vector.
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