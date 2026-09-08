"""Small, dependency-injected end-to-end RAG pipeline."""

from typing import Any, Callable, Dict, List, Sequence

from src.context_injector import assemble_context
from src.citations import build_citation_map, validate_citations
from src.similarity import cosine_similarity


NO_CONTEXT_ANSWER = "I could not find relevant context for that question."


def embed_query(
    query: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
) -> List[float]:
    """Embed one user query using the corpus embedding function."""

    embeddings = embed_fn([query])
    if len(embeddings) != 1:
        raise ValueError("embed_fn must return exactly one vector for one query")
    return embeddings[0]


def retrieve_context(
    query_vector: Sequence[float],
    chunk_records: List[Dict[str, Any]],
    k: int = 4,
    score_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Rank corpus chunks against an already-embedded query vector."""

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    scored = []
    for record in chunk_records:
        score = cosine_similarity(query_vector, record["embedding"])
        if score_threshold is None or score >= score_threshold:
            scored.append({
                "score": score,
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
            })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return [
        {**record, "rank": rank, "score": round(record["score"], 4)}
        for rank, record in enumerate(scored[:k], start=1)
    ]


def assemble_retrieved_context(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 5_000,
) -> Dict[str, Any]:
    """Assemble ranked chunks with source markers and token accounting."""

    context, used_tokens, chunks_included = assemble_context(
        chunks,
        max_tokens=max_tokens,
    )
    return {
        "context": context,
        "used_tokens": used_tokens,
        "chunks_included": chunks_included,
    }


def answer_query(
    query: str,
    chunk_records: List[Dict[str, Any]],
    embed_fn: Callable[[List[str]], List[List[float]]],
    generate_fn: Callable[[str, str], str],
    k: int = 4,
    score_threshold: float | None = None,
    max_context_tokens: int = 5_000,
) -> Dict[str, Any]:
    """Run embedding, retrieval, context assembly, and grounded generation."""

    query_vector = embed_query(query, embed_fn)
    chunks = retrieve_context(
        query_vector,
        chunk_records,
        k=k,
        score_threshold=score_threshold,
    )

    if not chunks:
        return {
            "answer": NO_CONTEXT_ANSWER,
            "sources": [],
            "citations": {},
            "context": "",
            "chunks_retrieved": 0,
            "status": "no_context",
        }

    assembled = assemble_retrieved_context(chunks, max_tokens=max_context_tokens)
    included_chunks = chunks[:assembled["chunks_included"]]
    sources = [chunk["metadata"] for chunk in included_chunks]
    citation_map = build_citation_map(included_chunks)
    answer = generate_fn(query, assembled["context"])
    citation_validation = validate_citations(answer, citation_map)
    return {
        "answer": answer,
        "sources": sources[:assembled["chunks_included"]],
        "citations": citation_map,
        "citation_validation": citation_validation,
        "context": assembled["context"],
        "chunks_retrieved": len(chunks),
        "chunks_included": assembled["chunks_included"],
        "status": "answered",
    }