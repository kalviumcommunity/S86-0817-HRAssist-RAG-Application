"""Citation mapping and validation for grounded RAG answers."""

import re
from typing import Any, Dict, List

from src.context_injector import assemble_context


def build_citation_map(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map numbered citation markers to the exact retrieved chunk evidence."""

    citation_map = {}
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        citation_map[f"[{index}]"] = {
            "source": metadata.get("source", "unknown"),
            "chunk_id": metadata.get("chunk_id", chunk.get("id")),
            "chunk_index": metadata.get("chunk_index"),
            "section": metadata.get("section"),
            "text": chunk.get("text", ""),
        }
    return citation_map


def build_cited_prompt(
    question: str,
    chunks: List[Dict[str, Any]],
    max_context_tokens: int = 5_000,
) -> str:
    """Build a prompt that restricts the model to real retrieved citations."""

    context, _, _ = assemble_context(chunks, max_tokens=max_context_tokens)
    return (
        "Answer using only the context below.\n"
        "Cite every factual claim using source markers like [1] or [2].\n"
        "Only use source markers that appear in the context.\n"
        "If the context does not support an answer, say you do not have "
        "enough information and do not invent citations.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


def cited_markers(answer: str) -> List[str]:
    """Return citation markers used in an answer, preserving their order."""

    return list(dict.fromkeys(re.findall(r"\[\d+\]", answer)))


def validate_citations(
    answer: str,
    citation_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Report whether every marker in an answer maps to retrieved evidence."""

    markers = cited_markers(answer)
    unknown = [marker for marker in markers if marker not in citation_map]
    return {
        "markers": markers,
        "unknown_markers": unknown,
        "valid": not unknown,
    }


def verify_citation(
    marker: str,
    citation_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the source evidence for one marker or reject an unknown marker."""

    if marker not in citation_map:
        raise KeyError(f"citation marker is not backed by retrieved context: {marker}")
    return citation_map[marker]