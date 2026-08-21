"""Simple chunking utilities for RAG document retrieval."""

from typing import Dict, Iterable, List


def fixed_chunks(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into fixed-size chunks with overlap between windows."""
    if not text:
        return []
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


def paragraph_chunks(text: str) -> List[str]:
    """Split text by paragraph boundaries while stripping empty sections."""
    if not text:
        return []
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def chunk_report(text: str, size: int = 500, overlap: int = 50) -> Dict[str, Dict[str, int]]:
    """Return summary metrics for both chunking strategies."""
    chunk_sets = {
        "fixed": fixed_chunks(text, size=size, overlap=overlap),
        "paragraph": paragraph_chunks(text),
    }

    report: Dict[str, Dict[str, int]] = {}
    for name, chunks in chunk_sets.items():
        sizes = [len(chunk) for chunk in chunks]
        report[name] = {
            "chunk_count": len(chunks),
            "average_size": int(sum(sizes) / len(sizes)) if sizes else 0,
            "largest_chunk": max(sizes) if sizes else 0,
        }
    return report
