"""
HRAssist - Chunk Metadata & Source Tracking Pipeline
Module 3.22: Chunk Metadata & Source Tracking

This module provides tools for:
1. Document chunking strategies (token-aware, sliding-window overlap, heading-aware, paragraph).
2. Attaching rich metadata to every chunk (source identifier, chunk index, character offsets, section, region).
3. Traceability of retrieved chunks back to exact document sources for LLM citations.
4. Metadata-based filtering (e.g. region scoping, document source filtering).
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    from src.document_loader import LoadedDocument
except ImportError:
    from document_loader import LoadedDocument


def estimate_tokens(text: str) -> int:
    """
    Estimates token count for a given text string.
    Rule of thumb: ~1.3 tokens per word or ~4 characters per token.
    """
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


@dataclass
class ChunkMetadata:
    """
    Metadata attached to every document chunk to ensure complete source traceability and citation ability.
    """
    source: str                          # Source document filename/identifier (required for citations)
    chunk_index: int                     # 0-indexed position within the document
    char_start: int = 0                  # Character start position in source document
    char_end: int = 0                    # Character end position in source document
    region: str = "Global"               # Region scope (e.g., India, US, Global)
    section: Optional[str] = None        # Section title or heading if available
    doc_title: Optional[str] = None      # Human-readable title of source document
    token_count: int = 0                 # Estimated token count of chunk text
    extra: Dict[str, Any] = field(default_factory=dict)  # Arbitrary additional custom metadata

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata object into a clean dictionary format."""
        res = {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "region": self.region,
            "section": self.section,
            "doc_title": self.doc_title,
            "token_count": self.token_count,
        }
        if self.extra:
            res["extra"] = self.extra
        return res

    def cite(self) -> str:
        """Generates a human-readable citation string for LLM response grounding."""
        parts = [f"Source: {self.source}"]
        if self.section:
            parts.append(f"Section: '{self.section}'")
        parts.append(f"Chunk #{self.chunk_index}")
        if self.region != "Global":
            parts.append(f"Region: {self.region}")
        return f"[{' | '.join(parts)}]"


@dataclass
class Chunk:
    """
    A chunk object encapsulating text content and its associated ChunkMetadata.
    """
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Chunk to dictionary shape (text + metadata)."""
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict()
        }

    def preview(self, max_chars: int = 80) -> str:
        """Returns a clean single-line preview of the chunk text."""
        clean_text = self.text.replace("\n", " ").strip()
        if len(clean_text) > max_chars:
            return clean_text[:max_chars] + "..."
        return clean_text


def tag_chunks(
    source: str,
    chunks: List[Union[str, Tuple[str, int], Tuple[str, int, int], Dict[str, Any]]],
    region: str = "Global",
    section: Optional[str] = None,
    doc_title: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Utility function to pair raw chunks with structured metadata dictionaries.
    Ensures every chunk is text plus metadata in a consistent structure.
    """
    tagged = []
    for idx, item in enumerate(chunks):
        if isinstance(item, tuple):
            if len(item) == 2:
                text, char_start = item
                char_end = char_start + len(text)
            elif len(item) >= 3:
                text, char_start, char_end = item[0], item[1], item[2]
            else:
                text = str(item[0])
                char_start, char_end = 0, len(text)
        elif isinstance(item, dict):
            text = item.get("text", "")
            char_start = item.get("char_start", 0)
            char_end = item.get("char_end", char_start + len(text))
        else:
            text = str(item)
            char_start = 0
            char_end = len(text)

        meta = {
            "source": source,
            "chunk_index": idx,
            "char_start": char_start,
            "char_end": char_end,
            "region": region,
            "section": section,
            "doc_title": doc_title or source,
            "token_count": estimate_tokens(text),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        tagged.append({
            "text": text,
            "metadata": meta
        })
    return tagged


class DocumentChunker:
    """
    Configurable document chunker supporting character and token-aware sizing,
    sliding window overlap, header awareness, and rich metadata tagging.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "fixed_size",  # "fixed_size", "sliding_window", "heading_aware", "paragraph"
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def _chunk_sliding_window(self, text: str) -> List[Tuple[str, int, int, Optional[str]]]:
        """
        Splits text into chunks using a sliding window with overlap.
        Returns list of (chunk_text, char_start, char_end, section_name).
        """
        chunks = []
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        start = 0
        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Adjust end to nearest word boundary if not at text end
            if end < text_len:
                space_idx = text.rfind(" ", start, end)
                if space_idx > start + (self.chunk_size // 2):
                    end = space_idx

            chunk_str = text[start:end].strip()
            if chunk_str:
                chunk_start_pos = start + (len(text[start:end]) - len(text[start:end].lstrip()))
                chunk_end_pos = chunk_start_pos + len(chunk_str)
                chunks.append((chunk_str, chunk_start_pos, chunk_end_pos, None))

            start += step
            if start >= text_len or end == text_len:
                break

        return chunks

    def _chunk_heading_aware(self, text: str) -> List[Tuple[str, int, int, Optional[str]]]:
        """
        Splits Markdown or structured text by headings (# Heading), preserving section titles.
        """
        chunks = []
        lines = text.splitlines(keepends=True)

        current_section = "Header"
        current_buffer = []
        current_start = 0
        current_pos = 0

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            line_len = len(line)
            match = heading_pattern.match(line.strip())

            if match:
                if current_buffer:
                    sec_text = "".join(current_buffer).strip()
                    if sec_text:
                        sub_chunks = self._sub_chunk_text(sec_text, current_start, current_section)
                        chunks.extend(sub_chunks)
                    current_buffer = []

                current_section = match.group(2).strip()
                current_start = current_pos

            current_buffer.append(line)
            current_pos += line_len

        if current_buffer:
            sec_text = "".join(current_buffer).strip()
            if sec_text:
                sub_chunks = self._sub_chunk_text(sec_text, current_start, current_section)
                chunks.extend(sub_chunks)

        return chunks

    def _chunk_paragraph(self, text: str) -> List[Tuple[str, int, int, Optional[str]]]:
        """
        Splits text by double newlines (paragraphs).
        """
        chunks = []
        paragraphs = re.split(r"\n\s*\n", text)
        pos = 0

        for para in paragraphs:
            para_str = para.strip()
            if para_str:
                para_start = text.find(para_str, pos)
                if para_start == -1:
                    para_start = pos
                para_end = para_start + len(para_str)
                pos = para_end
                chunks.append((para_str, para_start, para_end, None))

        return chunks

    def _sub_chunk_text(self, text: str, base_offset: int, section_name: str) -> List[Tuple[str, int, int, Optional[str]]]:
        """
        Sub-chunks long text blocks while maintaining base character offsets and section metadata.
        """
        raw = self._chunk_sliding_window(text)
        return [
            (c_text, base_offset + c_start, base_offset + c_end, section_name)
            for c_text, c_start, c_end, _ in raw
        ]

    def chunk_text(
        self,
        text: str,
        source: str = "document",
        region: str = "Global",
        section: Optional[str] = None,
        doc_title: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Splits raw text into chunks according to configured strategy and tags each with ChunkMetadata.
        """
        if not text or not text.strip():
            return []

        if self.strategy == "heading_aware":
            raw_chunks = self._chunk_heading_aware(text)
        elif self.strategy == "paragraph":
            raw_chunks = self._chunk_paragraph(text)
        else:
            raw_chunks = self._chunk_sliding_window(text)

        tagged_chunks = []
        for idx, (chunk_text, char_start, char_end, chunk_sec) in enumerate(raw_chunks):
            active_section = chunk_sec or section
            meta = ChunkMetadata(
                source=source,
                chunk_index=idx,
                char_start=char_start,
                char_end=char_end,
                region=region,
                section=active_section,
                doc_title=doc_title or source,
                token_count=estimate_tokens(chunk_text),
            )
            tagged_chunks.append(Chunk(text=chunk_text, metadata=meta))

        return tagged_chunks

    def chunk_document(
        self,
        doc: LoadedDocument,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Directly ingests a LoadedDocument object, chunks its text, and attaches document metadata.
        """
        if doc.status == "FAILED" or not doc.text:
            return []

        old_size, old_overlap = self.chunk_size, self.chunk_overlap
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap

        try:
            return self.chunk_text(
                text=doc.text,
                source=doc.source,
                region=doc.region,
                doc_title=Path(doc.source).stem.replace("_", " ").title(),
            )
        finally:
            self.chunk_size = old_size
            self.chunk_overlap = old_overlap

    @staticmethod
    def filter_chunks(
        chunks: List[Chunk],
        region: Optional[str] = None,
        source: Optional[str] = None,
        section: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Filters a list of Chunk objects based on metadata criteria.
        """
        filtered = chunks
        if region:
            filtered = [
                c for c in filtered
                if c.metadata.region.lower() in (region.lower(), "global")
            ]
        if source:
            filtered = [
                c for c in filtered
                if source.lower() in c.metadata.source.lower()
            ]
        if section:
            filtered = [
                c for c in filtered
                if c.metadata.section and section.lower() in c.metadata.section.lower()
            ]
        return filtered

    @staticmethod
    def summarize_chunks(chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Computes summary statistics for a set of Chunk objects.
        """
        total_chunks = len(chunks)
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "total_estimated_tokens": 0,
                "avg_chunk_chars": 0,
                "sources": [],
                "regions": [],
            }

        total_chars = sum(len(c.text) for c in chunks)
        total_tokens = sum(c.metadata.token_count for c in chunks)
        sources = sorted(list({c.metadata.source for c in chunks}))
        regions = sorted(list({c.metadata.region for c in chunks}))

        return {
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "total_estimated_tokens": total_tokens,
            "avg_chunk_chars": round(total_chars / total_chunks, 1),
            "sources": sources,
            "regions": regions,
        }
