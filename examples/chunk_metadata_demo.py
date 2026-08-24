"""
Example script demonstrating Module 3.22 concepts:
- Chunk Metadata & Source Tracking
- Document Chunking Strategies (Sliding window, Heading-aware, Paragraph)
- Traceability & Citation Generation for LLM Grounding
- Region & Source-based Metadata Filtering
- Chunk Corpus Summary Statistics
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_loader import DocumentLoader
from src.document_chunker import (
    DocumentChunker,
    Chunk,
    ChunkMetadata,
    tag_chunks,
    estimate_tokens,
)


def run_chunk_metadata_demonstration():
    print("=" * 70)
    print("1. CHUNK METADATA & SOURCE TRACKING DEMONSTRATION")
    print("=" * 70)

    loader = DocumentLoader(default_region="India")
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30, strategy="heading_aware")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample policy documents
        doc1_path = temp_path / "India_Leave_Policy.md"
        doc1_path.write_text(
            "# Leave Entitlements\n"
            "Full-time employees in India are entitled to 20 paid leave days annually.\n\n"
            "## Sick Leave\n"
            "Employees receive 10 days of sick leave per year with full pay allowance.\n\n"
            "## Carry Forward\n"
            "Up to 5 unused annual leave days can be carried forward into the next calendar year.",
            encoding="utf-8",
        )

        doc2_path = temp_path / "US_Benefits_Guide.txt"
        doc2_path.write_text(
            "US Health Insurance Section 1: Comprehensive medical, dental, and vision insurance coverage "
            "is provided to full-time employees beginning on their first day of employment. "
            "Section 2: 401(k) Retirement Plan offers matching up to 4% of annual base salary.",
            encoding="utf-8",
        )

        # 1. Ingest Documents
        docs = loader.load_directory(temp_path)
        print(f"\nLoaded {len(docs)} documents for chunking:\n")
        for doc in docs:
            print(f" • Document: {doc.source:<25} | Region: {doc.region:<6} | Chars: {doc.char_count}")

        # 2. Chunk Documents & Attach Metadata
        print("\n" + "=" * 70)
        print("2. CHUNKING & METADATA TAGGING")
        print("=" * 70)

        all_chunks = []
        for doc in docs:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        for chunk in all_chunks:
            meta = chunk.metadata
            print(f"\n[CHUNK #{meta.chunk_index}] Source: {meta.source}")
            print(f" Citation Tag: {meta.cite()}")
            print(f" Pos: chars {meta.char_start}..{meta.char_end} | Tokens: {meta.token_count} | Region: {meta.region}")
            if meta.section:
                print(f" Section Heading: '{meta.section}'")
            print(f" Preview: {chunk.preview(70)!r}")

        # 3. Direct Function Usage with tag_chunks
        print("\n" + "=" * 70)
        print("3. STANDARDIZED RAW CHUNK TAGGING (tag_chunks)")
        print("=" * 70)

        raw_text_chunks = [
            ("Employees must report absences by 9:00 AM.", 0),
            ("Unexcused absences may result in disciplinary review.", 45),
        ]
        tagged_dicts = tag_chunks(
            source="Attendance_Policy.txt",
            chunks=raw_text_chunks,
            region="Global",
            section="Absence Reporting",
        )
        print("Structured Chunk JSON Output:")
        print(json.dumps(tagged_dicts, indent=2))

        # 4. Metadata Filtering
        print("\n" + "=" * 70)
        print("4. METADATA-BASED FILTERING (Region: India)")
        print("=" * 70)

        india_chunks = DocumentChunker.filter_chunks(all_chunks, region="India")
        print(f"Retrieved {len(india_chunks)} chunks for India region scope:")
        for c in india_chunks:
            print(f" - [{c.metadata.source}] {c.preview(60)}")

        # 5. Corpus Summary Statistics
        print("\n" + "=" * 70)
        print("5. CORPUS SUMMARY STATISTICS")
        print("=" * 70)

        summary = DocumentChunker.summarize_chunks(all_chunks)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_chunk_metadata_demonstration()
