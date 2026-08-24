"""
Unit tests for Module 3.22: Chunk Metadata & Source Tracking
"""

import unittest
from pathlib import Path
from src.document_loader import LoadedDocument
from src.document_chunker import (
    DocumentChunker,
    Chunk,
    ChunkMetadata,
    tag_chunks,
    estimate_tokens,
)


class TestDocumentChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, strategy="fixed_size")

    def test_tag_chunks_structure(self):
        chunks_input = [
            ("Paid leave entitlement is 20 days per calendar year.", 0),
            ("Unused sick leave can be carried forward up to 5 days.", 55),
        ]
        tagged = tag_chunks(
            source="India_Leave_Policy.txt",
            chunks=chunks_input,
            region="India",
            section="Leave Entitlements",
        )

        self.assertEqual(len(tagged), 2)
        first = tagged[0]
        self.assertIn("text", first)
        self.assertIn("metadata", first)
        meta = first["metadata"]
        self.assertEqual(meta["source"], "India_Leave_Policy.txt")
        self.assertEqual(meta["chunk_index"], 0)
        self.assertEqual(meta["char_start"], 0)
        self.assertEqual(meta["char_end"], len(chunks_input[0][0]))
        self.assertEqual(meta["region"], "India")
        self.assertEqual(meta["section"], "Leave Entitlements")

    def test_chunk_metadata_citation(self):
        meta = ChunkMetadata(
            source="US_Benefits_Guide.pdf",
            chunk_index=2,
            char_start=200,
            char_end=400,
            region="US",
            section="Health Insurance",
        )
        citation = meta.cite()
        self.assertIn("Source: US_Benefits_Guide.pdf", citation)
        self.assertIn("Section: 'Health Insurance'", citation)
        self.assertIn("Chunk #2", citation)
        self.assertIn("Region: US", citation)

    def test_sliding_window_chunking(self):
        sample_text = (
            "Section 1: General Leave Policy. Employees are allowed paid time off. "
            "Section 2: Sick Leave. Employees receive 10 days sick leave annually. "
            "Section 3: Maternity Leave. Eligible employees receive 26 weeks paid leave."
        )

        chunks = self.chunker.chunk_text(
            text=sample_text,
            source="Leave_Policy.txt",
            region="Global"
        )

        self.assertGreater(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, Chunk)
            self.assertEqual(chunk.metadata.chunk_index, i)
            self.assertEqual(chunk.metadata.source, "Leave_Policy.txt")
            self.assertGreater(chunk.metadata.char_end, chunk.metadata.char_start)

    def test_heading_aware_chunking(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20, strategy="heading_aware")
        md_text = (
            "# Leave Policy\n\nAll employees receive 20 days off.\n\n"
            "## Health Insurance\n\nFull coverage for medical and dental expenses."
        )

        chunks = chunker.chunk_text(md_text, source="Policy.md", region="India")
        self.assertGreaterEqual(len(chunks), 2)
        
        sections = [c.metadata.section for c in chunks if c.metadata.section]
        self.assertTrue(any("Leave Policy" in s or "Health Insurance" in s for s in sections))

    def test_paragraph_chunking(self):
        chunker = DocumentChunker(strategy="paragraph")
        text = "Paragraph one text.\n\nParagraph two text.\n\nParagraph three text."
        chunks = chunker.chunk_text(text, source="Para.txt")

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "Paragraph one text.")
        self.assertEqual(chunks[1].text, "Paragraph two text.")
        self.assertEqual(chunks[2].text, "Paragraph three text.")

    def test_chunk_loaded_document(self):
        doc = LoadedDocument(
            source="India_Leave.txt",
            file_path="/path/to/India_Leave.txt",
            text="India leave policy details. Employees receive 22 annual leave days.",
            extension=".txt",
            region="India",
            status="OK"
        )

        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata.source, "India_Leave.txt")
        self.assertEqual(chunks[0].metadata.region, "India")
        self.assertEqual(chunks[0].metadata.doc_title, "India Leave")

    def test_filter_chunks(self):
        c1 = Chunk("Text 1", ChunkMetadata("Doc1.txt", 0, region="India", section="Leave"))
        c2 = Chunk("Text 2", ChunkMetadata("Doc2.txt", 0, region="US", section="Health"))
        c3 = Chunk("Text 3", ChunkMetadata("Doc3.txt", 0, region="Global", section="General"))

        all_chunks = [c1, c2, c3]

        # Filter by region
        india_chunks = DocumentChunker.filter_chunks(all_chunks, region="India")
        self.assertEqual(len(india_chunks), 2)  # India + Global

        # Filter by section
        health_chunks = DocumentChunker.filter_chunks(all_chunks, section="Health")
        self.assertEqual(len(health_chunks), 1)
        self.assertEqual(health_chunks[0].metadata.source, "Doc2.txt")

    def test_summarize_chunks(self):
        c1 = Chunk("Short text", ChunkMetadata("Doc1.txt", 0, region="India", token_count=3))
        c2 = Chunk("Another short text", ChunkMetadata("Doc2.txt", 0, region="US", token_count=4))

        summary = DocumentChunker.summarize_chunks([c1, c2])
        self.assertEqual(summary["total_chunks"], 2)
        self.assertEqual(summary["total_estimated_tokens"], 7)
        self.assertEqual(summary["sources"], ["Doc1.txt", "Doc2.txt"])
        self.assertEqual(summary["regions"], ["India", "US"])

    def test_invalid_overlap_raises_value_error(self):
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_text_handling(self):
        chunks = self.chunker.chunk_text("", source="empty.txt")
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
