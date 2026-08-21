"""
Unit tests for Module 3.19: Document Loading & Multi-Format Intake
"""

import unittest
import tempfile
from pathlib import Path
from src.document_loader import DocumentLoader, LoadedDocument


class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DocumentLoader(default_region="Global")

    def test_load_txt_document(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False, encoding="utf-8") as tmp:
            tmp.write("Leave policy text for India region.")
            tmp_path = Path(tmp.name)

        try:
            doc = self.loader.load_single_file(tmp_path, region="India")
            self.assertEqual(doc.status, "OK")
            self.assertEqual(doc.extension, ".txt")
            self.assertEqual(doc.region, "India")
            self.assertIn("Leave policy text", doc.text)
            self.assertGreater(doc.char_count, 0)
        finally:
            tmp_path.unlink()

    def test_load_md_document(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w+", delete=False, encoding="utf-8") as tmp:
            tmp.write("# Employee Handbook\n\nCode of Conduct details.")
            tmp_path = Path(tmp.name)

        try:
            doc = self.loader.load_single_file(tmp_path, region="UK")
            self.assertEqual(doc.status, "OK")
            self.assertEqual(doc.extension, ".md")
            self.assertEqual(doc.region, "UK")
            self.assertIn("Code of Conduct", doc.text)
        finally:
            tmp_path.unlink()

    def test_load_html_document(self):
        with tempfile.NamedTemporaryFile(suffix=".html", mode="w+", delete=False, encoding="utf-8") as tmp:
            tmp.write("<html><body><h1>Benefits Guide</h1><p>Health insurance details.</p></body></html>")
            tmp_path = Path(tmp.name)

        try:
            doc = self.loader.load_single_file(tmp_path)
            self.assertEqual(doc.status, "OK")
            self.assertEqual(doc.extension, ".html")
            self.assertIn("Health insurance details.", doc.text)
            self.assertNotIn("<html>", doc.text)
        finally:
            tmp_path.unlink()

    def test_unsupported_format_graceful_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".unsupported", mode="w+", delete=False) as tmp:
            tmp.write("some data")
            tmp_path = Path(tmp.name)

        try:
            doc = self.loader.load_single_file(tmp_path)
            self.assertEqual(doc.status, "FAILED")
            self.assertIsNotNone(doc.error)
            self.assertIn("Unsupported file format", doc.error)
        finally:
            tmp_path.unlink()

    def test_nonexistent_file_graceful_failure(self):
        fake_path = Path("/nonexistent/path/policy.pdf")
        doc = self.loader.load_single_file(fake_path)

        self.assertEqual(doc.status, "FAILED")
        self.assertIn("File does not exist", doc.error)

    def test_directory_intake_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            (temp_path / "doc1.txt").write_text("Text file content", encoding="utf-8")
            (temp_path / "doc2.md").write_text("# Markdown header", encoding="utf-8")

            documents = self.loader.load_directory(temp_path)
            self.assertEqual(len(documents), 2)

            summary = self.loader.summarize_intake(documents)
            self.assertEqual(summary["total_documents"], 2)
            self.assertEqual(summary["successful"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertGreater(summary["total_character_count"], 0)


if __name__ == "__main__":
    unittest.main()
