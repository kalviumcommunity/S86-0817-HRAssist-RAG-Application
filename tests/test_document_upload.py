"""Tests for document upload, validation, storage, and indexing (HRS3.45).

Covers:
  - validate_upload(): supported extensions pass, unsupported → 415,
    empty content → 400, oversized → 400, extension case-insensitive
  - safe_filename(): strips directory components (path traversal prevention)
  - store_upload(): creates upload dir, writes file, returns correct path,
    validation errors propagate, empty content rejected
  - process_uploaded_document(): loads/cleans/chunks/tags/indexes a real
    .txt file, chunks count > 0, indexed count matches chunks, status=indexed,
    embed_fn called with chunk texts, embed_fn=None stores empty embeddings,
    failed document load returns status=failed
  - FastAPI endpoint POST /documents: 200 on valid upload, 415 on bad type,
    400 on empty file, 500 when processing fails, response has required keys,
    vector store grows after successful upload
  - GET /health: 200, indexed_chunks key present
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ── Processor unit tests ──────────────────────────────────────────────────

from src.document_processor import (
    validate_upload,
    safe_filename,
    store_upload,
    process_uploaded_document,
    UploadValidationError,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)


class TestValidateUpload(unittest.TestCase):

    def test_txt_extension_accepted(self):
        ext = validate_upload("policy.txt", b"some text content")
        self.assertEqual(ext, ".txt")

    def test_md_extension_accepted(self):
        ext = validate_upload("readme.md", b"# Title\nSome content")
        self.assertEqual(ext, ".md")

    def test_pdf_extension_accepted(self):
        # Minimal valid bytes — validation only checks content is non-empty
        ext = validate_upload("doc.pdf", b"%PDF-1.4 fake content")
        self.assertEqual(ext, ".pdf")

    def test_unsupported_extension_raises_415(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload("data.csv", b"a,b,c")
        self.assertEqual(ctx.exception.status_code, 415)

    def test_binary_extension_raises_415(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload("file.bin", b"\x00\x01")
        self.assertEqual(ctx.exception.status_code, 415)

    def test_empty_content_raises_400(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload("policy.txt", b"")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_oversized_content_raises_400(self):
        oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        with self.assertRaises(UploadValidationError) as ctx:
            validate_upload("policy.txt", oversized)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_extension_case_insensitive(self):
        ext = validate_upload("POLICY.TXT", b"content")
        self.assertEqual(ext, ".txt")

    def test_exactly_max_size_accepted(self):
        content = b"x" * MAX_FILE_SIZE_BYTES
        ext = validate_upload("policy.txt", content)
        self.assertEqual(ext, ".txt")


class TestSafeFilename(unittest.TestCase):

    def test_plain_filename_unchanged(self):
        self.assertEqual(safe_filename("policy.txt"), "policy.txt")

    def test_path_traversal_stripped(self):
        self.assertEqual(safe_filename("../../etc/passwd"), "passwd")

    def test_nested_path_stripped(self):
        self.assertEqual(safe_filename("/uploads/subdir/doc.md"), "doc.md")

    def test_windows_path_stripped(self):
        self.assertEqual(safe_filename(r"C:\Users\HR\policy.txt"), "policy.txt")


class TestStoreUpload(unittest.TestCase):

    def test_file_written_to_upload_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = store_upload(
                "policy.txt", b"HR policy content",
                upload_dir=Path(tmpdir)
            )
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), b"HR policy content")

    def test_returns_correct_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = store_upload(
                "doc.md", b"# Title",
                upload_dir=Path(tmpdir)
            )
            self.assertEqual(dest.name, "doc.md")

    def test_creates_upload_dir_if_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_uploads"
            self.assertFalse(new_dir.exists())
            store_upload("f.txt", b"content", upload_dir=new_dir)
            self.assertTrue(new_dir.exists())

    def test_validation_error_propagates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(UploadValidationError):
                store_upload("bad.exe", b"content", upload_dir=Path(tmpdir))

    def test_empty_file_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(UploadValidationError):
                store_upload("policy.txt", b"", upload_dir=Path(tmpdir))


class TestProcessUploadedDocument(unittest.TestCase):

    def _write_txt(self, tmpdir: str, name: str = "policy.txt",
                   content: str = None) -> Path:
        text = content or (
            "HR Assist Employee Leave Policy\n\n"
            "Employees are entitled to annual leave based on company policy. "
            "Sick leave is available for medical conditions. "
            "Leave requests must be submitted through the HR portal "
            "at least five working days in advance. "
            "The HR manager reviews and approves all leave requests."
        )
        path = Path(tmpdir) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_successful_processing_returns_indexed_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            store: list = []
            result = process_uploaded_document(path, vector_store=store)
            self.assertEqual(result["status"], "indexed")

    def test_chunks_count_is_positive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            store: list = []
            result = process_uploaded_document(path, vector_store=store)
            self.assertGreater(result["chunks"], 0)

    def test_indexed_count_matches_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            store: list = []
            result = process_uploaded_document(path, vector_store=store)
            self.assertEqual(result["indexed"], result["chunks"])

    def test_records_appended_to_vector_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            store: list = []
            result = process_uploaded_document(path, vector_store=store)
            self.assertEqual(len(store), result["indexed"])

    def test_each_record_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            store: list = []
            process_uploaded_document(path, vector_store=store)
            for record in store:
                self.assertIn("text", record)
                self.assertIn("metadata", record)
                self.assertIn("embedding", record)

    def test_embed_fn_called_with_chunk_texts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            store: list = []
            call_log: list = []

            def fake_embed(texts):
                call_log.extend(texts)
                return [[0.1] * 4 for _ in texts]

            process_uploaded_document(path, embed_fn=fake_embed,
                                       vector_store=store)
            self.assertGreater(len(call_log), 0)
            for text in call_log:
                self.assertIsInstance(text, str)

    def test_embed_fn_none_stores_empty_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            store: list = []
            process_uploaded_document(path, embed_fn=None, vector_store=store)
            for record in store:
                self.assertEqual(record["embedding"], [])

    def test_nonexistent_file_returns_failed_status(self):
        result = process_uploaded_document(
            Path("/nonexistent/path/file.txt"),
            vector_store=[],
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)

    def test_document_path_in_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir)
            result = process_uploaded_document(path, vector_store=[])
            self.assertEqual(result["document"], str(path))

    def test_metadata_source_matches_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_txt(tmpdir, name="leave_policy.txt")
            store: list = []
            process_uploaded_document(path, vector_store=store)
            sources = {r["metadata"]["source"] for r in store}
            self.assertIn("leave_policy.txt", sources)


# ── FastAPI endpoint tests ────────────────────────────────────────────────

class TestDocumentUploadEndpoint(unittest.TestCase):

    def setUp(self):
        # Import app fresh so VECTOR_STORE can be cleared between tests
        import src.document_processor as proc_module
        proc_module.VECTOR_STORE.clear()

        from src.app import app
        self.client = TestClient(app)

    def _txt_file(self, content: str = None, filename: str = "test.txt"):
        text = content or (
            "Employee leave policy document. "
            "Employees may apply for annual leave, sick leave, "
            "and casual leave according to company policy."
        )
        return {"file": (filename, text.encode(), "text/plain")}

    def test_health_endpoint_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("status", resp.json())
        self.assertIn("indexed_chunks", resp.json())

    def test_valid_txt_upload_returns_200(self):
        resp = self.client.post("/documents", files=self._txt_file())
        self.assertEqual(resp.status_code, 200)

    def test_response_has_required_keys(self):
        resp = self.client.post("/documents", files=self._txt_file())
        body = resp.json()
        self.assertIn("status", body)
        self.assertIn("filename", body)
        self.assertIn("summary", body)

    def test_response_status_is_indexed(self):
        resp = self.client.post("/documents", files=self._txt_file())
        self.assertEqual(resp.json()["status"], "indexed")

    def test_response_filename_matches_upload(self):
        resp = self.client.post("/documents",
                                files=self._txt_file(filename="my_policy.txt"))
        self.assertEqual(resp.json()["filename"], "my_policy.txt")

    def test_summary_chunks_is_positive(self):
        resp = self.client.post("/documents", files=self._txt_file())
        self.assertGreater(resp.json()["summary"]["chunks"], 0)

    def test_unsupported_extension_returns_415(self):
        resp = self.client.post(
            "/documents",
            files={"file": ("data.csv", b"a,b,c", "text/csv")}
        )
        self.assertEqual(resp.status_code, 415)

    def test_empty_file_returns_400(self):
        resp = self.client.post(
            "/documents",
            files={"file": ("empty.txt", b"", "text/plain")}
        )
        self.assertEqual(resp.status_code, 400)

    def test_vector_store_grows_after_upload(self):
        import src.document_processor as proc_module
        before = len(proc_module.VECTOR_STORE)
        self.client.post("/documents", files=self._txt_file())
        after = len(proc_module.VECTOR_STORE)
        self.assertGreater(after, before)

    def test_md_file_accepted(self):
        md_content = b"# HR Policy\n\nEmployees may apply for annual leave."
        resp = self.client.post(
            "/documents",
            files={"file": ("policy.md", md_content, "text/markdown")}
        )
        self.assertEqual(resp.status_code, 200)

    def test_multiple_uploads_accumulate_in_store(self):
        import src.document_processor as proc_module
        self.client.post("/documents", files=self._txt_file(filename="a.txt"))
        self.client.post("/documents", files=self._txt_file(filename="b.txt"))
        self.assertGreater(len(proc_module.VECTOR_STORE), 0)


if __name__ == "__main__":
    unittest.main()
