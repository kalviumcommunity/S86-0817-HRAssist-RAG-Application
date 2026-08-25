import tempfile
import unittest
from pathlib import Path

from src.ingestion_pipeline import ingest


class TestIngestionPipeline(unittest.TestCase):

    def test_every_file_is_accounted_for(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "policy.txt").write_text(
                "HR ASSIST EMPLOYEE POLICY\n\nLeave requests must use the HR portal.",
                encoding="utf-8",
            )
            (folder / "unsupported.bin").write_bytes(b"unknown format")

            files, documents, chunks, failures = ingest(folder)

            self.assertEqual(len(files), 2)
            self.assertEqual(documents, 1)
            self.assertEqual(len(failures), 1)
            self.assertEqual(documents + len(failures), len(files))
            self.assertEqual(failures[0][0], "unsupported.bin")
            self.assertEqual(chunks[0]["metadata"]["source"], "policy.txt")
            self.assertNotIn("HR ASSIST EMPLOYEE POLICY", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()