import unittest

from src.chunker import fixed_chunks, paragraph_chunks, chunk_report


class TestChunkingStrategies(unittest.TestCase):

    def test_fixed_chunks_include_overlap_and_length(self):
        text = "word " * 100
        chunks = fixed_chunks(text, size=40, overlap=10)

        self.assertTrue(chunks)
        self.assertLessEqual(len(chunks[0]), 40)
        self.assertGreater(len(chunks), 1)

    def test_paragraph_chunks_split_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = paragraph_chunks(text)

        self.assertEqual(len(chunks), 3)
        self.assertIn("First paragraph.", chunks[0])
        self.assertIn("Third paragraph.", chunks[-1])

    def test_chunk_report_returns_summary(self):
        text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
        report = chunk_report(text)

        self.assertIn("fixed", report)
        self.assertIn("paragraph", report)
        self.assertIn("chunk_count", report["fixed"])
        self.assertGreater(report["fixed"]["average_size"], 0)


if __name__ == "__main__":
    unittest.main()
