import unittest

from src.citations import (
    build_citation_map,
    build_cited_prompt,
    validate_citations,
    verify_citation,
)


CHUNKS = [
    {
        "id": "policy.txt:0",
        "text": "Submit the request through the HR portal.",
        "metadata": {
            "source": "policy.txt",
            "chunk_index": 0,
            "section": "Requests",
        },
    },
]


class TestCitations(unittest.TestCase):
    def test_build_map_preserves_source_and_exact_text(self):
        citation_map = build_citation_map(CHUNKS)
        self.assertEqual(citation_map["[1]"]["source"], "policy.txt")
        self.assertEqual(citation_map["[1]"]["chunk_id"], "policy.txt:0")
        self.assertEqual(citation_map["[1]"]["text"], CHUNKS[0]["text"])

    def test_prompt_requires_real_markers(self):
        prompt = build_cited_prompt("How do I submit?", CHUNKS)
        self.assertIn("Cite every factual claim", prompt)
        self.assertIn("[1] policy.txt#0", prompt)
        self.assertIn("How do I submit?", prompt)

    def test_unknown_citations_are_rejected(self):
        result = validate_citations("Submit it [1]. Also call HR [9].", build_citation_map(CHUNKS))
        self.assertFalse(result["valid"])
        self.assertEqual(result["unknown_markers"], ["[9]"])

    def test_verify_citation_returns_original_evidence(self):
        evidence = verify_citation("[1]", build_citation_map(CHUNKS))
        self.assertEqual(evidence["text"], CHUNKS[0]["text"])
        with self.assertRaises(KeyError):
            verify_citation("[2]", build_citation_map(CHUNKS))


if __name__ == "__main__":
    unittest.main()