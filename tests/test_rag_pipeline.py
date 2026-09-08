import unittest

from src.rag_pipeline import (
    NO_CONTEXT_ANSWER,
    answer_query,
    assemble_retrieved_context,
    embed_query,
    retrieve_context,
)


def embed_fn(texts):
    return [[float(value) for value in text.split()] for text in texts]


CORPUS = [
    {
        "text": "Reset your password from the account settings page.",
        "embedding": [1.0, 0.0],
        "metadata": {"source": "account-guide.md", "chunk_index": 0},
    },
    {
        "text": "The cafeteria menu changes every Friday.",
        "embedding": [0.0, 1.0],
        "metadata": {"source": "campus-guide.md", "chunk_index": 3},
    },
]


class TestRagPipeline(unittest.TestCase):
    def test_stages_are_independently_testable(self):
        self.assertEqual(embed_query("1.0 0.0", embed_fn), [1.0, 0.0])
        chunks = retrieve_context([1.0, 0.0], CORPUS, k=1)
        self.assertEqual(chunks[0]["metadata"]["source"], "account-guide.md")

        assembled = assemble_retrieved_context(chunks)
        self.assertIn("[1] account-guide.md#0", assembled["context"])
        self.assertEqual(assembled["chunks_included"], 1)

    def test_answer_query_passes_context_to_generation(self):
        calls = []

        def generate(query, context):
            calls.append((query, context))
            return "Use the account settings page."

        result = answer_query("1.0 0.0", CORPUS, embed_fn, generate, k=1)

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "Use the account settings page.")
        self.assertEqual(result["sources"][0]["source"], "account-guide.md")
        self.assertEqual(calls[0][0], "1.0 0.0")
        self.assertIn("Reset your password", calls[0][1])

    def test_empty_retrieval_returns_fallback_without_generation(self):
        generate_called = []

        def generate(query, context):
            generate_called.append(True)
            return "unsupported"

        result = answer_query(
            "1.0 0.0",
            CORPUS,
            embed_fn,
            generate,
            score_threshold=1.01,
        )

        self.assertEqual(result["status"], "no_context")
        self.assertEqual(result["answer"], NO_CONTEXT_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(generate_called, [])


if __name__ == "__main__":
    unittest.main()