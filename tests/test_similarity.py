import unittest

from src.similarity import cosine_similarity, rank_chunks


class TestSimilarity(unittest.TestCase):

    def test_cosine_similarity_uses_vector_direction(self):
        self.assertAlmostEqual(cosine_similarity([3, 0], [10, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_rank_chunks_orders_by_similarity_and_preserves_metadata(self):
        records = [
            {"text": "cafeteria", "metadata": {"source": "campus.md"}, "embedding": [0, 1]},
            {"text": "password reset", "metadata": {"source": "account.md"}, "embedding": [1, 0]},
            {"text": "account recovery", "metadata": {"source": "account.md"}, "embedding": [0.8, 0.2]},
        ]

        ranked = rank_chunks([1, 0], records)

        self.assertEqual([record["text"] for record in ranked], [
            "password reset", "account recovery", "cafeteria"
        ])
        self.assertEqual(ranked[0]["metadata"]["source"], "account.md")
        self.assertAlmostEqual(ranked[0]["score"], 1.0)
        self.assertNotIn("score", records[0])

    def test_rank_chunks_supports_top_k(self):
        records = [
            {"text": "best", "embedding": [1, 0]},
            {"text": "other", "embedding": [0, 1]},
        ]
        self.assertEqual(len(rank_chunks([1, 0], records, top_k=1)), 1)

    def test_invalid_vectors_raise_value_error(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])


if __name__ == "__main__":
    unittest.main()