import unittest

from src.similarity import cosine_similarity, rank_chunks, compare_embeddings


class TestCosineSimilarity(unittest.TestCase):

    def test_identical_direction_scores_one(self):
        self.assertAlmostEqual(cosine_similarity([3, 0], [10, 0]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_direction_scores_negative_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_invalid_vectors_raise_value_error(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])

    def test_empty_vectors_raise_value_error(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])


class TestRankChunks(unittest.TestCase):

    def test_orders_by_similarity_and_preserves_metadata(self):
        records = [
            {"text": "cafeteria", "metadata": {"source": "campus.md"}, "embedding": [0, 1]},
            {"text": "password reset", "metadata": {"source": "account.md"}, "embedding": [1, 0]},
            {"text": "account recovery", "metadata": {"source": "account.md"}, "embedding": [0.8, 0.2]},
        ]

        ranked = rank_chunks([1, 0], records)

        self.assertEqual(
            [record["text"] for record in ranked],
            ["password reset", "account recovery", "cafeteria"],
        )
        self.assertEqual(ranked[0]["metadata"]["source"], "account.md")
        self.assertAlmostEqual(ranked[0]["score"], 1.0)

    def test_does_not_mutate_input_records(self):
        records = [{"text": "a", "embedding": [1, 0]}]
        rank_chunks([1, 0], records)
        self.assertNotIn("score", records[0])

    def test_supports_top_k(self):
        records = [
            {"text": "best", "embedding": [1, 0]},
            {"text": "other", "embedding": [0, 1]},
        ]
        self.assertEqual(len(rank_chunks([1, 0], records, top_k=1)), 1)

    def test_negative_top_k_raises(self):
        with self.assertRaises(ValueError):
            rank_chunks([1, 0], [], top_k=-1)


class TestCompareEmbeddings(unittest.TestCase):
    """Tests for the compare_embeddings helper introduced in HRS3.25."""

    def test_similar_pair_scores_higher_than_dissimilar(self):
        """
        The core embedding concept: vectors for semantically related texts
        point in a similar direction and therefore score higher via cosine
        similarity than vectors for unrelated texts.

        Synthetic 3-D vectors approximate what a real embedding model produces:
          - query and 'account recovery' share the account/access topic → close
          - query and 'cafeteria menu' share no topic → far apart
        """
        query_embedding = [0.9, 0.4, 0.1]          # "reset password"
        similar_embedding = [0.85, 0.45, 0.1]       # "account recovery"
        dissimilar_embedding = [0.1, 0.1, 0.99]     # "cafeteria menu"

        similar_score = cosine_similarity(query_embedding, similar_embedding)
        dissimilar_score = cosine_similarity(query_embedding, dissimilar_embedding)

        self.assertGreater(
            similar_score,
            dissimilar_score,
            msg=(
                "Expected the similar pair (password/login) to score higher "
                "than the dissimilar pair (password/cafeteria)."
            ),
        )

    def test_returns_sorted_results_with_ranks(self):
        query = [1.0, 0.0]
        candidates = [[0.8, 0.2], [0.0, 1.0]]
        labels = ["account recovery", "cafeteria"]

        results = compare_embeddings(query, candidates, labels)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["label"], "account recovery")
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[1]["rank"], 2)
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_works_without_labels(self):
        results = compare_embeddings([1, 0], [[1, 0], [0, 1]])
        self.assertEqual(results[0]["label"], "candidate_0")
        self.assertEqual(results[1]["label"], "candidate_1")

    def test_mismatched_labels_raises(self):
        with self.assertRaises(ValueError):
            compare_embeddings([1, 0], [[1, 0], [0, 1]], labels=["only_one"])

    def test_score_range_is_valid(self):
        """All cosine similarity scores must lie within [-1, 1]."""
        query = [0.9, 0.4, 0.1]
        candidates = [
            [0.85, 0.45, 0.1],
            [0.1, 0.1, 0.99],
            [-0.9, -0.4, -0.1],
        ]
        results = compare_embeddings(query, candidates)
        for item in results:
            self.assertGreaterEqual(item["score"], -1.0)
            self.assertLessEqual(item["score"], 1.0)


if __name__ == "__main__":
    unittest.main()