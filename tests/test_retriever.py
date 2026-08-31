"""Tests for the top-k similarity retrieval module (HRS3.32).

Covers:
  - retrieve(): top-k slicing, descending score order, required result keys,
    rank numbering, score threshold filtering, k=1, k > corpus size,
    invalid k raises ValueError, empty corpus raises ValueError,
    model mismatch produces unreliable results (documented behaviour)
  - retrieve_at_k_values(): all k values evaluated, shared scoring,
    invalid k raises, empty corpus raises, larger k never fewer results
    than smaller k (up to corpus size)
"""

import math
import unittest

from src.retriever import retrieve, retrieve_at_k_values


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_record(source: str, embedding: list, text: str = "") -> dict:
    return {
        "text": text or f"content from {source}",
        "metadata": {"source": source, "chunk_index": 0},
        "embedding": embedding,
    }


def _identity_embed(texts):
    """Stub embed_fn: returns the stored vector encoded in the text field.

    For tests we pass queries as space-separated floats so the stub can
    return the exact vector we want without any real API call.
    Example: query "1.0 0.0" → vector [1.0, 0.0]
    """
    result = []
    for text in texts:
        result.append([float(v) for v in text.split()])
    return result


def _corpus_2d():
    """
    Three chunks in a 2-D toy space:
      account-guide  → [1.0, 0.0]
      campus-guide   → [0.0, 1.0]
      leave-policy   → [0.7, 0.7]   (normalised diagonal)
    """
    mag = math.sqrt(0.7 ** 2 + 0.7 ** 2)
    diag = [0.7 / mag, 0.7 / mag]
    return [
        _make_record("account-guide.md",         [1.0, 0.0]),
        _make_record("campus-guide.md",           [0.0, 1.0]),
        _make_record("employee_leave_policy.txt", diag),
    ]


# ── retrieve() ────────────────────────────────────────────────────────────

class TestRetrieve(unittest.TestCase):

    def test_returns_top_k_results(self):
        """retrieve(k=2) returns exactly 2 results from a 3-chunk corpus."""
        results = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=2)
        self.assertEqual(len(results), 2)

    def test_results_sorted_descending_by_score(self):
        results = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=3)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_result_is_most_similar_chunk(self):
        """Query pointing along x-axis should rank account-guide first."""
        results = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=1)
        self.assertEqual(results[0]["metadata"]["source"], "account-guide.md")

    def test_rank_numbers_are_consecutive_from_one(self):
        results = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=3)
        ranks = [r["rank"] for r in results]
        self.assertEqual(ranks, list(range(1, len(results) + 1)))

    def test_result_contains_required_keys(self):
        results = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=1)
        required = {"rank", "score", "text", "metadata"}
        self.assertEqual(required, required & results[0].keys())

    def test_score_is_rounded_to_4_decimal_places(self):
        results = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=1)
        score_str = str(results[0]["score"])
        decimal_places = len(score_str.split(".")[-1]) if "." in score_str else 0
        self.assertLessEqual(decimal_places, 4)

    def test_k_equals_one_returns_single_best_result(self):
        results = retrieve("0.0 1.0", _corpus_2d(), _identity_embed, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["metadata"]["source"], "campus-guide.md")

    def test_k_larger_than_corpus_returns_all_chunks(self):
        """When k > corpus size the full corpus is returned, not an error."""
        corpus = _corpus_2d()   # 3 chunks
        results = retrieve("1.0 0.0", corpus, _identity_embed, k=100)
        self.assertEqual(len(results), len(corpus))

    def test_score_threshold_filters_low_confidence_results(self):
        """Chunks below score_threshold must be excluded from results."""
        # Query pointing along x-axis; campus-guide [0,1] scores 0.0
        results = retrieve(
            "1.0 0.0",
            _corpus_2d(),
            _identity_embed,
            k=3,
            score_threshold=0.5,
        )
        sources = [r["metadata"]["source"] for r in results]
        self.assertNotIn("campus-guide.md", sources)

    def test_score_threshold_zero_returns_all_positive_scores(self):
        results = retrieve(
            "1.0 0.0",
            _corpus_2d(),
            _identity_embed,
            k=3,
            score_threshold=0.0,
        )
        # campus-guide scores exactly 0.0, should be excluded (< threshold fails)
        # account-guide=1.0, leave-policy≈0.707 both pass
        for r in results:
            self.assertGreaterEqual(r["score"], 0.0)

    def test_invalid_k_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=0)

    def test_invalid_k_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=-1)

    def test_empty_corpus_raises_value_error(self):
        with self.assertRaises(ValueError):
            retrieve("1.0 0.0", [], _identity_embed, k=3)

    def test_metadata_is_forwarded_to_result(self):
        """Source and chunk_index from metadata must appear in results."""
        results = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=1)
        self.assertIn("source", results[0]["metadata"])
        self.assertIn("chunk_index", results[0]["metadata"])

    def test_text_is_forwarded_to_result(self):
        corpus = [_make_record("a.md", [1.0, 0.0], text="important HR text")]
        results = retrieve("1.0 0.0", corpus, _identity_embed, k=1)
        self.assertEqual(results[0]["text"], "important HR text")

    def test_embed_fn_called_exactly_once(self):
        """The embed function should be called once per retrieve() call."""
        call_count = [0]

        def counting_embed(texts):
            call_count[0] += len(texts)
            return _identity_embed(texts)

        retrieve("1.0 0.0", _corpus_2d(), counting_embed, k=2)
        # Only the query string is embedded — not the corpus chunks
        self.assertEqual(call_count[0], 1)

    def test_different_k_values_produce_subset_relationship(self):
        """Results for k=1 should be a prefix of results for k=3."""
        r1 = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=1)
        r3 = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=3)
        self.assertEqual(r1[0]["metadata"]["source"],
                         r3[0]["metadata"]["source"])

    def test_model_mismatch_still_returns_numbers_but_ranking_not_trusted(self):
        """
        If query and corpus use different embedding models, cosine similarity
        still returns floats but the ranking is meaningless. This test
        documents the risk by using deliberately mismatched random vectors.

        We verify only that retrieve() does not crash — the caller is
        responsible for ensuring model consistency.
        """
        corpus = [
            _make_record("a.md", [0.3, 0.7]),
            _make_record("b.md", [0.9, 0.1]),
        ]
        # Query vector is "from a different model" — completely arbitrary
        def mismatched_embed(texts):
            return [[0.5, 0.5]]

        results = retrieve("anything", corpus, mismatched_embed, k=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("score", r)


# ── retrieve_at_k_values() ────────────────────────────────────────────────

class TestRetrieveAtKValues(unittest.TestCase):

    def test_returns_result_for_every_k_value(self):
        results = retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[1, 2, 3]
        )
        self.assertEqual(set(results.keys()), {1, 2, 3})

    def test_each_k_returns_correct_number_of_results(self):
        results = retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[1, 2, 3]
        )
        for k, result_list in results.items():
            self.assertEqual(len(result_list), k)

    def test_k_larger_than_corpus_returns_full_corpus(self):
        corpus = _corpus_2d()   # 3 chunks
        results = retrieve_at_k_values(
            "1.0 0.0", corpus, _identity_embed, k_values=[100]
        )
        self.assertEqual(len(results[100]), len(corpus))

    def test_rank_1_is_consistent_across_k_values(self):
        """The best chunk should appear at rank 1 regardless of k."""
        results = retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[1, 2, 3]
        )
        rank_1_sources = {results[k][0]["metadata"]["source"] for k in [1, 2, 3]}
        self.assertEqual(len(rank_1_sources), 1)   # same source at every k

    def test_larger_k_results_are_superset_of_smaller_k(self):
        """Every result in k=1 should also appear in k=3."""
        results = retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[1, 3]
        )
        sources_k1 = {r["metadata"]["source"] for r in results[1]}
        sources_k3 = {r["metadata"]["source"] for r in results[3]}
        self.assertTrue(sources_k1.issubset(sources_k3))

    def test_embed_called_once_regardless_of_k_count(self):
        """Query embedding should happen once even with many k values."""
        call_count = [0]

        def counting_embed(texts):
            call_count[0] += len(texts)
            return _identity_embed(texts)

        retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), counting_embed,
            k_values=[1, 2, 3, 4, 5],
        )
        self.assertEqual(call_count[0], 1)

    def test_invalid_k_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            retrieve_at_k_values(
                "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[1, 0, 3]
            )

    def test_invalid_k_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            retrieve_at_k_values(
                "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[-1]
            )

    def test_empty_corpus_raises_value_error(self):
        with self.assertRaises(ValueError):
            retrieve_at_k_values(
                "1.0 0.0", [], _identity_embed, k_values=[1, 3]
            )

    def test_result_dicts_contain_required_keys(self):
        results = retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[2]
        )
        for r in results[2]:
            self.assertIn("rank", r)
            self.assertIn("score", r)
            self.assertIn("text", r)
            self.assertIn("metadata", r)

    def test_scores_are_descending_within_each_k(self):
        results = retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[3]
        )
        scores = [r["score"] for r in results[3]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_single_k_value_matches_retrieve(self):
        """retrieve_at_k_values with one k must match retrieve() output."""
        r_single = retrieve("1.0 0.0", _corpus_2d(), _identity_embed, k=2)
        r_multi  = retrieve_at_k_values(
            "1.0 0.0", _corpus_2d(), _identity_embed, k_values=[2]
        )
        sources_single = [r["metadata"]["source"] for r in r_single]
        sources_multi  = [r["metadata"]["source"] for r in r_multi[2]]
        self.assertEqual(sources_single, sources_multi)


if __name__ == "__main__":
    unittest.main()
