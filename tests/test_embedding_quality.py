"""Tests for the embedding quality / sanity-check module (HRS3.29).

Covers:
  - run_sanity_checks(): passing cases, failing cases, in-top-k detection,
    mismatched lengths raise ValueError, empty corpus raises ValueError
  - build_sanity_report(): correct aggregate counts (total/passed/failed/
    in_top_k_only)
  - check_dimension_consistency(): all-same, mismatch in corpus, mismatch
    in query vectors, empty corpus raises ValueError
  - detect_near_duplicate_chunks(): identical vectors flagged, distinct
    vectors not flagged, threshold boundary respected
"""

import unittest

from src.embedding_quality import (
    run_sanity_checks,
    build_sanity_report,
    check_dimension_consistency,
    detect_near_duplicate_chunks,
    DEFAULT_TEST_CASES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_chunk(source: str, embedding: list, text: str = "") -> dict:
    return {
        "text": text or f"content from {source}",
        "metadata": {"source": source},
        "embedding": embedding,
    }


def _corpus():
    """
    Three chunks in a 2-D toy vector space:
      account-guide  → [1.0, 0.0]   (points along x-axis)
      campus-guide   → [0.0, 1.0]   (points along y-axis)
      leave-policy   → [0.7, 0.7]   (diagonal — HR content)
    """
    return [
        _make_chunk("account-guide.md",          [1.0, 0.0]),
        _make_chunk("campus-guide.md",            [0.0, 1.0]),
        _make_chunk("employee_leave_policy.txt",  [0.7, 0.7]),
    ]


# ── run_sanity_checks() ────────────────────────────────────────────────────

class TestRunSanityChecks(unittest.TestCase):

    def test_passing_case_when_query_aligns_with_expected_source(self):
        """A query vector pointing toward account-guide should rank it first."""
        corpus = _corpus()
        cases = [{"query": "reset password", "expected_source": "account-guide.md"}]
        query_embeddings = [[0.99, 0.01]]   # almost pure x → account-guide wins

        results = run_sanity_checks(cases, corpus, query_embeddings, top_k=3)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["passed"])
        self.assertEqual(results[0]["top_source"], "account-guide.md")

    def test_failing_case_when_wrong_chunk_ranks_first(self):
        """A query pointing toward campus-guide should fail account-guide check."""
        corpus = _corpus()
        cases = [{"query": "cafeteria menu", "expected_source": "account-guide.md"}]
        query_embeddings = [[0.01, 0.99]]   # almost pure y → campus-guide wins

        results = run_sanity_checks(cases, corpus, query_embeddings, top_k=3)

        self.assertFalse(results[0]["passed"])
        self.assertEqual(results[0]["top_source"], "campus-guide.md")

    def test_in_top_k_true_when_expected_not_at_rank_1_but_in_top_k(self):
        """Expected source is at rank 2 → passed=False but in_top_k=True."""
        corpus = _corpus()
        # [0.6, 0.8] is closer to campus-guide [0,1] than account-guide [1,0]
        cases = [{"query": "q", "expected_source": "account-guide.md"}]
        query_embeddings = [[0.6, 0.8]]

        results = run_sanity_checks(cases, corpus, query_embeddings, top_k=3)

        self.assertFalse(results[0]["passed"])
        self.assertTrue(results[0]["in_top_k"])

    def test_note_is_forwarded_from_test_case(self):
        corpus = _corpus()
        cases = [
            {
                "query": "q",
                "expected_source": "account-guide.md",
                "note": "watch this edge case",
            }
        ]
        results = run_sanity_checks(cases, corpus, [[1.0, 0.0]], top_k=1)
        self.assertEqual(results[0]["note"], "watch this edge case")

    def test_top_k_sources_length_matches_top_k(self):
        corpus = _corpus()
        cases = [{"query": "q", "expected_source": "account-guide.md"}]
        results = run_sanity_checks(cases, corpus, [[1.0, 0.0]], top_k=2)
        self.assertEqual(len(results[0]["top_k_sources"]), 2)

    def test_mismatched_lengths_raise_value_error(self):
        """Passing 2 test cases but only 1 embedding should raise."""
        corpus = _corpus()
        cases = [
            {"query": "q1", "expected_source": "account-guide.md"},
            {"query": "q2", "expected_source": "campus-guide.md"},
        ]
        with self.assertRaises(ValueError):
            run_sanity_checks(cases, corpus, [[1.0, 0.0]], top_k=3)

    def test_empty_corpus_raises_value_error(self):
        cases = [{"query": "q", "expected_source": "any.md"}]
        with self.assertRaises(ValueError):
            run_sanity_checks(cases, [], [[1.0, 0.0]])

    def test_multiple_cases_in_one_call(self):
        corpus = _corpus()
        cases = [
            {"query": "password reset", "expected_source": "account-guide.md"},
            {"query": "cafeteria",      "expected_source": "campus-guide.md"},
        ]
        query_embeddings = [[0.99, 0.01], [0.01, 0.99]]

        results = run_sanity_checks(cases, corpus, query_embeddings, top_k=3)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["passed"])
        self.assertTrue(results[1]["passed"])

    def test_result_contains_required_keys(self):
        corpus = _corpus()
        results = run_sanity_checks(
            [{"query": "q", "expected_source": "account-guide.md"}],
            corpus,
            [[1.0, 0.0]],
            top_k=3,
        )
        required = {
            "query", "expected_source", "top_source",
            "top_score", "top_k_sources", "passed", "in_top_k", "note",
        }
        self.assertEqual(required, required & results[0].keys())


# ── build_sanity_report() ─────────────────────────────────────────────────

class TestBuildSanityReport(unittest.TestCase):

    def _results(self, passed_flags, in_top_k_flags=None):
        """Build a minimal results list from boolean flags."""
        if in_top_k_flags is None:
            in_top_k_flags = passed_flags
        return [
            {
                "query": f"q{i}",
                "expected_source": "x",
                "top_source": "x" if p else "y",
                "top_score": 0.9,
                "top_k_sources": ["x"] if k else ["y"],
                "passed": p,
                "in_top_k": k,
                "note": "",
            }
            for i, (p, k) in enumerate(zip(passed_flags, in_top_k_flags))
        ]

    def test_all_passing(self):
        report = build_sanity_report(self._results([True, True, True]))
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["passed"], 3)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["in_top_k_only"], 0)

    def test_all_failing(self):
        report = build_sanity_report(
            self._results([False, False], in_top_k_flags=[False, False])
        )
        self.assertEqual(report["failed"], 2)
        self.assertEqual(report["passed"], 0)

    def test_in_top_k_only_counts_near_misses(self):
        """passed=False but in_top_k=True → counted in in_top_k_only."""
        results = self._results(
            [True, False, False],
            in_top_k_flags=[True, True, False],
        )
        report = build_sanity_report(results)
        self.assertEqual(report["in_top_k_only"], 1)   # second row only
        self.assertEqual(report["failed"], 2)

    def test_total_equals_passed_plus_failed(self):
        results = self._results([True, False, True, False])
        report = build_sanity_report(results)
        self.assertEqual(report["total"], report["passed"] + report["failed"])

    def test_rows_are_forwarded(self):
        results = self._results([True])
        report = build_sanity_report(results)
        self.assertIs(report["rows"], results)


# ── check_dimension_consistency() ────────────────────────────────────────

class TestCheckDimensionConsistency(unittest.TestCase):

    def test_all_same_dimension_returns_consistent(self):
        corpus = [
            _make_chunk("a.md", [1.0, 0.0]),
            _make_chunk("b.md", [0.0, 1.0]),
        ]
        result = check_dimension_consistency(corpus)
        self.assertTrue(result["consistent"])
        self.assertEqual(result["corpus_dimension"], 2)
        self.assertEqual(result["mismatched_chunks"], [])

    def test_mismatched_chunk_dimension_detected(self):
        corpus = [
            _make_chunk("a.md", [1.0, 0.0]),       # dim 2
            _make_chunk("b.md", [0.0, 1.0, 0.5]),  # dim 3 — mismatch
        ]
        result = check_dimension_consistency(corpus)
        self.assertFalse(result["consistent"])
        self.assertIn(1, result["mismatched_chunks"])

    def test_query_dimension_mismatch_flagged(self):
        corpus = [_make_chunk("a.md", [1.0, 0.0])]
        result = check_dimension_consistency(corpus, query_embeddings=[[1.0, 0.0, 0.5]])
        self.assertFalse(result["consistent"])
        self.assertFalse(result["query_dimension_ok"])

    def test_query_dimension_match_passes(self):
        corpus = [_make_chunk("a.md", [1.0, 0.0])]
        result = check_dimension_consistency(corpus, query_embeddings=[[0.5, 0.5]])
        self.assertTrue(result["query_dimension_ok"])

    def test_no_query_embeddings_gives_none(self):
        corpus = [_make_chunk("a.md", [1.0, 0.0])]
        result = check_dimension_consistency(corpus)
        self.assertIsNone(result["query_dimension_ok"])

    def test_empty_corpus_raises(self):
        with self.assertRaises(ValueError):
            check_dimension_consistency([])


# ── detect_near_duplicate_chunks() ───────────────────────────────────────

class TestDetectNearDuplicateChunks(unittest.TestCase):

    def test_identical_embeddings_flagged(self):
        corpus = [
            _make_chunk("a.md", [1.0, 0.0], text="same text"),
            _make_chunk("b.md", [1.0, 0.0], text="same text"),
        ]
        dupes = detect_near_duplicate_chunks(corpus, threshold=0.98)
        self.assertEqual(len(dupes), 1)
        self.assertEqual(dupes[0]["index_a"], 0)
        self.assertEqual(dupes[0]["index_b"], 1)
        self.assertAlmostEqual(dupes[0]["score"], 1.0)

    def test_distinct_embeddings_not_flagged(self):
        corpus = [
            _make_chunk("a.md", [1.0, 0.0]),
            _make_chunk("b.md", [0.0, 1.0]),
        ]
        dupes = detect_near_duplicate_chunks(corpus, threshold=0.98)
        self.assertEqual(dupes, [])

    def test_below_threshold_not_flagged(self):
        # cosine([0.9, 0.1], [1.0, 0.0]) < 1.0 but > 0.98 only if very close
        # Use vectors that are close but below threshold
        import math
        a = [1.0, 0.0]
        # angle of ~12 degrees → cos≈0.978 < 0.98
        angle = 0.21   # radians
        b = [math.cos(angle), math.sin(angle)]
        corpus = [_make_chunk("a.md", a), _make_chunk("b.md", b)]
        dupes = detect_near_duplicate_chunks(corpus, threshold=0.99)
        self.assertEqual(dupes, [])

    def test_threshold_boundary_respected(self):
        """Pairs right at or above the threshold are included."""
        corpus = [
            _make_chunk("a.md", [1.0, 0.0]),
            _make_chunk("b.md", [1.0, 0.0]),  # score == 1.0
        ]
        dupes = detect_near_duplicate_chunks(corpus, threshold=1.0)
        self.assertEqual(len(dupes), 1)

    def test_result_contains_required_keys(self):
        corpus = [
            _make_chunk("a.md", [1.0, 0.0], text="hello world"),
            _make_chunk("b.md", [1.0, 0.0], text="hello world"),
        ]
        dupes = detect_near_duplicate_chunks(corpus)
        self.assertIn("index_a", dupes[0])
        self.assertIn("index_b", dupes[0])
        self.assertIn("score", dupes[0])
        self.assertIn("text_a", dupes[0])
        self.assertIn("text_b", dupes[0])

    def test_single_chunk_produces_no_duplicates(self):
        corpus = [_make_chunk("a.md", [1.0, 0.0])]
        self.assertEqual(detect_near_duplicate_chunks(corpus), [])

    def test_empty_corpus_produces_no_duplicates(self):
        self.assertEqual(detect_near_duplicate_chunks([]), [])


# ── DEFAULT_TEST_CASES sanity ─────────────────────────────────────────────

class TestDefaultTestCases(unittest.TestCase):

    def test_default_cases_have_required_keys(self):
        for case in DEFAULT_TEST_CASES:
            self.assertIn("query", case)
            self.assertIn("expected_source", case)
            self.assertIn("note", case)

    def test_default_cases_not_empty(self):
        self.assertGreater(len(DEFAULT_TEST_CASES), 0)

    def test_default_cases_queries_are_non_empty_strings(self):
        for case in DEFAULT_TEST_CASES:
            self.assertIsInstance(case["query"], str)
            self.assertGreater(len(case["query"]), 0)


if __name__ == "__main__":
    unittest.main()
