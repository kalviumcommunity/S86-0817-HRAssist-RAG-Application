"""Tests for the chunk re-ranking module (HRS3.35).

Covers:
  - keyword_overlap_score(): exact overlap, partial, empty query, no match,
    case-insensitive, multi-word
  - rerank(): order changes when rerank score differs from vector score,
    final_k slicing, rerank_score and rerank_rank keys added, original score
    preserved, invalid args raise ValueError, ties broken by vector score
  - rerank_with_llm(): delegates to rerank(), unparseable LLM response
    defaults to 0.0, LLM called once per candidate
  - build_reranking_report(): order_changed/top_source_changed flags,
    sources lists, query forwarded
  - Two-stage pipeline integration: retrieve candidates with large k then
    rerank to smaller final_k produces valid output
"""

import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace

from src.reranker import (
    keyword_overlap_score,
    rerank,
    rerank_with_llm,
    compare_reranking,
    build_reranking_report,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_candidate(source: str, text: str, vector_score: float) -> dict:
    return {
        "score": vector_score,
        "text": text,
        "metadata": {"source": source, "chunk_index": 0},
    }


def _fixed_score(value: float):
    """Return a score_fn that always returns the given value."""
    return lambda query, chunk: value


def _source_score_map(mapping: dict):
    """Return a score_fn that returns a score keyed on metadata source."""
    return lambda query, chunk: mapping.get(
        chunk.get("metadata", {}).get("source", ""), 0.0
    )


def _candidates():
    """
    Three candidates where vector order (score desc) is:
      1. policy.txt      (0.85)
      2. account.md      (0.75)
      3. campus.md       (0.60)
    After keyword rerank on "sick leave policy":
      "policy.txt" text contains "sick leave policy" → high overlap
      "account.md" text is about passwords → low overlap
      "campus.md"  text is about cafeteria → zero overlap
    """
    return [
        _make_candidate("policy.txt", "Employees may apply for sick leave under company policy.", 0.85),
        _make_candidate("account.md", "How to reset your password and recover your account.", 0.75),
        _make_candidate("campus.md",  "The cafeteria menu changes every Friday morning.", 0.60),
    ]


# ── keyword_overlap_score() ───────────────────────────────────────────────

class TestKeywordOverlapScore(unittest.TestCase):

    def test_perfect_overlap(self):
        chunk = {"text": "sick leave policy information"}
        score = keyword_overlap_score("sick leave policy", chunk)
        self.assertAlmostEqual(score, 1.0)

    def test_partial_overlap(self):
        chunk = {"text": "apply for sick leave"}
        # query: {"sick", "leave", "policy"} — "sick" and "leave" match, "policy" not
        score = keyword_overlap_score("sick leave policy", chunk)
        self.assertAlmostEqual(score, 2 / 3, places=5)

    def test_no_overlap_returns_zero(self):
        chunk = {"text": "the cafeteria menu changes every Friday"}
        score = keyword_overlap_score("sick leave policy", chunk)
        self.assertAlmostEqual(score, 0.0)

    def test_empty_query_returns_zero(self):
        chunk = {"text": "some text about leave"}
        self.assertEqual(keyword_overlap_score("", chunk), 0.0)

    def test_case_insensitive(self):
        chunk = {"text": "SICK LEAVE POLICY"}
        score = keyword_overlap_score("sick leave policy", chunk)
        self.assertAlmostEqual(score, 1.0)

    def test_score_in_zero_one_range(self):
        chunk = {"text": "a b c d e f g"}
        score = keyword_overlap_score("x y z a", chunk)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_missing_text_key_returns_zero(self):
        score = keyword_overlap_score("sick leave", {})
        self.assertEqual(score, 0.0)

    def test_duplicate_query_words_counted_once(self):
        """Unique query tokens: 'sick leave' — duplicates should not inflate score."""
        chunk = {"text": "sick leave information"}
        score = keyword_overlap_score("sick sick leave", chunk)
        # unique tokens: {"sick", "leave"} — both present → 1.0
        self.assertAlmostEqual(score, 1.0)


# ── rerank() ──────────────────────────────────────────────────────────────

class TestRerank(unittest.TestCase):

    def test_returns_final_k_items_as_first_element(self):
        candidates = _candidates()
        final_context, _ = rerank("sick leave policy", candidates,
                                  keyword_overlap_score, final_k=2)
        self.assertEqual(len(final_context), 2)

    def test_reranked_all_contains_all_candidates(self):
        candidates = _candidates()
        _, reranked_all = rerank("sick leave policy", candidates,
                                 keyword_overlap_score, final_k=2)
        self.assertEqual(len(reranked_all), len(candidates))

    def test_rerank_score_key_added_to_each_result(self):
        candidates = _candidates()
        final_context, _ = rerank("sick leave policy", candidates,
                                  keyword_overlap_score, final_k=2)
        for item in final_context:
            self.assertIn("rerank_score", item)

    def test_rerank_rank_key_added_and_starts_at_one(self):
        candidates = _candidates()
        _, reranked_all = rerank("sick leave policy", candidates,
                                 keyword_overlap_score, final_k=2)
        ranks = [item["rerank_rank"] for item in reranked_all]
        self.assertEqual(ranks, list(range(1, len(candidates) + 1)))

    def test_original_vector_score_preserved(self):
        candidates = _candidates()
        final_context, _ = rerank("sick leave policy", candidates,
                                  keyword_overlap_score, final_k=3)
        original_scores = {c["metadata"]["source"]: c["score"] for c in candidates}
        for item in final_context:
            source = item["metadata"]["source"]
            self.assertAlmostEqual(item["score"], original_scores[source])

    def test_reranking_changes_order_when_scores_differ(self):
        """policy.txt should move to rank 1 even though it was already rank 1 by
        vector score — verify the output is sorted by rerank_score not vector score."""
        candidates = _candidates()
        final_context, _ = rerank("sick leave policy", candidates,
                                  keyword_overlap_score, final_k=3)
        sources = [item["metadata"]["source"] for item in final_context]
        # policy.txt has highest keyword overlap → must be first
        self.assertEqual(sources[0], "policy.txt")
        # campus.md (cafeteria) has zero overlap → must be last
        self.assertEqual(sources[-1], "campus.md")

    def test_reranking_promotes_lower_vector_ranked_chunk(self):
        """Swap the scorer so account.md should rank above policy.txt."""
        candidates = _candidates()
        score_map = {
            "account.md":  9.0,
            "policy.txt":  5.0,
            "campus.md":   1.0,
        }
        final_context, _ = rerank("query", candidates,
                                  _source_score_map(score_map), final_k=1)
        self.assertEqual(final_context[0]["metadata"]["source"], "account.md")

    def test_ties_broken_by_vector_score(self):
        """When rerank scores are equal, higher vector score wins."""
        candidates = [
            _make_candidate("high-vec.md",  "text", vector_score=0.9),
            _make_candidate("low-vec.md",   "text", vector_score=0.5),
        ]
        # Both get the same rerank score → tie-break on vector score
        final_context, _ = rerank("q", candidates, _fixed_score(5.0), final_k=1)
        self.assertEqual(final_context[0]["metadata"]["source"], "high-vec.md")

    def test_final_k_equals_candidates_length(self):
        candidates = _candidates()
        final_context, _ = rerank("q", candidates,
                                  keyword_overlap_score, final_k=3)
        self.assertEqual(len(final_context), 3)

    def test_empty_candidates_raises_value_error(self):
        with self.assertRaises(ValueError):
            rerank("q", [], keyword_overlap_score, final_k=1)

    def test_final_k_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            rerank("q", _candidates(), keyword_overlap_score, final_k=0)

    def test_final_k_exceeds_candidates_raises_value_error(self):
        with self.assertRaises(ValueError):
            rerank("q", _candidates(), keyword_overlap_score, final_k=100)

    def test_input_candidates_not_mutated(self):
        """rerank() must not add keys to the original dicts in candidates."""
        candidates = _candidates()
        original_keys = [set(c.keys()) for c in candidates]
        rerank("q", candidates, keyword_overlap_score, final_k=2)
        for original, candidate in zip(original_keys, candidates):
            self.assertEqual(original, set(candidate.keys()))


# ── rerank_with_llm() ─────────────────────────────────────────────────────

class TestRerankWithLLM(unittest.TestCase):

    def _mock_client(self, responses):
        """responses: list of strings the LLM will return in order."""
        client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            content = responses[idx % len(responses)]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        client.chat.completions.create.side_effect = side_effect
        return client

    def test_llm_scores_used_for_ordering(self):
        """If the LLM gives account.md a higher score, it should rank first."""
        # Candidates in vector order: policy(0.85), account(0.75), campus(0.60)
        # LLM scores:                 policy→3,     account→9,     campus→1
        candidates = _candidates()
        client = self._mock_client(["3", "9", "1"])

        final_context, _ = rerank_with_llm(
            "password reset", candidates, client, "test-model", final_k=1
        )
        self.assertEqual(final_context[0]["metadata"]["source"], "account.md")

    def test_unparseable_llm_response_defaults_to_zero(self):
        """A response that isn't a number should not crash — chunk gets score 0."""
        candidates = _candidates()[:2]
        client = self._mock_client(["NOT_A_NUMBER", "7"])

        final_context, reranked_all = rerank_with_llm(
            "sick leave", candidates, client, "test-model", final_k=2
        )
        scores = {r["metadata"]["source"]: r["rerank_score"] for r in reranked_all}
        self.assertEqual(scores["policy.txt"], 0.0)
        self.assertEqual(scores["account.md"], 7.0)

    def test_llm_called_once_per_candidate(self):
        candidates = _candidates()
        client = self._mock_client(["5", "6", "7"])

        rerank_with_llm("query", candidates, client, "test-model", final_k=2)

        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_empty_candidates_raises(self):
        client = self._mock_client(["5"])
        with self.assertRaises(ValueError):
            rerank_with_llm("q", [], client, "model", final_k=1)

    def test_final_k_exceeds_candidates_raises(self):
        client = self._mock_client(["5"])
        with self.assertRaises(ValueError):
            rerank_with_llm("q", _candidates(), client, "model", final_k=100)


# ── build_reranking_report() ──────────────────────────────────────────────

class TestBuildRerankingReport(unittest.TestCase):

    def test_order_changed_true_when_sources_differ(self):
        before = [
            _make_candidate("a.md", "text", 0.9),
            _make_candidate("b.md", "text", 0.8),
        ]
        after = [
            _make_candidate("b.md", "text", 0.8),
            _make_candidate("a.md", "text", 0.9),
        ]
        report = build_reranking_report("q", before, after)
        self.assertTrue(report["order_changed"])
        self.assertTrue(report["top_source_changed"])

    def test_order_changed_false_when_sources_same(self):
        before = [_make_candidate("a.md", "text", 0.9)]
        after  = [_make_candidate("a.md", "text", 0.9)]
        report = build_reranking_report("q", before, after)
        self.assertFalse(report["order_changed"])
        self.assertFalse(report["top_source_changed"])

    def test_before_and_after_sources_listed(self):
        before = [_make_candidate("policy.txt", "text", 0.9)]
        after  = [_make_candidate("account.md", "text", 0.8)]
        report = build_reranking_report("q", before, after)
        self.assertEqual(report["before_sources"], ["policy.txt"])
        self.assertEqual(report["after_sources"],  ["account.md"])

    def test_query_forwarded(self):
        before = [_make_candidate("a.md", "text", 0.9)]
        report = build_reranking_report("sick leave steps", before, before)
        self.assertEqual(report["query"], "sick leave steps")

    def test_empty_lists_produce_false_flags(self):
        report = build_reranking_report("q", [], [])
        self.assertFalse(report["order_changed"])
        self.assertFalse(report["top_source_changed"])


# ── Two-stage pipeline integration ────────────────────────────────────────

class TestTwoStagePipeline(unittest.TestCase):
    """Simulate the full retrieve-then-rerank pattern from HRS3.35."""

    def test_full_pipeline_returns_smaller_final_context(self):
        """retrieve k=6 candidates, rerank to final_k=2."""
        import math

        # Build a synthetic corpus (6 chunks)
        corpus = [
            {"text": "sick leave policy document",    "metadata": {"source": "policy.txt"},  "embedding": [1.0, 0.0]},
            {"text": "annual leave entitlement info",  "metadata": {"source": "leave.md"},    "embedding": [0.9, 0.1]},
            {"text": "password reset instructions",    "metadata": {"source": "account.md"},  "embedding": [0.1, 0.9]},
            {"text": "cafeteria menu Friday",          "metadata": {"source": "campus.md"},   "embedding": [0.0, 1.0]},
            {"text": "HR portal login help",           "metadata": {"source": "portal.md"},   "embedding": [0.2, 0.8]},
            {"text": "submit leave request HR system", "metadata": {"source": "submit.md"},   "embedding": [0.8, 0.2]},
        ]

        from src.retriever import retrieve

        def stub_embed(texts):
            return [[1.0, 0.0] for _ in texts]   # query points toward "sick leave" chunks

        candidates = retrieve("sick leave policy", corpus, stub_embed, k=6)
        self.assertEqual(len(candidates), 6)

        final_context, reranked_all = rerank(
            "sick leave policy", candidates, keyword_overlap_score, final_k=2
        )

        self.assertEqual(len(final_context), 2)
        self.assertEqual(len(reranked_all), 6)
        # The top chunk should be about sick leave (highest keyword overlap)
        self.assertIn("sick", final_context[0]["text"].lower())

    def test_report_reflects_pipeline_output(self):
        candidates_before = _candidates()
        final_context, _ = rerank(
            "sick leave policy", candidates_before, keyword_overlap_score, final_k=2
        )
        report = build_reranking_report(
            "sick leave policy", candidates_before[:2], final_context
        )
        self.assertIn("before_sources", report)
        self.assertIn("after_sources", report)
        self.assertIn("order_changed", report)


if __name__ == "__main__":
    unittest.main()
