"""Tests for hallucination guardrails and refusal handling (HRS3.41).

Covers:
  - RetrievalStrengthConfig: default values, validation errors
  - retrieval_is_strong(): empty list, all above threshold, all below,
    partial, exact threshold boundary, custom config
  - assess_retrieval(): empty input, all-pass, all-fail, partial,
    correct top_score, correct chunks_above_threshold, failure_reason
    strings, is_strong matches retrieval_is_strong
  - guarded_answer(): STATUS_REFUSED_NO_CHUNKS for empty chunks,
    STATUS_REFUSED_WEAK_CONTEXT for low scores, STATUS_ANSWERED for
    strong context, generate_fn called only when strong, sources empty
    on refusal, retrieval_assessment key always present, generate_fn
    result merged into output
"""

import unittest
from unittest.mock import MagicMock

from src.guardrails import (
    RetrievalStrengthConfig,
    retrieval_is_strong,
    assess_retrieval,
    guarded_answer,
    REFUSAL_MESSAGE_EMPTY,
    REFUSAL_MESSAGE_LOW_SCORE,
    STATUS_ANSWERED,
    STATUS_REFUSED_WEAK_CONTEXT,
    STATUS_REFUSED_NO_CHUNKS,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _chunk(score: float, source: str = "policy.txt") -> dict:
    return {
        "score": score,
        "text": f"content from {source}",
        "metadata": {"source": source, "chunk_index": 0},
    }


def _strong_chunks(n: int = 2, score: float = 0.85) -> list:
    return [_chunk(score, f"doc{i}.txt") for i in range(n)]


def _weak_chunks(n: int = 2, score: float = 0.50) -> list:
    return [_chunk(score, f"doc{i}.txt") for i in range(n)]


def _stub_generate(answer: str = "Generated answer.", sources: list = None):
    """Return a generate_fn stub that returns a fixed answer."""
    def _fn(question, chunks):
        return {
            "answer": answer,
            "sources": sources or [c["metadata"] for c in chunks],
        }
    return _fn


# ── RetrievalStrengthConfig ───────────────────────────────────────────────

class TestRetrievalStrengthConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = RetrievalStrengthConfig()
        self.assertAlmostEqual(cfg.min_top_score, 0.72)
        self.assertEqual(cfg.min_supporting_chunks, 1)
        self.assertTrue(cfg.require_non_empty)

    def test_custom_values_accepted(self):
        cfg = RetrievalStrengthConfig(min_top_score=0.85, min_supporting_chunks=2)
        self.assertAlmostEqual(cfg.min_top_score, 0.85)
        self.assertEqual(cfg.min_supporting_chunks, 2)

    def test_min_top_score_above_one_raises(self):
        with self.assertRaises(ValueError):
            RetrievalStrengthConfig(min_top_score=1.1)

    def test_min_top_score_below_zero_raises(self):
        with self.assertRaises(ValueError):
            RetrievalStrengthConfig(min_top_score=-0.1)

    def test_min_supporting_chunks_zero_raises(self):
        with self.assertRaises(ValueError):
            RetrievalStrengthConfig(min_supporting_chunks=0)

    def test_boundary_values_accepted(self):
        cfg = RetrievalStrengthConfig(min_top_score=0.0)
        self.assertAlmostEqual(cfg.min_top_score, 0.0)
        cfg2 = RetrievalStrengthConfig(min_top_score=1.0)
        self.assertAlmostEqual(cfg2.min_top_score, 1.0)


# ── retrieval_is_strong() ─────────────────────────────────────────────────

class TestRetrievalIsStrong(unittest.TestCase):

    def test_empty_list_returns_false(self):
        self.assertFalse(retrieval_is_strong([]))

    def test_all_above_threshold_returns_true(self):
        self.assertTrue(retrieval_is_strong(_strong_chunks(3, score=0.90)))

    def test_all_below_threshold_returns_false(self):
        self.assertFalse(retrieval_is_strong(_weak_chunks(3, score=0.50)))

    def test_one_above_threshold_satisfies_min_one(self):
        chunks = [_chunk(0.80), _chunk(0.40), _chunk(0.30)]
        self.assertTrue(retrieval_is_strong(chunks))

    def test_exact_threshold_boundary_passes(self):
        """A chunk scoring exactly min_top_score should count as strong."""
        cfg = RetrievalStrengthConfig(min_top_score=0.72)
        chunks = [_chunk(0.72)]
        self.assertTrue(retrieval_is_strong(chunks, cfg))

    def test_just_below_threshold_fails(self):
        cfg = RetrievalStrengthConfig(min_top_score=0.72)
        chunks = [_chunk(0.719)]
        self.assertFalse(retrieval_is_strong(chunks, cfg))

    def test_custom_min_supporting_chunks(self):
        """Require 2 strong chunks — 1 strong chunk should fail."""
        cfg = RetrievalStrengthConfig(min_top_score=0.72, min_supporting_chunks=2)
        chunks = [_chunk(0.90), _chunk(0.50)]   # only 1 above threshold
        self.assertFalse(retrieval_is_strong(chunks, cfg))

    def test_custom_min_supporting_chunks_met(self):
        cfg = RetrievalStrengthConfig(min_top_score=0.72, min_supporting_chunks=2)
        chunks = [_chunk(0.90), _chunk(0.85)]   # both above threshold
        self.assertTrue(retrieval_is_strong(chunks, cfg))

    def test_uses_default_config_when_none_passed(self):
        # 0.72 is the default; score of 0.73 should pass
        self.assertTrue(retrieval_is_strong([_chunk(0.73)]))

    def test_chunk_without_score_key_treated_as_zero(self):
        chunks = [{"text": "no score", "metadata": {}}]
        self.assertFalse(retrieval_is_strong(chunks))


# ── assess_retrieval() ────────────────────────────────────────────────────

class TestAssessRetrieval(unittest.TestCase):

    def test_empty_chunks_returns_is_strong_false(self):
        report = assess_retrieval([])
        self.assertFalse(report["is_strong"])
        self.assertEqual(report["total_chunks"], 0)
        self.assertEqual(report["top_score"], 0.0)
        self.assertIsNotNone(report["failure_reason"])

    def test_strong_chunks_returns_is_strong_true(self):
        report = assess_retrieval(_strong_chunks(2, score=0.90))
        self.assertTrue(report["is_strong"])
        self.assertIsNone(report["failure_reason"])

    def test_weak_chunks_returns_is_strong_false(self):
        report = assess_retrieval(_weak_chunks(2, score=0.40))
        self.assertFalse(report["is_strong"])

    def test_top_score_is_max_of_all_chunk_scores(self):
        chunks = [_chunk(0.60), _chunk(0.85), _chunk(0.70)]
        report = assess_retrieval(chunks)
        self.assertAlmostEqual(report["top_score"], 0.85)

    def test_chunks_above_threshold_counted_correctly(self):
        cfg = RetrievalStrengthConfig(min_top_score=0.72)
        chunks = [_chunk(0.90), _chunk(0.80), _chunk(0.50), _chunk(0.40)]
        report = assess_retrieval(chunks, cfg)
        self.assertEqual(report["chunks_above_threshold"], 2)

    def test_failure_reason_mentions_score_when_all_below(self):
        chunks = [_chunk(0.30)]
        report = assess_retrieval(chunks)
        self.assertIn("top score", report["failure_reason"])

    def test_failure_reason_mentions_count_when_not_enough(self):
        cfg = RetrievalStrengthConfig(min_top_score=0.72, min_supporting_chunks=3)
        chunks = [_chunk(0.90), _chunk(0.85)]   # only 2 above, need 3
        report = assess_retrieval(chunks, cfg)
        self.assertIn("only", report["failure_reason"])

    def test_is_strong_matches_retrieval_is_strong(self):
        """assess_retrieval and retrieval_is_strong must agree."""
        for score in [0.40, 0.72, 0.90]:
            chunks = [_chunk(score)]
            self.assertEqual(
                assess_retrieval(chunks)["is_strong"],
                retrieval_is_strong(chunks),
            )

    def test_total_chunks_matches_input_length(self):
        chunks = _strong_chunks(5)
        report = assess_retrieval(chunks)
        self.assertEqual(report["total_chunks"], 5)

    def test_thresholds_forwarded_from_config(self):
        cfg = RetrievalStrengthConfig(min_top_score=0.80, min_supporting_chunks=2)
        report = assess_retrieval(_strong_chunks(1, score=0.90), cfg)
        self.assertAlmostEqual(report["min_top_score"], 0.80)
        self.assertEqual(report["min_supporting_chunks"], 2)


# ── guarded_answer() ──────────────────────────────────────────────────────

class TestGuardedAnswer(unittest.TestCase):

    def test_empty_chunks_returns_no_chunks_status(self):
        result = guarded_answer("q", [], _stub_generate())
        self.assertEqual(result["status"], STATUS_REFUSED_NO_CHUNKS)

    def test_empty_chunks_returns_empty_message(self):
        result = guarded_answer("q", [], _stub_generate())
        self.assertEqual(result["answer"], REFUSAL_MESSAGE_EMPTY)

    def test_empty_chunks_sources_is_empty_list(self):
        result = guarded_answer("q", [], _stub_generate())
        self.assertEqual(result["sources"], [])

    def test_weak_chunks_returns_weak_context_status(self):
        result = guarded_answer("q", _weak_chunks(2, score=0.40), _stub_generate())
        self.assertEqual(result["status"], STATUS_REFUSED_WEAK_CONTEXT)

    def test_weak_chunks_returns_low_score_message(self):
        result = guarded_answer("q", _weak_chunks(2, score=0.40), _stub_generate())
        self.assertEqual(result["answer"], REFUSAL_MESSAGE_LOW_SCORE)

    def test_weak_chunks_sources_is_empty_list(self):
        result = guarded_answer("q", _weak_chunks(2, score=0.40), _stub_generate())
        self.assertEqual(result["sources"], [])

    def test_strong_chunks_returns_answered_status(self):
        result = guarded_answer("q", _strong_chunks(2), _stub_generate())
        self.assertEqual(result["status"], STATUS_ANSWERED)

    def test_strong_chunks_returns_generated_answer(self):
        result = guarded_answer(
            "q", _strong_chunks(2), _stub_generate("The leave policy says...")
        )
        self.assertEqual(result["answer"], "The leave policy says...")

    def test_generate_fn_not_called_on_refusal(self):
        """generate_fn must not be called when retrieval is weak."""
        generate_fn = MagicMock(return_value={"answer": "x", "sources": []})
        guarded_answer("q", _weak_chunks(2, score=0.30), generate_fn)
        generate_fn.assert_not_called()

    def test_generate_fn_called_exactly_once_on_strong(self):
        generate_fn = MagicMock(return_value={"answer": "ok", "sources": []})
        guarded_answer("q", _strong_chunks(2), generate_fn)
        generate_fn.assert_called_once()

    def test_generate_fn_receives_question_and_chunks(self):
        received = {}

        def capture_fn(question, chunks):
            received["question"] = question
            received["chunks"] = chunks
            return {"answer": "x", "sources": []}

        chunks = _strong_chunks(2)
        guarded_answer("How do I apply for leave?", chunks, capture_fn)
        self.assertEqual(received["question"], "How do I apply for leave?")
        self.assertEqual(received["chunks"], chunks)

    def test_retrieval_assessment_always_present(self):
        for chunks in [[], _weak_chunks(1, 0.30), _strong_chunks(1)]:
            result = guarded_answer("q", chunks, _stub_generate())
            self.assertIn("retrieval_assessment", result)

    def test_generate_fn_result_merged_into_output(self):
        """Extra keys from generate_fn should appear in the result."""
        def rich_generate(q, chunks):
            return {"answer": "ok", "sources": [], "citations": ["[1]"]}

        result = guarded_answer("q", _strong_chunks(1), rich_generate)
        self.assertIn("citations", result)
        self.assertEqual(result["citations"], ["[1]"])

    def test_custom_config_respected(self):
        """Raising the threshold should turn a previously-passing score into a refusal."""
        strict_cfg = RetrievalStrengthConfig(min_top_score=0.95)
        # score=0.85 passes default but fails strict
        chunks = [_chunk(0.85)]
        result = guarded_answer("q", chunks, _stub_generate(), config=strict_cfg)
        self.assertEqual(result["status"], STATUS_REFUSED_WEAK_CONTEXT)

    def test_lowered_threshold_allows_previously_refused_answer(self):
        lenient_cfg = RetrievalStrengthConfig(min_top_score=0.30)
        chunks = [_chunk(0.40)]   # below default 0.72 but above lenient 0.30
        result = guarded_answer("q", chunks, _stub_generate(), config=lenient_cfg)
        self.assertEqual(result["status"], STATUS_ANSWERED)

    def test_refusal_status_not_answered(self):
        for chunks in [[], _weak_chunks(1, 0.20)]:
            result = guarded_answer("q", chunks, _stub_generate())
            self.assertNotEqual(result["status"], STATUS_ANSWERED)


if __name__ == "__main__":
    unittest.main()
