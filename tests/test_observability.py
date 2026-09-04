"""Tests for caching, logging, and usage monitoring (HRS3.48).

Covers:
  - cache_key(): normalisation (case/whitespace), filters included,
    different inputs produce different keys
  - get_cached_answer(): miss returns None, hit returns response, TTL
    expiry evicts and returns None, expired entry removed from store
  - save_cached_answer(): entry written with created_at, response stored
  - invalidate_cache(): single entry removed, full clear, missing key returns 0
  - cache_size(): empty, after saves, after invalidate
  - estimate_cost(): zero tokens, only input, only output, both, custom rates,
    rounded to 6 decimal places
  - build_usage_metadata(): required keys, cost matches estimate_cost,
    cache_hit forwarded
  - log_rag_request(): entry appended to log, JSON emitted, required keys
    in stored entry, answer_preview truncated to 180 chars, missing fields
    default gracefully
  - get_usage_log() / clear_usage_log(): copy semantics, clear returns count
  - summarize_usage(): empty log, single entry, cache hit rate, cost sum,
    average latency, status breakdown, top_questions ranking, total invariants
"""

import time
import logging
import unittest

from src.observability import (
    cache_key,
    get_cached_answer,
    save_cached_answer,
    invalidate_cache,
    cache_size,
    estimate_cost,
    build_usage_metadata,
    log_rag_request,
    get_usage_log,
    clear_usage_log,
    summarize_usage,
    new_request_id,
    CACHE_TTL_SECONDS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _fresh_cache():
    return {}


def _fresh_log():
    return []


def _sample_record(**overrides):
    base = {
        "request_id": "abc123",
        "question": "How do I apply for sick leave?",
        "answer": "Submit a leave request through the HR portal.",
        "sources": [{"source": "policy.txt", "chunk_index": 0}],
        "cache_hit": False,
        "input_tokens": 500,
        "output_tokens": 80,
        "estimated_cost": 0.000123,
        "latency_ms": 320.5,
        "status": "answered",
        "model": "gemini-test",
    }
    base.update(overrides)
    return base


# ── cache_key() ───────────────────────────────────────────────────────────

class TestCacheKey(unittest.TestCase):

    def test_returns_64_char_hex_string(self):
        key = cache_key("hello")
        self.assertEqual(len(key), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in key))

    def test_case_normalised(self):
        self.assertEqual(
            cache_key("How do I apply?"),
            cache_key("HOW DO I APPLY?"),
        )

    def test_whitespace_normalised(self):
        self.assertEqual(
            cache_key("  sick leave  "),
            cache_key("sick leave"),
        )

    def test_different_questions_produce_different_keys(self):
        self.assertNotEqual(cache_key("sick leave"), cache_key("annual leave"))

    def test_filters_change_the_key(self):
        k1 = cache_key("leave policy")
        k2 = cache_key("leave policy", filters={"region": "India"})
        self.assertNotEqual(k1, k2)

    def test_same_filters_produce_same_key(self):
        self.assertEqual(
            cache_key("q", {"region": "India"}),
            cache_key("q", {"region": "India"}),
        )

    def test_none_and_empty_filters_equivalent(self):
        self.assertEqual(cache_key("q", None), cache_key("q", {}))


# ── get_cached_answer() / save_cached_answer() ────────────────────────────

class TestCacheReadWrite(unittest.TestCase):

    def test_miss_returns_none(self):
        self.assertIsNone(get_cached_answer("unknown", cache=_fresh_cache()))

    def test_hit_returns_stored_response(self):
        store = _fresh_cache()
        response = {"answer": "yes", "sources": []}
        save_cached_answer("sick leave?", response, cache=store)
        result = get_cached_answer("sick leave?", cache=store)
        self.assertEqual(result, response)

    def test_case_insensitive_hit(self):
        store = _fresh_cache()
        save_cached_answer("sick leave?", {"answer": "ok"}, cache=store)
        result = get_cached_answer("SICK LEAVE?", cache=store)
        self.assertIsNotNone(result)

    def test_expired_entry_returns_none(self):
        store = _fresh_cache()
        save_cached_answer("q", {"answer": "old"}, cache=store)
        # Manually back-date the entry to simulate expiry
        key = cache_key("q")
        store[key]["created_at"] = time.time() - CACHE_TTL_SECONDS - 1
        result = get_cached_answer("q", ttl_seconds=CACHE_TTL_SECONDS,
                                   cache=store)
        self.assertIsNone(result)

    def test_expired_entry_evicted_from_store(self):
        store = _fresh_cache()
        save_cached_answer("q", {"answer": "old"}, cache=store)
        key = cache_key("q")
        store[key]["created_at"] = time.time() - CACHE_TTL_SECONDS - 1
        get_cached_answer("q", cache=store)
        self.assertNotIn(key, store)

    def test_non_expired_entry_kept_in_store(self):
        store = _fresh_cache()
        save_cached_answer("q", {"answer": "fresh"}, cache=store)
        get_cached_answer("q", cache=store)
        self.assertEqual(cache_size(store), 1)

    def test_save_returns_cache_key(self):
        store = _fresh_cache()
        key = save_cached_answer("q", {"answer": "x"}, cache=store)
        self.assertEqual(len(key), 64)

    def test_filters_separate_cache_entries(self):
        store = _fresh_cache()
        save_cached_answer("q", {"answer": "global"}, cache=store)
        save_cached_answer("q", {"answer": "india"},
                           filters={"region": "India"}, cache=store)
        r1 = get_cached_answer("q", cache=store)
        r2 = get_cached_answer("q", filters={"region": "India"}, cache=store)
        self.assertEqual(r1["answer"], "global")
        self.assertEqual(r2["answer"], "india")


# ── invalidate_cache() / cache_size() ────────────────────────────────────

class TestCacheInvalidation(unittest.TestCase):

    def test_single_entry_removed(self):
        store = _fresh_cache()
        save_cached_answer("q", {"answer": "x"}, cache=store)
        removed = invalidate_cache("q", cache=store)
        self.assertEqual(removed, 1)
        self.assertEqual(cache_size(store), 0)

    def test_full_clear_removes_all(self):
        store = _fresh_cache()
        save_cached_answer("q1", {"answer": "a"}, cache=store)
        save_cached_answer("q2", {"answer": "b"}, cache=store)
        removed = invalidate_cache(cache=store)
        self.assertEqual(removed, 2)
        self.assertEqual(cache_size(store), 0)

    def test_missing_key_returns_zero(self):
        store = _fresh_cache()
        self.assertEqual(invalidate_cache("not there", cache=store), 0)

    def test_cache_size_empty(self):
        self.assertEqual(cache_size(_fresh_cache()), 0)

    def test_cache_size_after_save(self):
        store = _fresh_cache()
        save_cached_answer("q", {"answer": "x"}, cache=store)
        self.assertEqual(cache_size(store), 1)


# ── estimate_cost() ───────────────────────────────────────────────────────

class TestEstimateCost(unittest.TestCase):

    def test_zero_tokens_returns_zero(self):
        self.assertEqual(estimate_cost(0, 0), 0.0)

    def test_only_input_tokens(self):
        cost = estimate_cost(1000, 0)
        self.assertAlmostEqual(cost, 0.00015, places=6)

    def test_only_output_tokens(self):
        cost = estimate_cost(0, 1000)
        self.assertAlmostEqual(cost, 0.00060, places=6)

    def test_both_tokens_combined(self):
        cost = estimate_cost(1000, 1000)
        self.assertAlmostEqual(cost, 0.00075, places=6)

    def test_custom_rates(self):
        cost = estimate_cost(1000, 0,
                             input_cost_per_1k=0.001,
                             output_cost_per_1k=0.002)
        self.assertAlmostEqual(cost, 0.001, places=6)

    def test_result_rounded_to_6_places(self):
        cost = estimate_cost(333, 111)
        decimal_places = len(str(cost).split(".")[-1]) if "." in str(cost) else 0
        self.assertLessEqual(decimal_places, 6)

    def test_cost_is_positive_for_positive_tokens(self):
        self.assertGreater(estimate_cost(100, 50), 0.0)


# ── build_usage_metadata() ───────────────────────────────────────────────

class TestBuildUsageMetadata(unittest.TestCase):

    def test_required_keys_present(self):
        meta = build_usage_metadata(500, 80)
        for key in ("input_tokens", "output_tokens",
                    "estimated_cost", "cache_hit"):
            self.assertIn(key, meta)

    def test_cost_matches_estimate_cost(self):
        meta = build_usage_metadata(500, 80)
        self.assertAlmostEqual(meta["estimated_cost"],
                               estimate_cost(500, 80), places=6)

    def test_cache_hit_forwarded(self):
        self.assertTrue(build_usage_metadata(0, 0, cache_hit=True)["cache_hit"])
        self.assertFalse(build_usage_metadata(0, 0, cache_hit=False)["cache_hit"])

    def test_token_counts_stored(self):
        meta = build_usage_metadata(300, 120)
        self.assertEqual(meta["input_tokens"], 300)
        self.assertEqual(meta["output_tokens"], 120)


# ── log_rag_request() ────────────────────────────────────────────────────

class TestLogRagRequest(unittest.TestCase):

    def test_entry_appended_to_usage_log(self):
        log = _fresh_log()
        log_rag_request(_sample_record(), usage_log=log)
        self.assertEqual(len(log), 1)

    def test_required_keys_in_stored_entry(self):
        log = _fresh_log()
        log_rag_request(_sample_record(), usage_log=log)
        entry = log[0]
        for key in ("timestamp", "request_id", "question", "answer_preview",
                    "sources", "cache_hit", "input_tokens", "output_tokens",
                    "estimated_cost", "latency_ms", "status"):
            self.assertIn(key, entry)

    def test_answer_preview_truncated_to_180(self):
        log = _fresh_log()
        long_answer = "word " * 200
        log_rag_request(_sample_record(answer=long_answer), usage_log=log)
        self.assertLessEqual(len(log[0]["answer_preview"]), 180)

    def test_missing_optional_fields_default_gracefully(self):
        log = _fresh_log()
        minimal = {"question": "q", "answer": "a"}
        log_rag_request(minimal, usage_log=log)
        entry = log[0]
        self.assertEqual(entry["cache_hit"], False)
        self.assertEqual(entry["input_tokens"], 0)

    def test_json_emitted_to_logger(self):
        log = _fresh_log()
        test_logger = logging.getLogger("test_obs")
        with self.assertLogs("test_obs", level="INFO") as captured:
            log_rag_request(_sample_record(), app_logger=test_logger,
                            usage_log=log)
        self.assertEqual(len(captured.output), 1)
        # Verify it is valid JSON
        raw = captured.output[0].split("INFO:test_obs:")[-1]
        import json
        parsed = json.loads(raw)
        self.assertIn("question", parsed)

    def test_timestamp_is_iso_format(self):
        log = _fresh_log()
        log_rag_request(_sample_record(), usage_log=log)
        ts = log[0]["timestamp"]
        from datetime import datetime, timezone
        # Should not raise
        datetime.fromisoformat(ts)

    def test_multiple_entries_appended(self):
        log = _fresh_log()
        for i in range(3):
            log_rag_request(_sample_record(request_id=str(i)), usage_log=log)
        self.assertEqual(len(log), 3)


# ── get_usage_log() / clear_usage_log() ──────────────────────────────────

class TestUsageLogManagement(unittest.TestCase):

    def test_get_usage_log_returns_copy(self):
        log = _fresh_log()
        log_rag_request(_sample_record(), usage_log=log)
        copy = get_usage_log(usage_log=log)
        copy.append({"injected": True})
        self.assertEqual(len(log), 1)   # original unaffected

    def test_clear_usage_log_returns_count(self):
        log = _fresh_log()
        log_rag_request(_sample_record(), usage_log=log)
        log_rag_request(_sample_record(), usage_log=log)
        removed = clear_usage_log(usage_log=log)
        self.assertEqual(removed, 2)
        self.assertEqual(len(log), 0)


# ── summarize_usage() ────────────────────────────────────────────────────

class TestSummarizeUsage(unittest.TestCase):

    def test_empty_log_returns_zero_totals(self):
        report = summarize_usage([])
        self.assertEqual(report["total_requests"], 0)
        self.assertEqual(report["cache_hit_rate"], 0.0)
        self.assertEqual(report["total_estimated_cost"], 0.0)

    def test_total_requests_correct(self):
        records = [_sample_record() for _ in range(5)]
        report = summarize_usage(records)
        self.assertEqual(report["total_requests"], 5)

    def test_cache_hits_counted(self):
        records = [
            _sample_record(cache_hit=True),
            _sample_record(cache_hit=False),
            _sample_record(cache_hit=True),
        ]
        report = summarize_usage(records)
        self.assertEqual(report["cache_hits"], 2)

    def test_cache_hit_rate_calculation(self):
        records = [_sample_record(cache_hit=True)] * 3 + \
                  [_sample_record(cache_hit=False)] * 1
        report = summarize_usage(records)
        self.assertAlmostEqual(report["cache_hit_rate"], 0.75, places=2)

    def test_total_cost_is_sum(self):
        records = [_sample_record(estimated_cost=0.001)] * 4
        report = summarize_usage(records)
        self.assertAlmostEqual(report["total_estimated_cost"], 0.004, places=6)

    def test_average_latency_correct(self):
        records = [
            _sample_record(latency_ms=100.0),
            _sample_record(latency_ms=300.0),
        ]
        report = summarize_usage(records)
        self.assertAlmostEqual(report["average_latency_ms"], 200.0, places=2)

    def test_status_breakdown_counts(self):
        records = [
            _sample_record(status="answered"),
            _sample_record(status="answered"),
            _sample_record(status="refused_weak_context"),
        ]
        report = summarize_usage(records)
        self.assertEqual(report["status_breakdown"]["answered"], 2)
        self.assertEqual(report["status_breakdown"]["refused_weak_context"], 1)

    def test_top_questions_most_frequent_first(self):
        records = (
            [_sample_record(question="sick leave?")] * 3 +
            [_sample_record(question="annual leave?")] * 2 +
            [_sample_record(question="resignation?")] * 1
        )
        report = summarize_usage(records)
        questions = [q for q, _ in report["top_questions"]]
        self.assertEqual(questions[0], "sick leave?")
        self.assertEqual(questions[1], "annual leave?")

    def test_top_questions_capped_at_five(self):
        records = [_sample_record(question=f"q{i}?") for i in range(10)]
        report = summarize_usage(records)
        self.assertLessEqual(len(report["top_questions"]), 5)

    def test_total_tokens_summed(self):
        records = [
            _sample_record(input_tokens=100, output_tokens=50),
            _sample_record(input_tokens=200, output_tokens=75),
        ]
        report = summarize_usage(records)
        self.assertEqual(report["total_input_tokens"], 300)
        self.assertEqual(report["total_output_tokens"], 125)

    def test_total_invariant_cache_hits_lte_total(self):
        records = [_sample_record(cache_hit=i % 2 == 0) for i in range(6)]
        report = summarize_usage(records)
        self.assertLessEqual(report["cache_hits"], report["total_requests"])


# ── new_request_id() ─────────────────────────────────────────────────────

class TestNewRequestId(unittest.TestCase):

    def test_returns_32_char_hex(self):
        rid = new_request_id()
        self.assertEqual(len(rid), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in rid))

    def test_unique_on_each_call(self):
        self.assertNotEqual(new_request_id(), new_request_id())


if __name__ == "__main__":
    unittest.main()
