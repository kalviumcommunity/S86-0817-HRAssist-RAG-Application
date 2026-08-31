"""Tests for the batch embedding pipeline (HRS3.28).

Covers:
  - batches() generator: correct slicing, edge cases, invalid size
  - estimate_tokens(): returns a positive integer
  - embed_with_retry(): success path, retry on transient error, fail-fast
  - run_batch_embedding(): skip-existing, summary counts, cost calculation,
    record structure, partial failure handling
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

from src.batch_embedding import (
    batches,
    estimate_tokens,
    embed_with_retry,
    run_batch_embedding,
    PRICE_PER_1K_TOKENS,
    DEFAULT_BATCH_SIZE,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_chunks(n: int, id_prefix: str = "chunk") -> list:
    """Build a minimal list of chunk dicts with text, metadata, and id."""
    return [
        {
            "id": f"{id_prefix}_{i}",
            "text": f"HR policy sentence number {i}.",
            "metadata": {"source": "policy.txt", "chunk_index": i},
        }
        for i in range(n)
    ]


def _fake_embed_response(texts: list):
    """Return an API-response-like object whose .data mirrors the input."""
    items = [
        SimpleNamespace(embedding=[0.1 * (i + 1)] * 4)
        for i in range(len(texts))
    ]
    return SimpleNamespace(data=items)


def _mock_client(texts_to_response=None):
    """Return a mock OpenAI-compatible client whose embeddings.create is stubbed."""
    client = MagicMock()
    if texts_to_response is None:
        client.embeddings.create.side_effect = (
            lambda model, input: _fake_embed_response(input)
        )
    else:
        client.embeddings.create.side_effect = texts_to_response
    return client


# ── batches() ─────────────────────────────────────────────────────────────

class TestBatches(unittest.TestCase):

    def test_even_split(self):
        """8 items with batch size 4 → exactly 2 batches of 4."""
        result = list(batches(list(range(8)), size=4))
        self.assertEqual(result, [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_uneven_split(self):
        """7 items with batch size 3 → [3, 3, 1]."""
        result = list(batches(list(range(7)), size=3))
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(len(result[2]), 1)

    def test_single_item_batch(self):
        """batch size 1 → each item in its own list."""
        result = list(batches([10, 20, 30], size=1))
        self.assertEqual(result, [[10], [20], [30]])

    def test_batch_larger_than_list(self):
        """batch size > len(items) → one batch containing all items."""
        result = list(batches([1, 2, 3], size=100))
        self.assertEqual(result, [[1, 2, 3]])

    def test_empty_list(self):
        """Empty input produces no batches."""
        self.assertEqual(list(batches([], size=10)), [])

    def test_invalid_size_raises(self):
        with self.assertRaises(ValueError):
            list(batches([1, 2, 3], size=0))

    def test_preserves_order(self):
        """Items must appear in the same order across batches."""
        items = list(range(100))
        reconstructed = [
            item for batch in batches(items, size=7) for item in batch
        ]
        self.assertEqual(reconstructed, items)


# ── estimate_tokens() ─────────────────────────────────────────────────────

class TestEstimateTokens(unittest.TestCase):

    def test_returns_positive_int(self):
        count = estimate_tokens(["Hello world", "How are you?"])
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_empty_list_returns_zero_or_positive(self):
        """Empty input should not crash — result >= 0."""
        count = estimate_tokens([])
        self.assertGreaterEqual(count, 0)

    def test_longer_text_produces_more_tokens(self):
        short = estimate_tokens(["Hi"])
        long = estimate_tokens(["Hi " * 100])
        self.assertGreater(long, short)


# ── embed_with_retry() ────────────────────────────────────────────────────

class TestEmbedWithRetry(unittest.TestCase):

    def test_success_on_first_attempt(self):
        client = _mock_client()
        response = embed_with_retry(client, "test-model", ["hello"])
        self.assertEqual(len(response.data), 1)
        client.embeddings.create.assert_called_once()

    def test_retries_on_transient_error_then_succeeds(self):
        """Fail once, succeed on second attempt — only 2 API calls total."""
        from openai import APIError

        call_count = [0]

        def side_effect(model, input):
            call_count[0] += 1
            if call_count[0] == 1:
                raise APIError("temporary failure", response=MagicMock(), body={})
            return _fake_embed_response(input)

        client = _mock_client(texts_to_response=side_effect)

        with patch("src.batch_embedding.time.sleep"):          # don't actually wait
            response = embed_with_retry(
                client, "test-model", ["hello"], max_attempts=3
            )

        self.assertEqual(len(response.data), 1)
        self.assertEqual(call_count[0], 2)

    def test_raises_after_all_attempts_exhausted(self):
        from openai import RateLimitError

        client = MagicMock()
        client.embeddings.create.side_effect = RateLimitError(
            "rate limited", response=MagicMock(), body={}
        )

        with patch("src.batch_embedding.time.sleep"):
            with self.assertRaises(RateLimitError):
                embed_with_retry(client, "test-model", ["text"], max_attempts=3)

        self.assertEqual(client.embeddings.create.call_count, 3)

    def test_non_transient_error_retries_then_raises(self):
        """Any exception is retried up to max_attempts then re-raised."""
        client = MagicMock()
        client.embeddings.create.side_effect = ValueError("bad input")

        with patch("src.batch_embedding.time.sleep"):
            with self.assertRaises(ValueError):
                embed_with_retry(
                    client, "test-model", ["text"], max_attempts=2
                )

        self.assertEqual(client.embeddings.create.call_count, 2)


# ── run_batch_embedding() ─────────────────────────────────────────────────

class TestRunBatchEmbedding(unittest.TestCase):

    def test_embeds_all_chunks_when_no_existing_ids(self):
        chunks = _make_chunks(5)
        client = _mock_client()

        result = run_batch_embedding(client, "model", chunks, batch_size=10)

        self.assertEqual(result["summary"]["total_chunks"], 5)
        self.assertEqual(result["summary"]["embedded"], 5)
        self.assertEqual(result["summary"]["skipped_existing"], 0)
        self.assertEqual(result["summary"]["failed"], 0)
        self.assertEqual(len(result["records"]), 5)

    def test_skips_chunks_with_existing_ids(self):
        """Chunks whose id is in existing_ids must not be re-embedded."""
        chunks = _make_chunks(6)
        existing = {"chunk_0", "chunk_2", "chunk_4"}   # 3 already embedded
        client = _mock_client()

        result = run_batch_embedding(
            client, "model", chunks,
            existing_ids=existing, batch_size=10,
        )

        self.assertEqual(result["summary"]["skipped_existing"], 3)
        self.assertEqual(result["summary"]["embedded"], 3)
        self.assertEqual(len(result["records"]), 3)

    def test_skip_all_chunks_produces_zero_api_calls(self):
        """When every chunk is already embedded the API should not be called."""
        chunks = _make_chunks(3)
        existing = {"chunk_0", "chunk_1", "chunk_2"}
        client = _mock_client()

        result = run_batch_embedding(
            client, "model", chunks,
            existing_ids=existing, batch_size=10,
        )

        client.embeddings.create.assert_not_called()
        self.assertEqual(result["summary"]["embedded"], 0)
        self.assertEqual(result["summary"]["skipped_existing"], 3)

    def test_batching_sends_correct_number_of_api_calls(self):
        """10 chunks with batch_size=3 → ceil(10/3) = 4 API requests."""
        chunks = _make_chunks(10)
        client = _mock_client()

        run_batch_embedding(client, "model", chunks, batch_size=3)

        self.assertEqual(client.embeddings.create.call_count, 4)

    def test_record_structure(self):
        """Every returned record must have text, metadata, embedding, and id."""
        chunks = _make_chunks(2)
        client = _mock_client()

        result = run_batch_embedding(client, "model", chunks, batch_size=10)

        for record in result["records"]:
            self.assertIn("text", record)
            self.assertIn("metadata", record)
            self.assertIn("embedding", record)
            self.assertIn("id", record)
            self.assertIsInstance(record["embedding"], list)

    def test_estimated_cost_is_positive_for_nonempty_input(self):
        chunks = _make_chunks(5)
        client = _mock_client()

        result = run_batch_embedding(
            client, "model", chunks,
            batch_size=10, price_per_1k_tokens=0.001,
        )

        self.assertGreater(result["estimated_cost_usd"], 0.0)

    def test_zero_cost_when_nothing_embedded(self):
        """Skipping all chunks means 0 tokens → $0.00 cost."""
        chunks = _make_chunks(3)
        existing = {"chunk_0", "chunk_1", "chunk_2"}
        client = _mock_client()

        result = run_batch_embedding(
            client, "model", chunks,
            existing_ids=existing, batch_size=10,
        )

        self.assertEqual(result["estimated_cost_usd"], 0.0)

    def test_failed_batch_counted_in_summary(self):
        """A batch that exhausts all retries increments summary['failed']."""
        chunks = _make_chunks(4)
        client = MagicMock()
        client.embeddings.create.side_effect = Exception("always fails")

        with patch("src.batch_embedding.time.sleep"):
            result = run_batch_embedding(
                client, "model", chunks,
                batch_size=4, max_attempts=2,
            )

        self.assertEqual(result["summary"]["failed"], 4)
        self.assertEqual(result["summary"]["embedded"], 0)
        self.assertEqual(result["summary"]["batches_failed"], 1)
        self.assertEqual(len(result["records"]), 0)

    def test_partial_failure_records_successful_batches(self):
        """If one batch fails and one succeeds, only the good records are kept."""
        chunks = _make_chunks(6)
        call_count = [0]

        def side_effect(model, input):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("first batch fails")
            return _fake_embed_response(input)

        client = _mock_client(texts_to_response=side_effect)

        with patch("src.batch_embedding.time.sleep"):
            result = run_batch_embedding(
                client, "model", chunks,
                batch_size=3, max_attempts=1,
            )

        self.assertEqual(result["summary"]["failed"], 3)
        self.assertEqual(result["summary"]["embedded"], 3)

    def test_summary_accounting_invariant(self):
        """embedded + failed + skipped_existing == total_chunks always holds."""
        chunks = _make_chunks(10)
        existing = {"chunk_0", "chunk_1"}
        client = _mock_client()

        result = run_batch_embedding(
            client, "model", chunks,
            existing_ids=existing, batch_size=3,
        )
        s = result["summary"]
        self.assertEqual(
            s["embedded"] + s["failed"] + s["skipped_existing"],
            s["total_chunks"],
        )

    def test_input_token_count_increases_with_more_chunks(self):
        small = _make_chunks(2)
        large = _make_chunks(20)
        client = _mock_client()

        r_small = run_batch_embedding(client, "model", small, batch_size=10)
        r_large = run_batch_embedding(client, "model", large, batch_size=10)

        self.assertGreater(
            r_large["summary"]["input_tokens"],
            r_small["summary"]["input_tokens"],
        )


if __name__ == "__main__":
    unittest.main()
