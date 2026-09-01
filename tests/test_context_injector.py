"""Tests for context injection and prompt augmentation (HRS3.38).

Covers:
  - format_chunk(): source marker format, chunk_index present/absent,
    text included, custom metadata
  - assemble_context(): single chunk, multiple chunks, token budget respected,
    chunks_included count, used_tokens <= max_tokens, separator accounting,
    invalid max_tokens raises, empty chunks list
  - assemble_context_with_truncation(): last chunk truncated when partial fit,
    truncation suffix present, full chunks still fit whole
  - build_augmented_prompt(): required keys, context tokens within budget,
    chunks_dropped = total - included, sources_used length matches included,
    no-context fallback when budget is tiny, custom system_instruction,
    messages structure, grounding instruction in prompt, use_truncation flag,
    invalid max_context_tokens raises
"""

import unittest

from src.context_injector import (
    format_chunk,
    assemble_context,
    assemble_context_with_truncation,
    build_augmented_prompt,
    _GROUNDED_SYSTEM_INSTRUCTION,
    DEFAULT_MAX_CONTEXT_TOKENS,
)
from src.token_counter import count_tokens


# ── Fixtures ───────────────────────────────────────────────────────────────

def _chunk(source: str, text: str, chunk_index: int = 0) -> dict:
    return {
        "score": 0.9,
        "text": text,
        "metadata": {"source": source, "chunk_index": chunk_index},
    }


def _short_chunks(n: int = 3) -> list:
    return [
        _chunk(f"doc{i}.txt", f"HR policy sentence number {i}.", chunk_index=i)
        for i in range(1, n + 1)
    ]


# ── format_chunk() ────────────────────────────────────────────────────────

class TestFormatChunk(unittest.TestCase):

    def test_marker_contains_index_source_and_chunk_index(self):
        chunk = _chunk("policy.txt", "Some text.", chunk_index=3)
        formatted = format_chunk(1, chunk)
        self.assertIn("[1]", formatted)
        self.assertIn("policy.txt", formatted)
        self.assertIn("#3", formatted)

    def test_text_appears_in_output(self):
        chunk = _chunk("doc.md", "Important HR information.")
        formatted = format_chunk(2, chunk)
        self.assertIn("Important HR information.", formatted)

    def test_marker_on_first_line_text_on_second(self):
        chunk = _chunk("source.txt", "The text content.", chunk_index=0)
        lines = format_chunk(1, chunk).split("\n")
        self.assertIn("[1]", lines[0])
        self.assertIn("The text content.", lines[1])

    def test_chunk_index_absent_omits_hash(self):
        """When chunk_index is not in metadata the '#N' part is omitted."""
        chunk = {
            "text": "No index here.",
            "metadata": {"source": "orphan.txt"},
        }
        formatted = format_chunk(1, chunk)
        self.assertNotIn("#", formatted)
        self.assertIn("[1] orphan.txt", formatted)

    def test_index_increments_correctly(self):
        chunks = _short_chunks(3)
        for i, chunk in enumerate(chunks, start=1):
            formatted = format_chunk(i, chunk)
            self.assertIn(f"[{i}]", formatted)

    def test_missing_text_key_produces_empty_body(self):
        chunk = {"metadata": {"source": "a.txt", "chunk_index": 0}}
        formatted = format_chunk(1, chunk)
        self.assertIn("[1] a.txt#0", formatted)

    def test_missing_metadata_uses_unknown_source(self):
        chunk = {"text": "some text"}
        formatted = format_chunk(1, chunk)
        self.assertIn("unknown", formatted)


# ── assemble_context() ────────────────────────────────────────────────────

class TestAssembleContext(unittest.TestCase):

    def test_single_chunk_returned_in_context(self):
        chunks = _short_chunks(1)
        context, tokens, included = assemble_context(chunks, max_tokens=500)
        self.assertEqual(included, 1)
        self.assertIn("doc1.txt", context)

    def test_all_chunks_included_when_budget_is_large(self):
        chunks = _short_chunks(3)
        _, _, included = assemble_context(chunks, max_tokens=10_000)
        self.assertEqual(included, 3)

    def test_token_budget_respected(self):
        chunks = _short_chunks(10)
        _, used_tokens, _ = assemble_context(chunks, max_tokens=200)
        self.assertLessEqual(used_tokens, 200)

    def test_used_tokens_matches_actual_context_tokens(self):
        chunks = _short_chunks(3)
        context, used_tokens, _ = assemble_context(chunks, max_tokens=10_000)
        self.assertEqual(count_tokens(context), used_tokens)

    def test_chunks_excluded_when_budget_too_small(self):
        # Each chunk is ~10-15 tokens; budget of 30 fits only 1-2
        chunks = _short_chunks(5)
        _, _, included = assemble_context(chunks, max_tokens=30)
        self.assertLess(included, 5)

    def test_chunks_separated_by_divider(self):
        chunks = _short_chunks(2)
        context, _, _ = assemble_context(chunks, max_tokens=10_000)
        self.assertIn("---", context)

    def test_single_chunk_has_no_separator(self):
        chunks = _short_chunks(1)
        context, _, _ = assemble_context(chunks, max_tokens=10_000)
        self.assertNotIn("---", context)

    def test_empty_chunks_returns_empty_string(self):
        context, tokens, included = assemble_context([], max_tokens=500)
        self.assertEqual(context, "")
        self.assertEqual(tokens, 0)
        self.assertEqual(included, 0)

    def test_invalid_max_tokens_raises(self):
        with self.assertRaises(ValueError):
            assemble_context(_short_chunks(1), max_tokens=0)

    def test_order_preserved(self):
        """Chunks must appear in input order (best-ranked first)."""
        chunks = _short_chunks(3)
        context, _, _ = assemble_context(chunks, max_tokens=10_000)
        pos1 = context.index("doc1.txt")
        pos2 = context.index("doc2.txt")
        pos3 = context.index("doc3.txt")
        self.assertLess(pos1, pos2)
        self.assertLess(pos2, pos3)


# ── assemble_context_with_truncation() ───────────────────────────────────

class TestAssembleContextWithTruncation(unittest.TestCase):

    def test_full_chunks_included_when_budget_is_large(self):
        chunks = _short_chunks(2)
        _, _, included = assemble_context_with_truncation(chunks, max_tokens=10_000)
        self.assertEqual(included, 2)

    def test_truncation_suffix_present_when_chunk_is_cut(self):
        # Use a very large chunk and a tight budget so it gets truncated
        long_text = "word " * 300   # ~300 tokens
        chunks = [_chunk("big.txt", long_text, chunk_index=0)]
        context, _, _ = assemble_context_with_truncation(
            chunks, max_tokens=50, truncation_suffix="... [truncated]"
        )
        self.assertIn("... [truncated]", context)

    def test_token_budget_respected_after_truncation(self):
        long_text = "word " * 500
        chunks = [_chunk("big.txt", long_text)]
        context, used_tokens, _ = assemble_context_with_truncation(
            chunks, max_tokens=60
        )
        self.assertLessEqual(used_tokens, 60)

    def test_invalid_max_tokens_raises(self):
        with self.assertRaises(ValueError):
            assemble_context_with_truncation(_short_chunks(1), max_tokens=0)

    def test_empty_chunks_returns_empty(self):
        context, tokens, included = assemble_context_with_truncation([], max_tokens=500)
        self.assertEqual(context, "")
        self.assertEqual(included, 0)


# ── build_augmented_prompt() ──────────────────────────────────────────────

class TestBuildAugmentedPrompt(unittest.TestCase):

    def test_required_keys_present(self):
        result = build_augmented_prompt("What is sick leave?", _short_chunks(2))
        required = {
            "prompt", "messages", "context_tokens",
            "total_prompt_tokens", "chunks_included",
            "chunks_dropped", "sources_used",
        }
        self.assertEqual(required, required & result.keys())

    def test_chunks_dropped_plus_included_equals_total(self):
        chunks = _short_chunks(5)
        result = build_augmented_prompt("question", chunks)
        self.assertEqual(
            result["chunks_included"] + result["chunks_dropped"],
            len(chunks),
        )

    def test_sources_used_length_matches_chunks_included(self):
        chunks = _short_chunks(3)
        result = build_augmented_prompt("question", chunks)
        self.assertEqual(len(result["sources_used"]), result["chunks_included"])

    def test_context_tokens_within_budget(self):
        chunks = _short_chunks(10)
        budget = 200
        result = build_augmented_prompt("q", chunks, max_context_tokens=budget)
        self.assertLessEqual(result["context_tokens"], budget)

    def test_grounding_instruction_in_prompt(self):
        result = build_augmented_prompt("How do I apply for leave?", _short_chunks(1))
        self.assertIn("only", result["prompt"].lower())

    def test_question_appears_in_prompt(self):
        result = build_augmented_prompt(
            "What is the sick leave policy?", _short_chunks(1)
        )
        self.assertIn("What is the sick leave policy?", result["prompt"])

    def test_source_markers_appear_in_prompt(self):
        chunks = _short_chunks(2)
        result = build_augmented_prompt("q", chunks)
        self.assertIn("[1]", result["prompt"])
        self.assertIn("[2]", result["prompt"])

    def test_custom_system_instruction_used(self):
        custom = "You are a strict HR bot. Use only approved documents."
        result = build_augmented_prompt("q", _short_chunks(1),
                                        system_instruction=custom)
        self.assertIn(custom, result["prompt"])
        self.assertNotIn(_GROUNDED_SYSTEM_INSTRUCTION, result["prompt"])

    def test_messages_structure_has_system_and_user(self):
        result = build_augmented_prompt("q", _short_chunks(1))
        roles = [m["role"] for m in result["messages"]]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

    def test_messages_system_content_is_instruction(self):
        result = build_augmented_prompt("q", _short_chunks(1))
        system_msg = next(m for m in result["messages"] if m["role"] == "system")
        self.assertIn("grounded", system_msg["content"].lower())

    def test_total_prompt_tokens_matches_actual_count(self):
        result = build_augmented_prompt("What is annual leave?", _short_chunks(2))
        self.assertEqual(
            count_tokens(result["prompt"]),
            result["total_prompt_tokens"],
        )

    def test_no_context_fallback_when_budget_exhausted(self):
        """With a budget of 1 token no chunks can fit; prompt still valid."""
        result = build_augmented_prompt(
            "question", _short_chunks(3), max_context_tokens=1
        )
        self.assertEqual(result["chunks_included"], 0)
        self.assertEqual(result["chunks_dropped"], 3)
        self.assertIn("question", result["prompt"])

    def test_empty_chunks_list_handled(self):
        result = build_augmented_prompt("q", [])
        self.assertEqual(result["chunks_included"], 0)
        self.assertEqual(result["chunks_dropped"], 0)
        self.assertIn("q", result["prompt"])

    def test_invalid_max_context_tokens_raises(self):
        with self.assertRaises(ValueError):
            build_augmented_prompt("q", _short_chunks(1), max_context_tokens=0)

    def test_use_truncation_flag_accepted(self):
        """use_truncation=True should not raise and should produce valid output."""
        result = build_augmented_prompt(
            "q", _short_chunks(3), max_context_tokens=50, use_truncation=True
        )
        self.assertIn("prompt", result)

    def test_sources_used_contain_source_key(self):
        chunks = _short_chunks(2)
        result = build_augmented_prompt("q", chunks)
        for meta in result["sources_used"]:
            self.assertIn("source", meta)

    def test_large_context_budget_includes_all_chunks(self):
        chunks = _short_chunks(5)
        result = build_augmented_prompt("q", chunks, max_context_tokens=50_000)
        self.assertEqual(result["chunks_included"], 5)
        self.assertEqual(result["chunks_dropped"], 0)

    def test_prompt_contains_context_section_header(self):
        result = build_augmented_prompt("q", _short_chunks(1))
        self.assertIn("Context:", result["prompt"])


if __name__ == "__main__":
    unittest.main()
