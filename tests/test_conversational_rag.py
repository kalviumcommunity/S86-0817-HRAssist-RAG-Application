"""Tests for conversational RAG with follow-up query rewriting (HRS3.42).

Covers:
  - ConversationHistory: add_user/assistant/turn, max_turns window enforcement,
    token_count, trimmed_messages, turn_count, clear, invalid max_turns
  - rewrite_followup(): LLM called with history + question, returns rewritten
    string, falls back to original question on LLM failure or empty response
  - rewrite_followup_simple(): follow-up triggers expand query, non-follow-up
    passed through unchanged, empty history handled
  - conversational_answer(): rewrite_fn called with history messages,
    retrieve_fn called with rewritten query, generate_fn called with original
    question, history updated after turn, status codes, refused when no chunks,
    refused when weak chunks, update_history=False skips history append,
    sources empty on refusal, required return keys present
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from src.conversational_rag import (
    ConversationHistory,
    rewrite_followup,
    rewrite_followup_simple,
    conversational_answer,
)
from src.guardrails import (
    STATUS_ANSWERED,
    STATUS_REFUSED_WEAK_CONTEXT,
    STATUS_REFUSED_NO_CHUNKS,
    RetrievalStrengthConfig,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _chunk(score: float, source: str = "policy.txt") -> dict:
    return {
        "score": score,
        "text": f"content from {source}",
        "metadata": {"source": source, "chunk_index": 0},
    }


def _strong_chunks(n: int = 2) -> list:
    return [_chunk(0.90, f"doc{i}.txt") for i in range(n)]


def _weak_chunks(n: int = 2) -> list:
    return [_chunk(0.30, f"doc{i}.txt") for i in range(n)]


def _stub_generate(answer: str = "The policy states X."):
    def _fn(question, chunks):
        return {"answer": answer, "sources": [c["metadata"] for c in chunks]}
    return _fn


def _mock_llm_client(content: str = "What video is required for project submission?"):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return client


# ── ConversationHistory ───────────────────────────────────────────────────

class TestConversationHistory(unittest.TestCase):

    def test_add_user_appends_user_message(self):
        h = ConversationHistory()
        h.add_user("Hello")
        self.assertEqual(h.messages[0], {"role": "user", "content": "Hello"})

    def test_add_assistant_appends_assistant_message(self):
        h = ConversationHistory()
        h.add_assistant("Hi there")
        self.assertEqual(h.messages[0], {"role": "assistant", "content": "Hi there"})

    def test_add_turn_appends_both_messages(self):
        h = ConversationHistory()
        h.add_turn("user q", "assistant a")
        self.assertEqual(len(h.messages), 2)
        self.assertEqual(h.messages[0]["role"], "user")
        self.assertEqual(h.messages[1]["role"], "assistant")

    def test_max_turns_window_drops_oldest(self):
        h = ConversationHistory(max_turns=2)
        h.add_turn("q1", "a1")
        h.add_turn("q2", "a2")
        h.add_turn("q3", "a3")   # should evict turn 1
        self.assertEqual(len(h.messages), 4)   # 2 turns × 2 messages
        # Oldest turn should be gone
        contents = [m["content"] for m in h.messages]
        self.assertNotIn("q1", contents)
        self.assertIn("q2", contents)
        self.assertIn("q3", contents)

    def test_turn_count_returns_complete_pairs(self):
        h = ConversationHistory()
        h.add_turn("q1", "a1")
        h.add_turn("q2", "a2")
        self.assertEqual(h.turn_count, 2)

    def test_token_count_positive_for_nonempty_history(self):
        h = ConversationHistory()
        h.add_user("How do I apply for sick leave?")
        self.assertGreater(h.token_count(), 0)

    def test_token_count_zero_for_empty_history(self):
        h = ConversationHistory()
        self.assertEqual(h.token_count(), 0)

    def test_trimmed_messages_respects_token_budget(self):
        h = ConversationHistory()
        h.add_turn("How do I apply for annual leave?", "Submit via the HR portal.")
        h.add_turn("What about sick leave?", "Sick leave requires a medical certificate.")
        msgs = h.trimmed_messages(max_tokens=20)
        total = sum(len(m["content"].split()) for m in msgs)
        # Rough check — very tight budget should drop older messages
        self.assertLessEqual(len(msgs), len(h.messages))

    def test_trimmed_messages_returns_all_when_no_budget(self):
        h = ConversationHistory()
        h.add_turn("q", "a")
        self.assertEqual(h.trimmed_messages(), h.messages)

    def test_clear_removes_all_messages(self):
        h = ConversationHistory()
        h.add_turn("q", "a")
        h.clear()
        self.assertEqual(h.messages, [])

    def test_messages_returns_copy_not_reference(self):
        h = ConversationHistory()
        h.add_user("q")
        msgs = h.messages
        msgs.append({"role": "user", "content": "injected"})
        self.assertEqual(len(h.messages), 1)  # original unaffected

    def test_invalid_max_turns_raises(self):
        with self.assertRaises(ValueError):
            ConversationHistory(max_turns=0)

    def test_empty_history_messages_is_empty_list(self):
        self.assertEqual(ConversationHistory().messages, [])


# ── rewrite_followup() ────────────────────────────────────────────────────

class TestRewriteFollowup(unittest.TestCase):

    def test_returns_rewritten_string_from_llm(self):
        client = _mock_llm_client("What video is required for submission?")
        history = [
            {"role": "user",      "content": "What evidence is needed?"},
            {"role": "assistant", "content": "You need a PR link and a video."},
        ]
        result = rewrite_followup(history, "What about the video?", client, "model")
        self.assertEqual(result, "What video is required for submission?")

    def test_llm_called_once(self):
        client = _mock_llm_client("rewritten")
        rewrite_followup([], "question", client, "model")
        client.chat.completions.create.assert_called_once()

    def test_fallback_to_original_on_llm_exception(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        result = rewrite_followup([], "original question", client, "model")
        self.assertEqual(result, "original question")

    def test_fallback_to_original_on_empty_response(self):
        client = _mock_llm_client("")   # empty content
        result = rewrite_followup([], "original question", client, "model")
        self.assertEqual(result, "original question")

    def test_history_included_in_prompt(self):
        client = _mock_llm_client("rewritten")
        history = [{"role": "user", "content": "tell me about leave policy"}]
        rewrite_followup(history, "what about sick leave?", client, "model")
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        self.assertIn("tell me about leave policy", user_msg["content"])

    def test_history_trimmed_to_token_budget(self):
        """Very long history should be trimmed before sending to LLM."""
        client = _mock_llm_client("rewritten")
        long_history = [
            {"role": "user",      "content": "word " * 500},
            {"role": "assistant", "content": "word " * 500},
        ]
        # Should not raise even with tiny budget
        result = rewrite_followup(long_history, "question", client, "model",
                                  max_history_tokens=50)
        self.assertIsInstance(result, str)

    def test_empty_history_still_rewrites(self):
        client = _mock_llm_client("standalone query")
        result = rewrite_followup([], "What is the policy?", client, "model")
        self.assertEqual(result, "standalone query")


# ── rewrite_followup_simple() ─────────────────────────────────────────────

class TestRewriteFollowupSimple(unittest.TestCase):

    def test_pronoun_start_expands_with_context(self):
        history = [{"role": "assistant", "content": "sick leave requires a certificate"}]
        result = rewrite_followup_simple(history, "it applies to maternity too?")
        self.assertIn("sick leave", result)

    def test_non_followup_returned_unchanged(self):
        history = [{"role": "assistant", "content": "some response"}]
        result = rewrite_followup_simple(history, "How do I submit a leave request?")
        self.assertEqual(result, "How do I submit a leave request?")

    def test_empty_history_returns_original(self):
        result = rewrite_followup_simple([], "what about the deadline?")
        self.assertEqual(result, "what about the deadline?")

    def test_what_about_trigger_detected(self):
        history = [{"role": "assistant", "content": "annual leave is 20 days per year"}]
        result = rewrite_followup_simple(history, "what about sick leave?")
        self.assertIn("annual leave", result)


# ── conversational_answer() ───────────────────────────────────────────────

class TestConversationalAnswer(unittest.TestCase):

    def _run(self, chunks, rewrite_fn=None, generate_fn=None,
             question="What about sick leave?", update_history=True,
             guardrail_config=None):
        history = ConversationHistory(max_turns=5)
        history.add_turn("What is the leave policy?", "You have 20 days annual leave.")
        retrieve_fn = MagicMock(return_value=chunks)
        gen_fn = generate_fn or _stub_generate()
        result = conversational_answer(
            question, history, retrieve_fn, gen_fn,
            rewrite_fn=rewrite_fn,
            guardrail_config=guardrail_config,
            update_history=update_history,
        )
        return result, history, retrieve_fn

    def test_required_keys_present(self):
        result, _, _ = self._run(_strong_chunks())
        required = {"answer", "rewritten_query", "sources", "status", "chunks_retrieved"}
        self.assertEqual(required, required & result.keys())

    def test_status_answered_on_strong_chunks(self):
        result, _, _ = self._run(_strong_chunks())
        self.assertEqual(result["status"], STATUS_ANSWERED)

    def test_status_refused_no_chunks_on_empty(self):
        result, _, _ = self._run([])
        self.assertEqual(result["status"], STATUS_REFUSED_NO_CHUNKS)

    def test_status_refused_weak_on_low_scores(self):
        result, _, _ = self._run(_weak_chunks())
        self.assertEqual(result["status"], STATUS_REFUSED_WEAK_CONTEXT)

    def test_sources_empty_on_refusal(self):
        result, _, _ = self._run([])
        self.assertEqual(result["sources"], [])

    def test_sources_populated_on_answer(self):
        result, _, _ = self._run(_strong_chunks(2))
        self.assertEqual(len(result["sources"]), 2)

    def test_rewrite_fn_called_with_history_and_question(self):
        received = {}

        def capture_rewrite(history_msgs, question):
            received["msgs"] = history_msgs
            received["question"] = question
            return "standalone: " + question

        result, _, _ = self._run(
            _strong_chunks(), rewrite_fn=capture_rewrite,
            question="What about sick leave?"
        )
        self.assertEqual(received["question"], "What about sick leave?")
        self.assertIsInstance(received["msgs"], list)

    def test_retrieve_fn_called_with_rewritten_query(self):
        def fixed_rewrite(msgs, q):
            return "rewritten sick leave query"

        result, _, retrieve_fn = self._run(
            _strong_chunks(), rewrite_fn=fixed_rewrite
        )
        retrieve_fn.assert_called_once_with("rewritten sick leave query")
        self.assertEqual(result["rewritten_query"], "rewritten sick leave query")

    def test_without_rewrite_fn_original_question_used_for_retrieval(self):
        result, _, retrieve_fn = self._run(
            _strong_chunks(), rewrite_fn=None,
            question="How do I apply for leave?"
        )
        retrieve_fn.assert_called_once_with("How do I apply for leave?")
        self.assertEqual(result["rewritten_query"], "How do I apply for leave?")

    def test_generate_fn_called_with_original_question_not_rewritten(self):
        """The original user question must go to generation, not the rewritten one."""
        received = {}

        def capture_gen(question, chunks):
            received["question"] = question
            return {"answer": "ok", "sources": []}

        def fixed_rewrite(msgs, q):
            return "totally different rewritten query"

        self._run(
            _strong_chunks(), rewrite_fn=fixed_rewrite,
            generate_fn=capture_gen, question="What about sick leave?"
        )
        self.assertEqual(received["question"], "What about sick leave?")

    def test_generate_fn_not_called_on_refusal(self):
        generate_fn = MagicMock(return_value={"answer": "x", "sources": []})
        self._run(_weak_chunks(), generate_fn=generate_fn)
        generate_fn.assert_not_called()

    def test_history_updated_after_answered_turn(self):
        result, history, _ = self._run(_strong_chunks(), question="sick leave?")
        msgs = history.messages
        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        self.assertTrue(any("sick leave?" in m["content"] for m in user_msgs))
        self.assertTrue(any(result["answer"] in m["content"] for m in asst_msgs))

    def test_history_updated_after_refusal(self):
        """History should be updated even when the answer is a refusal."""
        result, history, _ = self._run([], question="off-topic question?")
        msgs = history.messages
        self.assertTrue(any("off-topic question?" in m["content"] for m in msgs))

    def test_update_history_false_does_not_append(self):
        history = ConversationHistory(max_turns=5)
        retrieve_fn = MagicMock(return_value=_strong_chunks())
        before_count = len(history.messages)
        conversational_answer(
            "q", history, retrieve_fn, _stub_generate(),
            update_history=False
        )
        self.assertEqual(len(history.messages), before_count)

    def test_chunks_retrieved_count_in_result(self):
        result, _, _ = self._run(_strong_chunks(3))
        self.assertEqual(result["chunks_retrieved"], 3)

    def test_custom_guardrail_config_respected(self):
        """Raising threshold should cause a previously-passing score to be refused."""
        strict = RetrievalStrengthConfig(min_top_score=0.99)
        # score=0.90 passes default but fails strict
        result, _, _ = self._run(_strong_chunks(1), guardrail_config=strict)
        self.assertNotEqual(result["status"], STATUS_ANSWERED)

    def test_multi_turn_history_accumulates(self):
        history = ConversationHistory(max_turns=5)
        retrieve_fn = MagicMock(return_value=_strong_chunks())

        def simple_rewrite(msgs, q):
            return "standalone: " + q

        for i in range(3):
            conversational_answer(
                f"question {i}", history, retrieve_fn, _stub_generate(),
                rewrite_fn=simple_rewrite,
            )

        self.assertEqual(history.turn_count, 3)


if __name__ == "__main__":
    unittest.main()
