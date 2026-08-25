import unittest

from src.prompt_engine import total_tokens, trim_history, summarize_history


class TestContextWindowManager(unittest.TestCase):

    def test_total_tokens_counts_all_messages(self):
        messages = [
            {"role": "system", "content": "You are an HR assistant."},
            {"role": "user", "content": "How many leave days do I get?"},
        ]

        token_total = total_tokens(messages)

        self.assertGreater(token_total, 0)
        self.assertEqual(total_tokens(messages[:1]), total_tokens([messages[0]]))

    def test_trim_history_keeps_system_and_cuts_oldest_turns(self):
        messages = [
            {"role": "system", "content": "You are an HR assistant."},
            {"role": "user", "content": "A" * 300},
            {"role": "assistant", "content": "B" * 300},
            {"role": "user", "content": "C" * 300},
        ]

        trim_history(messages, budget=60)

        self.assertEqual(messages[0]["role"], "system")
        self.assertLessEqual(total_tokens(messages), 60)
        self.assertEqual(len(messages), 2)

    def test_summarize_history_replaces_old_turns_with_summary(self):
        messages = [
            {"role": "system", "content": "You are an HR assistant."},
            {"role": "user", "content": "A" * 180},
            {"role": "assistant", "content": "B" * 180},
            {"role": "user", "content": "C" * 180},
        ]

        summarized = summarize_history(messages, budget=75)

        self.assertTrue(any("summary" in msg.get("content", "").lower() for msg in summarized))
        self.assertLessEqual(total_tokens(summarized), 75)


if __name__ == "__main__":
    unittest.main()
