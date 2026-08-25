import unittest

from prompts.answer import ANSWER_TEMPLATE, render_template


class TestPromptTemplates(unittest.TestCase):

    def test_template_has_named_placeholders(self):
        self.assertIn("{context}", ANSWER_TEMPLATE)
        self.assertIn("{question}", ANSWER_TEMPLATE)

    def test_render_template_injects_runtime_values(self):
        rendered = render_template(
            ANSWER_TEMPLATE,
            context="Leave policy: up to 20 days.",
            question="How many leave days do I get?",
        )

        self.assertIn("Leave policy: up to 20 days.", rendered)
        self.assertIn("How many leave days do I get?", rendered)
        self.assertIn("Answer ONLY from the context.", rendered)

    def test_reusable_template_can_be_shared_across_features(self):
        first = render_template(ANSWER_TEMPLATE, context="Benefits", question="What is covered?")
        second = render_template(ANSWER_TEMPLATE, context="Leave", question="How many days?")

        self.assertIn("Benefits", first)
        self.assertIn("Leave", second)
        self.assertEqual(first.count("Context:"), 1)
        self.assertEqual(second.count("Context:"), 1)


if __name__ == "__main__":
    unittest.main()
