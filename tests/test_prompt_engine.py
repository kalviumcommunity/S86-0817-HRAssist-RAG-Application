"""
Unit tests for Module 3.13: Prompt Engine & System/User Roles
"""

import unittest
from src.prompt_engine import HRAssistPromptBuilder, Role, compare_prompt_variations


class TestPromptEngine(unittest.TestCase):

    def test_role_separation(self):
        builder = HRAssistPromptBuilder(region="India")
        sys_msg = builder.build_system_prompt()
        usr_msg = builder.build_user_prompt("How many leave days?")

        self.assertEqual(sys_msg.role, Role.SYSTEM)
        self.assertEqual(usr_msg.role, Role.USER)
        self.assertEqual(sys_msg.to_dict()["role"], "system")
        self.assertEqual(usr_msg.to_dict()["role"], "user")

    def test_system_prompt_constraints_and_region(self):
        builder = HRAssistPromptBuilder(region="UK")
        sys_msg = builder.build_system_prompt(
            constraints=["Reply in bullet points only."]
        )

        content = sys_msg.content
        self.assertIn("Target Region Scope: UK.", content)
        self.assertIn("Reply in bullet points only.", content)
        self.assertIn("If you are unsure or if the provided context does not contain sufficient information", content)

    def test_conversation_payload_generation(self):
        builder = HRAssistPromptBuilder(region="Global")
        payload = builder.build_conversation_payload(
            question="What is the work from home policy?",
            context="Global WFH policy: Up to 2 days a week.",
        )

        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["role"], "system")
        self.assertEqual(payload[1]["role"], "user")
        self.assertIn("Global WFH policy: Up to 2 days a week.", payload[1]["content"])

    def test_constrained_json_system_prompt(self):
        builder = HRAssistPromptBuilder(region="India")
        sys_msg = builder.build_constrained_json_system_prompt()

        self.assertIn("Reply with ONLY a valid JSON object", sys_msg.content)
        self.assertIn('"answer": "Concise answer based on policy"', sys_msg.content)

    def test_prompt_comparison_utility(self):
        results = compare_prompt_variations(
            prompts=["Vague prompt", "Clear specific prompt"],
            system_instruction="Be concise."
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["prompt"], "Vague prompt")
        self.assertEqual(results[1]["prompt"], "Clear specific prompt")
        self.assertEqual(results[0]["messages"][0]["role"], "system")
        self.assertEqual(results[0]["messages"][1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
