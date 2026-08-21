"""
Unit tests for Module 3.16: Model Parameters & Output Control
"""

import unittest
from src.model_config import ModelConfig, LLMController


class TestModelConfig(unittest.TestCase):

    def test_default_model_config_validation(self):
        config = ModelConfig()
        config.validate()  # Should not raise exception
        kwargs = config.to_api_kwargs()

        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 300)
        self.assertIn("stop", kwargs)

    def test_invalid_temperature(self):
        config = ModelConfig(temperature=2.5)
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_max_tokens(self):
        config = ModelConfig(max_tokens=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_rag_factual_preset(self):
        preset = ModelConfig.get_rag_factual_preset(max_tokens=150)
        self.assertEqual(preset.temperature, 0.1)
        self.assertEqual(preset.max_tokens, 150)
        self.assertIn("\n\nUser:", preset.stop)

    def test_creative_preset(self):
        preset = ModelConfig.get_creative_preset()
        self.assertEqual(preset.temperature, 1.0)
        self.assertEqual(preset.max_tokens, 800)

    def test_llm_controller_generation_simulation(self):
        config = ModelConfig.get_rag_factual_preset()
        controller = LLMController(config=config)
        messages = [{"role": "user", "content": "What is the policy?"}]

        result = controller.generate_completion(messages)
        self.assertIn("content", result)
        self.assertIn("finish_reason", result)
        self.assertEqual(result["parameters"]["temperature"], 0.1)

    def test_temperature_variation_comparison(self):
        controller = LLMController()
        messages = [{"role": "user", "content": "Sample question"}]

        variations = controller.compare_temperature_variations(messages, temperatures=[0.0, 1.5])
        self.assertEqual(len(variations), 2)
        self.assertEqual(variations[0]["temperature"], 0.0)
        self.assertEqual(variations[1]["temperature"], 1.5)


if __name__ == "__main__":
    unittest.main()
