"""
Example script demonstrating Module 3.16 concepts:
- Model parameter tuning (temperature, max_tokens, stop sequences, top_p)
- Grounded RAG factual presets vs Creative presets
- Temperature impact comparison (0.0 vs 1.0)
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prompt_engine import HRAssistPromptBuilder
from src.model_config import ModelConfig, LLMController


def run_rag_preset_demonstration():
    print("=" * 60)
    print("1. GROUNDED RAG FACTUAL PRESET CONFIGURATION")
    print("=" * 60)

    # 1. Load grounded RAG preset
    rag_config = ModelConfig.get_rag_factual_preset(model="gpt-3.5-turbo", max_tokens=250)
    print("\nRAG Factual Preset API Parameters:")
    print(json.dumps(rag_config.to_api_kwargs(), indent=2))

    # 2. Build prompt payload using HRAssistPromptBuilder
    builder = HRAssistPromptBuilder(region="India")
    messages = builder.build_conversation_payload(
        question="How many leave days am I entitled to?",
        context="India Leave Policy: Employees are entitled to 20 paid leave days annually."
    )

    controller = LLMController(config=rag_config)
    result = controller.generate_completion(messages=messages)

    print("\nGenerated RAG Answer:")
    print(f"Content: {result['content']}")
    print(f"Finish Reason: {result['finish_reason']}")


def run_temperature_comparison_demonstration():
    print("\n" + "=" * 60)
    print("2. TEMPERATURE IMPACT COMPARISON (0.0 vs 1.0)")
    print("=" * 60)

    builder = HRAssistPromptBuilder(region="US")
    messages = builder.build_conversation_payload(
        question="What is the work from home policy?",
        context="US WFH Policy: Remote work is permitted up to 2 days per week with manager approval."
    )

    controller = LLMController()
    comparisons = controller.compare_temperature_variations(
        messages=messages,
        temperatures=[0.0, 1.0]
    )

    for item in comparisons:
        temp = item["temperature"]
        print(f"\n--- Temperature: {temp} ---")
        print(f"Parameters: {item['parameters']}")
        print(f"Response: {item['response']}")


def run_max_tokens_and_stop_demo():
    print("\n" + "=" * 60)
    print("3. MAX_TOKENS & STOP SEQUENCES CONTROL")
    print("=" * 60)

    # Config with strict token limit (50 tokens) and custom stop sequence
    capped_config = ModelConfig(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=50,
        stop=["\n\n", "User:", "Section"]
    )

    print("Configured Parameters with strict max_tokens (50) & stop sequences:")
    print(json.dumps(capped_config.to_api_kwargs(), indent=2))


if __name__ == "__main__":
    run_rag_preset_demonstration()
    run_temperature_comparison_demonstration()
    run_max_tokens_and_stop_demo()
