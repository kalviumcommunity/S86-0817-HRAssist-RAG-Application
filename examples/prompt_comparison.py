"""
Example script demonstrating Module 3.13 concepts:
- System vs User roles separation
- Side-by-side prompt variation comparison
- Formatting constraints and refusal rules
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prompt_engine import HRAssistPromptBuilder, compare_prompt_variations, Role

def run_prompt_role_demonstration():
    print("=" * 60)
    print("1. DEMONSTRATING SYSTEM VS USER ROLES")
    print("=" * 60)

    builder = HRAssistPromptBuilder(region="India")
    
    # Payload for a standard HR question
    payload = builder.build_conversation_payload(
        question="How many paid leave days am I entitled to?",
        context="India Leave Policy Section 4: Employees in India are entitled to 20 paid leave days annually.",
        constraints=["Answer in 2 sentences max."]
    )

    print("\nGenerated OpenAI-compatible Messages Payload:")
    print(json.dumps(payload, indent=2))


def run_prompt_comparison_demonstration():
    print("\n" + "=" * 60)
    print("2. SIDE-BY-SIDE PROMPT VARIATION COMPARISON")
    print("=" * 60)

    vague_prompt = "Explain our leave policy."
    clear_prompt = "In 2 sentences, state the annual paid leave entitlement in days for India employees."

    prompts = [vague_prompt, clear_prompt]

    print("\nComparing Prompts:")
    results = compare_prompt_variations(
        prompts=prompts,
        system_instruction="You are a concise, factual HR support assistant. If unsure, say you don't know."
    )

    for res in results:
        print(f"\n[Prompt]: {res['prompt']}")
        print(f"[System Role]: {res['system_instruction']}")
        print(f"[Output]: {res['output']}")


def run_json_and_refusal_demonstration():
    print("\n" + "=" * 60)
    print("3. CONSTRAINED JSON FORMAT & REFUSAL FALLBACK DEMO")
    print("=" * 60)

    builder = HRAssistPromptBuilder(region="US")
    
    # Scenario A: Question with Context -> JSON Output
    json_payload = builder.build_conversation_payload(
        question="What is the pet policy in the office?",
        context="US Workplace Policy Sec 3: Service animals are permitted with prior approval.",
        json_format=True
    )
    
    print("\n[Scenario A - Constrained JSON Payload]:")
    print(json.dumps(json_payload, indent=2))

    # Scenario B: Unanswered Question -> Refusal Rule Trigger
    refusal_payload = builder.build_conversation_payload(
        question="What is the tuition reimbursement budget?",
        context=None  # No context available
    )
    
    print("\n[Scenario B - Refusal Fallback System Prompt]:")
    print(refusal_payload[0]["content"])


if __name__ == "__main__":
    run_prompt_role_demonstration()
    run_prompt_comparison_demonstration()
    run_json_and_refusal_demonstration()
