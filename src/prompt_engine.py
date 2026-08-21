"""
HRAssist - Prompt Engine & System/User Role Management
Module 3.13: Prompt Construction & System/User Roles

This module provides tools for:
1. Defining and validating explicit system and user roles.
2. Constructing grounded, region-aware system prompts with strict constraints.
3. Formatting structured JSON outputs and refusal fallbacks.
4. Comparing prompt variations side-by-side.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class HRAssistPromptBuilder:
    """
    Builder for constructing well-defined, role-separated prompts for HRAssist.
    """

    DEFAULT_PERSONA = (
        "You are HRAssist, an AI-powered HR policy assistant for company employees."
    )
    DEFAULT_REFUSAL_RULE = (
        "If you are unsure or if the provided context does not contain sufficient information to answer "
        "the question, clearly state: 'I couldn't find enough information in the available HR policies to answer this question. Please contact the HR team for clarification.' "
        "Do NOT invent or assume any policy details."
    )

    def __init__(self, region: str = "Global"):
        self.region = region

    def build_system_prompt(
        self,
        persona: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        format_instruction: Optional[str] = None,
    ) -> Message:
        """
        Constructs a structured system message defining role, scope, constraints, and fallback rules.
        """
        prompt_parts = []
        
        # 1. Persona & Role Definition
        active_persona = persona or self.DEFAULT_PERSONA
        prompt_parts.append(active_persona)
        
        # 2. Scope & Region Constraint
        prompt_parts.append(f"Target Region Scope: {self.region}.")
        
        # 3. Explicit Behavioral Constraints & Grounding Rules
        prompt_parts.append("\nBehavioral Rules & Constraints:")
        default_constraints = [
            "Answer strictly using approved HR documentation.",
            "Prioritize policies applicable to the target region.",
            self.DEFAULT_REFUSAL_RULE,
        ]
        all_constraints = default_constraints + (constraints or [])
        for idx, rule in enumerate(all_constraints, 1):
            prompt_parts.append(f"{idx}. {rule}")
            
        # 4. Output Format Instructions
        if format_instruction:
            prompt_parts.append(f"\nFormat Requirement:\n{format_instruction}")

        system_content = "\n".join(prompt_parts)
        return Message(role=Role.SYSTEM, content=system_content)

    def build_user_prompt(self, question: str, context: Optional[str] = None) -> Message:
        """
        Constructs a user message, optionally combining user question with RAG context.
        """
        if context:
            content = (
                f"Retrieved HR Context Documents (Region: {self.region}):\n"
                f"\"\"\"\n{context}\n\"\"\"\n\n"
                f"Employee Question: {question}"
            )
        else:
            content = question
            
        return Message(role=Role.USER, content=content)

    def build_constrained_json_system_prompt(self) -> Message:
        """
        Constructs a system prompt explicitly constraining output to a valid JSON format.
        """
        format_instruction = (
            "Reply with ONLY a valid JSON object matching this exact schema:\n"
            "{\n"
            '  "answer": "Concise answer based on policy",\n'
            '  "source": "Document name and section referenced",\n'
            '  "found": true | false\n'
            "}"
        )
        return self.build_system_prompt(format_instruction=format_instruction)

    def build_conversation_payload(
        self,
        question: str,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        json_format: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Returns a complete list of message dictionaries ready for LLM completion API call.
        """
        if json_format:
            system_msg = self.build_constrained_json_system_prompt()
        else:
            system_msg = self.build_system_prompt(constraints=constraints)
            
        user_msg = self.build_user_prompt(question=question, context=context)
        
        return [system_msg.to_dict(), user_msg.to_dict()]


def compare_prompt_variations(
    prompts: List[str],
    system_instruction: str = "You are concise and factual.",
    api_client: Optional[Any] = None,
    model_name: str = "gpt-3.5-turbo",
) -> List[Dict[str, Any]]:
    """
    Utility function to compare prompt variations side-by-side.
    If no API client is passed, returns formatted mock payloads for evaluation.
    """
    results = []
    for p in prompts:
        messages = [
            {"role": Role.SYSTEM.value, "content": system_instruction},
            {"role": Role.USER.value, "content": p},
        ]
        
        if api_client:
            try:
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                output = response.choices[0].message.content
            except Exception as e:
                output = f"API Error: {str(e)}"
        else:
            output = f"[Simulated Output for prompt: '{p}']"
            
        results.append({
            "prompt": p,
            "system_instruction": system_instruction,
            "messages": messages,
            "output": output,
        })
    return results
