"""
HRAssist - Model Parameters & Output Control
Module 3.16: Model Parameters & Output Control

This module provides tools for:
1. Defining and validating model hyper-parameters (temperature, max_tokens, stop, top_p).
2. Setting grounded RAG presets (low temperature, strict token limits, stop sequences).
3. Managing model API calls and parameter variation comparisons.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union


@dataclass
class ModelConfig:
    """
    Configuration dataclass for controlling LLM sampling parameters.
    """
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 300
    stop: Optional[List[str]] = field(default_factory=lambda: ["\n\nUser:", "Employee:"])
    top_p: Optional[float] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def validate(self) -> None:
        """Validates parameter ranges."""
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

    def to_api_kwargs(self) -> Dict[str, Any]:
        """
        Converts configuration to OpenAI-compatible API request parameters.
        Omits None values.
        """
        self.validate()
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.stop:
            kwargs["stop"] = self.stop
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs

    @classmethod
    def get_rag_factual_preset(cls, model: str = "gpt-3.5-turbo", max_tokens: int = 300) -> "ModelConfig":
        """
        Returns a tuned preset for grounded factual RAG Q&A:
        - Low temperature (0.1) for repeatable, factual responses.
        - Cap max_tokens (300) for conciseness and cost control.
        - Stop sequences to halt early generation.
        """
        return cls(
            model=model,
            temperature=0.1,
            max_tokens=max_tokens,
            stop=["\n\nUser:", "\n\nEmployee:"],
            top_p=None
        )

    @classmethod
    def get_creative_preset(cls, model: str = "gpt-3.5-turbo", max_tokens: int = 800) -> "ModelConfig":
        """
        Returns a preset for creative / brainstorming tasks (higher temperature).
        """
        return cls(
            model=model,
            temperature=1.0,
            max_tokens=max_tokens,
            stop=None,
            top_p=None
        )


class LLMController:
    """
    Controller for managing completion API requests with specified parameters.
    """

    def __init__(self, config: Optional[ModelConfig] = None, api_client: Optional[Any] = None):
        self.config = config or ModelConfig.get_rag_factual_preset()
        self.api_client = api_client

    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        override_config: Optional[ModelConfig] = None
    ) -> Dict[str, Any]:
        """
        Executes completion API call using configured parameters.
        Falls back to a detailed simulation dictionary if no API client is passed.
        """
        active_config = override_config or self.config
        api_kwargs = active_config.to_api_kwargs()

        if self.api_client:
            try:
                response = self.api_client.chat.completions.create(
                    messages=messages,
                    **api_kwargs
                )
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
            except Exception as e:
                content = f"API Error: {str(e)}"
                finish_reason = "error"
        else:
            # Simulated completion output reflecting parameter settings
            temp_desc = "deterministic/factual" if active_config.temperature <= 0.2 else "creative/variable"
            content = (
                f"[Simulated RAG Answer | Temp={active_config.temperature} ({temp_desc}) | "
                f"MaxTokens={active_config.max_tokens} | Stop={active_config.stop}]: "
                f"According to the policy, employees are entitled to 20 paid leave days annually."
            )
            finish_reason = "stop"

        return {
            "content": content,
            "finish_reason": finish_reason,
            "parameters": api_kwargs,
        }

    def compare_temperature_variations(
        self,
        messages: List[Dict[str, str]],
        temperatures: List[float] = [0.0, 1.0]
    ) -> List[Dict[str, Any]]:
        """
        Runs the same prompt messages across multiple temperature settings for side-by-side comparison.
        """
        results = []
        for temp in temperatures:
            temp_config = ModelConfig(
                model=self.config.model,
                temperature=temp,
                max_tokens=self.config.max_tokens,
                stop=self.config.stop
            )
            res = self.generate_completion(messages=messages, override_config=temp_config)
            results.append({
                "temperature": temp,
                "config": temp_config,
                "response": res["content"],
                "parameters": res["parameters"]
            })
        return results
