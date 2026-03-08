"""
LLM adapter: abstract interface and Anthropic implementation.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMAdapter(Protocol):
    """Interface for LLM completion (e.g. Claude)."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant text for the given system prompt and messages."""
        ...


class AnthropicLLMAdapter:
    """Anthropic Claude implementation of LLMAdapter."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
    ) -> str:
        from app.infrastructure.llm.anthropic_client import get_client
        client = get_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return (response.content[0].text or "").strip()


_default_llm_adapter: AnthropicLLMAdapter | None = None


def get_default_llm_adapter() -> LLMAdapter:
    """Return the default LLM adapter (Anthropic)."""
    global _default_llm_adapter
    if _default_llm_adapter is None:
        _default_llm_adapter = AnthropicLLMAdapter()
    return _default_llm_adapter
