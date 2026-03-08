"""
Co-DM LLM Brain — entry point (logic split into app.services.llm).
"""
from app.services.llm import classify_intent, handle_query

__all__ = ["classify_intent", "handle_query"]
