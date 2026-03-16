"""
Domain boundaries (Phase 5).

Subpackages define ownership of capabilities:
- voice: cloning, TTS, voice storage and metadata
- campaign: campaign persistence, adventure parsing, NPCs/scenes/locations
- live: WebSocket /ws/audio, Co-DM live board
- ai: LLM orchestration, RAG, NPC generation, dialogue
"""

__all__ = ["ai", "campaign", "live", "voice"]
