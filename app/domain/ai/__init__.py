"""
AI domain: LLM orchestration, RAG, NPC generation, dialogue, adventure parsing.

Ownership:
- Services: app.services.ai_service, npc_generator_service, llm_orchestrator
- Infrastructure: app.infrastructure.retrieval (pinecone_retriever, indexer),
  app.infrastructure.llm (anthropic_client)
- Routes (in routes_legacy): POST /rag/query, POST /brain/query, POST /npc/generate,
  POST /ai/dialogue; WebSocket /ws/audio uses handle_query and generate_dialogue
- Adventure parsing (campaign structure): ai_full_parse, assign_images_to_entities
"""

from app.services.ai_service import (
    ai_full_parse,
    assign_images_to_entities,
    generate_dialogue,
)
from app.services.llm_orchestrator import handle_query
from app.services.npc_generator_service import generate_npc_stream

__all__ = [
    "ai_full_parse",
    "assign_images_to_entities",
    "generate_dialogue",
    "handle_query",
    "generate_npc_stream",
]
