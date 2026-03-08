"""
Intent classification for Co-DM queries (keyword-based, no API call).
"""
import re

_NPC_PATTERNS = re.compile(
    r"\b(generate|create|make|build|give me|invent)\b.{0,30}\b(npc|character|villain|ally|enemy|patron|contact)\b"
    r"|\b(npc|character)\b.{0,20}\b(generate|create|make|build)\b",
    re.IGNORECASE,
)

_RULE_PATTERNS = re.compile(
    r"\b(rule|mechanic|how does|how do|what (is|are) the|stat|stats|AC|HP|CR|DC|attack|damage|ability|skill|"
    r"saving throw|spell|grapple|flanking|initiative|cover|condition|resistance|immunity|proficiency)\b",
    re.IGNORECASE,
)


def classify_intent(query: str) -> str:
    """
    Fast keyword-based intent classification.
    Returns one of: 'rule_lookup', 'npc_request', 'general_chat'
    """
    if _NPC_PATTERNS.search(query):
        return "npc_request"
    if _RULE_PATTERNS.search(query):
        return "rule_lookup"
    return "general_chat"
