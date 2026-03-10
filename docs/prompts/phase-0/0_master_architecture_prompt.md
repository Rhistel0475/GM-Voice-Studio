GM Voice Studio — Master Architecture Prompt

Role: Lead Software Architect.

Core Applications
- LiveBoard (real-time GM session control)
- Codex (lore & research database)
- NPC Workshop (NPC creation)
- Voice Studio (voice generation)

Shared Campaign Context
campaign.activeScene
campaign.scenes
campaign.npcs
campaign.locations
campaign.codexEntries
campaign.sessionLog
campaign.campaignMemories

Rules
- modular architecture
- separate UI, services, and state
- avoid rewriting the whole system
