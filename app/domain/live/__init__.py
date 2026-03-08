"""
Live domain: WebSocket session, Co-DM live board, streaming transcription and TTS.

Ownership:
- Service: app.services.live_board_service (WebSocket handler, session state, brain/tts integration)
- Routes (in routes_legacy): WebSocket /ws/audio
- Depends on: voice domain (TTS, voice list), AI domain (handle_query, generate_dialogue)
"""

from app.services import live_board_service

__all__ = ["live_board_service"]
