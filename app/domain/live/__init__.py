"""
Live domain: WebSocket session, Co-DM live board, streaming transcription and TTS.

Ownership:
- Service: app.services.live_board_service (WebSocket handler, session state, brain/tts integration)
- Service: app.services.scene_trigger_service (scene control trigger execution)
- Routes: WebSocket /ws/audio, POST /scene/trigger, POST /scene/activate, POST /scene/combat-start
- Depends on: voice domain (TTS, voice list), AI domain (handle_query, generate_dialogue)
"""

__all__ = ["scene_control", "scene_triggers", "session_control"]
