from types import SimpleNamespace

from app.services import ai_service, session_memory_service


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._rows)


class _FakeReadSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, model):
        return _FakeQuery(self._rows)

    def close(self):
        return None


class _FakeWriteSession:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)

    def commit(self):
        return None

    def refresh(self, record):
        record.id = 101

    def close(self):
        return None


def test_get_session_context_builds_general_and_npc_specific_summaries(monkeypatch):
    rows = [
        SimpleNamespace(
            id=3,
            session_id="sess-1",
            timestamp="2026-03-16T10:05:00+00:00",
            event_type="important_dialogue",
            npc_id="oleg",
            description="Oleg warned the party not to light a torch in the ruins.",
            tags='["warning","ruins"]',
        ),
        SimpleNamespace(
            id=2,
            session_id="sess-1",
            timestamp="2026-03-16T10:00:00+00:00",
            event_type="player_decision",
            npc_id=None,
            description="The party agreed to spare the captured scout.",
            tags='["mercy"]',
        ),
    ]
    monkeypatch.setattr(
        session_memory_service,
        "_resolve_session_reference",
        lambda **kwargs: {"id": "sess-1", "campaign_id": 7},
    )
    monkeypatch.setattr(session_memory_service, "SessionLocal", lambda: _FakeReadSession(rows))

    context = session_memory_service.get_session_context(campaign_id=7, npc_id="oleg")

    assert context["session_id"] == "sess-1"
    assert "Player Decision" in context["summary"]
    assert "Important Dialogue (oleg)" in context["npc_memory_summary"]
    assert "light a torch" in context["npc_memory_summary"]
    assert context["events"][0]["tags"] == ["mercy"]
    assert "[tags: warning, ruins]" in context["npc_memory_summary"]


def test_get_session_summary_alias_returns_same_summary_shape(monkeypatch):
    rows = [
        SimpleNamespace(
            id=1,
            session_id="sess-7",
            timestamp="2026-03-16T11:00:00+00:00",
            event_type="combat_outcome",
            npc_id=None,
            description="The bandits fled after losing their captain.",
            tags='["combat","bandits"]',
        ),
    ]
    monkeypatch.setattr(
        session_memory_service,
        "_resolve_session_reference",
        lambda **kwargs: {"id": "sess-7", "campaign_id": 99},
    )
    monkeypatch.setattr(session_memory_service, "SessionLocal", lambda: _FakeReadSession(rows))

    summary = session_memory_service.get_session_summary(campaign_id=99)

    assert summary["session_id"] == "sess-7"
    assert "Combat Outcome" in summary["summary"]
    assert summary["events"][0]["tags"] == ["combat", "bandits"]


def test_record_event_uses_resolved_active_session(monkeypatch):
    fake_db = _FakeWriteSession()
    monkeypatch.setattr(
        session_memory_service,
        "_resolve_session_reference",
        lambda **kwargs: {
            "id": "sess-42",
            "campaign_id": 12,
            "active_scene_id": "scene-9",
        },
    )
    monkeypatch.setattr(session_memory_service, "SessionLocal", lambda: fake_db)

    payload = session_memory_service.record_event(
        event_type="npc_interaction",
        description="Players insulted Oleg.",
        npc_id="oleg",
        tags=["hostile", "insult"],
        campaign_id=12,
    )

    assert payload["id"] == 101
    assert payload["session_id"] == "sess-42"
    assert payload["campaign_id"] == 12
    assert payload["scene_id"] == "scene-9"
    assert fake_db.records[0].event_type == "npc_interaction"
    assert fake_db.records[0].description == "Players insulted Oleg."
    assert payload["tags"] == ["hostile", "insult"]
    assert fake_db.records[0].tags == '["hostile", "insult"]'


def test_build_npc_system_prompt_includes_session_memory_sections():
    prompt = ai_service.build_npc_system_prompt(
        "Oleg",
        "Gruff scout with a suspicious streak.",
        faction="North Road Wardens",
        situation="The party is demanding answers.",
        session_context="- Player Decision: The party spared a captured scout.",
        npc_memory_summary="- NPC Interaction (oleg): The players insulted Oleg.",
    )

    assert "NPC memory from this session" in prompt
    assert "Session memory from this session" in prompt
    assert "Stay consistent with the session history" in prompt
