import { Lightbulb, MapPin, Mic, MicOff, ScrollText, Sparkles, Users } from "lucide-react";
import { FantasyButton, StatusPill } from "../shared";

const TYPE_LABELS = {
  npc_dialogue: "NPC",
  narration: "Narration",
  rule_check: "Rules",
  lore_reference: "Lore",
  scene: "Scene",
  quest: "Quest",
  encounter: "Encounter",
  npc: "NPC",
};

function roleLabel(role) {
  const normalized = String(role || "").trim().toLowerCase().replace(/[_-]+/g, " ");
  if (normalized === "player") return "Player";
  if (normalized === "assistant") return "Assistant";
  if (normalized === "npc") return "NPC";
  if (normalized === "narration") return "Narration";
  return normalized ? normalized[0].toUpperCase() + normalized.slice(1) : "System";
}

export default function SessionAssistantPanel({
  compact = false,
  supported = true,
  listening = false,
  analyzing = false,
  error = "",
  partialTranscript = "",
  suggestions = [],
  context = {},
  sessionLog = [],
  actionBusyId = "",
  onStartListening,
  onStopListening,
  onAnalyzeNow,
  onRunSuggestion,
  onNarrateSuggestion,
  onIgnoreSuggestion,
}) {
  const status = listening ? "recording" : analyzing ? "generating" : supported ? "ready" : "offline";
  const npcsInScene = Array.isArray(context?.npcsInScene) ? context.npcsInScene : [];
  const activeQuests = Array.isArray(context?.activeQuests) ? context.activeQuests : [];
  const recentLog = Array.isArray(sessionLog) ? sessionLog.slice().reverse().slice(0, 8) : [];

  return (
    <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden session-assistant-shell">
      <div className="panel-head">
        <div className="plaque">Session Assistant</div>
      </div>
      <div className="panel-body min-h-0 flex-1 overflow-hidden flex flex-col gap-3">
        <div className="session-assistant-header">
          <div className="flex items-center gap-2 min-w-0">
            <StatusPill status={status} />
            <span className="session-assistant-toggle" aria-label="Session Assistant">
              <Sparkles size={13} />
              <span>Co-GM online</span>
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <FantasyButton
              variant="secondary"
              className="text-xs"
              onClick={listening ? onStopListening : onStartListening}
              disabled={!supported}
            >
              {listening ? <><MicOff size={13} className="inline mr-1" />Stop</> : <><Mic size={13} className="inline mr-1" />Listen</>}
            </FantasyButton>
            <FantasyButton
              variant="ghost"
              className="text-xs"
              onClick={onAnalyzeNow}
              disabled={analyzing}
            >
              {analyzing ? "Refreshing..." : "Refresh"}
            </FantasyButton>
          </div>
        </div>

        {error ? <div className="session-assistant-error">{error}</div> : null}

        {partialTranscript ? (
          <div className="session-assistant-transcript">
            <div className="session-assistant-transcript-label">Listening now</div>
            <div className="session-assistant-transcript-text">{partialTranscript}</div>
          </div>
        ) : null}

        <div className="session-assistant-section">
          <div className="session-assistant-section-head">
            <Lightbulb size={13} />
            <span>Suggestions</span>
          </div>
          <div className="session-assistant-suggestions">
            {suggestions.length > 0 ? (
              suggestions.slice(0, 3).map((suggestion) => (
                <article key={suggestion.id} className="session-assistant-card">
                  <div className="session-assistant-card-head">
                    <div className="min-w-0">
                      <div className="session-assistant-card-title">{suggestion.title}</div>
                      <div className="session-assistant-card-text">{suggestion.description}</div>
                    </div>
                    <div className="session-assistant-card-badges">
                      <span className="session-assistant-card-type">
                        {TYPE_LABELS[suggestion.type] || "GM"}
                      </span>
                      {suggestion.npcName ? (
                        <span className="session-assistant-card-npc">{suggestion.npcName}</span>
                      ) : null}
                    </div>
                  </div>
                  <div className="session-assistant-card-actions">
                    <FantasyButton
                      variant="secondary"
                      className="text-xs"
                      onClick={() => onRunSuggestion?.(suggestion)}
                      disabled={actionBusyId === suggestion.id}
                    >
                      {actionBusyId === suggestion.id ? "Working..." : "Run"}
                    </FantasyButton>
                    <FantasyButton
                      variant="ghost"
                      className="text-xs"
                      onClick={() => onNarrateSuggestion?.(suggestion)}
                      disabled={actionBusyId === suggestion.id || !String(suggestion.narrateText || "").trim()}
                    >
                      Narrate
                    </FantasyButton>
                    <FantasyButton
                      variant="ghost"
                      className="text-xs"
                      onClick={() => onIgnoreSuggestion?.(suggestion)}
                      disabled={actionBusyId === suggestion.id}
                    >
                      Ignore
                    </FantasyButton>
                  </div>
                </article>
              ))
            ) : (
              <div className="session-assistant-empty">
                Suggestions appear once the session is active.
              </div>
            )}
          </div>
        </div>

        {!compact && <div className="session-assistant-section">
          <div className="session-assistant-section-head">
            <ScrollText size={13} />
            <span>Context</span>
          </div>
          <div className="session-assistant-context-grid">
            <article className="session-assistant-context-card">
              <div className="session-assistant-context-label">
                <Users size={12} />
                NPCs in Scene
              </div>
              {npcsInScene.length > 0 ? (
                <div className="session-assistant-context-list">
                  {npcsInScene.slice(0, 4).map((npc) => (
                    <div key={npc.name} className="session-assistant-context-item">
                      <span className="session-assistant-context-name">{npc.name}</span>
                      <span className="session-assistant-context-meta">{npc.tone || "Unknown tone"}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="session-assistant-empty session-assistant-empty--inline">No NPCs linked to this scene.</div>
              )}
            </article>

            <article className="session-assistant-context-card">
              <div className="session-assistant-context-label">
                <MapPin size={12} />
                Current Location
              </div>
              <div className="session-assistant-context-body">
                {context?.currentLocation || "No active location mapped to this scene yet."}
              </div>
            </article>

            <article className="session-assistant-context-card">
              <div className="session-assistant-context-label">
                <Sparkles size={12} />
                Active Quests
              </div>
              {activeQuests.length > 0 ? (
                <div className="session-assistant-context-list">
                  {activeQuests.slice(0, 3).map((quest) => (
                    <div key={quest.name} className="session-assistant-context-item">
                      <span className="session-assistant-context-name">{quest.name}</span>
                      <span className="session-assistant-context-meta">{quest.summary || "Linked to the active scene."}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="session-assistant-empty session-assistant-empty--inline">No active quests tied to this scene.</div>
              )}
            </article>
          </div>
        </div>}

        {!compact && <div className="session-assistant-section min-h-0 flex-1">
          <div className="session-assistant-section-head">
            <ScrollText size={13} />
            <span>Session Log</span>
          </div>
          <div className="session-assistant-log">
            {recentLog.length > 0 ? (
              recentLog.map((entry) => (
                <div key={entry.id} className="session-assistant-log-item">
                  <div className="session-assistant-log-meta">
                    <span>{roleLabel(entry.role)}</span>
                    {entry.meta ? <span>{entry.meta}</span> : null}
                  </div>
                  <div className="session-assistant-log-text">{entry.text}</div>
                </div>
              ))
            ) : (
              <div className="session-assistant-empty">Recent session events will appear here automatically.</div>
            )}
          </div>
        </div>}
      </div>
    </section>
  );
}
