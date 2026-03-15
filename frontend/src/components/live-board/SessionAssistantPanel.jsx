import { Lightbulb, Mic, MicOff, ScrollText, Sparkles, Swords } from "lucide-react";
import { FantasyButton, StatusPill } from "../shared";

const TYPE_META = {
  npc_dialogue: {
    label: "NPC Dialogue",
    icon: Swords,
    action: "Voice NPC",
  },
  narration: {
    label: "Narrate Scene",
    icon: ScrollText,
    action: "Narrate",
  },
  rule_check: {
    label: "Rule Check",
    icon: Lightbulb,
    action: "Explain Rule",
  },
  lore_reference: {
    label: "Lore Reference",
    icon: Sparkles,
    action: "Explain Lore",
  },
};

export default function SessionAssistantPanel({
  supported = true,
  listening = false,
  analyzing = false,
  error = "",
  partialTranscript = "",
  suggestions = [],
  recentEntries = [],
  actionBusyId = "",
  onStartListening,
  onStopListening,
  onAnalyzeNow,
  onRunSuggestion,
}) {
  const status = listening ? "recording" : analyzing ? "generating" : supported ? "ready" : "offline";

  return (
    <div className="session-assistant-panel">
      <div className="session-assistant-header">
        <div className="flex items-center gap-2 min-w-0">
          <StatusPill status={status} />
          <span className="session-assistant-toggle" aria-label="Session Assistant">
            <Sparkles size={13} />
            <span>Session Assistant</span>
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <FantasyButton
            variant="secondary"
            className="text-xs"
            onClick={listening ? onStopListening : onStartListening}
            disabled={!supported}
          >
            {listening ? <><MicOff size={13} className="inline mr-1" />Stop Listening</> : <><Mic size={13} className="inline mr-1" />Start Listening</>}
          </FantasyButton>
          <FantasyButton
            variant="ghost"
            className="text-xs"
            onClick={onAnalyzeNow}
            disabled={analyzing || recentEntries.length === 0}
          >
            {analyzing ? "Analyzing..." : "Analyze Now"}
          </FantasyButton>
        </div>
      </div>

      <div className="session-assistant-body">
        {!supported ? (
          <div className="session-assistant-empty">
            Browser speech recognition is not supported here.
          </div>
        ) : null}

        {error ? <div className="session-assistant-error">{error}</div> : null}

        {partialTranscript ? (
          <div className="session-assistant-transcript">
            <div className="session-assistant-transcript-label">Listening now</div>
            <div className="session-assistant-transcript-text">{partialTranscript}</div>
          </div>
        ) : null}

        {recentEntries.length > 0 ? (
          <div className="session-assistant-recent">
            <div className="session-assistant-recent-label">Recent table dialogue</div>
            <div className="session-assistant-recent-list">
              {recentEntries.slice(-3).map((entry, index) => (
                <div key={`${index}-${entry.slice(0, 16)}`} className="session-assistant-recent-item">
                  {entry}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {suggestions.length > 0 ? (
          <div className="session-assistant-suggestions">
            {suggestions.map((suggestion) => {
              const meta = TYPE_META[suggestion.type] || TYPE_META.rule_check;
              const Icon = meta.icon;
              return (
                <article key={suggestion.id || `${suggestion.type}-${suggestion.title}-${suggestion.text}`} className="session-assistant-card">
                  <div className="session-assistant-card-head">
                    <span className="session-assistant-card-type">
                      <Icon size={12} />
                      {meta.label}
                    </span>
                    {suggestion.npc_name ? (
                      <span className="session-assistant-card-npc">{suggestion.npc_name}</span>
                    ) : null}
                  </div>
                  <div className="session-assistant-card-text">{suggestion.text}</div>
                  <div className="session-assistant-card-actions">
                    <FantasyButton
                      variant={suggestion.type === "npc_dialogue" || suggestion.type === "narration" ? "secondary" : "ghost"}
                      className="text-xs"
                      onClick={() => onRunSuggestion?.(suggestion)}
                      disabled={actionBusyId === suggestion.id}
                    >
                      {actionBusyId === suggestion.id ? "Working..." : meta.action}
                    </FantasyButton>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <div className="session-assistant-empty">
            Listen to the table, then the assistant will surface NPC, narration, rule, and lore opportunities.
          </div>
        )}
      </div>
    </div>
  );
}
