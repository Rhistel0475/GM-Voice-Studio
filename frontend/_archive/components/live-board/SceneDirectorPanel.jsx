/**
 * SceneDirectorPanel — collapsible panel showing AI-generated dramatic suggestions
 * for the current scene (tension, environment, NPC reactions, twists, pacing).
 *
 * Calls getSceneDirectorSuggestions() which uses buildAiContext() to build the
 * prompt and returns structured SceneDirectorSuggestion[].
 *
 * Props:
 *   authFetch   — authenticated fetch from AppStateContext
 *   scene       — current legacy scene object (fallback context)
 *   sceneNpcs   — resolved NPC objects for context (fallback)
 *   onLogEntry  — callback(type, text, meta?) to append to session log
 */
import { useState } from "react";
import { ChevronDown, ChevronRight, Wand2 } from "lucide-react";
import { useAppState } from "../../context/AppStateContext";
import { useCampaignContextStore } from "../../store/campaignContext";
import { addSessionLogEntry } from "../../lib/liveboardCampaignContext";
import { getSceneDirectorSuggestions } from "../../lib/sceneDirector";
import { persistSessionEvent } from "../../lib/campaignPersistence";

const CATEGORY_LABELS = {
  tension:      "Tension",
  environment:  "Environment",
  npc_reaction: "NPC",
  twist:        "Twist",
  pacing:       "Pacing",
};

const CATEGORY_COLORS = {
  tension:      "text-[#e05a4a]",
  environment:  "text-[#7ab87a]",
  npc_reaction: "text-[#7ab8d4]",
  twist:        "text-[#c8a050]",
  pacing:       "text-[#b07ad4]",
};

export default function SceneDirectorPanel({ authFetch, onLogEntry }) {
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState("");

  const { apiKey } = useAppState();

  const handleFetchHints = async () => {
    setIsLoading(true);
    setError("");
    setSuggestions([]);
    try {
      const results = await getSceneDirectorSuggestions({ apiKey });
      if (!results.length) throw new Error("No suggestions returned.");

      setSuggestions(results);
      setOpen(true);

      const fullText = results.map((s) => `[${CATEGORY_LABELS[s.category]}] ${s.text}`).join("\n");
      onLogEntry?.("assistant", fullText, "Scene Director");

      const store = useCampaignContextStore.getState();
      addSessionLogEntry({
        type: "assistant",
        text: fullText,
        sceneId: store.activeSceneId ?? undefined,
        sessionId: store.activeSessionId ?? undefined,
      });
      persistSessionEvent(authFetch, {
        type: "assistant",
        text: fullText,
        scene_id: store.activeSceneId,
        session_id: store.activeSessionId,
      });
    } catch (err) {
      setError(err?.message || "Scene Director failed.");
      setOpen(true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="sd-panel">
      <div className="sd-header">
        <button
          type="button"
          className="sd-toggle"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          <span>Scene Director</span>
        </button>
        <button
          type="button"
          className="sd-fetch-btn"
          onClick={handleFetchHints}
          disabled={isLoading}
          title="Ask AI for scene direction suggestions"
        >
          <Wand2 size={12} className="inline-block mr-1" />
          {isLoading ? "Thinking…" : suggestions.length ? "Refresh" : "Get Hints"}
        </button>
      </div>

      {open && (
        <div className="sd-body">
          {error && <div className="sd-error">{error}</div>}

          {suggestions.length > 0 && !error && (
            <div className="sd-hints">
              {suggestions.map((s, i) => (
                <div key={i} className="sd-hint-line">
                  <span className={`sd-hint-category ${CATEGORY_COLORS[s.category]}`}>
                    {CATEGORY_LABELS[s.category]}
                  </span>
                  <span>{s.text}</span>
                </div>
              ))}
            </div>
          )}

          {!suggestions.length && !error && !isLoading && (
            <div className="sd-empty">
              Click "Get Hints" to receive dramatic suggestions for this scene.
            </div>
          )}

          {isLoading && (
            <div className="sd-loading">Consulting the narrative fates…</div>
          )}
        </div>
      )}
    </div>
  );
}
