/**
 * SceneDirectorPanel — collapsible panel showing AI-generated dramatic suggestions
 * for the current scene (tension, environment, NPC reactions, twists, pacing).
 *
 * Calls /brain/query with a scene director prompt and displays the response.
 *
 * Props:
 *   authFetch   — authenticated fetch from AppStateContext
 *   scene       — current legacy scene object
 *   sceneNpcs   — resolved NPC objects for context
 *   onLogEntry  — callback(type, text, meta?) to append to session log
 */
import { useState } from "react";
import { ChevronDown, ChevronRight, Wand2 } from "lucide-react";
import { useCampaignContextStore } from "../../store/campaignContext";
import { getAiContext } from "../../lib/aiContext";
import { persistSessionEvent } from "../../lib/campaignPersistence";

export default function SceneDirectorPanel({ authFetch, scene, sceneNpcs = [], onLogEntry }) {
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [hints, setHints] = useState("");
  const [error, setError] = useState("");

  const buildQuery = () => {
    const aiCtx = getAiContext();
    const hasStoreScene = Boolean(aiCtx.scene);

    const title = hasStoreScene ? aiCtx.scene.title : (scene?.title || "Current Scene");
    const location = hasStoreScene
      ? aiCtx.location?.name
      : (scene?.location || null);
    const npcNames = hasStoreScene
      ? aiCtx.npcs.map((n) => `${n.name}${n.role ? ` (${n.role})` : ""}`).join(", ")
      : sceneNpcs.map((n) => `${n.name}${n.role ? ` (${n.role})` : ""}`).join(", ");
    const summary = hasStoreScene
      ? aiCtx.scene.summary
      : (scene?.read_aloud?.slice(0, 250) || null);
    const recent = hasStoreScene
      ? aiCtx.recentEvents.slice(-3).map((e) => e.text).join(" · ")
      : null;

    return [
      `As a Scene Director for a tabletop RPG, give 4–5 specific, actionable dramatic suggestions for the GM running this scene.`,
      `Scene: "${title}".`,
      location ? `Location: ${location}.` : null,
      npcNames ? `NPCs present: ${npcNames}.` : null,
      summary ? `Setup: ${summary}.` : null,
      recent ? `Recent events: ${recent}.` : null,
      `Include ideas for: tension escalation, environment details, NPC reactions, and one optional twist. Be brief and specific.`,
    ].filter(Boolean).join(" ");
  };

  const handleFetchHints = async () => {
    setIsLoading(true);
    setError("");
    setHints("");
    try {
      const query = buildQuery();
      const res = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error(await res.text() || "Scene Director failed.");
      const data = await res.json();
      const text = (data.content || data.text || "").trim();
      if (!text) throw new Error("No suggestions returned.");
      setHints(text);
      setOpen(true);
      onLogEntry?.("assistant", text, "Scene Director");
      const store = useCampaignContextStore.getState();
      store.addActionLogEvent({ type: "assistant", text });
      persistSessionEvent(authFetch, {
        type: "assistant",
        text,
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
          {isLoading ? "Thinking…" : hints ? "Refresh" : "Get Hints"}
        </button>
      </div>

      {open && (
        <div className="sd-body">
          {error && <div className="sd-error">{error}</div>}
          {hints && !error && (
            <div className="sd-hints">
              {hints.split("\n").filter(Boolean).map((line, i) => (
                <p key={i} className="sd-hint-line">
                  {line.replace(/^[-•*]\s*/, "")}
                </p>
              ))}
            </div>
          )}
          {!hints && !error && !isLoading && (
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
