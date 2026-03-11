/**
 * AiNarrateButton — generates scene narration via Claude then speaks it via TTS.
 *
 * Two-step flow:
 *   1. POST /brain/query with a scene context prompt → get narration text from Claude
 *   2. POST /tts/narrate with the generated text → play audio
 *
 * Falls back to speaking scene.read_aloud directly when no voice is available.
 *
 * Props:
 *   authFetch       — authenticated fetch from AppStateContext
 *   scene           — current legacy scene object { title, read_aloud, type, location, npcs }
 *   sceneNpcs       — resolved NPC objects for context
 *   onLogEntry      — callback(type, text, meta?) to append to session log
 *   onAudioChange   — callback("loading"|"playing"|"idle")
 */
import { useState } from "react";
import { Sparkles } from "lucide-react";
import { useCampaignContextStore } from "../../store/campaignContext";
import { getAiContext } from "../../lib/aiContext";
import { persistSessionEvent } from "../../lib/campaignPersistence";
import AudioPlaybackCard from "./AudioPlaybackCard";

export default function AiNarrateButton({
  authFetch,
  scene,
  sceneNpcs = [],
  onLogEntry,
  onAudioChange,
}) {
  const [status, setStatus] = useState("idle"); // idle | generating | speaking | error
  const [generatedText, setGeneratedText] = useState("");
  const [error, setError] = useState("");
  const [audioStatus, setAudioStatus] = useState("idle");

  const storeState = useCampaignContextStore();

  const buildQuery = () => {
    // Try campaign context store first for richer context
    const aiCtx = getAiContext();
    const hasStoreScene = Boolean(aiCtx.scene);

    const title = hasStoreScene ? aiCtx.scene.title : (scene?.title || "Untitled Scene");
    const location = hasStoreScene
      ? aiCtx.location?.name
      : (scene?.location || null);
    const summary = hasStoreScene ? aiCtx.scene.summary : (scene?.read_aloud?.slice(0, 200) || null);
    const npcNames = hasStoreScene
      ? aiCtx.npcs.map((n) => n.name).join(", ")
      : sceneNpcs.map((n) => n.name).join(", ");
    const recent = hasStoreScene
      ? aiCtx.recentEvents.slice(-3).map((e) => e.text).join(" · ")
      : null;

    return [
      "Write a short, vivid 2–4 sentence scene narration for the GM to read aloud.",
      `Scene: "${title}".`,
      location ? `Location: ${location}.` : null,
      summary ? `Context: ${summary}.` : null,
      npcNames ? `NPCs present: ${npcNames}.` : null,
      recent ? `Recent events: ${recent}.` : null,
      "Return only the narration text with no preamble.",
    ].filter(Boolean).join(" ");
  };

  const resolveVoiceId = async () => {
    try {
      const res = await authFetch("/voices/list");
      if (!res.ok) return null;
      const voices = await res.json();
      const first = Array.isArray(voices) ? voices.find((v) => v?.voice_id) : null;
      return first?.voice_id ?? null;
    } catch {
      return null;
    }
  };

  const handleAiNarrate = async () => {
    setStatus("generating");
    setGeneratedText("");
    setError("");
    onAudioChange?.("loading");

    try {
      // Step 1: Generate narration text
      const query = buildQuery();
      const genRes = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!genRes.ok) throw new Error(await genRes.text() || "Generation failed.");
      const genData = await genRes.json();
      const text = (genData.content || genData.text || "").trim();
      if (!text) throw new Error("AI returned empty narration.");

      setGeneratedText(text);
      onLogEntry?.("narration", text, "AI Narrate");

      // Sync to campaign context store and backend
      const store = useCampaignContextStore.getState();
      store.addActionLogEvent({ type: "narration", text });
      persistSessionEvent(authFetch, {
        type: "narration",
        text,
        scene_id: store.activeSceneId,
        session_id: store.activeSessionId,
      });

      // Step 2: Speak via TTS
      setStatus("speaking");
      const voiceId = await resolveVoiceId();
      const ttsBody = { text };
      if (voiceId) ttsBody.voice_id = voiceId;

      const ttsRes = await authFetch("/tts/narrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ttsBody),
      });
      if (!ttsRes.ok) throw new Error(await ttsRes.text() || "TTS failed.");

      const blob = await ttsRes.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      setAudioStatus("playing");
      onAudioChange?.("playing");

      // Add narration clip to the campaign context store
      storeState.addNarrationClip({ title: scene?.title || "AI Narration", audioUrl });

      audio.onended = () => {
        setAudioStatus("idle");
        onAudioChange?.("idle");
        setStatus("idle");
      };
      audio.onerror = () => {
        setAudioStatus("idle");
        onAudioChange?.("idle");
        setStatus("idle");
      };
      audio.play();
    } catch (err) {
      setError(err?.message || "AI narration failed.");
      setStatus("error");
      onAudioChange?.("idle");
    }
  };

  const isDisabled = status === "generating" || status === "speaking";

  return (
    <div className="ai-narrate-wrap">
      <button
        type="button"
        className="cta-secondary ai-narrate-btn transition-all hover:brightness-110"
        onClick={handleAiNarrate}
        disabled={isDisabled}
        title="Generate new scene narration via Claude, then speak it"
      >
        <Sparkles size={13} className="inline-block mr-1" />
        {status === "generating" ? "Generating…" : status === "speaking" ? "Speaking…" : "AI Narrate"}
      </button>

      {error && (
        <span className="ai-narrate-error">{error}</span>
      )}

      {generatedText && (
        <div className="ai-narrate-result">
          <div className="ai-narrate-result-label">Generated narration</div>
          <div className="ai-narrate-result-text">{generatedText}</div>
        </div>
      )}

      {audioStatus !== "idle" && (
        <div className="mt-2">
          <AudioPlaybackCard
            audioStatus={audioStatus}
            voiceName="AI Narration"
            onPlayPause={() => {}}
          />
        </div>
      )}
    </div>
  );
}
