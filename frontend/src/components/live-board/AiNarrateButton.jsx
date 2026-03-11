/**
 * AiNarrateButton — generates scene narration via Claude then speaks it via TTS.
 *
 * Uses generateSceneNarration() (buildAiContext → /brain/query → /tts/narrate)
 * and writes the result to the session log via addSessionLogEntry().
 *
 * Props:
 *   authFetch       — authenticated fetch from AppStateContext
 *   scene           — current legacy scene object { title, read_aloud, type, location, npcs }
 *   sceneNpcs       — resolved NPC objects for context (used as fallback)
 *   onLogEntry      — callback(type, text, meta?) to append to session log
 *   onAudioChange   — callback("loading"|"playing"|"idle")
 */
import { useState } from "react";
import { Sparkles } from "lucide-react";
import { useAppState } from "../../context/AppStateContext";
import { useCampaignContextStore } from "../../store/campaignContext";
import { generateSceneNarration, addSessionLogEntry } from "../../lib/liveboardCampaignContext";
import { persistSessionEvent } from "../../lib/campaignPersistence";
import AudioPlaybackCard from "./AudioPlaybackCard";

export default function AiNarrateButton({
  authFetch,
  scene,
  onLogEntry,
  onAudioChange,
}) {
  const [status, setStatus] = useState("idle"); // idle | generating | speaking | error
  const [generatedText, setGeneratedText] = useState("");
  const [error, setError] = useState("");
  const [audioStatus, setAudioStatus] = useState("idle");

  const { apiKey } = useAppState();

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
      const voiceId = await resolveVoiceId();

      // Generate narration text and TTS audio via lib (uses buildAiContext internally)
      const { text, clip } = await generateSceneNarration({
        apiKey,
        voiceId: voiceId ?? undefined,
      });

      setGeneratedText(text);
      onLogEntry?.("narration", text, "AI Narrate");

      // Write to session log store and backend
      const store = useCampaignContextStore.getState();
      addSessionLogEntry({
        type: "narration",
        text,
        sceneId: store.activeSceneId ?? undefined,
        sessionId: store.activeSessionId ?? undefined,
      });
      persistSessionEvent(authFetch, {
        type: "narration",
        text,
        scene_id: store.activeSceneId,
        session_id: store.activeSessionId,
      });

      // Store the narration clip
      if (clip) {
        store.addNarrationClip({
          title: clip.title || scene?.title || "AI Narration",
          audioUrl: clip.audioUrl,
          voiceId: clip.voiceId,
        });
      }

      // Play the returned audio
      setStatus("speaking");
      const audioUrl = clip?.audioUrl;
      if (!audioUrl) throw new Error("No audio returned from TTS.");

      const audio = new Audio(audioUrl);
      setAudioStatus("playing");
      onAudioChange?.("playing");

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
        title="Generate scene narration via Claude, then speak it via TTS"
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
