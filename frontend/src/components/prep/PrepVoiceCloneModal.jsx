import { useCallback, useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import VoiceCloneWizard from "../voices/VoiceCloneWizard";
import { getVoices, submitClone, getCloneJobStatus } from "../../lib/api/voices";
import { useCampaignContextStore } from "../../store/campaignContext";

/**
 * Map API voice profile to Zustand Voice shape.
 * @param {import("../../lib/api/voices").VoiceProfile} p
 * @param {string | null} campaignId
 */
function profileToStoreVoice(p, campaignId) {
  const id = String(p.voice_id || p.id || "");
  if (!id) return null;
  return {
    id,
    campaignId: campaignId || undefined,
    name: p.name || "Voice",
    tags: Array.isArray(p.tags) ? p.tags : [],
    assignedNpcIds: [],
    tone: p.tone,
    accent: p.accent,
    status: p.status || "ready",
  };
}

async function mergeVoicesFromApi(authFetch) {
  const list = await getVoices(authFetch);
  const cid = useCampaignContextStore.getState().activeCampaignId;
  const upsertVoice = useCampaignContextStore.getState().upsertVoice;
  for (const p of list) {
    const v = profileToStoreVoice(p, cid);
    if (v) upsertVoice(v);
  }
}

/**
 * Modal wrapper for voice cloning (Kani TTS). Writes new voices into the campaign store after save.
 */
export default function PrepVoiceCloneModal({ open, onClose, authFetch }) {
  const [ttsProvider, setTtsProvider] = useState("hume");
  const [cloneStep, setCloneStep] = useState(1);
  const [cloneFile, setCloneFile] = useState(null);
  const [cloneName, setCloneName] = useState("");
  const [cloneTags, setCloneTags] = useState([]);
  const [isCloning, setIsCloning] = useState(false);
  const [cloneStatus, setCloneStatus] = useState("");
  const [cloneProgress, setCloneProgress] = useState(0);
  const [cloneVoiceId, setCloneVoiceId] = useState("");
  const [cloneVoiceName, setCloneVoiceName] = useState("");
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const [saving, setSaving] = useState(false);
  const [cloneSaved, setCloneSaved] = useState(false);
  const [cloneSaveError, setCloneSaveError] = useState("");
  const sampleAudioRef = useRef(null);
  const sampleAudioUrlRef = useRef("");

  const cloneAvailable = ttsProvider !== "hume";

  useEffect(() => {
    if (!open || !authFetch) return;
    let cancelled = false;
    authFetch("/config")
      .then((res) => (res.ok ? res.json() : {}))
      .then((cfg) => {
        if (!cancelled && cfg?.tts_provider) setTtsProvider(String(cfg.tts_provider));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [open, authFetch]);

  const resetWizard = useCallback(() => {
    setCloneStep(1);
    setCloneFile(null);
    setCloneName("");
    setCloneTags([]);
    setIsCloning(false);
    setCloneStatus("");
    setCloneProgress(0);
    setCloneVoiceId("");
    setCloneVoiceName("");
    setIsPlayingPreview(false);
    setSaving(false);
    setCloneSaved(false);
    setCloneSaveError("");
  }, []);

  useEffect(() => {
    if (open) resetWizard();
  }, [open, resetWizard]);

  const stopSampleAudio = useCallback(() => {
    const audio = sampleAudioRef.current;
    if (audio) {
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      audio.currentTime = 0;
      sampleAudioRef.current = null;
    }
    if (sampleAudioUrlRef.current) {
      URL.revokeObjectURL(sampleAudioUrlRef.current);
      sampleAudioUrlRef.current = "";
    }
  }, []);

  useEffect(() => () => stopSampleAudio(), [stopSampleAudio]);

  const playSample = useCallback(
    async (voiceId) => {
      if (!voiceId || !authFetch) return;
      stopSampleAudio();
      try {
        const res = await authFetch("/tts/narrate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: "The quick brown fox jumps over the lazy dog.",
            voice_id: voiceId,
          }),
        });
        if (!res.ok) return;
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        sampleAudioRef.current = audio;
        sampleAudioUrlRef.current = url;
        audio.onended = () => stopSampleAudio();
        audio.onerror = () => stopSampleAudio();
        await audio.play();
      } catch {
        stopSampleAudio();
      }
    },
    [authFetch, stopSampleAudio]
  );

  const handleClone = useCallback(async () => {
    if (!cloneFile || isCloning || !authFetch) return;
    setCloneStep(2);
    setIsCloning(true);
    setCloneStatus("Training voice…");
    setCloneProgress(10);
    try {
      const formData = new FormData();
      formData.append("audio", cloneFile);
      formData.append("consent_scope", "tts");
      if (cloneName.trim()) formData.append("name", cloneName.trim());
      const result = await submitClone(formData, authFetch);
      if (!result.ok) throw new Error(result.error || "Clone request failed.");
      setCloneProgress(30);
      if (result.voice_id) {
        setCloneVoiceId(result.voice_id);
        setCloneVoiceName(cloneName.trim() || result.voice_id);
        setCloneProgress(100);
        setCloneStatus("Ready");
        setCloneStep(3);
        await mergeVoicesFromApi(authFetch);
        return;
      }
      if (result.job_id) {
        setCloneStatus(`Queued: ${result.job_id}. Polling…`);
        for (let i = 0; i < 45; i += 1) {
          await new Promise((r) => setTimeout(r, 2000));
          setCloneProgress(30 + (i / 45) * 70);
          const job = await getCloneJobStatus(result.job_id, authFetch);
          const done = job?.status === "ready" || job?.status === "completed";
          if (done && job?.voice_id) {
            setCloneVoiceId(job.voice_id);
            setCloneVoiceName(cloneName.trim() || job.voice_id);
            setCloneProgress(100);
            setCloneStatus("Ready");
            setCloneStep(3);
            await mergeVoicesFromApi(authFetch);
            return;
          }
          if (job?.status === "failed") throw new Error(job?.error || "Clone failed.");
        }
        throw new Error("Clone timed out.");
      }
      throw new Error("No voice_id or job_id.");
    } catch (e) {
      setCloneStatus(e?.message || "Clone failed.");
      setCloneStep(1);
    } finally {
      setIsCloning(false);
    }
  }, [cloneFile, cloneName, isCloning, authFetch]);

  const handlePlayPreview = useCallback(
    (voiceId) => {
      if (!voiceId) return;
      setIsPlayingPreview(true);
      playSample(voiceId)
        .then(() => setIsPlayingPreview(false))
        .catch(() => setIsPlayingPreview(false));
    },
    [playSample]
  );

  const handleSaveClone = useCallback(async () => {
    setSaving(true);
    setCloneSaveError("");
    try {
      if (cloneVoiceId && authFetch) {
        const res = await authFetch(`/voices/${encodeURIComponent(cloneVoiceId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: cloneName.trim(), tags: cloneTags }),
        });
        if (!res.ok) throw new Error(await res.text());
      }
      setCloneSaved(true);
      setCloneStep(5);
      await mergeVoicesFromApi(authFetch);
    } catch (e) {
      setCloneSaveError(e?.message || "Save failed.");
    } finally {
      setSaving(false);
    }
  }, [cloneVoiceId, cloneName, cloneTags, authFetch]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="prep-voice-clone-title"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-lg border border-[#5c3e23] bg-[#120a04] shadow-2xl p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-2 mb-3">
          <h2 id="prep-voice-clone-title" className="font-heading text-[var(--gold)] text-sm tracking-wide">
            Clone a voice
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-[var(--text-2)] hover:text-[var(--text-1)] p-1"
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>

        {!cloneAvailable && (
          <p className="text-xs text-[var(--text-2)] mb-3 leading-relaxed">
            Voice cloning is only available when the server uses Kani TTS (not Hume). You can still assign
            existing voices to NPCs from the dropdowns.
          </p>
        )}

        {cloneAvailable && (
          <VoiceCloneWizard
            step={cloneStep}
            setStep={setCloneStep}
            cloneFile={cloneFile}
            onFileChange={setCloneFile}
            cloneName={cloneName}
            onNameChange={setCloneName}
            cloneTags={cloneTags}
            onTagsChange={setCloneTags}
            onTrain={handleClone}
            isCloning={isCloning}
            cloneStatus={cloneStatus}
            cloneProgress={cloneProgress}
            cloneVoiceId={cloneVoiceId}
            cloneVoiceName={cloneVoiceName}
            onPlayPreview={handlePlayPreview}
            isPlayingPreview={isPlayingPreview}
            onSave={handleSaveClone}
            saving={saving}
            saved={cloneSaved}
            saveError={cloneSaveError}
          />
        )}
      </div>
    </div>
  );
}
