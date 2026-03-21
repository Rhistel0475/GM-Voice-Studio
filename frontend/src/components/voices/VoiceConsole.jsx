import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LayoutGrid, List, Mic2, Plus, ScrollText, X } from "lucide-react";
import { defaultVoiceFilterState } from "../../types/voice";
import { VOICE_PRESETS } from "../../lib/voicePresets";
import {
  deleteVoice,
  getVoices,
  getGeneratedAudio,
  submitClone,
  getCloneJobStatus,
} from "../../lib/api/voices";
import { getNPCs } from "../../lib/api/npcs";
import { useCampaignOptional } from "../../context/CampaignContext";
import { useVoices as useVoicesFromStore, useNpcsForVoice } from "../../store/selectors";
import { EmptyState, FantasyButton } from "../shared";
import VoiceSearchInput from "./VoiceSearchInput";
import VoiceLibraryFilters from "./VoiceLibraryFilters";
import VoiceLibraryGrid from "./VoiceLibraryGrid";
import VoiceDetailPanel from "./VoiceDetailPanel";
import GeneratedAudioList from "./GeneratedAudioList";
import VoiceCloneWizard from "./VoiceCloneWizard";

/* ─── Filter helper ─────────────────────────────────────────────── */
function filterVoices(voices, filterState, selectedPreset) {
  let out = voices;
  const q = (filterState.query || "").trim().toLowerCase();
  if (q) {
    out = out.filter(
      (v) =>
        (v.name && v.name.toLowerCase().includes(q)) ||
        (v.description && v.description.toLowerCase().includes(q)) ||
        (v.tags && v.tags.some((t) => t.toLowerCase().includes(q)))
    );
  }
  if (filterState.source && filterState.source !== "all")
    out = out.filter((v) => v.source === filterState.source);
  if (filterState.status && filterState.status !== "all")
    out = out.filter((v) => v.status === filterState.status);
  if (filterState.tone && filterState.tone !== "all")
    out = out.filter((v) => v.tone === filterState.tone);
  if (selectedPreset) {
    const preset = VOICE_PRESETS[selectedPreset];
    if (preset) {
      const keywords = preset.tone.split(/[,\s]+/).filter(Boolean).map((w) => w.toLowerCase());
      out = out.filter((v) => {
        const tone = (v.tone || "").toLowerCase();
        const name = (v.name || "").toLowerCase();
        const tags = (v.tags || []).map((t) => t.toLowerCase());
        return keywords.some((kw) => tone.includes(kw) || name.includes(kw) || tags.some((t) => t.includes(kw)));
      });
    }
  }
  return out;
}

/* ─── Provider badge ─────────────────────────────────────────────── */
function ProviderBadge({ provider }) {
  const isHume = provider === "hume";
  const label = isHume ? "Hume AI" : provider === "kani" ? "KaniTTS-2" : provider || "Unknown";
  return (
    <span
      className={[
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] font-mono border",
        isHume
          ? "border-[#4f6db4] bg-[#1a2240] text-[#7da8e8]"
          : "border-[#6b4a28] bg-[#1e1208] text-[var(--gold)]",
      ].join(" ")}
    >
      {label}
    </span>
  );
}

/* ─── Section heading ────────────────────────────────────────────── */
function SectionLabel({ children }) {
  return (
    <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] font-heading mb-1.5">
      {children}
    </div>
  );
}

/**
 * VoiceConsole — 3-column voice workflow
 *
 * LEFT   (~260px): Voice library — search, filters, list
 * CENTER (flex-1): Selected voice detail + NPC assignment (Voice Console)
 * RIGHT  (~300px): Testing tools + provider status + recording (if !hume)
 */
export default function VoiceConsole({ campaignData, authFetch }) {
  const campaignCtx = useCampaignOptional();
  const voicesFromStore = useVoicesFromStore();

  /* ── Library ─────────────────────────────────────────────────── */
  const [voices, setVoices] = useState([]);
  const [ttsProvider, setTtsProvider] = useState("hume");
  const [filterState, setFilterState] = useState(defaultVoiceFilterState());
  const [selectedPreset, setSelectedPreset] = useState("");
  const [viewMode, setViewMode] = useState("list");
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const assignedNpcsForSelectedVoice = useNpcsForVoice(selectedVoiceId);

  /* ── Playback ────────────────────────────────────────────────── */
  const [isPlayingSample, setIsPlayingSample] = useState(false);
  const [playingClipId, setPlayingClipId] = useState("");
  const [playError, setPlayError] = useState("");
  const sampleAudioRef = useRef(null);
  const sampleAudioUrlRef = useRef("");

  /* ── Assignment ──────────────────────────────────────────────── */
  const [assignStatus, setAssignStatus] = useState("");
  const [isDeletingVoice, setIsDeletingVoice] = useState(false);

  /* ── NPC options ─────────────────────────────────────────────── */
  const [npcOptions, setNpcOptions] = useState([]);
  const npcList = useMemo(() => getNPCs(campaignData, authFetch), [campaignData, authFetch]);
  useEffect(() => {
    setNpcOptions(npcList);
    if (!npcVoiceNpcId && npcList[0]?.id) setNpcVoiceNpcId(String(npcList[0].id));
  }, [npcList]); // eslint-disable-line react-hooks/exhaustive-deps

  /* ── Testing (right col) ─────────────────────────────────────── */
  const [narrationText, setNarrationText] = useState("");
  const [isNarrating, setIsNarrating] = useState(false);
  const [narrateError, setNarrateError] = useState("");
  const [npcVoiceNpcId, setNpcVoiceNpcId] = useState("");
  const [npcVoiceText, setNpcVoiceText] = useState("");
  const [npcVoiceBusy, setNpcVoiceBusy] = useState(false);
  const [npcVoiceError, setNpcVoiceError] = useState("");
  const [generatedAudio, setGeneratedAudio] = useState([]);

  /* ── Clone state (kani-only) ─────────────────────────────────── */
  const [showClone, setShowClone] = useState(false);
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
  const [cloneSaved, setCloneSaved] = useState(false);
  const [cloneSaveError, setCloneSaveError] = useState("");
  const [saving, setSaving] = useState(false);

  const cloneAvailable = ttsProvider !== "hume";

  /* ── Derived ─────────────────────────────────────────────────── */
  const allVoices = useMemo(() => {
    const api = voices;
    const store = voicesFromStore ?? [];
    if (!campaignCtx || !store.length) return api;
    const byId = new Map(
      api.map((v) => [
        v.id || v.voice_id,
        { ...v, assignedNPCs: v.assignedNPCs ?? v.assignedNpcIds ?? [] },
      ])
    );
    store.forEach((v) =>
      byId.set(v.id, {
        ...v,
        id: v.id,
        voice_id: v.id,
        name: v.name,
        assignedNPCs: v.assignedNpcIds ?? [],
      })
    );
    return Array.from(byId.values());
  }, [voices, campaignCtx, voicesFromStore]);

  const filteredVoices = useMemo(
    () => filterVoices(allVoices, filterState, selectedPreset),
    [allVoices, filterState, selectedPreset]
  );

  const selectedVoice = useMemo(() => {
    const v = allVoices.find((x) => (x.voice_id || x.id) === selectedVoiceId) || null;
    if (v && assignedNpcsForSelectedVoice?.length) {
      return { ...v, assignedNPCs: assignedNpcsForSelectedVoice.map((n) => n.id) };
    }
    return v;
  }, [allVoices, selectedVoiceId, assignedNpcsForSelectedVoice]);

  /* ── Load voices ─────────────────────────────────────────────── */
  const loadVoices = useCallback(async () => {
    if (!authFetch) return;
    const list = await getVoices(authFetch);
    setVoices(list);
  }, [authFetch]);

  const loadGeneratedAudio = useCallback(async () => {
    const clips = await getGeneratedAudio(authFetch);
    setGeneratedAudio(clips);
  }, [authFetch]);

  useEffect(() => {
    let cancelled = false;
    if (!authFetch) return undefined;
    authFetch("/config")
      .then((res) => (res.ok ? res.json() : {}))
      .then((cfg) => {
        if (!cancelled && cfg?.tts_provider) setTtsProvider(String(cfg.tts_provider));
      })
      .catch(() => {
        if (!cancelled) setTtsProvider("hume");
      });
    return () => { cancelled = true; };
  }, [authFetch]);

  useEffect(() => { loadVoices(); }, [loadVoices]);
  useEffect(() => { loadGeneratedAudio(); }, [loadGeneratedAudio]);

  useEffect(() => {
    if (voices.length > 0 && !selectedVoiceId) {
      setSelectedVoiceId(voices[0].id || voices[0].voice_id);
    }
  }, [voices, selectedVoiceId]);

  /* ── Audio helpers ───────────────────────────────────────────── */
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

  useEffect(() => stopSampleAudio, [stopSampleAudio]);

  const playAudioBlob = useCallback(async (blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    audio.onerror = () => URL.revokeObjectURL(url);
    await audio.play();
  }, []);

  /* ── Handlers: library ───────────────────────────────────────── */
  const playSample = useCallback(
    async (voiceId, text = "The quick brown fox jumps over the lazy dog.") => {
      if (!voiceId || !authFetch) return;
      stopSampleAudio();
      setPlayError("");
      setIsPlayingSample(true);
      try {
        const res = await authFetch("/tts/narrate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: text.trim() || "The quick brown fox jumps over the lazy dog.",
            voice_id: voiceId,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          const msg = err.detail || err.error || res.statusText || "Play failed";
          setPlayError(typeof msg === "string" ? msg : "Play failed");
          setIsPlayingSample(false);
          return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        sampleAudioRef.current = audio;
        sampleAudioUrlRef.current = url;
        audio.onended = () => { stopSampleAudio(); setIsPlayingSample(false); };
        audio.onerror = () => {
          stopSampleAudio();
          setPlayError("Audio failed to play.");
          setIsPlayingSample(false);
        };
        await audio.play();
      } catch (e) {
        stopSampleAudio();
        setPlayError(e?.message || "Play failed");
        setIsPlayingSample(false);
      }
    },
    [authFetch, stopSampleAudio]
  );

  const handleAssignToNpc = useCallback(
    (voiceId, npcId) => {
      if (campaignCtx?.assignVoiceToNpc && voiceId && npcId) {
        campaignCtx.assignVoiceToNpc(npcId, voiceId);
        setAssignStatus("Voice assigned.");
      } else if (voiceId && npcId) {
        setAssignStatus("No campaign loaded — load a campaign first.");
      } else {
        setAssignStatus("Select an NPC to assign.");
      }
      setTimeout(() => setAssignStatus(""), 3000);
    },
    [campaignCtx]
  );

  const handleDeleteVoice = useCallback(
    async (voice) => {
      const voiceId = voice?.voice_id || voice?.id;
      if (!voiceId || !authFetch || isDeletingVoice) return;
      const confirmed = typeof window !== "undefined"
        ? window.confirm(`Delete voice "${voice?.name || voiceId}"? This cannot be undone.`)
        : false;
      if (!confirmed) return;
      setIsDeletingVoice(true);
      setPlayError("");
      setAssignStatus("");
      try {
        const result = await deleteVoice(voiceId, authFetch);
        if (!result.ok) { setPlayError(result.error || "Delete failed"); return; }
        const nextVoices = await getVoices(authFetch);
        setVoices(nextVoices);
        const fallback = nextVoices.find((v) => (v.voice_id || v.id) !== voiceId);
        setSelectedVoiceId(fallback ? (fallback.voice_id || fallback.id || "") : "");
        setAssignStatus(`Deleted ${voice?.name || voiceId}.`);
        setTimeout(() => setAssignStatus(""), 3000);
      } finally {
        setIsDeletingVoice(false);
      }
    },
    [authFetch, isDeletingVoice]
  );

  /* ── Handlers: testing ───────────────────────────────────────── */
  const handleNarrate = useCallback(async () => {
    if (!narrationText.trim() || !selectedVoiceId || !authFetch) return;
    setIsNarrating(true);
    setNarrateError("");
    try {
      const res = await authFetch("/tts/narrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: narrationText.trim(), voice_id: selectedVoiceId }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Narration failed.");
      const blob = await res.blob();
      await playAudioBlob(blob);
      loadGeneratedAudio();
    } catch (e) {
      setNarrateError(e?.message || "Narration failed.");
    } finally {
      setIsNarrating(false);
    }
  }, [narrationText, selectedVoiceId, authFetch, playAudioBlob, loadGeneratedAudio]);

  const handleSpeakAsNpc = useCallback(async () => {
    if (!npcVoiceNpcId || !npcVoiceText.trim() || !authFetch || npcVoiceBusy) return;
    setNpcVoiceBusy(true);
    setNpcVoiceError("");
    try {
      const res = await authFetch("/tts/npc-dialogue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ npc_id: npcVoiceNpcId, text: npcVoiceText.trim() }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "NPC dialogue failed.");
      const blob = await res.blob();
      await playAudioBlob(blob);
      loadGeneratedAudio();
    } catch (e) {
      setNpcVoiceError(e?.message || "NPC dialogue failed.");
    } finally {
      setNpcVoiceBusy(false);
    }
  }, [authFetch, npcVoiceBusy, npcVoiceNpcId, npcVoiceText, playAudioBlob, loadGeneratedAudio]);

  const handlePlayClip = useCallback(
    async (clip) => {
      if (!clip || !authFetch) return;
      stopSampleAudio();
      setPlayError("");
      setPlayingClipId(clip.id);
      try {
        const res = await authFetch("/tts/narrate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: clip.title || "Sample.", voice_id: clip.voiceId }),
        });
        if (!res.ok) { setPlayingClipId(""); return; }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        sampleAudioRef.current = audio;
        sampleAudioUrlRef.current = url;
        audio.onended = () => { stopSampleAudio(); setPlayingClipId(""); };
        audio.onerror = () => { stopSampleAudio(); setPlayingClipId(""); };
        await audio.play();
      } catch {
        setPlayingClipId("");
      }
    },
    [authFetch, stopSampleAudio]
  );

  /* ── Clone handlers (kani-only) ──────────────────────────────── */
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
        loadVoices();
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
            loadVoices();
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
  }, [cloneFile, cloneName, isCloning, authFetch, loadVoices]);

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
        const res = await authFetch(`/voices/${cloneVoiceId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: cloneName.trim(), tags: cloneTags }),
        });
        if (!res.ok) throw new Error(await res.text());
      }
      setCloneSaved(true);
      setCloneStep(5);
      loadVoices();
      if (selectedVoiceId !== cloneVoiceId) setSelectedVoiceId(cloneVoiceId);
    } catch (e) {
      setCloneSaveError(e?.message || "Save failed.");
    } finally {
      setSaving(false);
    }
  }, [cloneVoiceId, cloneName, cloneTags, authFetch, loadVoices, selectedVoiceId]);

  const handleResetClone = () => {
    setShowClone(false);
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
    setCloneSaved(false);
    setCloneSaveError("");
    setSaving(false);
  };

  /* ── Render ──────────────────────────────────────────────────── */
  return (
    <div className="flex h-full min-h-0 overflow-hidden">

      {/* ── LEFT: Voice Library ─────────────────────────────────── */}
      <aside
        className="flex-shrink-0 flex flex-col min-h-0 border-r border-[#3a2510] overflow-hidden"
        style={{ width: "260px" }}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-[#3a2510] flex-shrink-0">
          <span className="font-heading text-sm text-[var(--gold)]">Voice Library</span>
          <div className="flex border border-[#5c3e23] rounded overflow-hidden">
            <button
              type="button"
              className={`p-1 ${viewMode === "list" ? "bg-[rgba(202,167,75,0.2)] text-[var(--gold)]" : "text-[var(--text-2)] hover:text-[var(--gold)]"}`}
              onClick={() => setViewMode("list")}
              title="List view"
            >
              <List size={13} />
            </button>
            <button
              type="button"
              className={`p-1 ${viewMode === "grid" ? "bg-[rgba(202,167,75,0.2)] text-[var(--gold)]" : "text-[var(--text-2)] hover:text-[var(--gold)]"}`}
              onClick={() => setViewMode("grid")}
              title="Grid view"
            >
              <LayoutGrid size={13} />
            </button>
          </div>
        </div>

        <div className="flex flex-col gap-2 px-3 py-2 flex-shrink-0 border-b border-[#3a2510]">
          <VoiceSearchInput
            value={filterState.query}
            onChange={(q) => setFilterState((prev) => ({ ...prev, query: q }))}
          />
          {/* Preset pills */}
          <div className="flex flex-wrap gap-1">
            <button
              type="button"
              className={`vs-preset-pill${!selectedPreset ? " vs-preset-pill--active" : ""}`}
              onClick={() => setSelectedPreset("")}
            >
              All
            </button>
            {Object.values(VOICE_PRESETS).map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={`vs-preset-pill${selectedPreset === preset.id ? " vs-preset-pill--active" : ""}`}
                onClick={() => setSelectedPreset(selectedPreset === preset.id ? "" : preset.id)}
                title={preset.style}
              >
                {preset.name}
              </button>
            ))}
          </div>
          <VoiceLibraryFilters filterState={filterState} onFilterChange={setFilterState} />
        </div>

        <div className="px-2 py-1 flex-shrink-0">
          <span className="text-[10px] text-[var(--text-2)]">
            {filteredVoices.length} voice{filteredVoices.length !== 1 ? "s" : ""}
          </span>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-2">
          {filteredVoices.length === 0 ? (
            <EmptyState message="No voices match the current filters." />
          ) : (
            <VoiceLibraryGrid
              voices={filteredVoices}
              selectedVoiceId={selectedVoiceId}
              onSelectVoice={(id) => {
                setSelectedVoiceId((prev) => (prev === id ? "" : id));
                setPlayError("");
                setAssignStatus("");
              }}
              onPlaySample={(voice) =>
                playSample(voice?.voice_id || voice?.id, voice?.preview_text)
              }
              viewMode={viewMode}
            />
          )}
        </div>
      </aside>

      {/* ── CENTER: Voice Console (detail + assignment) ────────── */}
      <main className="flex-1 min-h-0 min-w-0 overflow-y-auto p-4 border-r border-[#3a2510] flex flex-col gap-3">
        {assignStatus && (
          <div className="rounded-md border border-[#5f7d63] bg-[rgba(94,124,99,0.12)] px-3 py-2 text-xs text-[#9ec24f] flex-shrink-0">
            {assignStatus}
          </div>
        )}
        <VoiceDetailPanel
          voice={selectedVoice}
          npcOptions={npcOptions}
          isPlayingSample={isPlayingSample}
          playError={playError}
          isDeletingVoice={isDeletingVoice}
          onPlaySample={(text) => playSample(selectedVoiceId, text)}
          onAssignToNpc={handleAssignToNpc}
          onUnassignNpc={() => setAssignStatus("Unassign: coming soon.")}
          onReuseForNarration={(voice) => {
            setNarrationText(voice?.preview_text || voice?.name || "");
          }}
          onDeleteVoice={handleDeleteVoice}
        />
      </main>

      {/* ── RIGHT: Testing + Provider + Clone ───────────────────── */}
      <aside
        className="flex-shrink-0 flex flex-col min-h-0 overflow-hidden"
        style={{ width: "280px" }}
      >
        {/* Provider status */}
        <div className="px-3 py-2 border-b border-[#3a2510] flex items-center gap-2 flex-shrink-0 bg-[#130d07]">
          <span className="text-[10px] text-[var(--text-2)] uppercase tracking-[0.12em]">Provider</span>
          <ProviderBadge provider={ttsProvider} />
        </div>

        <div className="flex flex-col flex-1 min-h-0 overflow-y-auto">

          {/* Narration test */}
          <div className="voice-section p-3 border-b border-[#2a1a0a]">
            <SectionLabel>
              <ScrollText size={10} className="inline mr-1" />
              Narration Test
            </SectionLabel>
            <textarea
              value={narrationText}
              onChange={(e) => setNarrationText(e.target.value)}
              rows={2}
              placeholder="Enter text to test with the selected voice…"
              className="w-full rounded border border-[#4a3018] bg-[#1c1008] px-2.5 py-1.5 text-xs text-[var(--text-1)] placeholder:text-[var(--text-2)] resize-none focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
            />
            {narrateError && (
              <p className="text-[11px] text-[#ff9d82] mt-1">{narrateError}</p>
            )}
            <FantasyButton
              variant="primary"
              className="text-xs mt-1.5 w-full"
              onClick={handleNarrate}
              disabled={!narrationText.trim() || !selectedVoiceId || isNarrating}
            >
              {isNarrating ? "Narrating…" : "Speak Narration"}
            </FantasyButton>
            {!selectedVoiceId && (
              <p className="text-[10px] text-[var(--text-2)] mt-1">
                Select a voice first.
              </p>
            )}
          </div>

          {/* NPC dialogue test */}
          <div className="voice-section p-3 border-b border-[#2a1a0a]">
            <SectionLabel>
              <Mic2 size={10} className="inline mr-1" />
              NPC Dialogue
            </SectionLabel>
            <select
              value={npcVoiceNpcId}
              onChange={(e) => setNpcVoiceNpcId(e.target.value)}
              className="w-full mb-1.5 rounded border border-[#4a3018] bg-[#1c1008] px-2.5 py-1.5 text-xs text-[var(--text-1)] focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
            >
              <option value="">— Select NPC —</option>
              {npcOptions.map((npc) => (
                <option key={npc.id} value={String(npc.id)}>
                  {npc.name}
                </option>
              ))}
            </select>
            <textarea
              value={npcVoiceText}
              onChange={(e) => setNpcVoiceText(e.target.value)}
              rows={2}
              placeholder="Dialogue line…"
              className="w-full rounded border border-[#4a3018] bg-[#1c1008] px-2.5 py-1.5 text-xs text-[var(--text-1)] placeholder:text-[var(--text-2)] resize-none focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
            />
            {npcVoiceError && (
              <p className="text-[11px] text-[#ff9d82] mt-1">{npcVoiceError}</p>
            )}
            <FantasyButton
              variant="secondary"
              className="text-xs mt-1.5 w-full"
              onClick={handleSpeakAsNpc}
              disabled={!npcVoiceNpcId || !npcVoiceText.trim() || npcVoiceBusy}
            >
              {npcVoiceBusy ? "Speaking…" : "Speak as NPC"}
            </FantasyButton>
          </div>

          {/* Recent clips */}
          {generatedAudio.length > 0 && (
            <div className="voice-section p-3 border-b border-[#2a1a0a]">
              <SectionLabel>Recent Audio</SectionLabel>
              <GeneratedAudioList
                clips={generatedAudio}
                onPlayClip={handlePlayClip}
                playingClipId={playingClipId}
              />
            </div>
          )}

          {/* Voice upload / clone — kani only */}
          {cloneAvailable && (
            <div className="voice-section p-3">
              <div className="flex items-center justify-between mb-2">
                <SectionLabel>Add Voice</SectionLabel>
                {showClone ? (
                  <button
                    type="button"
                    onClick={handleResetClone}
                    className="text-[var(--text-2)] hover:text-[var(--text-1)]"
                    title="Close"
                  >
                    <X size={13} />
                  </button>
                ) : (
                  <FantasyButton
                    variant="ghost"
                    className="text-[10px] px-2 py-1"
                    onClick={() => setShowClone(true)}
                  >
                    <Plus size={10} className="inline mr-1" />
                    New
                  </FantasyButton>
                )}
              </div>

              {showClone ? (
                <div className="rounded border border-[#5c3e23] bg-[#150d07]/80 p-2.5">
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
                </div>
              ) : (
                <p className="text-[10px] text-[var(--text-2)] leading-relaxed">
                  Upload a voice sample to clone it into your library.
                </p>
              )}
            </div>
          )}

          {/* Hume: explain what's available */}
          {!cloneAvailable && (
            <div className="voice-section p-3">
              <SectionLabel>Note</SectionLabel>
              <p className="text-[11px] text-[var(--text-2)] leading-relaxed">
                Using Hume AI — local voice upload is unavailable. Select a voice to assign it to NPCs.
              </p>
            </div>
          )}

        </div>
      </aside>

    </div>
  );
}
