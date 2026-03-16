import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LayoutGrid, Library, List, Mic2, ScrollText, Wand2 } from "lucide-react";
import { defaultVoiceFilterState } from "../../types/voice";
import { VOICE_PRESETS } from "../../lib/voicePresets";
import { deleteVoice, getVoices, getGeneratedAudio, submitClone, getCloneJobStatus } from "../../lib/api/voices";
import { getNPCs } from "../../lib/api/npcs";
import { useCampaignOptional } from "../../context/CampaignContext";
import { useVoices as useVoicesFromStore, useNpcsForVoice } from "../../store/selectors";
import { FantasyButton, ModalShell } from "../shared";
import VoiceActionCard from "./VoiceActionCard";
import VoiceSearchInput from "./VoiceSearchInput";
import VoiceLibraryFilters from "./VoiceLibraryFilters";
import VoiceLibraryGrid from "./VoiceLibraryGrid";
import GeneratedAudioList from "./GeneratedAudioList";
import VoiceDetailPanel from "./VoiceDetailPanel";
import VoiceCloneWizard from "./VoiceCloneWizard";

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
  if (filterState.source && filterState.source !== "all") {
    out = out.filter((v) => v.source === filterState.source);
  }
  if (filterState.status && filterState.status !== "all") {
    out = out.filter((v) => v.status === filterState.status);
  }
  if (filterState.tone && filterState.tone !== "all") {
    out = out.filter((v) => v.tone === filterState.tone);
  }
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

/**
 * Voice Studio main screen: fantasy tool deck.
 * Action cards launch the existing library, narration, NPC voice, and AI dialogue tools in modals.
 */
export default function VoiceStudioScreen({ campaignData, authFetch }) {
  const campaignCtx = useCampaignOptional();
  const voicesFromStore = useVoicesFromStore();
  const [voices, setVoices] = useState([]);
  const [generatedAudio, setGeneratedAudio] = useState([]);
  const [ttsProvider, setTtsProvider] = useState("kani");
  const [filterState, setFilterState] = useState(defaultVoiceFilterState());
  const [viewMode, setViewMode] = useState("grid");
  const [selectedPreset, setSelectedPreset] = useState("");
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [activeTool, setActiveTool] = useState("");
  const assignedNpcsForSelectedVoice = useNpcsForVoice(selectedVoiceId);
  const [isPlayingSample, setIsPlayingSample] = useState(false);
  const [playingClipId, setPlayingClipId] = useState("");
  const [npcOptions, setNpcOptions] = useState([]);
  const [npcVoiceNpcId, setNpcVoiceNpcId] = useState("");
  const [npcVoiceText, setNpcVoiceText] = useState("");
  const [npcVoiceBusy, setNpcVoiceBusy] = useState(false);
  const [aiDialogueNpcId, setAiDialogueNpcId] = useState("");
  const [aiDialoguePrompt, setAiDialoguePrompt] = useState("");
  const [aiDialogueGeneratedText, setAiDialogueGeneratedText] = useState("");
  const [aiDialogueBusy, setAiDialogueBusy] = useState(false);

  // Clone wizard state
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

  // Assignment feedback
  const [assignStatus, setAssignStatus] = useState("");
  // Play sample/preview error (e.g. "Voice not found")
  const [playError, setPlayError] = useState("");
  const [isDeletingVoice, setIsDeletingVoice] = useState(false);
  const sampleAudioRef = useRef(null);
  const sampleAudioUrlRef = useRef("");

  // Narration state
  const [narrationText, setNarrationText] = useState("");

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

  const playBase64Audio = useCallback(
    async (encoded, mimeType = "audio/wav") => {
      if (!encoded) return;
      const binary = atob(encoded);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      await playAudioBlob(new Blob([bytes], { type: mimeType }));
    },
    [playAudioBlob]
  );

  const loadVoices = useCallback(async () => {
    if (!authFetch) return;
    const list = await getVoices(authFetch);
    setVoices(list);
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
        if (!cancelled) setTtsProvider("kani");
      });
    return () => {
      cancelled = true;
    };
  }, [authFetch]);

  const loadGeneratedAudio = useCallback(async () => {
    const clips = await getGeneratedAudio(authFetch);
    setGeneratedAudio(clips);
  }, [authFetch]);

  useEffect(() => {
    loadVoices();
  }, [loadVoices]);

  useEffect(() => {
    if (voices.length > 0 && !selectedVoiceId) {
      setSelectedVoiceId(voices[0].id || voices[0].voice_id);
    }
  }, [voices, selectedVoiceId]);

  useEffect(() => {
    loadGeneratedAudio();
  }, [loadGeneratedAudio]);

  const npcList = useMemo(() => getNPCs(campaignData, authFetch), [campaignData, authFetch]);
  useEffect(() => {
    setNpcOptions(npcList);
  }, [npcList]);

  useEffect(() => {
    if (!npcVoiceNpcId && npcList[0]?.id) {
      setNpcVoiceNpcId(String(npcList[0].id));
    }
    if (!aiDialogueNpcId && npcList[0]?.id) {
      setAiDialogueNpcId(String(npcList[0].id));
    }
  }, [aiDialogueNpcId, npcList, npcVoiceNpcId]);

  const allVoices = useMemo(() => {
    const api = voices;
    const store = voicesFromStore ?? [];
    if (!campaignCtx || !store.length) return api;
    const byId = new Map(api.map((v) => [v.id || v.voice_id, { ...v, assignedNPCs: v.assignedNPCs ?? v.assignedNpcIds ?? [] }]));
    store.forEach((v) => byId.set(v.id, { ...v, id: v.id, voice_id: v.id, name: v.name, assignedNPCs: v.assignedNpcIds ?? [] }));
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

  const cloneAvailable = ttsProvider !== "hume";

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
          body: JSON.stringify({ text: text.trim() || "The quick brown fox jumps over the lazy dog.", voice_id: voiceId }),
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          const msg = errBody.detail || errBody.error || res.statusText || "Play failed";
          setPlayError(typeof msg === "string" ? msg : JSON.stringify(msg));
          setIsPlayingSample(false);
          if (typeof window !== "undefined" && window.toast) window.toast(typeof msg === "string" ? msg : "Play failed");
          return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        sampleAudioRef.current = audio;
        sampleAudioUrlRef.current = url;
        audio.onended = () => {
          stopSampleAudio();
          setIsPlayingSample(false);
        };
        audio.onerror = () => {
          stopSampleAudio();
          setPlayError("Audio failed to play.");
          setIsPlayingSample(false);
          if (typeof window !== "undefined" && window.toast) window.toast("Audio failed to play.");
        };
        await audio.play();
      } catch (e) {
        stopSampleAudio();
        setPlayError(e?.message || "Play failed");
        setIsPlayingSample(false);
        if (typeof window !== "undefined" && window.toast) window.toast(e?.message || "Play failed");
      }
    },
    [authFetch, stopSampleAudio]
  );

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
          body: JSON.stringify({
            text: clip.title || "Sample.",
            voice_id: clip.voiceId,
          }),
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          const msg = errBody.detail || errBody.error || res.statusText || "Play failed";
          setPlayError(typeof msg === "string" ? msg : JSON.stringify(msg));
          setPlayingClipId("");
          if (typeof window !== "undefined" && window.toast) window.toast(typeof msg === "string" ? msg : "Play failed");
          return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        sampleAudioRef.current = audio;
        sampleAudioUrlRef.current = url;
        audio.onended = () => {
          stopSampleAudio();
          setPlayingClipId("");
        };
        audio.onerror = () => {
          stopSampleAudio();
          setPlayError("Audio failed to play.");
          setPlayingClipId("");
          if (typeof window !== "undefined" && window.toast) window.toast("Audio failed to play.");
        };
        await audio.play();
      } catch (e) {
        stopSampleAudio();
        setPlayError(e?.message || "Play failed");
        setPlayingClipId("");
        if (typeof window !== "undefined" && window.toast) window.toast(e?.message || "Play failed");
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

  const handleUnassignNpc = useCallback((_voiceId, _npcId) => {
    setAssignStatus("Unassign: coming soon.");
    setTimeout(() => setAssignStatus(""), 3000);
  }, []);

  const handleReuseForNarration = useCallback((voice) => {
    setSelectedVoiceId(voice?.voice_id || voice?.id);
    setActiveTool("narration");
    if (typeof window !== "undefined" && window.toast) window.toast("Voice selected for narration.");
  }, []);

  const handleDeleteVoice = useCallback(
    async (voice) => {
      const voiceId = voice?.voice_id || voice?.id;
      if (!voiceId || !authFetch || isDeletingVoice) return;
      const voiceName = voice?.name || voiceId;

      if (typeof window !== "undefined") {
        const confirmed = window.confirm(`Delete voice "${voiceName}"? This cannot be undone.`);
        if (!confirmed) return;
      }

      setIsDeletingVoice(true);
      setPlayError("");
      setAssignStatus("");
      try {
        const result = await deleteVoice(voiceId, authFetch);
        if (!result.ok) {
          const msg = result.error || "Delete failed";
          setPlayError(msg);
          if (typeof window !== "undefined" && window.toast) window.toast(msg);
          return;
        }

        const nextVoices = await getVoices(authFetch);
        setVoices(nextVoices);
        const fallbackVoice = nextVoices.find((item) => (item.voice_id || item.id) !== voiceId);
        setSelectedVoiceId(fallbackVoice ? (fallbackVoice.voice_id || fallbackVoice.id || "") : "");
        setAssignStatus(`Deleted ${voiceName}.`);
        setTimeout(() => setAssignStatus(""), 3000);
        if (typeof window !== "undefined" && window.toast) window.toast(`Deleted ${voiceName}.`);
      } finally {
        setIsDeletingVoice(false);
      }
    },
    [authFetch, isDeletingVoice]
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
      playSample(voiceId).then(() => setIsPlayingPreview(false)).catch(() => setIsPlayingPreview(false));
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

  const handleNarrate = useCallback(async () => {
    if (!narrationText.trim() || !selectedVoiceId || !authFetch) return;
    try {
      const res = await authFetch("/tts/narrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: narrationText.trim(), voice_id: selectedVoiceId }),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      await playAudioBlob(blob);
      loadGeneratedAudio();
    } catch (e) {
      if (typeof window !== "undefined" && window.toast) window.toast(e?.message || "Narration failed.");
    }
  }, [narrationText, selectedVoiceId, authFetch, loadGeneratedAudio, playAudioBlob]);

  const handleSpeakAsNpc = useCallback(async () => {
    if (!npcVoiceNpcId || !npcVoiceText.trim() || !authFetch || npcVoiceBusy) return;
    setNpcVoiceBusy(true);
    try {
      const res = await authFetch("/tts/npc-dialogue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ npc_id: npcVoiceNpcId, text: npcVoiceText.trim() }),
      });
      if (!res.ok) throw new Error((await res.text()) || "NPC dialogue failed.");
      const blob = await res.blob();
      await playAudioBlob(blob);
      loadGeneratedAudio();
    } catch (e) {
      if (typeof window !== "undefined" && window.toast) window.toast(e?.message || "NPC dialogue failed.");
    } finally {
      setNpcVoiceBusy(false);
    }
  }, [authFetch, loadGeneratedAudio, npcVoiceBusy, npcVoiceNpcId, npcVoiceText, playAudioBlob]);

  const handleGenerateNpcResponse = useCallback(async () => {
    if (!aiDialogueNpcId || !aiDialoguePrompt.trim() || !authFetch || aiDialogueBusy) return;
    setAiDialogueBusy(true);
    try {
      const res = await authFetch("/npc/generate-dialogue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ npc_id: aiDialogueNpcId, player_input: aiDialoguePrompt.trim() }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload?.detail || payload?.error || "NPC response generation failed.");
      }
      setAiDialogueGeneratedText(payload.generated_text || "");
      if (payload.audio_base64) {
        await playBase64Audio(payload.audio_base64, payload.mime_type || "audio/wav");
      }
      loadGeneratedAudio();
    } catch (e) {
      if (typeof window !== "undefined" && window.toast) window.toast(e?.message || "NPC response generation failed.");
    } finally {
      setAiDialogueBusy(false);
    }
  }, [aiDialogueBusy, aiDialogueNpcId, aiDialoguePrompt, authFetch, loadGeneratedAudio, playBase64Audio]);

  useEffect(() => {
    setAiDialogueGeneratedText("");
  }, [aiDialogueNpcId, aiDialoguePrompt]);

  const closeTool = useCallback(() => setActiveTool(""), []);

  const voiceOptions = useMemo(() => (voices.length ? voices : allVoices), [voices, allVoices]);
  const selectedVoiceLabel = selectedVoice?.name?.trim()
    || voiceOptions.find((voice) => (voice.voice_id || voice.id) === selectedVoiceId)?.name?.trim()
    || selectedVoiceId
    || "Choose a voice in the library";
  const selectedNpcVoiceName = npcOptions.find((npc) => String(npc.id) === npcVoiceNpcId)?.name || "No NPC selected";
  const selectedAiNpcName = npcOptions.find((npc) => String(npc.id) === aiDialogueNpcId)?.name || "No NPC selected";
  const campaignTitle = campaignCtx?.campaign?.title || campaignData?.title || "No campaign loaded";
  const cloneSummary = cloneAvailable
    ? "Clone, edit, and assign local voices from the library workbench."
    : "Hosted Hume voices are available in the library; local cloning is disabled in this mode.";

  const renderLibraryWorkbench = () => (
    <div className="grid min-h-[72vh] grid-cols-1 gap-3 xl:grid-cols-12">
      <div className="xl:col-span-5 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
        <div className="plaque mb-2 shrink-0">Voice library</div>
        <div className="flex flex-col gap-2 shrink-0">
          <label className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
            Search
          </label>
          <VoiceSearchInput
            value={filterState.query}
            onChange={(q) => setFilterState((prev) => ({ ...prev, query: q }))}
          />
        </div>
        <div className="shrink-0 pt-2">
          <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-1">
            Presets
          </div>
          <div className="vs-preset-strip">
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
        </div>
        <div className="shrink-0 py-2">
          <VoiceLibraryFilters filterState={filterState} onFilterChange={setFilterState} />
        </div>
        <div className="flex items-center justify-between gap-2 mb-2 shrink-0">
          <span className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
            Voices ({filteredVoices.length})
          </span>
          <div className="flex rounded border border-[#5c3e23] overflow-hidden">
            <button
              type="button"
              className={`p-1.5 ${viewMode === "grid" ? "bg-[rgba(202,167,75,0.2)] text-[var(--gold)] border-[var(--gold)]" : "text-[var(--text-2)] hover:text-[var(--gold)]"}`}
              onClick={() => setViewMode("grid")}
              title="Grid view"
              aria-label="Grid view"
            >
              <LayoutGrid size={16} />
            </button>
            <button
              type="button"
              className={`p-1.5 border-l border-[#5c3e23] ${viewMode === "list" ? "bg-[rgba(202,167,75,0.2)] text-[var(--gold)]" : "text-[var(--text-2)] hover:text-[var(--gold)]"}`}
              onClick={() => setViewMode("list")}
              title="List view"
              aria-label="List view"
            >
              <List size={16} />
            </button>
          </div>
        </div>
        <div className="panel-body min-h-0 flex-1 overflow-auto flex flex-col">
          <VoiceLibraryGrid
            voices={filteredVoices}
            selectedVoiceId={selectedVoiceId}
            onPlaySample={playSample}
            onSelectVoice={(v) => setSelectedVoiceId(v?.voice_id || v?.id || "")}
            viewMode={viewMode}
          />
        </div>
        <div className="shrink-0 mt-2 pt-2 border-t border-[#5c3e23]">
          <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-2">
            Recent generated ({generatedAudio.length})
          </div>
          <GeneratedAudioList
            clips={generatedAudio}
            onPlayClip={handlePlayClip}
            playingClipId={playingClipId}
          />
        </div>
      </div>

      <div className="xl:col-span-7 min-h-0 flex flex-col gap-3 min-w-0">
        <div className="panel-ornate rounded border border-[#734f2c] p-2 flex-1 min-h-0 flex flex-col overflow-hidden">
          <div className="plaque mb-2 shrink-0">Voice detail</div>
          <div className="panel-body min-h-0 overflow-auto">
            <VoiceDetailPanel
              voice={selectedVoice}
              npcOptions={npcOptions}
              onPlaySample={playSample}
              isPlayingSample={isPlayingSample}
              playError={playError}
              onAssignToNpc={handleAssignToNpc}
              onUnassignNpc={handleUnassignNpc}
              onReuseForNarration={handleReuseForNarration}
              onDeleteVoice={selectedVoice?.deletable ? handleDeleteVoice : undefined}
              isDeletingVoice={isDeletingVoice}
            />
            {assignStatus && (
              <p className="text-xs text-[var(--gold)] mt-2 px-1">{assignStatus}</p>
            )}
          </div>
        </div>

        <div className="panel-ornate rounded border border-[#734f2c] p-2 shrink-0">
          <div className="plaque mb-2">Clone voice wizard</div>
          <div className="panel-body">
            {cloneAvailable ? (
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
            ) : (
              <div className="rounded border border-[#5c3e23] bg-[#1a1008] p-3 text-sm text-[var(--text-2)]">
                Voice cloning is disabled in Hume mode. Create custom Hume voices in Hume first, then use them here through the voice library.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <section className="voice-studio-screen min-h-0 flex-1 flex flex-col gap-4 p-3 md:p-4 bg-[var(--wood-2)]/50">
        <div className="panel-ornate overflow-hidden rounded-xl border border-[#7e5630]">
          <div className="panel-body relative gap-4 bg-[radial-gradient(circle_at_top_right,rgba(224,182,111,0.16),transparent_32%),linear-gradient(135deg,rgba(46,30,18,0.9),rgba(23,14,8,0.96))]">
            <div className="space-y-2">
              <p className="text-xs font-heading uppercase tracking-[0.32em] text-[#c8a050]">
                Voice Studio
              </p>
              <h1 className="font-heading text-3xl text-[var(--gold)]">
                Tool Deck
              </h1>
              <p className="max-w-3xl text-sm leading-relaxed text-[var(--text-2)]">
                Open the desk you need, run the voice action, then return to the deck. Narration, live NPC performance,
                AI-generated replies, and the full voice library all live behind dedicated action cards now.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-2 text-xs text-[var(--text-2)] md:grid-cols-3">
              <div className="rounded-lg border border-[#5c3e23] bg-[#120a05]/80 px-3 py-2">
                <span className="font-heading uppercase tracking-wider text-[#d7b77d]">Campaign</span>
                <p className="mt-1 text-sm text-[var(--text-1)]">{campaignTitle}</p>
              </div>
              <div className="rounded-lg border border-[#5c3e23] bg-[#120a05]/80 px-3 py-2">
                <span className="font-heading uppercase tracking-wider text-[#d7b77d]">Selected Voice</span>
                <p className="mt-1 text-sm text-[var(--text-1)]">{selectedVoiceLabel}</p>
              </div>
              <div className="rounded-lg border border-[#5c3e23] bg-[#120a05]/80 px-3 py-2">
                <span className="font-heading uppercase tracking-wider text-[#d7b77d]">Prepared Assets</span>
                <p className="mt-1 text-sm text-[var(--text-1)]">{allVoices.length} voices · {generatedAudio.length} recent clips</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <VoiceActionCard
            icon={ScrollText}
            title="Narration"
            description="Write boxed text, choose a voice, and send long-form narration straight to playback."
            launchLabel="Open Narration"
            tone="gold"
            meta={`Current voice: ${selectedVoiceLabel}\nRecent clips: ${generatedAudio.length}`}
            onLaunch={() => setActiveTool("narration")}
          />
          <VoiceActionCard
            icon={Mic2}
            title="NPC Voice"
            description="Speak a prepared line as an NPC using their assigned voice for immediate table playback."
            launchLabel="Open NPC Voice"
            tone="sage"
            meta={npcOptions.length ? `Ready NPC: ${selectedNpcVoiceName}\nNPCs available: ${npcOptions.length}` : "Load a campaign with NPCs to unlock direct NPC voice playback."}
            onLaunch={() => setActiveTool("npc-voice")}
            disabled={!npcOptions.length}
          />
          <VoiceActionCard
            icon={Wand2}
            title="AI Dialogue"
            description="Feed an NPC a player prompt and let the AI produce a short in-character response."
            launchLabel="Open AI Dialogue"
            tone="ember"
            meta={npcOptions.length ? `Focused NPC: ${selectedAiNpcName}\nLast result: ${aiDialogueGeneratedText ? "generated" : "waiting"}` : "Load a campaign with NPCs to generate AI dialogue."}
            onLaunch={() => setActiveTool("ai-dialogue")}
            disabled={!npcOptions.length}
          />
          <VoiceActionCard
            icon={Library}
            title="Voice Library"
            description="Browse voices, preview samples, assign them to NPCs, and open the clone workbench."
            launchLabel="Open Library"
            tone="arcane"
            meta={`Voices loaded: ${filteredVoices.length}\n${cloneSummary}`}
            onLaunch={() => setActiveTool("voice-library")}
          />
        </div>
      </section>

      {activeTool === "voice-library" && (
        <ModalShell title="Voice Library" onClose={closeTool} className="max-w-7xl">
          {renderLibraryWorkbench()}
        </ModalShell>
      )}

      {activeTool === "narration" && (
        <ModalShell title="Narration" onClose={closeTool} className="max-w-4xl">
          <div className="space-y-4">
            <div className="rounded-lg border border-[#5c3e23] bg-[#120a05]/80 px-3 py-3 text-sm text-[var(--text-2)]">
              <div className="font-heading uppercase tracking-wider text-[#d7b77d] text-xs">Narration Desk</div>
              <p className="mt-1">Compose long-form table narration and send it through the currently selected voice.</p>
            </div>
            <div className="parchment rounded border border-[#a17a42] p-4 space-y-3">
              <select
                className="chat-input w-full"
                value={selectedVoiceId}
                onChange={(e) => setSelectedVoiceId(e.target.value)}
              >
                <option value="">Select voice</option>
                {voiceOptions.map((v) => (
                  <option key={v.voice_id || v.id} value={v.voice_id || v.id}>
                    {v.name?.trim() || v.voice_id || v.id}
                  </option>
                ))}
              </select>
              <textarea
                className="chat-input w-full min-h-[180px] resize-y"
                placeholder="Enter narration text..."
                value={narrationText}
                onChange={(e) => setNarrationText(e.target.value)}
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs text-[#6b3e10]">Selected voice: {selectedVoiceLabel}</p>
                <FantasyButton
                  variant="primary"
                  onClick={handleNarrate}
                  disabled={!narrationText.trim() || !selectedVoiceId}
                >
                  Generate &amp; Play
                </FantasyButton>
              </div>
            </div>
            <div className="panel-ornate rounded border border-[#734f2c] p-2">
              <div className="plaque mb-2">Recent generated</div>
              <GeneratedAudioList
                clips={generatedAudio}
                onPlayClip={handlePlayClip}
                playingClipId={playingClipId}
              />
            </div>
          </div>
        </ModalShell>
      )}

      {activeTool === "npc-voice" && (
        <ModalShell title="NPC Voice" onClose={closeTool} className="max-w-3xl">
          <div className="space-y-4">
            <div className="rounded-lg border border-[#5c3e23] bg-[#120a05]/80 px-3 py-3 text-sm text-[var(--text-2)]">
              <div className="font-heading uppercase tracking-wider text-[#d7b77d] text-xs">NPC Performance</div>
              <p className="mt-1">Choose an NPC, type the exact line, and perform it with that NPC's assigned voice.</p>
            </div>
            <div className="parchment rounded border border-[#a17a42] p-4 space-y-3">
              <select
                className="chat-input w-full"
                value={npcVoiceNpcId}
                onChange={(e) => setNpcVoiceNpcId(e.target.value)}
              >
                <option value="">Select NPC</option>
                {npcOptions.map((npc) => (
                  <option key={npc.id} value={npc.id}>
                    {npc.name || npc.id}
                  </option>
                ))}
              </select>
              <textarea
                className="chat-input w-full min-h-[180px] resize-y"
                placeholder="Enter the exact line you want this NPC to speak..."
                value={npcVoiceText}
                onChange={(e) => setNpcVoiceText(e.target.value)}
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs text-[#6b3e10]">Selected NPC: {selectedNpcVoiceName}</p>
                <FantasyButton
                  variant="primary"
                  onClick={handleSpeakAsNpc}
                  disabled={!npcVoiceNpcId || !npcVoiceText.trim() || npcVoiceBusy}
                >
                  {npcVoiceBusy ? "Working..." : "Speak as NPC"}
                </FantasyButton>
              </div>
            </div>
          </div>
        </ModalShell>
      )}

      {activeTool === "ai-dialogue" && (
        <ModalShell title="AI Dialogue" onClose={closeTool} className="max-w-4xl">
          <div className="space-y-4">
            <div className="rounded-lg border border-[#5c3e23] bg-[#120a05]/80 px-3 py-3 text-sm text-[var(--text-2)]">
              <div className="font-heading uppercase tracking-wider text-[#d7b77d] text-xs">Improvisation Desk</div>
              <p className="mt-1">Provide player input or scene pressure, then generate and play a short in-character NPC response.</p>
            </div>
            <div className="parchment rounded border border-[#a17a42] p-4 space-y-3">
              <select
                className="chat-input w-full"
                value={aiDialogueNpcId}
                onChange={(e) => setAiDialogueNpcId(e.target.value)}
              >
                <option value="">Select NPC</option>
                {npcOptions.map((npc) => (
                  <option key={npc.id} value={npc.id}>
                    {npc.name || npc.id}
                  </option>
                ))}
              </select>
              <textarea
                className="chat-input w-full min-h-[180px] resize-y"
                placeholder="Enter player dialogue, pressure, or scene context for the AI..."
                value={aiDialoguePrompt}
                onChange={(e) => setAiDialoguePrompt(e.target.value)}
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs text-[#6b3e10]">Selected NPC: {selectedAiNpcName}</p>
                <FantasyButton
                  variant="primary"
                  onClick={handleGenerateNpcResponse}
                  disabled={!aiDialogueNpcId || !aiDialoguePrompt.trim() || aiDialogueBusy}
                >
                  {aiDialogueBusy ? "Working..." : "Generate Dialogue"}
                </FantasyButton>
              </div>
            </div>
            {aiDialogueGeneratedText ? (
              <div className="panel-ornate rounded border border-[#734f2c] p-2">
                <div className="plaque mb-2">Generated dialogue</div>
                <div className="panel-body">
                  <div className="rounded border border-[#5c3e23] bg-[#1a1008]/70 p-3 text-sm text-[var(--text-1)]">
                    <p>{aiDialogueGeneratedText}</p>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </ModalShell>
      )}
    </>
  );
}
