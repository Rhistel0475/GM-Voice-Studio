import React, { useState, useMemo, useEffect, useCallback } from "react";
import { LayoutGrid, List } from "lucide-react";
import { defaultVoiceFilterState } from "../../types/voice";
import { getVoices, getGeneratedAudio, submitClone, getCloneJobStatus } from "../../lib/api/voices";
import { getNPCs } from "../../lib/api/npcs";
import { useCampaignOptional } from "../../context/CampaignContext";
import { useVoices as useVoicesFromStore, useNpcsForVoice } from "../../store/selectors";
import { FantasyButton } from "../shared";
import VoiceSearchInput from "./VoiceSearchInput";
import VoiceLibraryFilters from "./VoiceLibraryFilters";
import VoiceLibraryGrid from "./VoiceLibraryGrid";
import GeneratedAudioList from "./GeneratedAudioList";
import VoiceDetailPanel from "./VoiceDetailPanel";
import VoiceCloneWizard from "./VoiceCloneWizard";

function filterVoices(voices, filterState) {
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
  return out;
}

/**
 * Voice Studio main screen: 2-column layout.
 * Left: library (search, filters, voice grid, generated audio list).
 * Right: detail panel (selected voice, sample player, metadata, assignment) and clone wizard.
 */
export default function VoiceStudioScreen({ campaignData, authFetch }) {
  const campaignCtx = useCampaignOptional();
  const voicesFromStore = useVoicesFromStore();
  const [voices, setVoices] = useState([]);
  const [generatedAudio, setGeneratedAudio] = useState([]);
  const [filterState, setFilterState] = useState(defaultVoiceFilterState());
  const [viewMode, setViewMode] = useState("grid");
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const assignedNpcsForSelectedVoice = useNpcsForVoice(selectedVoiceId);
  const [isPlayingSample, setIsPlayingSample] = useState(false);
  const [playingClipId, setPlayingClipId] = useState("");
  const [npcOptions, setNpcOptions] = useState([]);

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

  // Narration panel (right column, below detail)
  const [narrationText, setNarrationText] = useState("");

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

  const allVoices = useMemo(() => {
    const api = voices;
    const store = voicesFromStore ?? [];
    if (!campaignCtx || !store.length) return api;
    const byId = new Map(api.map((v) => [v.id || v.voice_id, { ...v, assignedNPCs: v.assignedNPCs ?? v.assignedNpcIds ?? [] }]));
    store.forEach((v) => byId.set(v.id, { ...v, id: v.id, voice_id: v.id, name: v.name, assignedNPCs: v.assignedNpcIds ?? [] }));
    return Array.from(byId.values());
  }, [voices, campaignCtx, voicesFromStore]);

  const filteredVoices = useMemo(
    () => filterVoices(allVoices, filterState),
    [allVoices, filterState]
  );

  const selectedVoice = useMemo(() => {
    const v = allVoices.find((x) => (x.voice_id || x.id) === selectedVoiceId) || null;
    if (v && assignedNpcsForSelectedVoice?.length) {
      return { ...v, assignedNPCs: assignedNpcsForSelectedVoice.map((n) => n.id) };
    }
    return v;
  }, [allVoices, selectedVoiceId, assignedNpcsForSelectedVoice]);

  const playSample = useCallback(
    async (voiceId, text = "The quick brown fox jumps over the lazy dog.") => {
      if (!voiceId || !authFetch) return;
      setIsPlayingSample(true);
      try {
        const res = await authFetch("/tts/narrate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text.trim() || "The quick brown fox jumps over the lazy dog.", voice_id: voiceId }),
        });
        if (!res.ok) return;
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => {
          URL.revokeObjectURL(url);
          setIsPlayingSample(false);
        };
        audio.play();
      } catch {
        setIsPlayingSample(false);
      }
    },
    [authFetch]
  );

  const handlePlayClip = useCallback(
    async (clip) => {
      if (!clip || !authFetch) return;
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
        if (!res.ok) throw new Error("Play failed");
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => {
          URL.revokeObjectURL(url);
          setPlayingClipId("");
        };
        audio.play();
      } catch {
        setPlayingClipId("");
      }
    },
    [authFetch]
  );

  const handleAssignToNpc = useCallback(
    (voiceId, npcId) => {
      if (campaignCtx?.assignVoiceToNpc && voiceId && npcId) {
        campaignCtx.assignVoiceToNpc(npcId, voiceId);
        if (typeof window !== "undefined" && window.toast) window.toast("Voice assigned to NPC (campaign state updated).");
        else if (typeof window !== "undefined") window.alert("Voice assigned to NPC.");
      } else {
        if (typeof window !== "undefined" && window.toast) window.toast("Assign to NPC: coming soon.");
        else if (typeof window !== "undefined") window.alert("Assign to NPC: coming soon.");
      }
      // TODO: Backend — persist voice assignment via PATCH /api/npcs/:id or PATCH /api/campaigns/:id/npcs.
    },
    [campaignCtx]
  );

  const handleUnassignNpc = useCallback((voiceId, npcId) => {
    if (typeof window !== "undefined" && window.toast) window.toast("Unassign: coming soon.");
    else window.alert("Unassign: coming soon.");
  }, []);

  const handleReuseForNarration = useCallback((voice) => {
    setSelectedVoiceId(voice?.voice_id || voice?.id);
    if (typeof window !== "undefined" && window.toast) window.toast("Voice selected for narration.");
  }, []);

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
      if (!result.ok) throw new Error("Clone request failed.");
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
        for (let i = 0; i < 45; i++) {
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
      const url = URL.createObjectURL(blob);
      new Audio(url).play();
      loadGeneratedAudio();
    } catch (e) {
      if (typeof window !== "undefined" && window.toast) window.toast(e?.message || "Narration failed.");
    }
  }, [narrationText, selectedVoiceId, authFetch, loadGeneratedAudio]);

  return (
    <section className="voice-studio-screen min-h-0 flex-1 grid grid-cols-1 md:grid-cols-12 gap-3 p-2 md:p-3 bg-[var(--wood-2)]/50">
      {/* Left: library */}
      <div className="md:col-span-5 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
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
            onAssign={(v) => handleAssignToNpc(v?.voice_id || v?.id, null)}
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

      {/* Right: detail + clone wizard + narration */}
      <div className="md:col-span-7 min-h-0 flex flex-col gap-3 min-w-0">
        <div className="panel-ornate rounded border border-[#734f2c] p-2 flex-1 min-h-0 flex flex-col overflow-hidden">
          <div className="plaque mb-2 shrink-0">Voice detail</div>
          <div className="panel-body min-h-0 overflow-auto">
            <VoiceDetailPanel
              voice={selectedVoice}
              npcOptions={npcOptions}
              onPlaySample={playSample}
              isPlayingSample={isPlayingSample}
              onAssignToNpc={handleAssignToNpc}
              onUnassignNpc={handleUnassignNpc}
              onReuseForNarration={handleReuseForNarration}
            />
          </div>
        </div>

        <div className="panel-ornate rounded border border-[#734f2c] p-2 shrink-0">
          <div className="plaque mb-2">Clone voice wizard</div>
          <div className="panel-body">
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
        </div>

        <div className="panel-ornate rounded border border-[#734f2c] p-2 shrink-0">
          <div className="plaque mb-2">Narration</div>
          <p className="text-xs text-[var(--text-2)] mb-2">Long-form text with selected voice.</p>
          <textarea
            className="chat-input w-full min-h-[80px] resize-y"
            placeholder="Enter narration text…"
            value={narrationText}
            onChange={(e) => setNarrationText(e.target.value)}
          />
          <div className="flex flex-wrap gap-2 mt-2">
            <select
              className="chat-input flex-1 min-w-[120px]"
              value={selectedVoiceId}
              onChange={(e) => setSelectedVoiceId(e.target.value)}
            >
              <option value="">Select voice</option>
              {voices.map((v) => (
                <option key={v.voice_id || v.id} value={v.voice_id || v.id}>
                  {v.name?.trim() || v.voice_id || v.id}
                </option>
              ))}
            </select>
            <FantasyButton
              variant="primary"
              onClick={handleNarrate}
              disabled={!narrationText.trim() || !selectedVoiceId}
            >
              Generate & play
            </FantasyButton>
          </div>
        </div>
      </div>
    </section>
  );
}
