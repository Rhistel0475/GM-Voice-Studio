import React, { useState, useCallback, useEffect } from "react";
import SectionHeader from "../components/layout/SectionHeader";
import VoiceLibraryGrid from "../components/voices/VoiceLibraryGrid";
import VoiceCloneWizard from "../components/voices/VoiceCloneWizard";
import { FantasyButton } from "../components/shared";

export default function VoiceStudioPage({ authFetch }) {
  const [voices, setVoices] = useState([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [cloneStep, setCloneStep] = useState(1);
  const [cloneFile, setCloneFile] = useState(null);
  const [cloneName, setCloneName] = useState("");
  const [isCloning, setIsCloning] = useState(false);
  const [cloneStatus, setCloneStatus] = useState("");
  const [cloneProgress, setCloneProgress] = useState(0);
  const [narrationText, setNarrationText] = useState("");
  const [isNarrating, setIsNarrating] = useState(false);

  const reloadVoices = useCallback(() => {
    authFetch("/voices/list")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => {
        const list = Array.isArray(data) ? data.filter((v) => v?.voice_id) : [];
        setVoices(list);
        if (!selectedVoiceId && list[0]) setSelectedVoiceId(list[0].voice_id);
      })
      .catch(() => setVoices([]));
  }, [authFetch, selectedVoiceId]);

  useEffect(() => {
    reloadVoices();
  }, [reloadVoices]);

  const playSample = useCallback(
    async (voiceId, text = "The quick brown fox jumps over the lazy dog.") => {
      if (!voiceId) return;
      try {
        const res = await authFetch("/tts/narrate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, voice_id: voiceId }),
        });
        if (!res.ok) return;
        const blob = await res.blob();
        new Audio(URL.createObjectURL(blob)).play();
      } catch { /* ignore */ }
    },
    [authFetch]
  );

  const handleClone = async () => {
    if (!cloneFile || isCloning) return;
    setCloneStep(2);
    setIsCloning(true);
    setCloneStatus("Training voice…");
    setCloneProgress(10);
    try {
      const formData = new FormData();
      formData.append("audio", cloneFile);
      formData.append("consent_scope", "tts");
      if (cloneName.trim()) formData.append("name", cloneName.trim());
      const res = await authFetch("/voices/clone", { method: "POST", body: formData });
      if (!res.ok) throw new Error((await res.text()) || "Clone failed.");
      const payload = await res.json();
      setCloneProgress(50);
      if (payload?.voice_id) {
        setCloneStatus(`Voice created: ${payload.voice_id}`);
        setCloneProgress(100);
        reloadVoices();
        setSelectedVoiceId(payload.voice_id);
        setCloneStep(3);
        return;
      }
      if (payload?.job_id) {
        setCloneStatus(`Queued: ${payload.job_id}. Polling…`);
        for (let i = 0; i < 45; i++) {
          await new Promise((r) => setTimeout(r, 2000));
          setCloneProgress(50 + (i / 45) * 50);
          const statusRes = await authFetch(`/jobs/${payload.job_id}`);
          if (!statusRes.ok) continue;
          const statusPayload = await statusRes.json();
          const status = (statusPayload?.status || "").toLowerCase();
          if (status === "completed" && statusPayload?.voice_id) {
            setCloneStatus(`Voice created: ${statusPayload.voice_id}`);
            setCloneProgress(100);
            reloadVoices();
            setSelectedVoiceId(statusPayload.voice_id);
            setCloneStep(3);
            return;
          }
          if (status === "failed" || status === "error")
            throw new Error(statusPayload?.error || "Clone failed.");
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
  };

  const handleNarrate = async () => {
    if (!narrationText.trim() || !selectedVoiceId || isNarrating) return;
    setIsNarrating(true);
    try {
      const res = await authFetch("/tts/narrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: narrationText.trim(), voice_id: selectedVoiceId }),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      new Audio(URL.createObjectURL(blob)).play();
    } catch (e) {
      setCloneStatus(e?.message || "Narration failed.");
    } finally {
      setIsNarrating(false);
    }
  };

  return (
    <section className="min-h-0 flex-1 grid grid-cols-1 xl:grid-cols-12 gap-4 p-3">
      <div className="xl:col-span-6 min-h-0 flex flex-col gap-3">
        <SectionHeader title="Voice library" />
        <div className="panel-ornate rounded border border-[#734f2c] p-2 flex-1 min-h-0 flex flex-col">
          <div className="panel-head">
            <div className="plaque">Voice library</div>
          </div>
          <div className="panel-body min-h-0 overflow-auto">
            <VoiceLibraryGrid voices={voices} onPlaySample={playSample} />
          </div>
        </div>
        <div className="panel-ornate rounded border border-[#734f2c] p-2">
          <div className="panel-head">
            <div className="plaque">Narration</div>
          </div>
          <div className="panel-body">
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
                  <option key={v.voice_id} value={v.voice_id}>
                    {v.name?.trim() || v.voice_id}
                  </option>
                ))}
              </select>
              <FantasyButton
                variant="primary"
                onClick={handleNarrate}
                disabled={!narrationText.trim() || !selectedVoiceId || isNarrating}
              >
                {isNarrating ? "Generating…" : "Generate & play"}
              </FantasyButton>
            </div>
          </div>
        </div>
      </div>
      <div className="xl:col-span-6 min-h-0">
        <SectionHeader title="Clone voice wizard" />
        <div className="panel-ornate rounded border border-[#734f2c] p-2 mt-2">
          <div className="panel-head">
            <div className="plaque">Clone voice wizard</div>
          </div>
          <div className="panel-body">
            <VoiceCloneWizard
              step={cloneStep}
              cloneFile={cloneFile}
              onFileChange={setCloneFile}
              cloneName={cloneName}
              onNameChange={setCloneName}
              onTrain={handleClone}
              isCloning={isCloning}
              cloneStatus={cloneStatus}
              cloneProgress={cloneProgress}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
