import React, { useState, useCallback, useEffect } from "react";
import SectionHeader from "../components/layout/SectionHeader";
import NPCListSidebar from "../components/npcs/NPCListSidebar";
import NPCGeneratorForm from "../components/npcs/NPCGeneratorForm";
import NPCPortraitCard from "../components/npcs/NPCPortraitCard";
import NPCDetailCard from "../components/npcs/NPCDetailCard";
import NPCVoiceAssignment from "../components/npcs/NPCVoiceAssignment";

export default function NPCWorkshopPage({ campaignData, authFetch }) {
  const [genre, setGenre] = useState("1930s noir fantasy");
  const [location, setLocation] = useState("");
  const [npcName, setNpcName] = useState("");
  const [role, setRole] = useState("");
  const [personality, setPersonality] = useState("");
  const [npcGenStreaming, setNpcGenStreaming] = useState(false);
  const [npcGenError, setNpcGenError] = useState("");
  const [npcGenSpeaking, setNpcGenSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [selectedNpc, setSelectedNpc] = useState(null);

  const npcs = campaignData?.npcs?.length ? campaignData.npcs : [];

  const resolveVoiceId = useCallback(async () => {
    const res = await authFetch("/voices/list");
    if (!res.ok) return null;
    const data = await res.json();
    const list = Array.isArray(data) ? data.filter((v) => v?.voice_id) : [];
    const id = selectedVoiceId && list.some((v) => v.voice_id === selectedVoiceId) ? selectedVoiceId : list[0]?.voice_id;
    return id || null;
  }, [authFetch, selectedVoiceId]);

  useEffect(() => {
    let cancelled = false;
    authFetch("/voices/list")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => {
        if (cancelled) return;
        const list = Array.isArray(data) ? data.filter((v) => v?.voice_id) : [];
        setVoices(list);
        if (!selectedVoiceId && list[0]) setSelectedVoiceId(list[0].voice_id);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [authFetch]);

  const runGenerate = useCallback(async () => {
    if (!npcName.trim() || npcGenStreaming) return;
    setNpcGenStreaming(true);
    setPersonality("");
    setNpcGenError("");
    try {
      const res = await authFetch("/npc/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          genre,
          location: location || undefined,
          name: npcName.trim(),
          role: role || undefined,
        }),
      });
      if (!res.ok) throw new Error((await res.text()) || "Generation failed.");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.token) setPersonality((t) => t + evt.token);
            if (evt.error) setNpcGenError(evt.error);
          } catch { /* skip */ }
        }
      }
    } catch (e) {
      setNpcGenError(e?.message || "NPC generation failed.");
    } finally {
      setNpcGenStreaming(false);
    }
  }, [authFetch, genre, location, npcName, role, npcGenStreaming]);

  const playProfile = useCallback(async () => {
    const text = personality || selectedNpc?.personality;
    if (!text || npcGenSpeaking) return;
    setNpcGenSpeaking(true);
    try {
      const voiceId = await resolveVoiceId();
      if (!voiceId) throw new Error("No voice available.");
      const res = await authFetch("/tts/narrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice_id: voiceId }),
      });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      new Audio(URL.createObjectURL(blob)).play();
    } catch (e) {
      setNpcGenError(e?.message || "Play failed.");
    } finally {
      setNpcGenSpeaking(false);
    }
  }, [authFetch, personality, selectedNpc, resolveVoiceId, npcGenSpeaking]);

  const selectNpc = useCallback((npc) => {
    setSelectedNpc(npc);
    setNpcName(npc.name || "");
    setRole(npc.role || "");
    setPersonality(npc.personality || "");
  }, []);

  const voiceSelect = (
    <label className="field-wrap flex-1 min-w-[120px]">
      <span>Assign Voice</span>
      <select
        className="chat-input w-full"
        value={selectedVoiceId}
        onChange={(e) => setSelectedVoiceId(e.target.value)}
      >
        <option value="">—</option>
        {voices.map((v) => (
          <option key={v.voice_id} value={v.voice_id}>
            {v.name?.trim() || v.voice_id}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <section className="min-h-0 flex-1 grid grid-cols-1 xl:grid-cols-12 gap-4 p-3">
      <div className="xl:col-span-5 min-h-0 flex flex-col gap-3">
        <section className="panel-ornate rounded border border-[#734f2c]">
          <div className="panel-head">
            <div className="plaque">Summon NPC</div>
          </div>
          <div className="panel-body">
            <NPCGeneratorForm
            genre={genre}
            onGenreChange={setGenre}
            location={location}
            onLocationChange={setLocation}
            npcName={npcName}
            onNpcNameChange={setNpcName}
            role={role}
            onRoleChange={setRole}
            onGenerate={runGenerate}
            onRegenerate={runGenerate}
            generating={npcGenStreaming}
            error={npcGenError}
            personalityText={personality || selectedNpc?.personality}
            voiceSelect={voiceSelect}
            onPlaySample={playProfile}
            playing={npcGenSpeaking}
          />
          </div>
        </section>
      </div>
      <div className="xl:col-span-7 min-h-0 flex flex-col gap-3">
        <SectionHeader title="Saved NPCs" />
        <div className="panel-ornate flex-1 min-h-0 flex flex-col rounded border border-[#734f2c] p-2">
          <div className="panel-head">
            <div className="plaque">Saved NPCs</div>
          </div>
          <div className="panel-body min-h-0">
            <NPCListSidebar npcs={npcs} selectedNpc={selectedNpc} onSelectNpc={selectNpc} />
          </div>
        </div>
        {(selectedNpc || personality) && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <NPCPortraitCard npc={selectedNpc || (npcName ? { name: npcName } : null)} />
            <div className="md:col-span-2 space-y-2">
              <NPCDetailCard npc={selectedNpc || (personality ? { name: npcName, role, personality } : null)} />
              <NPCVoiceAssignment
                voices={voices}
                selectedVoiceId={selectedVoiceId}
                onVoiceChange={setSelectedVoiceId}
                onPlaySample={playProfile}
                playing={npcGenSpeaking}
              />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
