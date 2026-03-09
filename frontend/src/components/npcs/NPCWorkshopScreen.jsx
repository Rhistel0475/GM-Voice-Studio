import React, { useState, useMemo, useCallback, useEffect } from "react";
import { defaultNpcFilterState, defaultNpcDraft } from "../../types/npc";
import { getNPCs } from "../../lib/api/npcs";
import { saveNPC, pushToLiveBoard } from "../../lib/api/npcs";
import { useCampaignOptional } from "../../context/CampaignContext";
import NPCRosterSidebar from "./NPCRosterSidebar";
import NPCGeneratorForm from "./NPCGeneratorForm";
import NPCPreviewCard from "./NPCPreviewCard";

function filterNpcs(npcs, filterState) {
  let out = npcs;
  const q = (filterState.query || "").trim().toLowerCase();
  if (q) {
    out = out.filter(
      (n) =>
        (n.name && n.name.toLowerCase().includes(q)) ||
        (n.role && n.role.toLowerCase().includes(q)) ||
        (n.summary && n.summary.toLowerCase().includes(q)) ||
        (n.tags && n.tags.some((t) => t.toLowerCase().includes(q)))
    );
  }
  if (filterState.faction) {
    out = out.filter((n) => n.faction === filterState.faction);
  }
  if (filterState.location) {
    out = out.filter((n) => n.location === filterState.location);
  }
  if (filterState.favoritesOnly) {
    out = out.filter((n) => n.favorite);
  }
  return out;
}

function profileToDraft(profile) {
  if (!profile) return defaultNpcDraft();
  return {
    role: profile.role || "",
    profession: profile.profession || "",
    faction: profile.faction || "",
    location: profile.location || "",
    personalityTraits: Array.isArray(profile.personalityTraits) ? [...profile.personalityTraits] : [],
    goals: Array.isArray(profile.goals) ? [...profile.goals] : [],
    secrets: Array.isArray(profile.secrets) ? [...profile.secrets] : [],
    quirks: Array.isArray(profile.quirks) ? [...profile.quirks] : [],
    notes: profile.notes || "",
    preferredVoice: profile.voiceId || profile.voice_id || "",
  };
}

/**
 * Merged profile for preview: draft fields override selectedNpc.
 */
function mergeProfile(selectedNpc, draft, npcName, personalityText) {
  const base = selectedNpc || {};
  const name = npcName || base.name || "";
  const summary = base.summary || personalityText || "";
  return {
    id: base.id,
    name,
    role: draft?.role ?? base.role ?? "",
    profession: draft?.profession ?? base.profession ?? "",
    faction: draft?.faction ?? base.faction ?? "",
    location: draft?.location ?? base.location ?? "",
    personalityTraits: draft?.personalityTraits ?? base.personalityTraits ?? [],
    goals: draft?.goals ?? base.goals ?? [],
    secrets: draft?.secrets ?? base.secrets ?? [],
    quirks: draft?.quirks ?? base.quirks ?? [],
    summary: summary || "No summary yet.",
    notes: draft?.notes ?? base.notes ?? "",
    voiceId: base.voiceId ?? base.voice_id,
    portraitUrl: base.portraitUrl ?? base.portrait_url,
    disposition: base.disposition,
    tags: base.tags ?? [],
    campaign: base.campaign,
    updatedAt: base.updatedAt,
    personality: personalityText || base.personality,
  };
}

export default function NPCWorkshopScreen({ campaignData, authFetch }) {
  const campaignCtx = useCampaignOptional();
  const [filterState, setFilterState] = useState(defaultNpcFilterState());
  const [selectedNpc, setSelectedNpc] = useState(null);
  const [draft, setDraft] = useState(defaultNpcDraft());
  const [npcName, setNpcName] = useState("");
  const [genre, setGenre] = useState("1930s noir fantasy");
  const [personalityText, setPersonalityText] = useState("");
  const [npcGenStreaming, setNpcGenStreaming] = useState(false);
  const [npcGenError, setNpcGenError] = useState("");
  const [npcGenSpeaking, setNpcGenSpeaking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [npcs, setNpcs] = useState([]);

  const npcList = useMemo(() => getNPCs(campaignData, authFetch), [campaignData, authFetch]);
  useEffect(() => {
    setNpcs(npcList);
  }, [npcList]);

  const filteredNpcs = useMemo(() => filterNpcs(npcs, filterState), [npcs, filterState]);

  const factions = useMemo(() => {
    const set = new Set();
    npcs.forEach((n) => n.faction && set.add(n.faction));
    return Array.from(set).sort();
  }, [npcs]);
  const locations = useMemo(() => {
    const set = new Set();
    npcs.forEach((n) => n.location && set.add(n.location));
    return Array.from(set).sort();
  }, [npcs]);

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

  const selectNpc = useCallback((npc) => {
    setSelectedNpc(npc);
    setDraft(profileToDraft(npc));
    setNpcName(npc?.name || "");
    setPersonalityText(npc?.summary || npc?.personality || "");
  }, []);

  const resolveVoiceId = useCallback(async () => {
    const res = await authFetch("/voices/list");
    if (!res.ok) return null;
    const data = await res.json();
    const list = Array.isArray(data) ? data.filter((v) => v?.voice_id) : [];
    const id = selectedVoiceId && list.some((v) => v.voice_id === selectedVoiceId) ? selectedVoiceId : list[0]?.voice_id;
    return id || null;
  }, [authFetch, selectedVoiceId]);

  const runGenerate = useCallback(async () => {
    if (!npcName.trim() || npcGenStreaming) return;
    setNpcGenStreaming(true);
    setPersonalityText("");
    setNpcGenError("");
    try {
      const res = await authFetch("/npc/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          genre,
          location: draft.location || undefined,
          name: npcName.trim(),
          role: draft.role || undefined,
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
            if (evt.token) setPersonalityText((t) => t + evt.token);
            if (evt.error) setNpcGenError(evt.error);
          } catch { /* skip */ }
        }
      }
    } catch (e) {
      setNpcGenError(e?.message || "NPC generation failed.");
    } finally {
      setNpcGenStreaming(false);
    }
  }, [authFetch, genre, npcName, draft.location, draft.role, npcGenStreaming]);

  const playProfile = useCallback(async () => {
    const text = personalityText || selectedNpc?.summary || selectedNpc?.personality;
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
  }, [authFetch, personalityText, selectedNpc, resolveVoiceId, npcGenSpeaking]);

  const regenerateBackstory = useCallback(() => {
    setNpcGenError("");
    if (typeof window !== "undefined") {
      if (window.toast) window.toast("Regenerate backstory: coming soon.");
      else console.log("Regenerate backstory: coming soon.");
    }
  }, []);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const profile = mergeProfile(selectedNpc, draft, npcName, personalityText);
      profile.name = npcName || profile.name || "Unnamed NPC";
      profile.summary = personalityText || profile.summary || "";
      profile.voiceId = selectedVoiceId || profile.voiceId;
      const saved = await saveNPC(profile, authFetch);
      if (saved) {
        setNpcs((prev) => {
          const idx = prev.findIndex((n) => n.id === saved.id);
          if (idx >= 0) return prev.map((n, i) => (i === idx ? saved : n));
          return [...prev, saved];
        });
        if (campaignCtx?.assignVoiceToNpc && (saved.name || profile.name) && (selectedVoiceId || saved.voiceId || saved.voice_id)) {
          campaignCtx.assignVoiceToNpc(saved.name || profile.name, selectedVoiceId || saved.voiceId || saved.voice_id);
        }
        if (typeof window !== "undefined") {
          if (window.toast) window.toast("NPC saved.");
          else console.log("NPC saved.");
        }
      } else {
        // No backend: add to local list so the roster updates
        const localProfile = {
          ...profile,
          id: profile.id || `local-${Date.now()}`,
          updatedAt: new Date().toISOString().slice(0, 10),
          campaign: campaignData?.title || "Campaign",
        };
        setNpcs((prev) => [...prev, localProfile]);
        setSelectedNpc(localProfile);
        if (campaignCtx?.assignVoiceToNpc && localProfile.name && selectedVoiceId) {
          campaignCtx.assignVoiceToNpc(localProfile.name, selectedVoiceId);
        }
        if (typeof window !== "undefined") {
          if (window.toast) window.toast("Saved locally (no backend).");
          else console.log("Saved locally (no backend).");
        }
      }
    } finally {
      setSaving(false);
    }
  }, [selectedNpc, draft, npcName, personalityText, selectedVoiceId, authFetch, campaignData?.title, campaignCtx]);

  const handlePushToLiveBoard = useCallback(async () => {
    const profile = mergeProfile(selectedNpc, draft, npcName, personalityText);
    const name = npcName || profile.name || selectedNpc?.name;
    if (campaignCtx?.assignNpcToScene && name) {
      campaignCtx.assignNpcToScene(name);
    }
    const id = profile.id || "new";
    const ok = await pushToLiveBoard(id, authFetch);
    if (typeof window !== "undefined") {
      if (window.toast) window.toast(ok ? "Pushed to Live Board." : "Push to Live Board: coming soon.");
      else console.log(ok ? "Pushed to Live Board." : "Push to Live Board: coming soon.");
    }
  }, [selectedNpc, draft, npcName, authFetch, campaignCtx]);

  const previewProfile = useMemo(
    () => mergeProfile(selectedNpc, draft, npcName, personalityText),
    [selectedNpc, draft, npcName, personalityText]
  );

  return (
    <section className="npc-workshop-screen min-h-0 flex-1 grid grid-cols-1 md:grid-cols-12 gap-3 p-2 md:p-3 bg-[var(--wood-2)]/80">
      {/* Left: roster — same panel pattern as Codex */}
      <div className="md:col-span-3 lg:col-span-3 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
        <NPCRosterSidebar
          filteredNpcs={filteredNpcs}
          filterState={filterState}
          onFilterChange={setFilterState}
          selectedNpc={selectedNpc}
          onSelectNpc={selectNpc}
          factions={factions}
          locations={locations}
        />
      </div>

      {/* Center: creation form */}
      <div className="md:col-span-5 lg:col-span-5 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
        <div className="plaque mb-2 shrink-0">Create / Edit</div>
        <div className="panel-body min-h-0 overflow-auto">
          <NPCGeneratorForm
            draft={draft}
            onDraftChange={setDraft}
            genre={genre}
            onGenreChange={setGenre}
            npcName={npcName}
            onNpcNameChange={setNpcName}
            onGenerate={runGenerate}
            onRegenerate={runGenerate}
            onRegenerateBackstory={regenerateBackstory}
            generating={npcGenStreaming}
            error={npcGenError}
            personalityText={personalityText}
            voices={voices}
            selectedVoiceId={selectedVoiceId}
            onVoiceChange={setSelectedVoiceId}
            onPlaySample={playProfile}
            playing={npcGenSpeaking}
          />
        </div>
      </div>

      {/* Right: preview */}
      <div className="md:col-span-4 lg:col-span-4 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
        <NPCPreviewCard
          profile={previewProfile}
          voices={voices}
          selectedVoiceId={selectedVoiceId}
          onVoiceChange={setSelectedVoiceId}
          onPlaySample={playProfile}
          playing={npcGenSpeaking}
          onGenerate={runGenerate}
          onRegeneratePersonality={runGenerate}
          onRegenerateBackstory={regenerateBackstory}
          onSave={handleSave}
          onAssignVoice={() => {}}
          onPushToLiveBoard={handlePushToLiveBoard}
          generating={npcGenStreaming}
          saving={saving}
        />
      </div>
    </section>
  );
}
