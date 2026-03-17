import { useState, useMemo, useCallback, useEffect } from "react";
import { defaultNpcFilterState, defaultNpcDraft } from "../../types/npc";
import { getNPCs } from "../../lib/api/npcs";
import { saveNPC, pushToLiveBoard, suggestNpcVoice } from "../../lib/api/npcs";
import { useCampaignOptional } from "../../context/CampaignContext";
import { useNpcsForActiveCampaign } from "../../store/selectors";
import { suggestVoiceForNpc } from "../../lib/voiceSuggestions";
import { persistNpcVoice } from "../../lib/campaignPersistence";
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

function isBackendNpcId(value) {
  return /^\d+$/.test(String(value || "").trim());
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

function normalizeBackendSuggestion(suggestion, voices) {
  const voiceId = String(suggestion?.voice_id || suggestion?.voiceId || "").trim();
  if (!voiceId) return null;
  const matchedVoice = (voices || []).find((voice) => (voice.voice_id || voice.id) === voiceId) || null;
  return {
    provider: suggestion?.provider || matchedVoice?.provider || "",
    voice_id: voiceId,
    voice_name: suggestion?.voice_name || suggestion?.name || matchedVoice?.name || voiceId,
    confidence: typeof suggestion?.confidence === "number" ? suggestion.confidence : null,
    matched_tags: Array.isArray(suggestion?.matched_tags) ? suggestion.matched_tags : [],
  };
}

function normalizeLocalSuggestion(suggestion) {
  const candidate = suggestion?.candidateVoices?.[0];
  const voiceId = String(candidate?.voice_id || candidate?.id || "").trim();
  if (!voiceId) return null;
  return {
    provider: candidate?.provider || "",
    voice_id: voiceId,
    voice_name: candidate?.name || voiceId,
    confidence: typeof suggestion?.confidence === "number" ? suggestion.confidence : null,
    matched_tags: Array.isArray(candidate?.matchedTags) ? candidate.matchedTags : [],
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
  const npcsFromStore = useNpcsForActiveCampaign();
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
  const [backendSuggestedVoice, setBackendSuggestedVoice] = useState(null);
  const [npcs, setNpcs] = useState([]);

  const playAudioBlob = useCallback(async (blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    audio.onerror = () => URL.revokeObjectURL(url);
    await audio.play();
  }, []);

  const npcList = useMemo(() => {
    const apiList = getNPCs(campaignData, authFetch);
    if (!campaignCtx || !npcsFromStore?.length) return apiList;
    const byId = new Map(apiList.map((n) => [n.id, n]));
    npcsFromStore.forEach((n) => byId.set(n.id, { ...n, voice_id: n.voiceId }));
    return Array.from(byId.values());
  }, [campaignData, authFetch, campaignCtx, npcsFromStore]);
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
    setSelectedVoiceId(String(npc?.voiceId || npc?.voice_id || "").trim());
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
        const resolvedName = saved.name || profile.name;
        const resolvedVoiceId = selectedVoiceId || saved.voiceId || saved.voice_id;
        if (campaignCtx?.assignVoiceToNpc && resolvedName && resolvedVoiceId) {
          campaignCtx.assignVoiceToNpc(resolvedName, resolvedVoiceId);
        }
        if (resolvedName && resolvedVoiceId) {
          persistNpcVoice(authFetch, resolvedName, resolvedVoiceId);
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
        if (localProfile.name && selectedVoiceId) {
          persistNpcVoice(authFetch, localProfile.name, selectedVoiceId);
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

  const handleSpeakNpc = useCallback(async () => {
    const profile = mergeProfile(selectedNpc, draft, npcName, personalityText);
    const npcId = profile?.id;
    if (!npcId) {
      if (typeof window !== "undefined" && window.toast) window.toast("Save or select an NPC first.");
      return;
    }
    const promptText = typeof window !== "undefined"
      ? window.prompt(`What should ${profile.name || "this NPC"} say?`, "")
      : "";
    const text = (promptText || "").trim();
    if (!text) return;

    try {
      const res = await authFetch("/tts/npc-dialogue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ npc_id: npcId, text }),
      });
      if (!res.ok) throw new Error((await res.text()) || "Speak failed.");
      const blob = await res.blob();
      await playAudioBlob(blob);
    } catch (e) {
      if (typeof window !== "undefined") {
        if (window.toast) window.toast(e?.message || "Speak failed.");
        else console.log(e?.message || "Speak failed.");
      }
    }
  }, [selectedNpc, draft, npcName, personalityText, authFetch, playAudioBlob]);

  const previewProfile = useMemo(
    () => mergeProfile(selectedNpc, draft, npcName, personalityText),
    [selectedNpc, draft, npcName, personalityText]
  );

  useEffect(() => {
    let cancelled = false;
    const npcId = String(selectedNpc?.id || "").trim();
    if (!isBackendNpcId(npcId)) {
      setBackendSuggestedVoice(null);
      return () => { cancelled = true; };
    }

    setBackendSuggestedVoice(null);
    suggestNpcVoice(npcId, authFetch)
      .then((suggestion) => {
        if (!cancelled) setBackendSuggestedVoice(suggestion);
      })
      .catch(() => {
        if (!cancelled) setBackendSuggestedVoice(null);
      });

    return () => { cancelled = true; };
  }, [authFetch, selectedNpc?.id]);

  const localVoiceSuggestion = useMemo(
    () => normalizeLocalSuggestion(
      previewProfile?.name || previewProfile?.role ? suggestVoiceForNpc(previewProfile, voices) : null
    ),
    [previewProfile, voices]
  );

  const voiceSuggestion = useMemo(
    () => normalizeBackendSuggestion(backendSuggestedVoice, voices) || localVoiceSuggestion,
    [backendSuggestedVoice, voices, localVoiceSuggestion]
  );

  const handleApplySuggestedVoice = useCallback((voiceId) => {
    const resolvedVoiceId = String(voiceId || "").trim();
    if (!resolvedVoiceId) return;

    setSelectedVoiceId(resolvedVoiceId);

    if (selectedNpc?.id != null) {
      setSelectedNpc((current) => (current ? { ...current, voiceId: resolvedVoiceId, voice_id: resolvedVoiceId } : current));
      setNpcs((prev) => prev.map((npc) => (
        String(npc.id) === String(selectedNpc.id)
          ? { ...npc, voiceId: resolvedVoiceId, voice_id: resolvedVoiceId }
          : npc
      )));
    }

    const targetNpc = selectedNpc || previewProfile;
    if (campaignCtx?.assignVoiceToNpc && (targetNpc?.id || targetNpc?.name)) {
      campaignCtx.assignVoiceToNpc(targetNpc.id || targetNpc.name, resolvedVoiceId);
    }
    if (selectedNpc?.name) {
      persistNpcVoice(authFetch, selectedNpc.name, resolvedVoiceId);
    }

    if (typeof window !== "undefined") {
      const message = selectedNpc?.name
        ? "Suggested voice applied."
        : "Suggested voice selected. Save the NPC to keep it.";
      if (window.toast) window.toast(message);
      else console.log(message);
    }
  }, [selectedNpc, previewProfile, campaignCtx, authFetch]);

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
          onSpeak={handleSpeakNpc}
          onAssignVoice={() => {}}
          onPushToLiveBoard={handlePushToLiveBoard}
          generating={npcGenStreaming}
          saving={saving}
          suggestion={voiceSuggestion}
          onApplySuggestion={handleApplySuggestedVoice}
        />
      </div>
    </section>
  );
}
