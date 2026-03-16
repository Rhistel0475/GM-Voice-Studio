import { useEffect, useMemo, useState } from "react";
import { BookOpen, Map as MapIcon, Mic2, Play } from "lucide-react";
import { FantasyButton } from "../shared";

function isBackendCampaignId(value) {
  return /^\d+$/.test(String(value || "").trim());
}

function mergeCampaignOptions(initialCampaign, campaigns) {
  const byId = new Map();
  if (initialCampaign?.id != null && initialCampaign.id !== "" && isBackendCampaignId(initialCampaign.id)) {
    byId.set(String(initialCampaign.id), {
      id: String(initialCampaign.id),
      title: initialCampaign.title || `Campaign #${initialCampaign.id}`,
      summary: initialCampaign.summary || "",
    });
  }
  (Array.isArray(campaigns) ? campaigns : []).forEach((campaign) => {
    if (campaign?.id == null || campaign.id === "" || !isBackendCampaignId(campaign.id)) return;
    byId.set(String(campaign.id), {
      id: String(campaign.id),
      title: campaign.title || `Campaign #${campaign.id}`,
      summary: campaign.summary || "",
    });
  });
  return Array.from(byId.values());
}

function getCampaignNarratorVoice(campaign) {
  return String(
    campaign?.narrator_voice_id
    || campaign?.narrator_voice
    || ""
  ).trim();
}

export default function StartSessionPanel({
  authFetch,
  initialCampaign = null,
  onSessionStarted,
  onOpenLibrary,
}) {
  const [campaignOptions, setCampaignOptions] = useState(() => mergeCampaignOptions(initialCampaign, []));
  const [selectedCampaignId, setSelectedCampaignId] = useState(
    initialCampaign?.id != null && initialCampaign.id !== "" && isBackendCampaignId(initialCampaign.id)
      ? String(initialCampaign.id)
      : ""
  );
  const [selectedCampaign, setSelectedCampaign] = useState(initialCampaign);
  const [voices, setVoices] = useState([]);
  const [selectedSceneId, setSelectedSceneId] = useState("");
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [loadingCampaigns, setLoadingCampaigns] = useState(false);
  const [loadingCampaignDetail, setLoadingCampaignDetail] = useState(false);
  const [loadingVoices, setLoadingVoices] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setCampaignOptions((current) => mergeCampaignOptions(initialCampaign, current));
    if (!selectedCampaignId && initialCampaign?.id != null && initialCampaign.id !== "" && isBackendCampaignId(initialCampaign.id)) {
      setSelectedCampaignId(String(initialCampaign.id));
    }
    if (!selectedCampaign && initialCampaign?.id != null && initialCampaign.id !== "" && isBackendCampaignId(initialCampaign.id)) {
      setSelectedCampaign(initialCampaign);
    }
  }, [initialCampaign, selectedCampaign, selectedCampaignId]);

  useEffect(() => {
    let cancelled = false;
    setLoadingCampaigns(true);
    authFetch("/api/campaigns")
      .then((response) => (response.ok ? response.json() : []))
      .then((payload) => {
        if (cancelled) return;
        const options = mergeCampaignOptions(initialCampaign, payload);
        setCampaignOptions(options);
        if (!selectedCampaignId && options[0]?.id) {
          setSelectedCampaignId(String(options[0].id));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCampaignOptions((current) => mergeCampaignOptions(initialCampaign, current));
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingCampaigns(false);
      });
    return () => {
      cancelled = true;
    };
  }, [authFetch, initialCampaign]);

  useEffect(() => {
    let cancelled = false;
    setLoadingVoices(true);
    authFetch("/voices/list")
      .then((response) => (response.ok ? response.json() : []))
      .then((payload) => {
        if (cancelled) return;
        const list = Array.isArray(payload) ? payload.filter((voice) => voice?.voice_id) : [];
        setVoices(list);
      })
      .catch(() => {
        if (!cancelled) setVoices([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingVoices(false);
      });
    return () => {
      cancelled = true;
    };
  }, [authFetch]);

  useEffect(() => {
    if (!selectedCampaignId) {
      setSelectedCampaign(initialCampaign?.id != null && String(initialCampaign.id) === selectedCampaignId ? initialCampaign : null);
      return;
    }

    if (initialCampaign?.id != null && isBackendCampaignId(initialCampaign.id) && String(initialCampaign.id) === selectedCampaignId) {
      setSelectedCampaign(initialCampaign);
      return;
    }

    let cancelled = false;
    setLoadingCampaignDetail(true);
    setError("");
    authFetch(`/api/campaigns/${selectedCampaignId}`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error((await response.text()) || "Could not load campaign.");
        }
        return response.json();
      })
      .then((payload) => {
        if (!cancelled) setSelectedCampaign(payload);
      })
      .catch((err) => {
        if (!cancelled) {
          setSelectedCampaign(null);
          setError(err?.message || "Could not load campaign.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingCampaignDetail(false);
      });

    return () => {
      cancelled = true;
    };
  }, [authFetch, initialCampaign, selectedCampaignId]);

  const scenes = Array.isArray(selectedCampaign?.scenes) ? selectedCampaign.scenes : [];
  const selectedScene = useMemo(
    () => scenes.find((scene) => String(scene?.id) === selectedSceneId) || scenes[0] || null,
    [scenes, selectedSceneId]
  );

  useEffect(() => {
    if (!scenes.length) {
      setSelectedSceneId("");
      return;
    }
    if (selectedSceneId && scenes.some((scene) => String(scene?.id) === selectedSceneId)) return;
    setSelectedSceneId(String(scenes[0]?.id || ""));
  }, [scenes, selectedSceneId]);

  useEffect(() => {
    const sceneVoiceId = String(selectedScene?.narrator_voice_id || "").trim();
    const campaignVoiceId = getCampaignNarratorVoice(selectedCampaign);
    const preferredVoiceId = sceneVoiceId || campaignVoiceId || voices[0]?.voice_id || "";
    if (!preferredVoiceId) {
      setSelectedVoiceId("");
      return;
    }
    if (selectedVoiceId && voices.some((voice) => voice.voice_id === selectedVoiceId)) return;
    setSelectedVoiceId(preferredVoiceId);
  }, [selectedCampaign, selectedScene, selectedVoiceId, voices]);

  const selectedVoice = useMemo(
    () => voices.find((voice) => voice.voice_id === selectedVoiceId) || null,
    [selectedVoiceId, voices]
  );

  const handleStartSession = async () => {
    if (!selectedCampaignId || !selectedSceneId || !selectedVoiceId || starting) return;

    setStarting(true);
    setError("");
    try {
      const response = await authFetch("/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          campaign_id: Number(selectedCampaignId),
          scene_id: String(selectedSceneId),
          narrator_voice: selectedVoiceId,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || "Could not start session.");
      }
      await onSessionStarted?.(payload);
    } catch (err) {
      setError(err?.message || "Could not start session.");
    } finally {
      setStarting(false);
    }
  };

  const noCampaigns = !loadingCampaigns && campaignOptions.length === 0;
  const hasScenes = scenes.length > 0;
  const canStart = Boolean(selectedCampaignId && selectedSceneId && selectedVoiceId && hasScenes && !starting);

  return (
    <section className="panel-ornate w-full max-w-3xl rounded-xl border border-[#7d5a30] overflow-hidden">
      <div className="panel-head">
        <div className="plaque">Start Session</div>
      </div>
      <div className="panel-body flex flex-col gap-5">
        <div className="rounded-xl border border-[#5c3e23] bg-[linear-gradient(180deg,rgba(31,20,11,0.96),rgba(17,10,5,0.98))] px-4 py-4">
          <h3 className="font-heading text-xl text-[var(--text-1)]">Guided Session Mode</h3>
          <p className="mt-2 text-sm text-[var(--text-2)]">
            Pick a campaign, choose the opening scene, set the narrator voice, and launch straight into LiveBoard.
          </p>
        </div>

        {noCampaigns ? (
          <div className="rounded-lg border border-dashed border-[#5c3e23] bg-[#140c07]/85 px-4 py-8 text-center">
            <p className="font-heading text-[var(--text-1)]">No saved campaigns are ready yet.</p>
            <p className="mt-2 text-sm text-[var(--text-2)]">
              Load or parse a campaign in the Library first, then come back to start a guided session.
            </p>
            {onOpenLibrary ? (
              <FantasyButton variant="secondary" className="mt-4" onClick={onOpenLibrary}>
                Open Library
              </FantasyButton>
            ) : null}
          </div>
        ) : (
          <>
            <div className="grid gap-3 md:grid-cols-3">
              <label className="rounded-lg border border-[#5c3e23] bg-[#140d08]/80 p-3 field-wrap">
                <span className="inline-flex items-center gap-2 font-heading uppercase tracking-wide text-xs text-[#d0ac69]">
                  <BookOpen size={14} />
                  1. Campaign
                </span>
                <select
                  value={selectedCampaignId}
                  onChange={(event) => setSelectedCampaignId(event.target.value)}
                  disabled={loadingCampaigns || starting}
                >
                  {campaignOptions.map((campaign) => (
                    <option key={campaign.id} value={campaign.id}>
                      {campaign.title}
                    </option>
                  ))}
                </select>
              </label>

              <label className="rounded-lg border border-[#5c3e23] bg-[#140d08]/80 p-3 field-wrap">
                <span className="inline-flex items-center gap-2 font-heading uppercase tracking-wide text-xs text-[#d0ac69]">
                  <MapIcon size={14} />
                  2. Scene
                </span>
                <select
                  value={selectedSceneId}
                  onChange={(event) => setSelectedSceneId(event.target.value)}
                  disabled={loadingCampaignDetail || !hasScenes || starting}
                >
                  {hasScenes ? (
                    scenes.map((scene) => (
                      <option key={String(scene.id)} value={String(scene.id)}>
                        {scene.title || `Scene ${scene.id}`}
                      </option>
                    ))
                  ) : (
                    <option value="">No scenes available</option>
                  )}
                </select>
              </label>

              <label className="rounded-lg border border-[#5c3e23] bg-[#140d08]/80 p-3 field-wrap">
                <span className="inline-flex items-center gap-2 font-heading uppercase tracking-wide text-xs text-[#d0ac69]">
                  <Mic2 size={14} />
                  3. Narrator Voice
                </span>
                <select
                  value={selectedVoiceId}
                  onChange={(event) => setSelectedVoiceId(event.target.value)}
                  disabled={loadingVoices || voices.length === 0 || starting}
                >
                  {voices.length > 0 ? (
                    voices.map((voice) => (
                      <option key={voice.voice_id} value={voice.voice_id}>
                        {voice.name || voice.voice_id}
                      </option>
                    ))
                  ) : (
                    <option value="">No voices available</option>
                  )}
                </select>
              </label>
            </div>

            <div className="grid gap-3 md:grid-cols-[1.2fr_0.8fr]">
              <div className="rounded-lg border border-[#5c3e23] bg-[#140d08]/80 p-4">
                <div className="text-xs font-heading uppercase tracking-[0.24em] text-[#9c7a3a]">Opening Scene</div>
                {loadingCampaignDetail ? (
                  <p className="mt-3 text-sm text-[var(--text-2)]">Loading campaign details...</p>
                ) : selectedScene ? (
                  <>
                    <h4 className="mt-2 font-heading text-lg text-[var(--text-1)]">{selectedScene.title}</h4>
                    <p className="mt-2 text-sm text-[var(--text-2)] whitespace-pre-wrap">
                      {selectedScene.read_aloud || selectedScene.notes || selectedScene.summary || "No opening description is set for this scene yet."}
                    </p>
                  </>
                ) : (
                  <p className="mt-3 text-sm text-[var(--text-2)]">Choose a campaign with at least one scene to begin.</p>
                )}
              </div>

              <div className="rounded-lg border border-[#5c3e23] bg-[#140d08]/80 p-4">
                <div className="text-xs font-heading uppercase tracking-[0.24em] text-[#9c7a3a]">Session Summary</div>
                <div className="mt-3 space-y-3 text-sm text-[var(--text-2)]">
                  <div>
                    <div className="text-[10px] uppercase tracking-[0.22em] text-[#8f6a39]">Campaign</div>
                    <div className="mt-1 text-[var(--text-1)]">{selectedCampaign?.title || "None selected"}</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-[0.22em] text-[#8f6a39]">Scene</div>
                    <div className="mt-1 text-[var(--text-1)]">{selectedScene?.title || "None selected"}</div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-[0.22em] text-[#8f6a39]">Narrator</div>
                    <div className="mt-1 text-[var(--text-1)]">{selectedVoice?.name || selectedVoiceId || "None selected"}</div>
                  </div>
                </div>
              </div>
            </div>

            {!hasScenes && !loadingCampaignDetail ? (
              <div className="text-sm text-[#c69055]">
                This campaign does not have any scenes yet. Add scenes in the Library or Prep Room before starting.
              </div>
            ) : null}
            {voices.length === 0 && !loadingVoices ? (
              <div className="text-sm text-[#c69055]">
                No narrator voices are available yet. Clone or load a voice in Voice Studio before starting.
              </div>
            ) : null}
            {error ? <div className="text-sm text-red-400">{error}</div> : null}

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#3d2815] pt-4">
              <div className="text-xs uppercase tracking-[0.24em] text-[#8f6a39] font-heading">
                4. Start session
              </div>
              <FantasyButton
                variant="primary"
                className="inline-flex items-center gap-2"
                onClick={handleStartSession}
                disabled={!canStart}
              >
                <Play size={14} />
                {starting ? "Starting..." : "Start Session"}
              </FantasyButton>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
