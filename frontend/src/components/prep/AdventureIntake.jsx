import { useState, useEffect, useMemo } from "react";
import { useAppState } from "../../context/AppStateContext";
import { useExtractionReviewQueueStore } from "../../store/extractionReview";
import { parseResultToExtractionBatch } from "../../lib/parseResultToExtractionBatch";
import { importParseResultToStore } from "../../lib/campaignImport";
import { getBackendCampaignId, setBackendCampaignId } from "../../lib/campaignPersistence";
import {
  DEFAULT_GAME_SYSTEM_ID,
  listGameSystemPlugins,
  normalizeGameSystemId,
  normalizeGameSystemPlugin,
  resolveGameSystemPlugin,
} from "../../lib/gameSystemPlugins";
import { getScenePlaceholder } from "../../lib/placeholders";
import ExtractionReviewQueue from "../intake/ExtractionReviewQueue";
import PrepPanel from "./PrepPanel";
import { Map, Upload, Zap, Save, CheckCircle, Trash2 } from "lucide-react";

const ViewTabs = ({ view, onNavigate, className = "" }) => (
  <div className={className}>
    <button type="button" className={`nav-glyph-btn ${view === "live" ? "is-active" : ""}`} onClick={() => onNavigate("live")}>
      Live Board
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "codex" ? "is-active" : ""}`} onClick={() => onNavigate("codex")}>
      Campaign Codex
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "npc-workshop" ? "is-active" : ""}`} onClick={() => onNavigate("npc-workshop")}>
      NPC Workshop
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "voice-studio" ? "is-active" : ""}`} onClick={() => onNavigate("voice-studio")}>
      Voice Studio
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "prep" ? "is-active" : ""}`} onClick={() => onNavigate("prep")}>
      Prep Room
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "intake" ? "is-active" : ""}`} onClick={() => onNavigate("intake")}>
      Library
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "settings" ? "is-active" : ""}`} onClick={() => onNavigate("settings")}>
      Settings
    </button>
  </div>
);

const TYPE_BADGE = { hook: "green", secret: "red", clue: "amber", twist: "amber" };

const IntakeHeader = ({ view, onNavigate, campaignData }) => (
  <header className="prep-header intake-header">
    <div className="header-glow" />
    <ViewTabs view={view} onNavigate={onNavigate} className="prep-header-bar nav-tab-bar" />
    <div className="relative z-10 text-center">
      <h1 className="font-heading text-[clamp(1.8rem,2.35vw,3rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
        GM Voice Studio - Library
      </h1>
      <p className="font-heading text-[clamp(1rem,1.5vw,1.85rem)] leading-[1.05] text-[#d8b36f]">
        {campaignData?.title ? `Active Campaign: ${campaignData.title}` : "Upload docs · AI extracts · Feed your session"}
      </p>
    </div>
  </header>
);

const DetailDrawer = ({ item, onClose, onLightbox }) => {
  if (!item) return null;
  const { type, data } = item;
  return (
    <div className="fixed inset-0 z-40 flex justify-end" onClick={onClose} style={{ backdropFilter:"blur(4px)", WebkitBackdropFilter:"blur(4px)" }}>
      <div className="w-full max-w-sm h-full bg-[#1a0f06] border-l-2 border-[#4f341f] shadow-2xl flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between px-3 py-2 border-b border-[#4f341f] bg-[#120a04]">
          <div className="font-heading text-[#d4af37] text-sm truncate">
            {type === "scene" && `Scene: ${data.title}`}
            {type === "npc" && `NPC: ${data.name}`}
            {type === "location" && `Location: ${data.name}`}
            {type === "reveal" && `${data.type || "Reveal"}: ${data.name}`}
            {type === "item" && `Item: ${data.name}`}
          </div>
          <button type="button" onClick={onClose} className="text-[#9b7440] hover:text-[#d4af37] text-2xl ml-2 flex-shrink-0 leading-none">×</button>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {type === "scene" && <>
            <div className="flex gap-2 flex-wrap text-xs">
              {data.act && <span className="border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.act}</span>}
              {data.type && <span className="border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.type}</span>}
              {data.difficulty && data.difficulty !== "none" && (
                <span className={`border rounded px-2 py-0.5 ${data.difficulty === "deadly" ? "border-red-900 text-red-400" : data.difficulty === "hard" ? "border-orange-900 text-orange-400" : "border-[#4f341f] text-[#9b7440]"}`}>{data.difficulty}</span>
              )}
            </div>
            {data.location && <p className="text-xs text-[#7a5a30]">Location: {data.location}</p>}
            {data.image_url && <img src={data.image_url} alt={data.title} className="w-full rounded border border-[#4f341f] cursor-pointer object-cover" style={{maxHeight:"140px"}} onClick={() => onLightbox(data.image_url)} />}
            {data.read_aloud && <><div className="subhead">Read-Aloud</div><div className="parchment text-sm">{data.read_aloud}</div></>}
            {data.npcs?.length > 0 && <><div className="subhead">NPCs in Scene</div><div className="flex flex-wrap gap-1">{data.npcs.map(n => <span key={n} className="border border-[#4f341f] rounded px-2 py-0.5 text-xs text-[#9b7440]">{n}</span>)}</div></>}
            {data.rewards && <><div className="subhead">Rewards</div><p className="text-xs text-[#9b7440]">{data.rewards}</p></>}
            {data.notes && <><div className="subhead">GM Notes</div><p className="text-xs text-[#7a5a30] italic">{data.notes}</p></>}
          </>}

          {type === "npc" && <>
            <div className="flex gap-3 items-start">
              {data.image_url
                ? <img src={data.image_url} alt={data.name} className="w-20 h-20 object-cover rounded border border-[#4f341f] flex-shrink-0 cursor-pointer" onClick={() => onLightbox(data.image_url)} />
                : <div className="w-20 h-20 flex-shrink-0 rounded border border-[#4f341f] bg-[#120a04] flex items-center justify-center text-[#4f341f] text-2xl font-bold">{(data.name || "?").slice(0,2).toUpperCase()}</div>
              }
              <div>
                <div className="flex gap-2 flex-wrap text-xs mb-1">
                  <span className="border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.role}</span>
                  {data.faction && <span className="text-[#7a5a30]">{data.faction}</span>}
                </div>
                <div className="flex gap-3 text-xs text-[#c8a050]">
                  {data.hp && <span>HP {data.hp}</span>}
                  {data.ac ? <span>AC {data.ac}</span> : null}
                  {data.cr && <span>{data.cr}</span>}
                </div>
              </div>
            </div>
            {data.personality && <><div className="subhead">Personality</div><p className="text-xs text-[#9b7440] italic">{data.personality}</p></>}
            {data.motivation && <><div className="subhead">Motivation</div><p className="text-xs text-[#b08040]">{data.motivation}</p></>}
            {data.secrets && <><div className="subhead">Secrets</div><p className="text-xs text-[#7a5a30] italic">{data.secrets}</p></>}
          </>}

          {type === "location" && <>
            {data.scene && <p className="text-xs text-[#7a5a30]">Scene: {data.scene}</p>}
            <div className="subhead">Description</div>
            <p className="text-sm text-[#c8a050]">{data.description || "No description available."}</p>
          </>}

          {type === "reveal" && <>
            <div className="flex gap-2 items-center mb-1">
              <span className={`prep-reveal-dot ${TYPE_BADGE[data.type] || "amber"}`} />
              <span className="text-xs border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.type}</span>
            </div>
            {data.when && <><div className="subhead">When Triggered</div><p className="text-sm text-[#c8a050]">{data.when}</p></>}
          </>}

          {type === "item" && <>
            {data.magical && <span className="text-xs border border-amber-800 rounded px-2 py-0.5 text-amber-400">Magical</span>}
            {data.scene && <p className="text-xs text-[#7a5a30] mt-1">Found in: {data.scene}</p>}
            <div className="subhead">Description</div>
            <p className="text-sm text-[#c8a050]">{data.description || "No description."}</p>
          </>}
        </div>
      </div>
    </div>
  );
};

// Review tab button — shows a live count badge from the review queue store.
const ReviewTabButton = ({ activePanel, setActivePanel }) => {
  const items = useExtractionReviewQueueStore((s) => s.items);
  const pendingCount = items.filter(
    (i) => i.reviewStatus === "pending" || i.reviewStatus === "needs_review"
  ).length;
  return (
    <button
      type="button"
      className={activePanel === "review" ? "tab-active" : ""}
      onClick={() => setActivePanel("review")}
    >
      Review{items.length > 0 ? ` (${items.length}${pendingCount > 0 ? ` · ${pendingCount} pending` : ""})` : ""}
    </button>
  );
};

// LEGACY — still injected into PrepPage as libraryContent (Upload Adventure mode).
// IntakeHeader suppressed when embedded to avoid duplicate nav inside PrepPage.
const AdventureIntake = ({ view, onNavigate, campaignData, onSaveCampaign, authFetch, embedded = false }) => {
  const { clearCampaignData } = useAppState();
  const { enqueueBatch } = useExtractionReviewQueueStore();
  const [files, setFiles] = useState([]);
  const [isParsing, setIsParsing] = useState(false);
  const [isExtractingImages, setIsExtractingImages] = useState(false);
  const [parseError, setParseError] = useState("");
  const [parseResult, setParseResult] = useState(() => {
    try {
      const s = localStorage.getItem("gm_parse_result");
      if (!s) return null;
      const parsed = JSON.parse(s);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
    } catch { return null; }
  });
  const [images, setImages] = useState(() => {
    try {
      const s = localStorage.getItem("gm_parse_images");
      if (!s) return { embedded: [], pages: [] };
      const parsed = JSON.parse(s);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed) &&
          Array.isArray(parsed.embedded) && Array.isArray(parsed.pages)) {
        return parsed;
      }
      return { embedded: [], pages: [] };
    } catch { return { embedded: [], pages: [] }; }
  });
  const [saved, setSaved] = useState(false);
  const [activePanel, setActivePanel] = useState("outline");
  const [lightbox, setLightbox] = useState(null); // URL of enlarged image
  const [detailItem, setDetailItem] = useState(null); // {type, data} for detail drawer
  const [expandedActs, setExpandedActs] = useState(new Set()); // which chapter headers are open
  const [campaignSystems, setCampaignSystems] = useState(() => listGameSystemPlugins());
  const [selectedSystemId, setSelectedSystemId] = useState(DEFAULT_GAME_SYSTEM_ID);
  const selectedSystem = useMemo(
    () => resolveGameSystemPlugin(selectedSystemId, campaignSystems),
    [selectedSystemId, campaignSystems]
  );

  const deleteAssignedImage = (idx) =>
    setParseResult(r => ({ ...r, images: r.images.filter(img => img.idx !== idx) }));
  const deleteEmbeddedImage = (i) =>
    setImages(s => ({ ...s, embedded: s.embedded.filter((_, j) => j !== i) }));
  const deletePageImage = (i) =>
    setImages(s => ({ ...s, pages: s.pages.filter((_, j) => j !== i) }));
  const assignImageTo = (idx, entityName) =>
    setParseResult(r => {
      const targetImg = r.images.find(img => img.idx === idx);
      const url = targetImg?.url || null;
      const updatedImages = r.images.map(img =>
        img.idx === idx ? { ...img, assigned_to: entityName } : img
      );
      const updatedNpcs = (r.npcs || []).map(n => {
        if (entityName && n.name === entityName) return { ...n, image_url: url };
        if (!entityName && n.image_url === url) return { ...n, image_url: null };
        return n;
      });
      const updatedScenes = (r.scenes || []).map(s => {
        if (entityName && s.title === entityName) return { ...s, image_url: url };
        if (!entityName && s.image_url === url) return { ...s, image_url: null };
        return s;
      });
      return { ...r, images: updatedImages, npcs: updatedNpcs, scenes: updatedScenes };
    });

  const handleAssignImage = (imageUrl, selection) =>
    setParseResult(r => {
      if (!r) return r;
      const [type, idxStr] = (selection || "").split(":");
      const i = parseInt(idxStr, 10);
      const updatedNpcs = (r.npcs || []).map((n, ni) =>
        type === "npc" && ni === i ? { ...n, image_url: imageUrl }
        : n.image_url === imageUrl ? { ...n, image_url: null } : n
      );
      const updatedScenes = (r.scenes || []).map((s, si) =>
        type === "scene" && si === i ? { ...s, image_url: imageUrl }
        : s.image_url === imageUrl ? { ...s, image_url: null } : s
      );
      return { ...r, npcs: updatedNpcs, scenes: updatedScenes };
    });

  const getAssignedLabel = (url) => {
    if (!parseResult) return null;
    const npc = (parseResult.npcs || []).find(n => n.image_url === url);
    if (npc) return `NPC: ${npc.name}`;
    const scene = (parseResult.scenes || []).find(s => s.image_url === url);
    if (scene) return `Scene: ${scene.title}`;
    return null;
  };

  const [savedCampaigns, setSavedCampaigns] = useState([]);
  const [loadingCampaigns, setLoadingCampaigns] = useState(false);
  const [loadingCampaignId, setLoadingCampaignId] = useState(null);

  useEffect(() => {
    try {
      if (parseResult) localStorage.setItem("gm_parse_result", JSON.stringify(parseResult));
      else localStorage.removeItem("gm_parse_result");
    } catch {}
  }, [parseResult]);

  useEffect(() => {
    try {
      if (images.embedded.length || images.pages.length)
        localStorage.setItem("gm_parse_images", JSON.stringify(images));
      else localStorage.removeItem("gm_parse_images");
    } catch {}
  }, [images]);

  useEffect(() => {
    setLoadingCampaigns(true);
    authFetch("/api/campaigns")
      .then(r => r.ok ? r.json() : [])
      .then(data => setSavedCampaigns(Array.isArray(data) ? data : []))
      .catch(() => setSavedCampaigns([]))
      .finally(() => setLoadingCampaigns(false));
  }, [authFetch]);

  useEffect(() => {
    let cancelled = false;
    authFetch("/api/campaign-systems")
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (cancelled || !payload) return;
        const nextSystems = Array.isArray(payload.systems)
          ? payload.systems
            .map((system) => normalizeGameSystemPlugin(system))
            .filter(Boolean)
          : [];
        if (nextSystems.length > 0) {
          setCampaignSystems(nextSystems);
        }
        const defaultSystemId = normalizeGameSystemId(payload.default_system_id);
        setSelectedSystemId((current) => normalizeGameSystemId(current || defaultSystemId));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [authFetch]);

  useEffect(() => {
    const parsedSystemId = parseResult?.system_id ?? parseResult?.systemId;
    if (parsedSystemId) {
      setSelectedSystemId(normalizeGameSystemId(parsedSystemId));
    }
  }, [parseResult?.system_id, parseResult?.systemId]);

  const onFileChange = (e) => {
    setFiles(Array.from(e.target.files || []));
    setParseError("");
    setSaved(false);
  };

  const runParse = async (endpoint) => {
    if (!files.length) {
      setParseError("Select at least one .txt, .md, or .pdf file.");
      return;
    }
    setParseError("");
    setIsParsing(true);
    setParseResult(null);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      formData.append("campaign_system", selectedSystemId);
      const res = await authFetch(endpoint, { method: "POST", body: formData });
      const raw = await res.text();
      let payload = null;
      try { payload = raw ? JSON.parse(raw) : null; } catch { payload = null; }
      if (!res.ok) throw new Error((payload?.detail) || raw || `Parse failed (${res.status})`);
      if (!payload) throw new Error("Parse returned no data.");
      setParseResult(payload);
      // Persist backend campaign ID for sync operations
      if (payload.campaign_id) setBackendCampaignId(payload.campaign_id);
      // Enqueue extracted entities into review queue
      try {
        const batch = parseResultToExtractionBatch(
          payload,
          files.length === 1 ? files[0].name : (payload.title || undefined)
        );
        if (batch.entities.length > 0) {
          enqueueBatch(batch);
          setActivePanel("review");
        }
      } catch { /* non-fatal — review queue is optional */ }
      // Refresh saved campaigns list (new campaign was just persisted to DB)
      authFetch("/api/campaigns")
        .then(r => r.ok ? r.json() : [])
        .then(data => setSavedCampaigns(Array.isArray(data) ? data : []))
        .catch(() => {});
      // Auto-expand the first act/chapter in the outline
      const firstAct = payload?.scenes?.[0]?.act || (payload?.acts?.[0]?.title);
      if (firstAct) setExpandedActs(new Set([firstAct]));
      // Auto-switch to images tab if AI parse returned images
      if (payload?.images?.length > 0) {
        setActivePanel("images");
      }
    } catch (err) {
      setParseError(err.message || "Unable to parse documents.");
    } finally {
      setIsParsing(false);
    }
  };

  const runImageExtract = async () => {
    if (!files.length) {
      setParseError("Select at least one PDF file first.");
      return;
    }
    setParseError("");
    setIsExtractingImages(true);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      const res = await authFetch("/adventure/images", { method: "POST", body: formData });
      const raw = await res.text();
      let payload = null;
      try { payload = raw ? JSON.parse(raw) : null; } catch { payload = null; }
      if (!res.ok) throw new Error((payload?.detail) || `Image extraction failed (${res.status})`);
      setImages({ embedded: payload.embedded || [], pages: payload.pages || [] });
      setActivePanel("images");
    } catch (err) {
      setParseError(err.message || "Image extraction failed.");
    } finally {
      setIsExtractingImages(false);
    }
  };

  const saveToSession = () => {
    if (!parseResult) return;
    onSaveCampaign(parseResult);           // legacy AppState + localStorage (unchanged)
    try { importParseResultToStore(parseResult); } catch { /* non-fatal */ }
    setSaved(true);
  };

  const loadSavedCampaign = async (id) => {
    setLoadingCampaignId(id);
    try {
      const res = await authFetch(`/api/campaigns/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      const normalized = {
        ...data,
        party: data.party ?? [],
        reveals: data.reveals ?? [],
        items: data.items ?? [],
        images: data.images ?? [],
      };
      setParseResult(normalized);
      onSaveCampaign(normalized);          // sync to AppState so legacy views update
      try { importParseResultToStore(normalized); } catch { /* non-fatal */ }
      setBackendCampaignId(id);            // persist backend ID for sync operations
      setSaved(true);
      setActivePanel("outline");
      const firstAct = data?.scenes?.[0]?.act;
      if (firstAct) setExpandedActs(new Set([firstAct]));
    } finally {
      setLoadingCampaignId(null);
    }
  };

  const deleteSavedCampaign = async (id) => {
    if (!window.confirm("Delete this campaign? This cannot be undone.")) return;
    await authFetch(`/api/campaigns/${id}`, { method: "DELETE" });
    setSavedCampaigns(prev => prev.filter(c => c.id !== id));
    if (parseResult?.id === id) { setParseResult(null); setSaved(false); }
  };

  const clearAllCampaignData = async () => {
    if (!window.confirm("Clear all campaign data? This will remove your current parse, any loaded campaign, and delete all saved campaigns from the server. You can then upload new documents. This cannot be undone.")) return;
    await clearCampaignData({ deleteBackendCampaigns: true });
    setParseResult(null);
    setSaved(false);
    setFiles([]);
    setImages({ embedded: [], pages: [] });
    setSavedCampaigns([]);
    setParseError("");
    setActivePanel("outline");
    setDetailItem(null);
    setLightbox(null);
    setExpandedActs(new Set());
    setSelectedSystemId(DEFAULT_GAME_SYSTEM_ID);
  };

  const npcs = parseResult?.npcs?.length ? parseResult.npcs : [];
  const party = parseResult?.party?.length ? parseResult.party : [];
  const scenes = parseResult?.scenes?.length ? parseResult.scenes : [];
  const locations = parseResult?.locations?.length ? parseResult.locations : [];
  const reveals = parseResult?.reveals?.length ? parseResult.reveals : [];
  const items = parseResult?.items?.length ? parseResult.items : [];

  // Quick parse result (regex-based) also has acts/npcs as strings
  const acts = parseResult?.acts || [];
  const isAiResult = parseResult && typeof parseResult.npcs?.[0] === "object";

  return (
    <div className={embedded ? "flex flex-col gap-3" : "dm-shell dm-fit prep-shell intake-shell mx-auto"}>
      {!embedded && <IntakeHeader view={view} onNavigate={onNavigate} campaignData={campaignData} />}
      <section className={embedded ? "flex flex-col gap-3" : "min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3"}>

        {/* Left: Upload */}
        <div className="xl:col-span-3 min-h-0">
          <PrepPanel title="Upload Adventure Docs" className="h-full">
            {/* Saved Campaigns */}
            <div style={{ marginBottom: "1rem" }}>
              <h3 style={{ color: "#e7c27a", fontFamily: "Cinzel, serif", fontSize: "0.85rem",
                letterSpacing: "0.05em", marginBottom: "0.5rem", borderBottom: "1px solid #5a3e1b",
                paddingBottom: "0.35rem" }}>
                Saved Campaigns
              </h3>
              {loadingCampaigns ? (
                <p style={{ color: "#9c7a3a", fontSize: "0.78rem" }}>Loading…</p>
              ) : (
                <>
                  <label style={{ display:"block", color:"#9c7a3a", fontSize:"0.72rem",
                    fontFamily:"Cinzel,serif", letterSpacing:"0.04em", marginBottom:"0.3rem" }}>
                    Active Campaign:
                  </label>
                  <div style={{ display:"flex", gap:"0.4rem", alignItems:"center" }}>
                    <select
                      value={parseResult?.id ?? ""}
                      onChange={e => { if (e.target.value) loadSavedCampaign(Number(e.target.value)); }}
                      style={{ flex:1, background:"#1a0f06", color:"#e7c27a",
                        border:"1px solid #5a3e1b", borderRadius:"4px",
                        padding:"0.4rem 0.5rem", fontFamily:"Cinzel,serif", fontSize:"0.8rem",
                        cursor:"pointer" }}
                    >
                      <option value="" style={{ color:"#6b5230" }}>— Select Campaign —</option>
                      {savedCampaigns.map(c => (
                        <option key={c.id} value={c.id}>
                          {c.title || `Campaign #${c.id}`}{c.system_label ? ` · ${c.system_label}` : ""}
                        </option>
                      ))}
                    </select>
                    {parseResult?.id && (
                      <button
                        onClick={() => deleteSavedCampaign(parseResult.id)}
                        title="Delete campaign"
                        style={{ background:"none", border:"1px solid #7a2020", borderRadius:"4px",
                          color:"#c05050", padding:"0.35rem 0.5rem", cursor:"pointer", flexShrink:0,
                          display:"flex", alignItems:"center" }}
                        onMouseEnter={e => { e.currentTarget.style.background="#3a0f0f"; e.currentTarget.style.color="#ff6b6b"; }}
                        onMouseLeave={e => { e.currentTarget.style.background="none"; e.currentTarget.style.color="#c05050"; }}
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>
                  <div style={{ marginTop: "0.6rem" }}>
                    <button
                      type="button"
                      onClick={clearAllCampaignData}
                      title="Clear parse result, loaded campaign, and all saved campaigns so you can upload new documents"
                      style={{ background: "none", border: "1px solid #5a3e1b", borderRadius: "4px",
                        color: "#9c7a3a", padding: "0.35rem 0.6rem", cursor: "pointer", fontSize: "0.75rem",
                        fontFamily: "Cinzel, serif" }}
                      onMouseEnter={e => { e.currentTarget.style.background = "#2a1f10"; e.currentTarget.style.color = "#c8a050"; }}
                      onMouseLeave={e => { e.currentTarget.style.background = "none"; e.currentTarget.style.color = "#9c7a3a"; }}
                    >
                      Clear all campaign data (new upload)
                    </button>
                  </div>
                  {loadingCampaignId && (
                    <p style={{ color:"#9c7a3a", fontSize:"0.72rem", marginTop:"0.3rem" }}>Loading…</p>
                  )}
                </>
              )}
            </div>

            <p className="intake-hint">
              Drop in session notes, module PDFs, or campaign text. AI Parse uses Claude to extract full campaign data.
            </p>

            <div style={{ marginBottom: "0.9rem" }}>
              <h3 style={{ color: "#e7c27a", fontFamily: "Cinzel, serif", fontSize: "0.85rem",
                letterSpacing: "0.05em", marginBottom: "0.5rem", borderBottom: "1px solid #5a3e1b",
                paddingBottom: "0.35rem" }}>
                Campaign System
              </h3>
              <label style={{ display:"block", color:"#9c7a3a", fontSize:"0.72rem",
                fontFamily:"Cinzel,serif", letterSpacing:"0.04em", marginBottom:"0.3rem" }}>
                Rules Preset:
              </label>
              <select
                value={selectedSystemId}
                onChange={(e) => setSelectedSystemId(normalizeGameSystemId(e.target.value))}
                style={{ width:"100%", background:"#1a0f06", color:"#e7c27a",
                  border:"1px solid #5a3e1b", borderRadius:"4px", padding:"0.45rem 0.55rem",
                  fontFamily:"Cinzel,serif", fontSize:"0.82rem", cursor:"pointer" }}
              >
                {campaignSystems.map((system) => (
                  <option key={system.id} value={system.id}>{system.label}</option>
                ))}
              </select>
              {selectedSystem && (
                <div style={{ marginTop: "0.55rem", border: "1px solid #4f341f", borderRadius: "6px",
                  background: "rgba(32, 18, 8, 0.72)", padding: "0.65rem 0.75rem" }}>
                  <div style={{ color: "#e7c27a", fontFamily: "Cinzel, serif", fontSize: "0.84rem", marginBottom: "0.35rem" }}>
                    {selectedSystem.label}
                  </div>
                  <p style={{ color: "#b89a62", fontSize: "0.74rem", lineHeight: 1.5, marginBottom: "0.35rem" }}>
                    {selectedSystem.rules_flavor}
                  </p>
                  <p style={{ color: "#9c7a3a", fontSize: "0.72rem", lineHeight: 1.45, marginBottom: "0.25rem" }}>
                    Terms: {selectedSystem.skill_check_terminology.skill_term}, {selectedSystem.skill_check_terminology.check_term}, {selectedSystem.skill_check_terminology.difficulty_term}
                  </p>
                  <p style={{ color: "#9c7a3a", fontSize: "0.72rem", lineHeight: 1.45, marginBottom: "0.25rem" }}>
                    Encounters: {selectedSystem.encounter_assumptions}
                  </p>
                  <p style={{ color: "#9c7a3a", fontSize: "0.72rem", lineHeight: 1.45 }}>
                    Tone: {selectedSystem.thematic_guidance}
                  </p>
                </div>
              )}
            </div>

            <div
              style={{ border:"2px dashed #4f341f", borderRadius:"6px",
                padding:"0.75rem", marginBottom:"0.5rem", transition:"border-color 0.2s" }}
              onMouseEnter={e => e.currentTarget.style.borderColor="#9b7440"}
              onMouseLeave={e => e.currentTarget.style.borderColor="#4f341f"}
            >
              <label className="intake-file-pick">
                <Upload size={16} className="inline mr-1" />
                <span>Select Files</span>
                <input type="file" multiple accept=".txt,.md,.pdf" onChange={onFileChange} />
              </label>
            </div>

            <div className="flex flex-col gap-2 mt-2">
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn is-active"
                onClick={() => runParse("/adventure/ai-parse")}
                disabled={isParsing || isExtractingImages}
              >
                <Zap size={14} className="inline mr-1" />
                {isParsing ? "Parsing with AI..." : "AI Parse (Claude)"}
              </button>
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn"
                onClick={() => runParse("/adventure/parse")}
                disabled={isParsing || isExtractingImages}
              >
                {isParsing ? "Parsing..." : "Quick Parse (fast)"}
              </button>
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn"
                onClick={runImageExtract}
                disabled={isParsing || isExtractingImages}
              >
                <Map size={14} className="inline mr-1" />
                {isExtractingImages ? "Extracting images..." : "Extract Images"}
              </button>
            </div>

            {parseError && <div className="intake-error mt-2">{parseError}</div>}

            <div className="subhead mt-3">Queued Files</div>
            <div className="intake-file-list">
              {files.length ? files.map((f) => (
                <div key={`${f.name}-${f.size}`} className="intake-file-item">
                  <span>{f.name}</span>
                  <small>{Math.max(1, Math.round(f.size / 1024))} KB</small>
                </div>
              )) : (
                <div className="intake-empty">No files selected.</div>
              )}
            </div>

            {parseResult && (
              <button
                type="button"
                className={`nav-glyph-btn intake-parse-btn mt-4 ${saved ? "is-active" : ""}`}
                onClick={saveToSession}
              >
                {saved
                  ? <><CheckCircle size={14} className="inline mr-1" />Saved to Campaign</>
                  : <><Save size={14} className="inline mr-1" />Save to Campaign</>
                }
              </button>
            )}

            {parseResult?.files?.length && (
              <div className="mt-3 text-xs text-[#7a5a30]">
                {parseResult.files.map(f => (
                  <div key={f.name}>{f.name} — {(f.characters / 1000).toFixed(1)}k chars{f.page_count ? ` · ${f.page_count}p` : ""}</div>
                ))}
              </div>
            )}
          </PrepPanel>
        </div>

        {/* Middle: Outline / Summary / Images */}
        <div className="xl:col-span-4 min-h-0">
          <PrepPanel title="Parsed Adventure Outline" className="h-full">
            <div className="tab-strip mb-2">
              <button type="button" className={activePanel === "outline" ? "tab-active" : ""} onClick={() => setActivePanel("outline")}>
                Outline
              </button>
              <button type="button" className={activePanel === "locations" ? "tab-active" : ""} onClick={() => setActivePanel("locations")}>
                Locations
              </button>
              <button type="button" className={activePanel === "party" ? "tab-active" : ""} onClick={() => setActivePanel("party")}>
                Party
              </button>
              <button type="button" className={activePanel === "items" ? "tab-active" : ""} onClick={() => setActivePanel("items")}>
                Items {items.length > 0 ? `(${items.length})` : ""}
              </button>
              <button type="button" className={activePanel === "images" ? "tab-active" : ""} onClick={() => setActivePanel("images")}>
                Images {(images.embedded.length + images.pages.length) > 0 ? `(${images.embedded.length + images.pages.length})` : ""}
              </button>
              <ReviewTabButton activePanel={activePanel} setActivePanel={setActivePanel} />
            </div>

            {parseResult?.summary && activePanel !== "images" && (
              <div className="parchment intake-summary mb-3">
                {parseResult.summary}
              </div>
            )}

            {!parseResult && activePanel !== "images" && (
              <div className="intake-empty">Upload docs and click a Parse button to see your adventure outline here.</div>
            )}

            {/* Images gallery panel */}
            {activePanel === "images" && (
              <div className="overflow-y-auto space-y-3">
                {/* AI-assigned images from ai-parse (rich metadata) */}
                {parseResult?.images?.length > 0 ? (
                  <>
                    <div className="subhead">Extracted Images ({parseResult.images.length}) — auto-assigned</div>
                    <div className="grid grid-cols-2 gap-2">
                      {parseResult.images.map((img) => {
                        const assignOptions = [
                          ...(parseResult.npcs || []).map(n => ({ label: `NPC: ${n.name}`, value: n.name })),
                          ...(parseResult.scenes || []).map(s => ({ label: `Scene: ${s.title}`, value: s.title })),
                        ];
                        return (
                          <div key={img.idx} style={{ position: "relative" }}>
                            <img
                              src={img.url}
                              alt={img.label || `Image ${img.idx}`}
                              style={{ maxHeight: "120px", width: "100%", borderRadius: "4px", border: "1px solid #4f341f", objectFit: "cover", cursor: "pointer", display: "block" }}
                              onClick={() => setLightbox(img.url)}
                            />
                            <button
                              type="button"
                              onClick={() => deleteAssignedImage(img.idx)}
                              title="Remove image"
                              style={{ position:"absolute", top:"5px", right:"5px", width:"20px", height:"20px", borderRadius:"50%", background:"#1a0f06", border:"1px solid #c8a050", color:"#c8a050", fontSize:"12px", lineHeight:1, display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", zIndex:2, transition:"border-color 0.15s, box-shadow 0.15s, color 0.15s" }}
                              onMouseEnter={e => { e.currentTarget.style.borderColor="#c05050"; e.currentTarget.style.color="#ff6b6b"; e.currentTarget.style.boxShadow="0 0 6px #c0505088"; }}
                              onMouseLeave={e => { e.currentTarget.style.borderColor="#c8a050"; e.currentTarget.style.color="#c8a050"; e.currentTarget.style.boxShadow="none"; }}
                            >×</button>
                            <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"rgba(0,0,0,0.72)", borderBottomLeftRadius:"4px", borderBottomRightRadius:"4px", padding:"3px 5px" }}>
                              <div style={{ fontSize:"10px", marginBottom:"2px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                                {img.assigned_to
                                  ? <span style={{ color:"#d4af37", fontWeight:600 }}>Linked to: {img.assigned_to}</span>
                                  : <span style={{ color:"#7a6040" }}>{img.type || "illustration"}{img.label ? ` · ${img.label}` : ""}</span>
                                }
                              </div>
                              <select
                                value={img.assigned_to || ""}
                                onChange={e => assignOptions.length && assignImageTo(img.idx, e.target.value)}
                                onClick={e => e.stopPropagation()}
                                disabled={assignOptions.length === 0}
                                style={{ width:"100%", background:"#1a0f06", border:"1px solid #c8a050", color: assignOptions.length ? "#c8a050" : "#5a3e1b", fontSize:"10px", borderRadius:"3px", padding:"1px 2px", cursor: assignOptions.length ? "pointer" : "default", opacity: assignOptions.length ? 1 : 0.5 }}
                              >
                                {assignOptions.length === 0
                                  ? <option value="">Run AI Parse to assign</option>
                                  : <><option value="">Assign to...</option>{assignOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</>
                                }
                              </select>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : images.embedded.length > 0 ? (
                  <>
                    <div className="subhead">Maps &amp; Artwork ({images.embedded.length})</div>
                    <div className="grid grid-cols-2 gap-2">
                      {images.embedded.map((url, i) => {
                        const assignedLabel = getAssignedLabel(url);
                        const npcIdx = (parseResult?.npcs || []).findIndex(n => n.image_url === url);
                        const sceneIdx = (parseResult?.scenes || []).findIndex(s => s.image_url === url);
                        const currentVal = npcIdx >= 0 ? `npc:${npcIdx}` : sceneIdx >= 0 ? `scene:${sceneIdx}` : "";
                        const embedAssignOptions = [
                          ...(parseResult?.npcs || []).map((n, ni) => ({ label: `NPC: ${n.name}`, value: `npc:${ni}` })),
                          ...(parseResult?.scenes || []).map((s, si) => ({ label: `Scene: ${s.title}`, value: `scene:${si}` })),
                        ];
                        return (
                          <div key={i} style={{ position: "relative" }}>
                            <img src={url} alt={`Image ${i + 1}`}
                              style={{ maxHeight: "130px", width: "100%", borderRadius: "4px", border: "1px solid #4f341f", objectFit: "cover", cursor: "pointer", display: "block" }}
                              onClick={() => setLightbox(url)} />
                            <button
                              type="button"
                              onClick={() => deleteEmbeddedImage(i)}
                              title="Remove image"
                              style={{ position:"absolute", top:"5px", right:"5px", width:"20px", height:"20px", borderRadius:"50%", background:"#1a0f06", border:"1px solid #c8a050", color:"#c8a050", fontSize:"12px", lineHeight:1, display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", zIndex:2, transition:"border-color 0.15s, box-shadow 0.15s, color 0.15s" }}
                              onMouseEnter={e => { e.currentTarget.style.borderColor="#c05050"; e.currentTarget.style.color="#ff6b6b"; e.currentTarget.style.boxShadow="0 0 6px #c0505088"; }}
                              onMouseLeave={e => { e.currentTarget.style.borderColor="#c8a050"; e.currentTarget.style.color="#c8a050"; e.currentTarget.style.boxShadow="none"; }}
                            >×</button>
                            <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"rgba(0,0,0,0.72)", borderBottomLeftRadius:"4px", borderBottomRightRadius:"4px", padding:"3px 5px" }}>
                              {assignedLabel && (
                                <div style={{ fontSize:"10px", marginBottom:"2px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                                  <span style={{ color:"#d4af37", fontWeight:600 }}>Linked to: {assignedLabel.replace(/^(NPC|Scene): /, "")}</span>
                                </div>
                              )}
                              <select
                                value={currentVal}
                                onChange={e => handleAssignImage(url, e.target.value)}
                                onClick={e => e.stopPropagation()}
                                disabled={embedAssignOptions.length === 0}
                                style={{ width:"100%", background:"#1a0f06", border:"1px solid #c8a050", color: embedAssignOptions.length ? "#c8a050" : "#5a3e1b", fontSize:"10px", borderRadius:"3px", padding:"1px 2px", cursor: embedAssignOptions.length ? "pointer" : "default", opacity: embedAssignOptions.length ? 1 : 0.5 }}
                              >
                                {embedAssignOptions.length === 0
                                  ? <option value="">Run AI Parse to assign</option>
                                  : <><option value="">Assign to...</option>{embedAssignOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</>
                                }
                              </select>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div className="intake-empty">Run "AI Parse" to automatically extract and assign images, or use "Extract Images" for raw extraction.</div>
                )}
                {images.pages.length > 0 && (
                  <>
                    <div className="subhead mt-2">Page Thumbnails ({images.pages.length})</div>
                    <div className="grid grid-cols-2 gap-2">
                      {images.pages.map((url, i) => {
                        const assignedLabel = getAssignedLabel(url);
                        const npcIdx = (parseResult?.npcs || []).findIndex(n => n.image_url === url);
                        const sceneIdx = (parseResult?.scenes || []).findIndex(s => s.image_url === url);
                        const currentVal = npcIdx >= 0 ? `npc:${npcIdx}` : sceneIdx >= 0 ? `scene:${sceneIdx}` : "";
                        const pageAssignOptions = [
                          ...(parseResult?.npcs || []).map((n, ni) => ({ label: `NPC: ${n.name}`, value: `npc:${ni}` })),
                          ...(parseResult?.scenes || []).map((s, si) => ({ label: `Scene: ${s.title}`, value: `scene:${si}` })),
                        ];
                        return (
                          <div key={i} style={{ position: "relative" }}>
                            <img src={url} alt={`Page ${i + 1}`}
                              style={{ maxHeight: "130px", width: "100%", borderRadius: "4px", border: "1px solid #4f341f", objectFit: "cover", cursor: "pointer", display: "block" }}
                              onClick={() => setLightbox(url)} />
                            <button
                              type="button"
                              onClick={() => deletePageImage(i)}
                              title="Remove image"
                              style={{ position:"absolute", top:"5px", right:"5px", width:"20px", height:"20px", borderRadius:"50%", background:"#1a0f06", border:"1px solid #c8a050", color:"#c8a050", fontSize:"12px", lineHeight:1, display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", zIndex:2, transition:"border-color 0.15s, box-shadow 0.15s, color 0.15s" }}
                              onMouseEnter={e => { e.currentTarget.style.borderColor="#c05050"; e.currentTarget.style.color="#ff6b6b"; e.currentTarget.style.boxShadow="0 0 6px #c0505088"; }}
                              onMouseLeave={e => { e.currentTarget.style.borderColor="#c8a050"; e.currentTarget.style.color="#c8a050"; e.currentTarget.style.boxShadow="none"; }}
                            >×</button>
                            <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"rgba(0,0,0,0.72)", borderBottomLeftRadius:"4px", borderBottomRightRadius:"4px", padding:"3px 5px" }}>
                              {assignedLabel && (
                                <div style={{ fontSize:"10px", marginBottom:"2px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                                  <span style={{ color:"#d4af37", fontWeight:600 }}>Linked to: {assignedLabel.replace(/^(NPC|Scene): /, "")}</span>
                                </div>
                              )}
                              <select
                                value={currentVal}
                                onChange={e => handleAssignImage(url, e.target.value)}
                                onClick={e => e.stopPropagation()}
                                disabled={pageAssignOptions.length === 0}
                                style={{ width:"100%", background:"#1a0f06", border:"1px solid #c8a050", color: pageAssignOptions.length ? "#c8a050" : "#5a3e1b", fontSize:"10px", borderRadius:"3px", padding:"1px 2px", cursor: pageAssignOptions.length ? "pointer" : "default", opacity: pageAssignOptions.length ? 1 : 0.5 }}
                              >
                                {pageAssignOptions.length === 0
                                  ? <option value="">Run AI Parse to assign</option>
                                  : <><option value="">Assign to...</option>{pageAssignOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</>
                                }
                              </select>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            )}

            {activePanel === "outline" && (
              <>
                {isAiResult && scenes.length > 0 && (() => {
                  const actGroups = scenes.reduce((acc, scene) => {
                    const act = scene.act || "Scenes";
                    if (!acc[act]) acc[act] = [];
                    acc[act].push(scene);
                    return acc;
                  }, {});
                  return (
                    <div className="space-y-1">
                      {Object.entries(actGroups).map(([act, actScenes]) => {
                        const isOpen = expandedActs.has(act);
                        const toggle = () => setExpandedActs(prev => {
                          const next = new Set(prev);
                          isOpen ? next.delete(act) : next.add(act);
                          return next;
                        });
                        return (
                          <div key={act} className="border border-[#4f341f] rounded overflow-hidden">
                            {/* Chapter header — clickable to expand/collapse */}
                            <button
                              type="button"
                              className="w-full flex items-center justify-between px-3 py-2 bg-[#1a0f06] hover:bg-[#2a1a0a] text-left"
                              onClick={toggle}
                            >
                              <span className="font-heading text-[#d4af37] text-sm">{act}</span>
                              <span className="text-[#9b7440] text-xs flex items-center gap-2">
                                <span>{actScenes.length} scene{actScenes.length !== 1 ? "s" : ""}</span>
                                <span className="text-base leading-none">{isOpen ? "▲" : "▼"}</span>
                              </span>
                            </button>

                            {/* Scenes — only visible when chapter is open */}
                            {isOpen && (
                              <ul className="divide-y divide-[#2a1a0a]">
                            {actScenes.map((scene, i) => {
                              const thumbSrc = scene.image_url || getScenePlaceholder(scene.type);
                              return (
                                  <li key={i}
                                    className="flex gap-2 items-start cursor-pointer hover:bg-[#2a1a0a] px-3 py-2"
                                    title="Click to view scene details"
                                    onClick={() => setDetailItem({type:"scene", data:scene})}>
                                      <img src={thumbSrc} alt={scene.title}
                                        className="w-12 h-10 object-cover rounded border border-[#4f341f] flex-shrink-0 mt-0.5"
                                        onClick={e => { e.stopPropagation(); if (scene.image_url) setLightbox(scene.image_url); }} />
                                    <div className="flex-1 min-w-0">
                                      <div className="font-semibold text-[#c8a050] text-sm">{scene.title}</div>
                                      <div className="flex gap-1 flex-wrap mt-0.5">
                                        {scene.type && <span className="text-[#9b7440] text-xs">({scene.type})</span>}
                                        {scene.difficulty && scene.difficulty !== "none" && (
                                          <span className={`text-xs ${scene.difficulty === "deadly" ? "text-red-400" : scene.difficulty === "hard" ? "text-orange-400" : "text-[#7a5a30]"}`}>{scene.difficulty}</span>
                                        )}
                                        {scene.location && <span className="text-xs text-[#7a5a30]">· {scene.location}</span>}
                                      </div>
                                      {scene.read_aloud && (
                                        <div className="text-xs text-[#7a5a30] italic mt-0.5 line-clamp-2">{scene.read_aloud.slice(0, 100)}…</div>
                                      )}
                                    </div>
                                    <span className="text-[#4f341f] text-xs flex-shrink-0 mt-1">›</span>
                                  </li>
                                );})}
                              </ul>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  );
                })()}
                {!isAiResult && acts.length > 0 && (() => {
                  return (
                    <div className="space-y-1">
                      {acts.map((act) => {
                        const isOpen = expandedActs.has(act.title);
                        const toggle = () => setExpandedActs(prev => {
                          const next = new Set(prev);
                          isOpen ? next.delete(act.title) : next.add(act.title);
                          return next;
                        });
                        return (
                          <div key={act.title} className="border border-[#4f341f] rounded overflow-hidden">
                            <button type="button"
                              className="w-full flex items-center justify-between px-3 py-2 bg-[#1a0f06] hover:bg-[#2a1a0a] text-left"
                              onClick={toggle}>
                              <span className="font-heading text-[#d4af37] text-sm">{act.title}</span>
                              <span className="text-[#9b7440] text-xs">{isOpen ? "▲" : "▼"}</span>
                            </button>
                            {isOpen && (
                              <ul className="divide-y divide-[#2a1a0a]">
                                {(act.scenes || []).map((scene, i) => (
                                  <li key={i} className="px-3 py-1.5 text-sm text-[#c8a050]">{scene}</li>
                                ))}
                              </ul>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  );
                })()}
              </>
            )}

            {activePanel === "items" && (
              <div className="space-y-2">
                {items.length ? items.map((item, i) => (
                  <div key={item.name || i} className="intake-act-card cursor-pointer hover:border-[#d4af37]"
                    onClick={() => setDetailItem({type:"item", data:item})}>
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="truncate">{item.name}</h3>
                      {item.magical && <span className="text-xs border border-amber-800 rounded px-1 text-amber-400 flex-shrink-0">magical</span>}
                    </div>
                    {item.scene && <p className="text-xs text-[#7a5a30]">{item.scene}</p>}
                    {item.description && <p className="text-xs text-[#9b7440] mt-0.5">{item.description}</p>}
                  </div>
                )) : (
                  <div className="intake-empty">No items/treasure extracted yet. Run AI Parse on an adventure with loot.</div>
                )}
              </div>
            )}

            {activePanel === "locations" && (
              <div className="space-y-2">
                {locations.length ? locations.filter(Boolean).map((loc) => (
                  <div key={loc.name || String(loc)} className="intake-act-card cursor-pointer hover:border-[#d4af37]"
                    onClick={() => isAiResult && setDetailItem({type:"location", data:loc})}>
                    <h3>{loc.name || loc}</h3>
                    {loc.description && <p className="text-xs text-[#9b7440]">{loc.description}</p>}
                  </div>
                )) : (
                  <div className="intake-empty">No locations extracted yet.</div>
                )}
                {!isAiResult && parseResult?.locations?.length > 0 && (
                  <ul className="intake-pill-list">
                    {parseResult.locations.map(l => <li key={l}>{l}</li>)}
                  </ul>
                )}
              </div>
            )}

            {activePanel === "party" && (
              <div className="space-y-2">
                {party.length ? party.map((pc) => (
                  <div key={pc.name} className="intake-act-card">
                    <h3>{pc.name}</h3>
                    <p className="text-xs text-[#9b7440]">
                      {[pc.race, pc.class_].filter(Boolean).join(" ")}
                      {pc.level ? ` · Lv ${pc.level}` : ""}
                      {pc.hp ? ` · HP ${pc.hp}` : ""}
                      {pc.ac ? ` · AC ${pc.ac}` : ""}
                    </p>
                  </div>
                )) : (
                  <div className="intake-empty">No player characters found in the uploaded docs.</div>
                )}
              </div>
            )}

            {activePanel === "review" && (
              <ExtractionReviewQueue
                documentName={parseResult?.title || (files.length === 1 ? files[0].name : undefined)}
              />
            )}
          </PrepPanel>
        </div>

        {/* Right: NPCs + Reveals */}
        <div className="xl:col-span-5 min-h-0">
          <PrepPanel title="Extracted Campaign Data" className="h-full">

            <div className="subhead">NPC Roster</div>
            {isAiResult && npcs.length ? (
              <div className="space-y-2 mb-4">
                {npcs.slice(0, 8).map((npc) => (
                  <div key={npc.name} className="intake-act-card flex gap-2 cursor-pointer hover:border-[#d4af37]"
                    onClick={() => setDetailItem({type:"npc", data:npc})}>
                    {npc.image_url ? (
                      <img
                        src={npc.image_url}
                        alt={npc.name}
                        className="w-14 h-14 object-cover rounded border border-[#4f341f] flex-shrink-0"
                        onClick={e => { e.stopPropagation(); setLightbox(npc.image_url); }}
                      />
                    ) : (
                      <div className="w-14 h-14 flex-shrink-0 rounded border border-[#4f341f] bg-[#1a0f06] flex items-center justify-center text-[#4f341f] text-lg font-bold">
                        {(npc.name || "?").slice(0, 2).toUpperCase()}
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1">
                        <h3 className="truncate">{npc.name}</h3>
                        <span className="text-xs text-[#9b7440] border border-[#4f341f] rounded px-1 flex-shrink-0">{typeof npc.role === "string" ? npc.role : ""}</span>
                      </div>
                      {npc.faction && <p className="text-xs text-[#7a5a30]">{typeof npc.faction === "string" ? npc.faction : ""}</p>}
                      {npc.personality && typeof npc.personality === "string" && <p className="text-xs text-[#9b7440] italic mt-0.5">{npc.personality.slice(0, 80)}{npc.personality.length > 80 ? "…" : ""}</p>}
                      {npc.motivation && typeof npc.motivation === "string" && <p className="text-xs text-[#7a5a30]">Wants: {npc.motivation.slice(0, 60)}{npc.motivation.length > 60 ? "…" : ""}</p>}
                    </div>
                  </div>
                ))}
              </div>
            ) : !isAiResult && parseResult?.npcs?.length ? (
              <ul className="intake-pill-list mb-4">
                {parseResult.npcs.map(n => <li key={n}>{n}</li>)}
              </ul>
            ) : (
              <div className="intake-empty mb-3">No NPCs extracted yet.</div>
            )}

            <div className="subhead">Reveals &amp; Plot Hooks</div>
            {reveals.length ? (
              <div className="prep-reveal-list">
                {reveals.map((reveal) => (
                  <div key={reveal.name || reveal} className="prep-reveal-row cursor-pointer hover:bg-[#2a1a0a] rounded px-1 -mx-1"
                    onClick={() => isAiResult && setDetailItem({type:"reveal", data:reveal})}>
                    <span className={`prep-reveal-dot ${TYPE_BADGE[reveal.type] || "amber"}`} />
                    <span className="prep-reveal-name">{reveal.name || reveal}</span>
                    <span className="prep-reveal-status text-xs">{reveal.when || reveal.type || ""}</span>
                  </div>
                ))}
              </div>
            ) : !isAiResult && parseResult?.reveals?.length ? (
              <ul className="intake-pill-list">
                {parseResult.reveals.map(r => <li key={r}>{r}</li>)}
              </ul>
            ) : (
              <div className="intake-empty">No reveals extracted yet.</div>
            )}
          </PrepPanel>
        </div>

      </section>

      {/* Detail drawer */}
      <DetailDrawer item={detailItem} onClose={() => setDetailItem(null)} onLightbox={setLightbox} />

      {/* Lightbox overlay */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 cursor-pointer"
          onClick={() => setLightbox(null)}
        >
          <img
            src={lightbox}
            alt="Full size"
            className="max-w-[90vw] max-h-[90vh] rounded border-2 border-[#d4af37] shadow-2xl"
            onClick={e => e.stopPropagation()}
          />
          <button
            type="button"
            className="absolute top-4 right-6 text-[#d4af37] text-3xl font-bold"
            onClick={() => setLightbox(null)}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
};

export default AdventureIntake;
