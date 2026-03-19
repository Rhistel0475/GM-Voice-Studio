import { useState } from "react";
import PrepPanel from "./PrepPanel";

// Fallback defaults (shown when no campaign data is loaded)
const DEFAULT_SCENES = [
  { title: "No scenes loaded", act: "Upload docs in Library", type: "exploration", atmosphere_type: "forest", read_aloud: "", npcs: [], location: "", notes: "" },
];
const DEFAULT_REVEALS = [
  { name: "Upload adventure docs to see plot hooks", when: "", type: "hook" },
];

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

const PrepHeader = ({ view, onNavigate, campaignData }) => (
  <header className="prep-header">
    <div className="header-glow" />
    <ViewTabs view={view} onNavigate={onNavigate} className="prep-header-bar nav-tab-bar" />
    <div className="relative z-10 text-center">
      <h1 className="font-heading text-[clamp(1.8rem,2.35vw,3rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
        GM Voice Studio - Prep Room
      </h1>
      <p className="font-heading text-[clamp(1rem,1.5vw,1.85rem)] leading-[1.05] text-[#d8b36f]">
        {campaignData?.title ? `Active Campaign: ${campaignData.title}` : "Campaign Planning Console"}
      </p>
    </div>
  </header>
);

const BLANK_SCENE = { title: "", act: "", type: "exploration", atmosphere_type: "forest", location: "", read_aloud: "", notes: "", npcs: [] };

const PrepLeftColumn = ({ campaignData, selectedIdx, onSelectScene, onUpdateCampaign }) => {
  const scenes = campaignData?.scenes?.length ? campaignData.scenes : DEFAULT_SCENES;
  const [showForm, setShowForm] = useState(false);
  const [editIdx, setEditIdx] = useState(null); // null = new, number = editing existing
  const [form, setForm] = useState(BLANK_SCENE);

  const existingActs = [...new Set(scenes.map(s => s.act).filter(Boolean))];
  const typeGlyph = { combat: "x", social: "o", exploration: "*", mystery: "?", travel: "~" };

  const actGroups = scenes.reduce((acc, scene, idx) => {
    const act = scene.act || "Adventure";
    if (!acc[act]) acc[act] = [];
    acc[act].push({ ...scene, idx });
    return acc;
  }, {});

  const openNew = () => {
    setForm(BLANK_SCENE);
    setEditIdx(null);
    setShowForm(true);
  };

  const openEdit = (e, scene, idx) => {
    e.stopPropagation();
    setForm({
      title: scene.title || "",
      act: scene.act || "",
      type: scene.type || "exploration",
      atmosphere_type: scene.atmosphere_type || "forest",
      location: scene.location || "",
      read_aloud: scene.read_aloud || "",
      notes: scene.notes || "",
      npcs: scene.npcs || [],
    });
    setEditIdx(idx);
    setShowForm(true);
  };

  const saveScene = () => {
    if (!form.title.trim() || !onUpdateCampaign) return;
    const base = campaignData?.scenes?.length ? campaignData.scenes : [];
    let updated;
    if (editIdx !== null) {
      updated = base.map((s, i) => i === editIdx ? { ...s, ...form } : s);
    } else {
      updated = [...base, { ...form, npcs: [] }];
    }
    onUpdateCampaign({ ...(campaignData || {}), scenes: updated });
    if (editIdx === null) onSelectScene(updated.length - 1);
    setShowForm(false);
    setEditIdx(null);
  };

  return (
    <div className="h-full min-h-0 flex flex-col">
      <PrepPanel title="Adventure Outline" className="flex-1 min-h-0">
        <div className="overflow-y-auto flex-1">
          {Object.entries(actGroups).map(([act, actScenes]) => (
            <div key={act}>
              <div className="prep-act-row"><span>{act}</span><span>▾</span></div>
              {actScenes.map((scene) => (
                <div
                  key={scene.idx}
                  className={`prep-outline-item group flex items-center gap-1 ${scene.idx === selectedIdx ? "is-active" : ""}`}
                  style={{cursor:"pointer"}}
                  onClick={() => onSelectScene(scene.idx)}
                >
                  <div className="prep-outline-glyph">{typeGlyph[scene.type] || "*"}</div>
                  <div className="prep-outline-copy flex-1 min-w-0">
                    <div className="prep-outline-title">{scene.title}</div>
                    <div className="prep-outline-meta">{scene.type}{scene.location ? ` · ${scene.location}` : ""}</div>
                  </div>
                  {onUpdateCampaign && (
                    <button
                      type="button"
                      className="opacity-0 group-hover:opacity-100 text-[#9b7440] hover:text-[#d4af37] text-xs px-1 flex-shrink-0"
                      onClick={(e) => openEdit(e, scene, scene.idx)}
                      title="Edit scene"
                    >✎</button>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>

        {onUpdateCampaign && !showForm && (
          <button type="button" className="nav-glyph-btn intake-parse-btn mt-2 w-full" onClick={openNew}>
            ＋ New Scene
          </button>
        )}

        {showForm && (
          <div className="mt-2 border border-[#4f341f] rounded p-2 space-y-2 bg-[#120a04]">
            <div className="text-xs text-[#d4af37] font-heading mb-1">{editIdx !== null ? "Edit Scene" : "New Scene"}</div>
            <input
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f]"
              placeholder="Title *"
              value={form.title}
              onChange={e => setForm(f => ({...f, title: e.target.value}))}
            />
            <input
              list="act-options"
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f]"
              placeholder="Act (e.g. Chapter 1)"
              value={form.act}
              onChange={e => setForm(f => ({...f, act: e.target.value}))}
            />
            <datalist id="act-options">{existingActs.map(a => <option key={a} value={a} />)}</datalist>
            <div className="flex gap-2">
              <select
                className="flex-1 bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1"
                value={form.type}
                onChange={e => setForm(f => ({...f, type: e.target.value}))}
              >
                {["combat","social","exploration","mystery","travel"].map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              <input
                className="flex-1 bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f]"
                placeholder="Location"
                value={form.location}
                onChange={e => setForm(f => ({...f, location: e.target.value}))}
              />
            </div>
            <select
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1"
              value={form.atmosphere_type || "forest"}
              onChange={e => setForm(f => ({ ...f, atmosphere_type: e.target.value }))}
            >
              {["forest", "tavern", "town", "dungeon", "combat", "mystery"].map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <textarea
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f] resize-none"
              placeholder="Read-aloud text..."
              rows={3}
              value={form.read_aloud}
              onChange={e => setForm(f => ({...f, read_aloud: e.target.value}))}
            />
            <textarea
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f] resize-none"
              placeholder="GM notes..."
              rows={2}
              value={form.notes}
              onChange={e => setForm(f => ({...f, notes: e.target.value}))}
            />
            <div className="flex gap-2">
              <button type="button" className="flex-1 nav-glyph-btn intake-parse-btn is-active text-xs py-1" onClick={saveScene} disabled={!form.title.trim()}>
                Save Scene
              </button>
              <button type="button" className="flex-1 nav-glyph-btn intake-parse-btn text-xs py-1" onClick={() => { setShowForm(false); setEditIdx(null); }}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </PrepPanel>
    </div>
  );
};

const PrepMiddleColumn = ({ campaignData, selectedIdx, onUpdateCampaign }) => {
  const [tab, setTab] = useState("readaloud");
  const scenes = campaignData?.scenes?.length ? campaignData.scenes : DEFAULT_SCENES;
  const scene = scenes[selectedIdx] || scenes[0];
  const npcs = campaignData?.npcs?.length ? campaignData.npcs : [];
  const allReveals = campaignData?.reveals?.length ? campaignData.reveals : DEFAULT_REVEALS;
  const sceneNpcs = npcs.filter(n => scene.npcs?.includes(n.name));
  const sceneReveals = (scene.reveals || [])
    .map(name => allReveals.find(r => r.name === name) || { name, type: "clue", when: "" });

  const removeFromScene = (field, value) => {
    if (!onUpdateCampaign || !campaignData) return;
    const updatedScenes = scenes.map((s, i) => {
      if (i !== selectedIdx) return s;
      return { ...s, [field]: (s[field] || []).filter(v => v !== value) };
    });
    onUpdateCampaign({ ...campaignData, scenes: updatedScenes });
  };

  return (
    <div className="h-full min-h-0">
      <PrepPanel title={`Scene: ${scene.title}`} className="h-full">
        {(scene.difficulty || scene.rewards || scene.location) && (
          <div className="flex gap-2 flex-wrap text-xs mb-2">
            {scene.location && <span className="text-[#7a5a30]">📍 {scene.location}</span>}
            {scene.atmosphere_type && <span className="text-[#9b7440]">Atmosphere: {scene.atmosphere_type}</span>}
            {scene.difficulty && scene.difficulty !== "none" && (
              <span className={`border rounded px-2 py-0.5 ${scene.difficulty === "deadly" ? "border-red-900 text-red-400" : scene.difficulty === "hard" ? "border-orange-900 text-orange-400" : "border-[#4f341f] text-[#9b7440]"}`}>{scene.difficulty}</span>
            )}
            {scene.rewards && <span className="text-[#9b7440]">Rewards: {scene.rewards}</span>}
          </div>
        )}
        <div className="tab-strip prep-main-tabs">
          <button type="button" className={tab === "readaloud" ? "tab-active" : ""} onClick={() => setTab("readaloud")}>
            Read-aloud
          </button>
          <button type="button" className={tab === "npcs" ? "tab-active" : ""} onClick={() => setTab("npcs")}>
            Important NPCs
          </button>
          <button type="button" className={tab === "secrets" ? "tab-active" : ""} onClick={() => setTab("secrets")}>
            Secrets &amp; Clues
          </button>
          <button type="button" className={tab === "notes" ? "tab-active" : ""} onClick={() => setTab("notes")}>
            GM Notes
          </button>
        </div>

        {tab === "readaloud" && (
          <div className="parchment prep-readaloud">
            {scene.read_aloud || "No read-aloud text extracted for this scene."}
          </div>
        )}

        {tab === "npcs" && (
          <div className="prep-npc-grid mt-2">
            {sceneNpcs.length ? sceneNpcs.map((npc) => (
              <article key={npc.name} className="prep-npc-card" style={{ position:"relative" }}>
                {onUpdateCampaign && (
                  <button
                    type="button"
                    onClick={() => removeFromScene("npcs", npc.name)}
                    title="Remove from scene"
                    style={{ position:"absolute", top:"2px", right:"2px", background:"none",
                      border:"none", color:"#7a3a3a", fontSize:"0.8rem", cursor:"pointer",
                      lineHeight:1, padding:"0 2px" }}
                  >×</button>
                )}
                <div className="prep-npc-face">{npc.name.slice(0, 2).toUpperCase()}</div>
                <p>{npc.name}</p>
                <p className="text-xs text-[#9b7440]">{npc.role}</p>
              </article>
            )) : (
              <div className="intake-empty">No NPCs assigned. Use Library Assets → +</div>
            )}
          </div>
        )}

        {tab === "secrets" && (
          <div className="prep-reveal-list mt-2">
            {sceneReveals.length ? sceneReveals.map((reveal) => (
              <div key={reveal.name} className="prep-reveal-row" style={{ display:"flex", alignItems:"center" }}>
                <span className={`prep-reveal-dot ${reveal.type === "hook" ? "green" : reveal.type === "secret" ? "red" : "amber"}`} />
                <span className="prep-reveal-name" style={{ flex:1 }}>{reveal.name}</span>
                <span className="prep-reveal-status">{reveal.when || ""}</span>
                {onUpdateCampaign && (
                  <button
                    type="button"
                    onClick={() => removeFromScene("reveals", reveal.name)}
                    title="Remove from scene"
                    style={{ background:"none", border:"none", color:"#7a3a3a",
                      fontSize:"0.8rem", cursor:"pointer", lineHeight:1,
                      marginLeft:"0.25rem", padding:"0 2px" }}
                  >×</button>
                )}
              </div>
            )) : (
              <div className="intake-empty">No secrets assigned. Use Library Assets → +</div>
            )}
          </div>
        )}

        {tab === "notes" && (
          <div className="parchment prep-readaloud mt-2">
            {scene.notes || "No GM notes for this scene."}
          </div>
        )}
      </PrepPanel>
    </div>
  );
};

const PrepRightColumn = ({ campaignData, selectedIdx, onUpdateCampaign }) => {
  const allNpcs = campaignData?.npcs || [];
  const allReveals = campaignData?.reveals || [];
  const allItems = campaignData?.items || [];
  const scenes = campaignData?.scenes?.length ? campaignData.scenes : DEFAULT_SCENES;
  const [assetTab, setAssetTab] = useState("npcs");

  const addToScene = (field, value) => {
    if (!onUpdateCampaign || !campaignData) return;
    const updatedScenes = scenes.map((s, i) => {
      if (i !== selectedIdx) return s;
      const arr = s[field] || [];
      if (arr.includes(value)) return s;
      return { ...s, [field]: [...arr, value] };
    });
    onUpdateCampaign({ ...campaignData, scenes: updatedScenes });
  };

  const hasAny = allNpcs.length || allReveals.length || allItems.length;

  const AssetRow = ({ label, field }) => (
    <div style={{ display:"flex", alignItems:"center", gap:"0.4rem", padding:"0.3rem 0.25rem",
      borderBottom:"1px solid #2e1e0a" }}>
      <span style={{ flex:1, fontSize:"0.8rem", color:"#c9a85c", fontFamily:"Cinzel,serif",
        overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{label}</span>
      <button
        type="button"
        onClick={() => addToScene(field, label)}
        title="Add to scene"
        style={{ background:"none", border:"1px solid #5a3e1b", color:"#d4af37",
          borderRadius:"3px", width:"18px", height:"18px", lineHeight:"14px",
          fontSize:"0.9rem", cursor:"pointer", flexShrink:0, textAlign:"center" }}
      >+</button>
    </div>
  );

  return (
    <div className="h-full min-h-0">
      <PrepPanel title="Library Assets" className="h-full">
        {hasAny ? (
          <>
            <div className="tab-strip prep-main-tabs">
              <button type="button" className={assetTab === "npcs" ? "tab-active" : ""} onClick={() => setAssetTab("npcs")}>All NPCs</button>
              <button type="button" className={assetTab === "secrets" ? "tab-active" : ""} onClick={() => setAssetTab("secrets")}>All Secrets</button>
              <button type="button" className={assetTab === "items" ? "tab-active" : ""} onClick={() => setAssetTab("items")}>All Items</button>
            </div>
            <div style={{ overflowY:"auto", flex:1, marginTop:"0.25rem" }}>
              {assetTab === "npcs" && (
                allNpcs.length
                  ? allNpcs.map(n => <AssetRow key={n.name} label={n.name} field="npcs" />)
                  : <div className="intake-empty">No NPCs in campaign.</div>
              )}
              {assetTab === "secrets" && (
                allReveals.length
                  ? allReveals.map(r => <AssetRow key={r.name} label={r.name} field="reveals" />)
                  : <div className="intake-empty">No secrets in campaign.</div>
              )}
              {assetTab === "items" && (
                allItems.length
                  ? allItems.map(it => <AssetRow key={it.name} label={it.name} field="items" />)
                  : <div className="intake-empty">No items in campaign.</div>
              )}
            </div>
          </>
        ) : (
          <div className="intake-empty">Use Library to import a campaign.</div>
        )}
      </PrepPanel>
    </div>
  );
};

// LEGACY — still injected into PrepPage as prepContent (Scene Builder mode).
// PrepHeader suppressed when embedded to avoid duplicate nav inside PrepPage.
const PrepRoom = ({ view, onNavigate, campaignData, onUpdateCampaign, embedded = false }) => {
  const [selectedIdx, setSelectedIdx] = useState(0);
  if (embedded) {
    return (
      <div className="flex flex-col gap-3">
        <PrepLeftColumn campaignData={campaignData} selectedIdx={selectedIdx} onSelectScene={setSelectedIdx} onUpdateCampaign={onUpdateCampaign} />
        <PrepMiddleColumn campaignData={campaignData} selectedIdx={selectedIdx} onUpdateCampaign={onUpdateCampaign} />
        <PrepRightColumn campaignData={campaignData} selectedIdx={selectedIdx} onUpdateCampaign={onUpdateCampaign} />
      </div>
    );
  }
  return (
    <div className="dm-shell dm-fit prep-shell mx-auto">
      <PrepHeader view={view} onNavigate={onNavigate} campaignData={campaignData} />
      <section className="min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3">
        <div className="xl:col-span-3 min-h-0">
          <PrepLeftColumn campaignData={campaignData} selectedIdx={selectedIdx} onSelectScene={setSelectedIdx} onUpdateCampaign={onUpdateCampaign} />
        </div>
        <div className="xl:col-span-5 min-h-0">
          <PrepMiddleColumn campaignData={campaignData} selectedIdx={selectedIdx} onUpdateCampaign={onUpdateCampaign} />
        </div>
        <div className="xl:col-span-4 min-h-0">
          <PrepRightColumn campaignData={campaignData} selectedIdx={selectedIdx} onUpdateCampaign={onUpdateCampaign} />
        </div>
      </section>
    </div>
  );
};

export default PrepRoom;
