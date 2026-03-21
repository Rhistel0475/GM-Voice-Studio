import { useState } from "react";
import { useAppState } from "../context/AppStateContext";
import SettingsPage from "./SettingsPage";
import { ParchmentCard } from "../components/shared";

// ── Helpers ───────────────────────────────────────────────────────────────────

function roleBadgeClass(role) {
  const r = (role || "").toLowerCase();
  if (r === "villain" || r === "enemy") return "camp-badge camp-badge--villain";
  if (r === "ally") return "camp-badge camp-badge--ally";
  if (r === "quest-giver" || r === "questgiver") return "camp-badge camp-badge--quest";
  return "camp-badge camp-badge--neutral";
}

function typeBadgeClass(type) {
  const t = (type || "").toLowerCase();
  if (t === "combat") return "camp-badge camp-badge--combat";
  if (t === "social") return "camp-badge camp-badge--social";
  if (t === "exploration") return "camp-badge camp-badge--explore";
  if (t === "mystery" || t === "investigation") return "camp-badge camp-badge--mystery";
  return "camp-badge camp-badge--neutral";
}

function loreBadgeClass(type) {
  const t = (type || "").toLowerCase();
  if (t === "faction") return "camp-badge camp-badge--faction";
  if (t === "quest") return "camp-badge camp-badge--quest";
  if (t === "item") return "camp-badge camp-badge--item";
  return "camp-badge camp-badge--lore";
}

// ── NPC Card ──────────────────────────────────────────────────────────────────

function NpcCard({ npc }) {
  const [open, setOpen] = useState(false);
  const name = npc.name || "(unnamed)";
  const role = typeof npc.role === "string" ? npc.role : "";
  const description = npc.description || npc.personality || "";
  const motivation = npc.motivation || "";
  const secrets = npc.secrets || "";
  const voice_notes = npc.voice_notes || "";
  const read_aloud = npc.read_aloud || "";
  const stats = [npc.hp && `HP ${npc.hp}`, npc.ac && `AC ${npc.ac}`, npc.cr].filter(Boolean).join(" · ");

  return (
    <div className="camp-entity-card">
      <div className="camp-entity-header" onClick={() => setOpen(v => !v)}>
        <div className="camp-entity-title-row">
          {npc.image_url && (
            <img src={npc.image_url} alt={name} className="camp-npc-avatar" />
          )}
          <div className="camp-entity-name-block">
            <span className="camp-entity-name">{name}</span>
            {role && <span className={roleBadgeClass(role)}>{role}</span>}
            {npc.faction && <span className="camp-entity-faction">{npc.faction}</span>}
          </div>
        </div>
        {description && <p className="camp-entity-desc">{description}</p>}
        {stats && <p className="camp-entity-stats">{stats}</p>}
        <button type="button" className="camp-expand-btn">{open ? "▲ less" : "▼ more"}</button>
      </div>

      {open && (
        <div className="camp-entity-body">
          {motivation && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Motivation</span>
              <span className="camp-detail-text">{motivation}</span>
            </div>
          )}
          {secrets && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Secrets</span>
              <span className="camp-detail-text camp-detail-text--secret">{secrets}</span>
            </div>
          )}
          {voice_notes && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Voice</span>
              <span className="camp-detail-text">{voice_notes}</span>
            </div>
          )}
          {read_aloud && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Read-Aloud</span>
              <div className="camp-read-aloud">{read_aloud}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Scene Card ────────────────────────────────────────────────────────────────

function SceneCard({ scene }) {
  const [open, setOpen] = useState(false);
  const title = scene.title || "(untitled)";
  const type = scene.type || "";
  const read_aloud = scene.read_aloud || "";
  const gm_notes = scene.gm_notes || scene.notes || "";
  const npcs = Array.isArray(scene.npcs) ? scene.npcs : [];

  return (
    <div className="camp-entity-card">
      <div className="camp-entity-header" onClick={() => setOpen(v => !v)}>
        <div className="camp-entity-title-row">
          <span className="camp-entity-name">{title}</span>
          {type && <span className={typeBadgeClass(type)}>{type}</span>}
          {scene.act && <span className="camp-entity-faction">{scene.act}</span>}
        </div>
        {scene.location && <p className="camp-entity-desc">Location: {scene.location}</p>}
        {read_aloud && (
          <p className="camp-entity-desc camp-entity-desc--italic">
            {read_aloud.slice(0, 120)}{read_aloud.length > 120 ? "…" : ""}
          </p>
        )}
        <button type="button" className="camp-expand-btn">{open ? "▲ less" : "▼ more"}</button>
      </div>

      {open && (
        <div className="camp-entity-body">
          {read_aloud && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Read-Aloud</span>
              <div className="camp-read-aloud">{read_aloud}</div>
            </div>
          )}
          {gm_notes && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">GM Notes</span>
              <span className="camp-detail-text">{gm_notes}</span>
            </div>
          )}
          {npcs.length > 0 && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">NPCs</span>
              <div className="camp-npc-chips">
                {npcs.map(n => <span key={n} className="camp-npc-chip">{n}</span>)}
              </div>
            </div>
          )}
          {scene.difficulty && scene.difficulty !== "none" && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Difficulty</span>
              <span className="camp-detail-text">{scene.difficulty}</span>
            </div>
          )}
          {scene.rewards && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Rewards</span>
              <span className="camp-detail-text">{scene.rewards}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Location Card ─────────────────────────────────────────────────────────────

function LocationCard({ location }) {
  const name = typeof location === "string" ? location : (location.name || "(unnamed)");
  const description = typeof location === "object" ? (location.description || "") : "";

  return (
    <div className="camp-entity-card">
      <div className="camp-entity-header">
        <div className="camp-entity-title-row">
          <span className="camp-entity-name">{name}</span>
        </div>
        {description && <p className="camp-entity-desc">{description}</p>}
      </div>
    </div>
  );
}

// ── Lore Card ─────────────────────────────────────────────────────────────────

function LoreCard({ entry, entryType }) {
  const [open, setOpen] = useState(false);
  const title = entry.title || entry.name || "(untitled)";
  const summary = entry.summary || entry.description || "";
  const content = entry.content || "";
  const type = entryType || entry.type || "lore";

  return (
    <div className="camp-entity-card">
      <div className="camp-entity-header" onClick={() => (content || entry.objective) ? setOpen(v => !v) : null}>
        <div className="camp-entity-title-row">
          <span className="camp-entity-name">{title}</span>
          <span className={loreBadgeClass(type)}>{type}</span>
        </div>
        {summary && <p className="camp-entity-desc">{summary}</p>}
        {(content || entry.objective) && (
          <button type="button" className="camp-expand-btn">{open ? "▲ less" : "▼ more"}</button>
        )}
      </div>

      {open && (
        <div className="camp-entity-body">
          {entry.objective && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Objective</span>
              <span className="camp-detail-text">{entry.objective}</span>
            </div>
          )}
          {entry.stakes && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Stakes</span>
              <span className="camp-detail-text">{entry.stakes}</span>
            </div>
          )}
          {content && (
            <div className="camp-detail-row">
              <span className="camp-detail-label">Details</span>
              <span className="camp-detail-text">{content}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tab panels ────────────────────────────────────────────────────────────────

function EmptyMsg({ children }) {
  return <div className="camp-empty">{children}</div>;
}

function NpcsTab({ campaignData }) {
  const npcs = campaignData?.npcs || [];
  if (!npcs.length) return <EmptyMsg>No NPCs extracted yet. Parse an adventure document in Prep → Upload Adventure.</EmptyMsg>;
  return (
    <div className="camp-entity-list">
      {npcs.map((npc, i) => <NpcCard key={npc.name || i} npc={typeof npc === "string" ? { name: npc } : npc} />)}
    </div>
  );
}

function ScenesTab({ campaignData }) {
  const scenes = campaignData?.scenes || [];
  if (!scenes.length) return <EmptyMsg>No scenes extracted yet.</EmptyMsg>;
  return (
    <div className="camp-entity-list">
      {scenes.map((scene, i) => <SceneCard key={scene.title || i} scene={scene} />)}
    </div>
  );
}

function LocationsTab({ campaignData }) {
  const locations = campaignData?.locations || [];
  if (!locations.length) return <EmptyMsg>No locations extracted yet.</EmptyMsg>;
  return (
    <div className="camp-entity-list">
      {locations.map((loc, i) => <LocationCard key={(typeof loc === "string" ? loc : loc.name) || i} location={loc} />)}
    </div>
  );
}

function LoreTab({ campaignData }) {
  const entries = [
    ...(campaignData?.codex_entries || []).map(e => ({ ...e, _src: "codex" })),
    ...(campaignData?.factions || []).map(e => ({ ...e, _src: "faction", type: "faction" })),
    ...(campaignData?.quests || []).map(e => ({ ...e, _src: "quest", type: "quest" })),
    ...(campaignData?.items || []).map(e => ({ ...e, _src: "item", type: "item" })),
    ...(campaignData?.lore || []).map(e => ({ ...e, _src: "lore" })),
  ];
  if (!entries.length) return <EmptyMsg>No lore, factions, items or quests extracted yet.</EmptyMsg>;
  return (
    <div className="camp-entity-list">
      {entries.map((entry, i) => (
        <LoreCard key={entry.id || entry.title || entry.name || i} entry={entry} entryType={entry.type || entry._src} />
      ))}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

const TABS = ["overview", "npcs", "scenes", "locations", "lore", "settings"];

const TAB_LABELS = {
  overview: "Overview",
  npcs: "NPCs",
  scenes: "Scenes",
  locations: "Locations",
  lore: "Lore",
  settings: "Settings",
};

/**
 * CampaignPage — campaign data hub and app settings.
 */
export default function CampaignPage() {
  const [tab, setTab] = useState("overview");
  const { campaignData } = useAppState();

  const npcCount = campaignData?.npcs?.length ?? 0;
  const sceneCount = campaignData?.scenes?.length ?? 0;
  const locationCount = campaignData?.locations?.length ?? 0;
  const questCount = campaignData?.quests?.length ?? 0;
  const factionCount = campaignData?.factions?.length ?? 0;
  const loreCount = (campaignData?.codex_entries?.length ?? 0)
    + (campaignData?.lore?.length ?? 0)
    + factionCount
    + questCount
    + (campaignData?.items?.length ?? 0);
  const hasCampaign = Boolean(campaignData?.title);

  return (
    <div className="flex flex-col min-h-0 h-full">
      <div className="tab-strip mb-3 flex-shrink-0">
        {TABS.map(t => (
          <button
            key={t}
            type="button"
            className={tab === t ? "tab-active" : ""}
            onClick={() => setTab(t)}
          >
            {TAB_LABELS[t]}
            {t === "npcs" && npcCount > 0 && ` (${npcCount})`}
            {t === "scenes" && sceneCount > 0 && ` (${sceneCount})`}
            {t === "locations" && locationCount > 0 && ` (${locationCount})`}
            {t === "lore" && loreCount > 0 && ` (${loreCount})`}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {tab === "overview" && (
          <section className="max-w-3xl mx-auto p-5 space-y-5">
            {!hasCampaign ? (
              <div className="campaign-hero flex flex-col items-center text-center gap-4">
                <div className="font-heading text-[var(--gold)] text-xl tracking-widest uppercase">
                  No Campaign Loaded
                </div>
                <p className="text-sm text-[var(--text-2)] leading-relaxed max-w-sm">
                  Upload a campaign document in{" "}
                  <strong className="text-[var(--text-1)]">Prep → Upload Adventure</strong>{" "}
                  to get started. Parsed campaign data will appear here.
                </p>
              </div>
            ) : (
              <>
                <div className="campaign-hero">
                  <div className="font-heading text-2xl text-[var(--gold)] tracking-widest uppercase leading-tight">
                    {campaignData.title}
                  </div>
                  {(campaignData.system || campaignData.setting) && (
                    <div className="flex gap-4 mt-2 flex-wrap">
                      {campaignData.system && (
                        <span className="text-xs text-[var(--text-2)] uppercase tracking-[0.14em]">
                          <span className="text-[var(--text-2)]">System: </span>
                          <span className="text-[var(--text-1)] font-heading">
                            {typeof campaignData.system === "string"
                              ? campaignData.system
                              : campaignData.system?.label ?? ""}
                          </span>
                        </span>
                      )}
                      {campaignData.setting && (
                        <span className="text-xs text-[var(--text-2)]">
                          <span className="text-[var(--text-2)]">Setting: </span>
                          <span className="text-[var(--text-1)]">{campaignData.setting}</span>
                        </span>
                      )}
                    </div>
                  )}
                  {campaignData.summary && (
                    <p className="mt-3 text-sm text-[var(--text-1)] leading-relaxed border-t border-[#5c3e23]/50 pt-3">
                      {campaignData.summary}
                    </p>
                  )}
                  {campaignData.description && !campaignData.summary && (
                    <p className="mt-3 text-sm text-[var(--text-1)] leading-relaxed border-t border-[#5c3e23]/50 pt-3">
                      {campaignData.description}
                    </p>
                  )}
                </div>

                <div>
                  <div className="sidebar-section-label mb-3">Campaign Contents</div>
                  <div className="grid grid-cols-5 gap-2">
                    {[
                      { label: "NPCs", count: npcCount, tab: "npcs" },
                      { label: "Scenes", count: sceneCount, tab: "scenes" },
                      { label: "Locations", count: locationCount, tab: "locations" },
                      { label: "Quests", count: questCount, tab: "lore" },
                      { label: "Factions", count: factionCount, tab: "lore" },
                    ].map(({ label, count, tab: targetTab }) => (
                      <button
                        key={label}
                        type="button"
                        className="stat-tile"
                        onClick={() => count > 0 && setTab(targetTab)}
                        style={{ cursor: count > 0 ? "pointer" : "default" }}
                      >
                        <div className="stat-tile-number">{count}</div>
                        <div className="stat-tile-label">{label}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {campaignData.narrator_voice && (
                  <ParchmentCard title="Narrator Voice">
                    <p className="text-sm text-[var(--text-1)]">{campaignData.narrator_voice}</p>
                  </ParchmentCard>
                )}
              </>
            )}
          </section>
        )}

        {tab === "npcs" && (
          <div className="max-w-3xl mx-auto p-4">
            <NpcsTab campaignData={campaignData} />
          </div>
        )}

        {tab === "scenes" && (
          <div className="max-w-3xl mx-auto p-4">
            <ScenesTab campaignData={campaignData} />
          </div>
        )}

        {tab === "locations" && (
          <div className="max-w-3xl mx-auto p-4">
            <LocationsTab campaignData={campaignData} />
          </div>
        )}

        {tab === "lore" && (
          <div className="max-w-3xl mx-auto p-4">
            <LoreTab campaignData={campaignData} />
          </div>
        )}

        {tab === "settings" && <SettingsPage />}
      </div>
    </div>
  );
}
