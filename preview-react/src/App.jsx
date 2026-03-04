import React, { useEffect, useState } from "react";
import {
  Book,
  Dice5,
  Map,
  Pause,
  Play,
  ScrollText,
  Settings,
  SlidersHorizontal,
  Square,
  Swords,
} from "lucide-react";

const normalizePath = (path) => (path || "/").replace(/\/+$/, "") || "/";
const BASE_PATH = normalizePath(import.meta.env.BASE_URL || "/");
const LIVE_PATH = BASE_PATH;
const PREP_PATH = LIVE_PATH === "/" ? "/prep" : `${LIVE_PATH}/prep`;
const INTAKE_PATH = LIVE_PATH === "/" ? "/intake" : `${LIVE_PATH}/intake`;
const resolveViewFromPath = (path) => {
  const normalized = normalizePath(path);
  if (normalized.endsWith("/prep")) {
    return "prep";
  }
  if (normalized.endsWith("/intake")) {
    return "intake";
  }
  return "live";
};

const WAVE = [20, 45, 28, 60, 35, 70, 26, 52, 41, 63, 30, 47, 22, 58, 37, 66, 29, 49, 31, 54, 24, 62, 34, 57, 27, 64];

const TOOL_ICONS = [Dice5, Book, Map, ScrollText, Swords, Settings];

const ENEMIES = [
  { name: "Goblin Scout", hp: 8, max: 12 },
  { name: "Goblin Archer", hp: 6, max: 10 },
  { name: "Goblin Shaman", hp: 9, max: 15 },
  { name: "Captive Wolf", hp: 11, max: 20 },
];

const NPCS = ["Temple Guardian", "Lost Pilgrim", "Ancient Priest"];
const PARTY = [
  { name: "ATHELRED", hp: "333/33", ac: 13 },
  { name: "SERAPINA", hp: "333/33", ac: 15 },
  { name: "BORWIN", hp: "390/44", ac: 16 },
  { name: "DUNCAN", hp: "390/44", ac: 16 },
];
const PREP_SCENES = [
  { title: "Temple Entrance", detail: "Read-aloud", glyph: "*" },
  { title: "Haunted Halls", detail: "Social scene, 3 NPCs", glyph: "o" },
  { title: "Guardian Chamber", detail: "Combat encounter", glyph: "x" },
];
const PREP_NPCS = [
  { name: "Temple Guardian", initials: "TG" },
  { name: "Lost Pilgrim", initials: "LP" },
  { name: "Ancient Priest", initials: "AP" },
];
const PREP_REVEALS = [
  { name: "Hook line", status: "", tone: "green" },
  { name: "Temple history", status: "after combat", tone: "amber" },
  { name: "Secret passage", status: "Locked until clue", tone: "red" },
];

const Panel = ({ title, children, className = "" }) => (
  <section className={`panel-ornate ${className}`}>
    <div className="panel-head">
      <div className="plaque">{title}</div>
    </div>
    <div className="panel-body">{children}</div>
  </section>
);

const ViewTabs = ({ view, onNavigate, className = "" }) => (
  <div className={className}>
    <button type="button" className={`nav-glyph-btn ${view === "live" ? "is-active" : ""}`} onClick={() => onNavigate("live")}>
      Live Board
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "prep" ? "is-active" : ""}`} onClick={() => onNavigate("prep")}>
      Prep Room
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "intake" ? "is-active" : ""}`} onClick={() => onNavigate("intake")}>
      Doc Intake
    </button>
  </div>
);

const Header = ({ view, onNavigate }) => (
  <header className="dm-header">
    <div className="header-glow" />
    <ViewTabs view={view} onNavigate={onNavigate} className="header-actions nav-tab-bar" />
    <div className="relative z-10 text-center">
      <h1 className="font-heading text-[clamp(1.6rem,2.25vw,2.85rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
        GM Voice Studio - Live Board
      </h1>
      <p className="font-heading text-[clamp(1.1rem,1.7vw,2.1rem)] leading-[1.05] text-[#d9b878]">Campaign : The Shattered Crown</p>
    </div>
  </header>
);

const LeftColumn = () => (
  <div className="h-full min-h-0 grid grid-rows-[1.1fr_1fr_.75fr] gap-3">
    <Panel title="Quick Tools">
      <div className="grid grid-cols-3 gap-2 h-full">
        {TOOL_ICONS.map((Icon, idx) => (
          <button key={idx} type="button" className="quick-tile">
            <Icon size={30} className="text-[#a98547]" />
          </button>
        ))}
      </div>
    </Panel>

    <Panel title="Encounter Tracker">
      <div className="space-y-2">
        {ENEMIES.map((enemy) => (
          <div key={enemy.name} className="tracker-row">
            <span className="encounter-name">{enemy.name}</span>
            <div className="flex items-center gap-2 w-[40%]">
              <span className="encounter-hp">
                {enemy.hp}/{enemy.max}
              </span>
              <div className="grow h-3 border border-[#4f341f] bg-[#0f0b08]">
                <div className="h-full bg-gradient-to-r from-[#a82e17] to-[#d46e2f]" style={{ width: `${(enemy.hp / enemy.max) * 100}%` }} />
              </div>
            </div>
          </div>
        ))}
      </div>
    </Panel>

    <Panel title="Party Roster">
      <div className="map-slot">
        <Map size={42} className="text-[#9b7440]/70" />
      </div>
    </Panel>
  </div>
);

const MiddleColumn = () => (
  <div className="h-full min-h-0">
    <Panel title="Scene Detail Editor" className="h-full">
      <div className="tab-strip">
        <button type="button" className="tab-active">
          Scene Text
        </button>
        <button type="button">Important NPCs</button>
        <button type="button">Secrets & Clues</button>
      </div>

      <div className="parchment mt-2">
        &quot;As you enter the temple, dust motes dance in shafts of light that pierce the cracked shadows where the ruins meet
        the eternal vault.&quot;
      </div>

      <div className="subhead mt-3">Important NPCs</div>
      <div className="grid grid-cols-3 gap-2">
        {NPCS.map((npc) => (
          <article key={npc} className="portrait-card">
            <div className="portrait-slot" />
            <p>{npc}</p>
          </article>
        ))}
      </div>

      <div className="subhead mt-3">Party Roster</div>
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-2">
        {PARTY.map((char) => (
          <article key={char.name} className="party-card">
            <div className="party-face" />
            <div className="party-meta">
              <div className="name">{char.name}</div>
              <div className="stats">
                HP {char.hp} <span>/ AC {char.ac}</span>
              </div>
            </div>
          </article>
        ))}
      </div>
    </Panel>
  </div>
);

const RightColumn = () => (
  <div className="h-full min-h-0 grid grid-rows-[.95fr_1.25fr] gap-3">
    <Panel title="Voice Studio">
      <div className="space-y-2">
        <label className="field-wrap">
          <span>Choose Voices:</span>
          <select>
            <option>Albia (preset)</option>
          </select>
        </label>
        <label className="field-wrap">
          <select>
            <option>Dragon Queen (cloned)</option>
            <option>Temple Guardian (cloned)</option>
          </select>
        </label>

        <div className="wave-box">
          {WAVE.map((h, idx) => (
            <span key={idx} style={{ height: `${h}%` }} />
          ))}
        </div>

        <div className="flex items-center justify-between gap-2 mt-1">
          <div className="flex gap-1">
            <button type="button" className="ctl-btn">
              <Play size={14} />
            </button>
            <button type="button" className="ctl-btn">
              <Pause size={14} />
            </button>
            <button type="button" className="ctl-btn">
              <Square size={14} />
            </button>
          </div>
          <div className="voice-temp-row">
            <span>Temperature</span>
            <input type="range" className="accent-[#d4a857] w-24" />
          </div>
        </div>
      </div>
    </Panel>

    <Panel title="NPC Profile">
      <div className="flex items-start gap-2 border-b border-[#54351f] pb-2 mb-2">
        <div className="size-16 border border-[#6e4a28] rounded-full bg-[radial-gradient(circle_at_30%_25%,#74522f,#2b1a10)]" />
        <div className="flex-1">
          <div className="voice-label">Voice:</div>
          <div className="voice-name">Temple Guardian</div>
          <div className="voice-tag">
            Undead Sentinel
          </div>
        </div>
      </div>

      <div className="subhead">The events unfold:</div>
      <div className="event-line">-&gt; Players approach temple entrance</div>
      <div className="event-quote">
        &quot;As you enter the temple, dust motes dance in shafts of light...&quot;
      </div>

      <div className="mt-auto">
        <div className="flex justify-end mb-2">
          <button type="button" className="cta-secondary">
            Speak as NPC <SlidersHorizontal size={13} />
          </button>
        </div>
        <div className="flex gap-2">
          <input type="text" placeholder="They attack the guardian...." className="chat-input" />
          <button type="button" className="send-btn">
            Send
          </button>
        </div>
      </div>
    </Panel>
  </div>
);

const LiveBoard = ({ view, onNavigate }) => (
  <div className="dm-shell dm-fit mx-auto">
    <Header view={view} onNavigate={onNavigate} />
    <section className="min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3">
      <div className="xl:col-span-3 min-h-0">
        <LeftColumn />
      </div>
      <div className="xl:col-span-5 min-h-0">
        <MiddleColumn />
      </div>
      <div className="xl:col-span-4 min-h-0">
        <RightColumn />
      </div>
    </section>
  </div>
);

const PrepPanel = ({ title, children, className = "" }) => (
  <section className={`panel-ornate prep-panel ${className}`}>
    <div className="panel-head">
      <div className="plaque">{title}</div>
    </div>
    <div className="panel-body">{children}</div>
  </section>
);

const PrepHeader = ({ view, onNavigate }) => (
  <header className="prep-header">
    <div className="header-glow" />
    <ViewTabs view={view} onNavigate={onNavigate} className="prep-header-bar nav-tab-bar" />
    <div className="relative z-10 text-center">
      <h1 className="font-heading text-[clamp(1.8rem,2.35vw,3rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
        GM Voice Studio - Prep Room
      </h1>
      <p className="font-heading text-[clamp(1rem,1.5vw,1.85rem)] leading-[1.05] text-[#d8b36f]">Campaign Planning Console</p>
    </div>
  </header>
);

const PrepLeftColumn = () => (
  <div className="h-full min-h-0">
    <PrepPanel title="Adventure Outline" className="h-full">
      <div className="prep-act-row">
        <span>Act I - The Sunken Temple</span>
        <span>v</span>
      </div>
      {PREP_SCENES.map((scene, idx) => (
        <button key={scene.title} type="button" className={`prep-outline-item ${idx === 0 ? "is-active" : ""}`}>
          <div className="prep-outline-glyph">{scene.glyph}</div>
          <div className="prep-outline-copy">
            <div className="prep-outline-title">{scene.title}</div>
            <div className="prep-outline-meta">{scene.detail}</div>
          </div>
        </button>
      ))}
      <div className="prep-act-row">
        <span>Act II - The Shattered Crown</span>
        <span>v</span>
      </div>
      <div className="prep-act-row">
        <span>Act III - Dragon&apos;s Lair</span>
        <span>v</span>
      </div>
    </PrepPanel>
  </div>
);

const PrepMiddleColumn = () => (
  <div className="h-full min-h-0">
    <PrepPanel title="Scene Detail Editor" className="h-full">
      <div className="tab-strip prep-main-tabs">
        <button type="button" className="tab-active">
          Read-aloud
        </button>
        <button type="button">Important NPCs</button>
        <button type="button">Secrets &amp; Clues</button>
        <button type="button">Environment</button>
      </div>

      <div className="parchment prep-readaloud">
        - As you enter the temple, dust motes dance in shafts of light, filtering while cracked stained glass. A deep voice echoes
        from the shadows: "Who dares disturb the eternal resort?"
      </div>

      <div className="tab-strip prep-subtabs">
        <button type="button" className="tab-active">
          Important NPCs
        </button>
        <button type="button">Important NPCs</button>
        <button type="button">Secrets &amp; Clues</button>
      </div>

      <div className="prep-card-wrap">
        <div className="prep-card-head">Important NPCs</div>
        <div className="prep-npc-grid">
          {PREP_NPCS.map((npc) => (
            <article key={npc.name} className="prep-npc-card">
              <div className="prep-npc-face">{npc.initials}</div>
              <p>{npc.name}</p>
            </article>
          ))}
        </div>
      </div>
    </PrepPanel>
  </div>
);

const PrepRightColumn = () => (
  <div className="h-full min-h-0">
    <PrepPanel title="NPC Configuration" className="h-full">
      <div className="prep-profile-head">
        <div className="prep-avatar">TG</div>
        <div className="prep-name-block">
          <span>Name:</span>
          <strong>Temple Guardian</strong>
        </div>
      </div>

      <div className="prep-stack">
        <label className="prep-field">
          <span>Archetype</span>
          <select>
            <option>Undead Sentinel</option>
            <option>Ethereal Guardian</option>
          </select>
        </label>
        <label className="prep-field">
          <span>Voice</span>
          <select>
            <option>Dragon Queen (cloned)</option>
            <option>Temple Guardian (cloned)</option>
          </select>
        </label>
      </div>

      <div className="subhead">Personality Notes:</div>
      <ul className="prep-notes">
        <li>Speaks in formal, archaic language.</li>
        <li>Protective of the temple.</li>
        <li>Reveals secret passage after combat.</li>
      </ul>

      <div className="subhead">Reveals:</div>
      <div className="prep-reveal-list">
        {PREP_REVEALS.map((reveal) => (
          <div key={reveal.name} className="prep-reveal-row">
            <span className={`prep-reveal-dot ${reveal.tone}`} />
            <span className="prep-reveal-name">{reveal.name}</span>
            <span className="prep-reveal-status">{reveal.status || " "}</span>
          </div>
        ))}
      </div>
    </PrepPanel>
  </div>
);

const PrepRoom = ({ view, onNavigate }) => (
  <div className="dm-shell dm-fit prep-shell mx-auto">
    <PrepHeader view={view} onNavigate={onNavigate} />
    <section className="min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3">
      <div className="xl:col-span-3 min-h-0">
        <PrepLeftColumn />
      </div>
      <div className="xl:col-span-5 min-h-0">
        <PrepMiddleColumn />
      </div>
      <div className="xl:col-span-4 min-h-0">
        <PrepRightColumn />
      </div>
    </section>
  </div>
);

const IntakeHeader = ({ view, onNavigate }) => (
  <header className="prep-header intake-header">
    <div className="header-glow" />
    <ViewTabs view={view} onNavigate={onNavigate} className="prep-header-bar nav-tab-bar" />
    <div className="relative z-10 text-center">
      <h1 className="font-heading text-[clamp(1.8rem,2.35vw,3rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
        GM Voice Studio - Adventure Intake
      </h1>
      <p className="font-heading text-[clamp(1rem,1.5vw,1.85rem)] leading-[1.05] text-[#d8b36f]">
        Upload docs, parse structure, prep faster
      </p>
    </div>
  </header>
);

const AdventureIntake = ({ view, onNavigate }) => {
  const [files, setFiles] = useState([]);
  const [isParsing, setIsParsing] = useState(false);
  const [parseError, setParseError] = useState("");
  const [parseResult, setParseResult] = useState(null);

  const onFileChange = (event) => {
    setFiles(Array.from(event.target.files || []));
    setParseError("");
  };

  const onParse = async () => {
    if (!files.length) {
      setParseError("Choose at least one .txt, .md, or .pdf file.");
      return;
    }
    setParseError("");
    setIsParsing(true);
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      const response = await fetch("/adventure/parse", { method: "POST", body: formData });
      const raw = await response.text();
      let payload = null;
      try {
        payload = raw ? JSON.parse(raw) : null;
      } catch (error) {
        payload = null;
      }

      if (!response.ok) {
        throw new Error((payload && payload.detail) || raw || `Parse failed (${response.status})`);
      }
      if (!payload) {
        throw new Error("Parse returned no data.");
      }
      setParseResult(payload);
    } catch (error) {
      setParseError(error.message || "Unable to parse adventure docs.");
    } finally {
      setIsParsing(false);
    }
  };

  const acts = parseResult?.acts?.length
    ? parseResult.acts
    : [{ title: "Act I - Waiting for uploads", scenes: ["Upload your adventure docs, then click Parse Documents."] }];
  const npcs = parseResult?.npcs?.length ? parseResult.npcs : ["No NPCs parsed yet."];
  const locations = parseResult?.locations?.length ? parseResult.locations : ["No locations parsed yet."];
  const reveals = parseResult?.reveals?.length ? parseResult.reveals : ["No clues/reveals parsed yet."];

  return (
    <div className="dm-shell dm-fit prep-shell intake-shell mx-auto">
      <IntakeHeader view={view} onNavigate={onNavigate} />
      <section className="min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3">
        <div className="xl:col-span-4 min-h-0">
          <PrepPanel title="Upload Adventure Docs" className="h-full">
            <p className="intake-hint">Drop in session notes, module text, or PDF handouts and parse them into prep-ready blocks.</p>
            <label className="intake-file-pick">
              <span>Select Files</span>
              <input type="file" multiple accept=".txt,.md,.pdf" onChange={onFileChange} />
            </label>
            <button type="button" className="nav-glyph-btn intake-parse-btn" onClick={onParse} disabled={isParsing}>
              {isParsing ? "Parsing..." : "Parse Documents"}
            </button>
            {parseError ? <div className="intake-error">{parseError}</div> : null}
            <div className="subhead">Queued Files</div>
            <div className="intake-file-list">
              {files.length ? (
                files.map((file) => (
                  <div key={`${file.name}-${file.size}`} className="intake-file-item">
                    <span>{file.name}</span>
                    <small>{Math.max(1, Math.round(file.size / 1024))} KB</small>
                  </div>
                ))
              ) : (
                <div className="intake-empty">No files selected.</div>
              )}
            </div>
          </PrepPanel>
        </div>

        <div className="xl:col-span-4 min-h-0">
          <PrepPanel title="Parsed Adventure Outline" className="h-full">
            <div className="parchment intake-summary">
              {parseResult?.summary || "Summary appears here after parsing."}
            </div>
            <div className="subhead">Acts & Scenes</div>
            <div className="intake-act-list">
              {acts.map((act) => (
                <article key={act.title} className="intake-act-card">
                  <h3>{act.title}</h3>
                  <ul>
                    {(act.scenes || []).map((scene, sceneIdx) => (
                      <li key={`${act.title}-${sceneIdx}`}>{scene}</li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </PrepPanel>
        </div>

        <div className="xl:col-span-4 min-h-0">
          <PrepPanel title="Prep Essentials" className="h-full">
            <div className="subhead">NPC Candidates</div>
            <ul className="intake-pill-list">
              {npcs.map((npc) => (
                <li key={npc}>{npc}</li>
              ))}
            </ul>

            <div className="subhead">Location Candidates</div>
            <ul className="intake-pill-list">
              {locations.map((location) => (
                <li key={location}>{location}</li>
              ))}
            </ul>

            <div className="subhead">Clues & Reveals</div>
            <ul className="intake-pill-list">
              {reveals.map((reveal) => (
                <li key={reveal}>{reveal}</li>
              ))}
            </ul>
          </PrepPanel>
        </div>
      </section>
    </div>
  );
};

export default function App() {
  const [view, setView] = useState(() => {
    if (typeof window === "undefined") {
      return "live";
    }
    return resolveViewFromPath(window.location.pathname);
  });

  useEffect(() => {
    const onPopState = () => setView(resolveViewFromPath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigateTo = (nextView) => {
    if (typeof window === "undefined") {
      setView(nextView);
      return;
    }
    setView(nextView);
    const target = nextView === "prep" ? PREP_PATH : nextView === "intake" ? INTAKE_PATH : LIVE_PATH;
    if (normalizePath(window.location.pathname) !== normalizePath(target)) {
      window.history.pushState({ view: nextView }, "", target);
    }
  };

  return (
    <main className={`app-stage min-h-[100dvh] overflow-x-hidden overflow-y-auto p-2 md:p-3 ${view !== "live" ? "prep-mode" : ""}`}>
      {view === "prep" ? <PrepRoom view={view} onNavigate={navigateTo} /> : null}
      {view === "intake" ? <AdventureIntake view={view} onNavigate={navigateTo} /> : null}
      {view === "live" ? <LiveBoard view={view} onNavigate={navigateTo} /> : null}
    </main>
  );
}
