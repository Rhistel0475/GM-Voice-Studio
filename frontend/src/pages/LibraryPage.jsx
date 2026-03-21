import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Clipboard, GripVertical, Upload } from "lucide-react";
import PrepPanel from "../components/prep/PrepPanel";
import { createId } from "../lib/utils/ids";
import { persistSceneContent, setBackendCampaignId } from "../lib/campaignPersistence";
import { useCampaignContextStore } from "../store/campaignContext";
import {
  DEFAULT_GAME_SYSTEM_ID,
  listGameSystemPlugins,
  normalizeGameSystemId,
  normalizeGameSystemPlugin,
  resolveGameSystemPlugin,
} from "../lib/gameSystemPlugins";

/** @param {unknown[]} rawNpcs @param {string} name */
function findRawNpc(rawNpcs, name) {
  const target = String(name || "").trim().toLowerCase();
  if (!target) return null;
  for (const n of rawNpcs) {
    if (typeof n === "string") {
      if (n.trim().toLowerCase() === target) return { name: n.trim() };
    } else if (n && typeof n === "object" && typeof n.name === "string") {
      if (n.name.trim().toLowerCase() === target) return n;
    }
  }
  return null;
}

function splitDocumentParagraphs(text) {
  if (!text || typeof text !== "string") return [];
  return text
    .split(/\n\n+/)
    .map((p) => p.trim())
    .filter(Boolean);
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** @param {string} text @param {string} query */
function highlightParagraphContent(text, query) {
  const q = (query || "").trim();
  if (!q) return text;
  const re = new RegExp(`(${escapeRegExp(q)})`, "gi");
  const parts = text.split(re);
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark
        key={`m-${i}-${part.slice(0, 24)}`}
        className="text-amber-300 bg-amber-950/50 rounded px-0.5"
      >
        {part}
      </mark>
    ) : (
      <span key={`t-${i}`}>{part}</span>
    )
  );
}

/**
 * @param {Record<string, unknown>} payload
 * @returns {{ id: string, parsedRef: number | null, title: string, readAloud: string, gmNotes: string, npcs: { id: string, name: string, role: string }[], saved: boolean }[]}
 */
function buildInitialWorkflowScenes(payload) {
  const scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  const rawNpcs = Array.isArray(payload.npcs) ? payload.npcs : [];

  const indexed = scenes
    .map((s, idx) => ({ s, idx }))
    .filter(({ s }) => s && typeof s === "object" && String(s.title || "").trim());

  if (indexed.length === 0) {
    return [
      {
        id: createId("import_scene"),
        parsedRef: null,
        title: "Scene 1",
        readAloud: "",
        gmNotes: "",
        npcs: [{ id: createId("import_npc"), name: "", role: "" }],
        saved: false,
      },
    ];
  }

  return indexed.map(({ s, idx }) => {
    const names = Array.isArray(s.npcs) ? s.npcs.map((x) => String(x || "").trim()).filter(Boolean) : [];
    const npcRows =
      names.length === 0
        ? [{ id: createId("import_npc"), name: "", role: "" }]
        : names.map((nm) => {
            const n = findRawNpc(rawNpcs, nm);
            const role =
              n && typeof n === "object"
                ? String((n.role || n.profession || "").trim())
                : "";
            return { id: createId("import_npc"), name: nm, role };
          });

    return {
      id: createId("import_scene"),
      parsedRef: idx,
      title: String(s.title).trim(),
      readAloud: "",
      gmNotes: "",
      npcs: npcRows,
      saved: false,
    };
  });
}

export default function LibraryPage() {
  const navigate = useNavigate();

  const upsertScene = useCampaignContextStore((s) => s.upsertScene);
  const upsertNpc = useCampaignContextStore((s) => s.upsertNpc);
  const upsertCampaign = useCampaignContextStore((s) => s.upsertCampaign);
  const setActiveCampaign = useCampaignContextStore((s) => s.setActiveCampaign);
  const activeCampaignId = useCampaignContextStore((s) => s.activeCampaignId);
  const campaigns = useCampaignContextStore((s) => s.campaigns);
  const storeScenes = useCampaignContextStore((s) => s.scenes);

  const [step, setStep] = useState(1);
  const [files, setFiles] = useState([]);
  const [isParsing, setIsParsing] = useState(false);
  const [parseError, setParseError] = useState("");
  const [documentName, setDocumentName] = useState("");

  /** Full raw text from POST /adventure/extract-text (first selected file) */
  const [documentText, setDocumentText] = useState("");
  const [isExtractingText, setIsExtractingText] = useState(false);
  const [extractTextError, setExtractTextError] = useState("");
  const extractAbortRef = useRef(null);
  const extractGenRef = useRef(0);
  const [docSearchQuery, setDocSearchQuery] = useState("");
  const [libraryImageUrls, setLibraryImageUrls] = useState([]);
  const [imageCopyToast, setImageCopyToast] = useState("");
  const imageToastTimerRef = useRef(null);

  /** Raw parse JSON — workflow scenes from Parse Document */
  const [parsePayload, setParsePayload] = useState(null);
  const [workflowScenes, setWorkflowScenes] = useState([]);
  const [activeSceneIndex, setActiveSceneIndex] = useState(0);
  const activeSceneIndexRef = useRef(0);
  activeSceneIndexRef.current = activeSceneIndex;
  const [builderError, setBuilderError] = useState("");

  const [requireApiKey, setRequireApiKey] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const [campaignSystems, setCampaignSystems] = useState(() => listGameSystemPlugins());
  const [selectedSystemId, setSelectedSystemId] = useState(DEFAULT_GAME_SYSTEM_ID);
  const selectedSystem = useMemo(
    () => resolveGameSystemPlugin(selectedSystemId, campaignSystems),
    [selectedSystemId, campaignSystems]
  );

  useEffect(() => {
    let cancelled = false;
    fetch("/config")
      .then((r) => (r.ok ? r.json() : { require_api_key: false }))
      .then((cfg) => {
        if (!cancelled) setRequireApiKey(Boolean(cfg?.require_api_key));
      })
      .catch(() => {
        if (!cancelled) setRequireApiKey(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const headers = new Headers();
    const key = apiKey.trim();
    if (key) headers.set("X-API-Key", key);
    fetch("/api/campaign-systems", { headers })
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (cancelled || !payload) return;
        const nextSystems = Array.isArray(payload.systems)
          ? payload.systems.map((system) => normalizeGameSystemPlugin(system)).filter(Boolean)
          : [];
        if (nextSystems.length > 0) {
          setCampaignSystems(nextSystems);
        }
        const defaultSystemId = normalizeGameSystemId(payload.default_system_id);
        setSelectedSystemId((current) => normalizeGameSystemId(current || defaultSystemId));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [apiKey]);

  const documentParagraphs = useMemo(() => splitDocumentParagraphs(documentText), [documentText]);
  const searchLower = docSearchQuery.trim().toLowerCase();
  const visibleParagraphs = useMemo(() => {
    if (!searchLower) return documentParagraphs;
    return documentParagraphs.filter((p) => p.toLowerCase().includes(searchLower));
  }, [documentParagraphs, searchLower]);

  const authFetch = useCallback(
    (input, init = {}) => {
      const headers = new Headers(init.headers || {});
      const key = apiKey.trim();
      if (key) headers.set("X-API-Key", key);
      return fetch(input, { ...init, headers });
    },
    [apiKey]
  );

  useEffect(() => {
    if (step !== 2) {
      setLibraryImageUrls([]);
      return;
    }
    const cid = parsePayload?.campaign_id;
    if (cid == null) {
      setLibraryImageUrls([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await authFetch(`/api/campaigns/${cid}/images`);
        const data = res.ok ? await res.json() : { images: [] };
        if (cancelled) return;
        setLibraryImageUrls(Array.isArray(data.images) ? data.images : []);
      } catch {
        if (!cancelled) setLibraryImageUrls([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [parsePayload?.campaign_id, step, authFetch]);

  useEffect(() => {
    return () => {
      if (imageToastTimerRef.current) clearTimeout(imageToastTimerRef.current);
    };
  }, []);

  const onFileChange = useCallback(
    async (e) => {
      const gen = ++extractGenRef.current;
      const list = Array.from(e.target.files || []);
      setFiles(list);
      setParseError("");
      setExtractTextError("");

      extractAbortRef.current?.abort();
      extractAbortRef.current = null;

      if (!list.length) {
        setDocumentText("");
        if (gen === extractGenRef.current) setIsExtractingText(false);
        return;
      }

      const first = list[0];
      setIsExtractingText(true);

      if (requireApiKey && !apiKey.trim()) {
        setDocumentText("");
        setExtractTextError("Enter your API key to load document text.");
        if (gen === extractGenRef.current) setIsExtractingText(false);
        return;
      }

      const ac = new AbortController();
      extractAbortRef.current = ac;

      try {
        const formData = new FormData();
        formData.append("files", first);
        const res = await authFetch("/adventure/extract-text", {
          method: "POST",
          body: formData,
          signal: ac.signal,
        });
        const raw = await res.text();
        let payload = null;
        try {
          payload = raw ? JSON.parse(raw) : null;
        } catch {
          payload = null;
        }
        if (!res.ok) {
          throw new Error(payload?.detail || raw || `Extract failed (${res.status})`);
        }
        const text = typeof payload?.text === "string" ? payload.text : "";
        if (gen !== extractGenRef.current) return;
        setDocumentText(text);
        setExtractTextError("");
      } catch (err) {
        if (err?.name === "AbortError") return;
        if (gen !== extractGenRef.current) return;
        setDocumentText("");
        const isUnreachable =
          err?.name === "TypeError" ||
          (typeof err?.message === "string" &&
            (err.message.includes("NetworkError") || err.message.includes("Failed to fetch")));
        setExtractTextError(
          isUnreachable
            ? "Could not reach the API. Start the backend (e.g. python server.py on port 7862), restart npm run dev after proxy changes, and use http://localhost:5173/preview/"
            : err.message || "Could not extract text from file."
        );
      } finally {
        if (extractAbortRef.current === ac) {
          extractAbortRef.current = null;
        }
        if (gen === extractGenRef.current) {
          setIsExtractingText(false);
        }
      }
    },
    [authFetch, requireApiKey, apiKey]
  );

  const deriveDocumentName = (payload) => {
    const fromTitle = typeof payload?.title === "string" ? payload.title.trim() : "";
    if (fromTitle) return fromTitle;
    const first = files[0]?.name;
    if (first) return first.replace(/\.[^.]+$/, "");
    return "Imported adventure";
  };

  const resolveOrCreateCampaignId = useCallback(
    (name) => {
      if (activeCampaignId) return activeCampaignId;
      if (campaigns.length === 1) {
        setActiveCampaign(campaigns[0].id);
        return campaigns[0].id;
      }
      const campaignId = createId("campaign");
      upsertCampaign({ id: campaignId, name: name || "Imported Campaign" });
      setActiveCampaign(campaignId);
      return campaignId;
    },
    [activeCampaignId, campaigns, setActiveCampaign, upsertCampaign]
  );

  const runParse = async (endpoint) => {
    if (!files.length) {
      setParseError("Select at least one .txt, .md, or .pdf file.");
      return;
    }
    if (requireApiKey && !apiKey.trim()) {
      setParseError("Enter your API key (required by server).");
      return;
    }
    setParseError("");
    setIsParsing(true);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      formData.append("campaign_system", selectedSystemId);
      const res = await authFetch(endpoint, { method: "POST", body: formData });
      const raw = await res.text();
      let payload = null;
      try {
        payload = raw ? JSON.parse(raw) : null;
      } catch {
        payload = null;
      }
      if (!res.ok) throw new Error(payload?.detail || raw || `Parse failed (${res.status})`);
      if (!payload) throw new Error("Parse returned no data.");

      if (payload.campaign_id != null) setBackendCampaignId(payload.campaign_id);

      const docName = deriveDocumentName(payload);
      setDocumentName(docName);

      setParsePayload(payload);
      const initial = buildInitialWorkflowScenes(payload);
      setWorkflowScenes(initial);
      setActiveSceneIndex(0);
      setBuilderError("");

      setStep(2);
    } catch (err) {
      setParseError(err.message || "Unable to parse documents.");
    } finally {
      setIsParsing(false);
    }
  };

  const handleStartOver = () => {
    extractAbortRef.current?.abort();
    extractAbortRef.current = null;
    setStep(1);
    setParseError("");
    setDocumentName("");
    setDocumentText("");
    setExtractTextError("");
    setIsExtractingText(false);
    setParsePayload(null);
    setWorkflowScenes([]);
    setActiveSceneIndex(0);
    setBuilderError("");
    setDocSearchQuery("");
    setLibraryImageUrls([]);
    setImageCopyToast("");
  };

  const activeWorkflow = workflowScenes[activeSceneIndex] ?? null;

  const copyParagraphToClipboard = useCallback((para) => {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(para);
    }
  }, []);

  const copyImageUrlWithToast = useCallback((url) => {
    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) return;
    void navigator.clipboard.writeText(url).then(() => {
      setImageCopyToast("Copied!");
      if (imageToastTimerRef.current) clearTimeout(imageToastTimerRef.current);
      imageToastTimerRef.current = setTimeout(() => {
        setImageCopyToast("");
        imageToastTimerRef.current = null;
      }, 2000);
    });
  }, []);

  const allScenesSaved = workflowScenes.length > 0 && workflowScenes.every((w) => w.saved);

  const updateWorkflowField = useCallback(
    (index, patch) => {
      setWorkflowScenes((prev) => prev.map((w, i) => (i === index ? { ...w, ...patch, saved: false } : w)));
    },
    []
  );

  const updateNpcRow = useCallback((sceneIndex, npcId, patch) => {
    setWorkflowScenes((prev) =>
      prev.map((w, i) => {
        if (i !== sceneIndex) return w;
        return {
          ...w,
          saved: false,
          npcs: w.npcs.map((n) => (n.id === npcId ? { ...n, ...patch } : n)),
        };
      })
    );
  }, []);

  const addNpcRow = useCallback((sceneIndex) => {
    setWorkflowScenes((prev) =>
      prev.map((w, i) =>
        i === sceneIndex
          ? { ...w, saved: false, npcs: [...w.npcs, { id: createId("import_npc"), name: "", role: "" }] }
          : w
      )
    );
  }, []);

  const removeNpcRow = useCallback((sceneIndex, npcId) => {
    setWorkflowScenes((prev) =>
      prev.map((w, i) => {
        if (i !== sceneIndex) return w;
        const next = w.npcs.filter((n) => n.id !== npcId);
        return {
          ...w,
          saved: false,
          npcs: next.length ? next : [{ id: createId("import_npc"), name: "", role: "" }],
        };
      })
    );
  }, []);

  const addWorkflowScene = useCallback(() => {
    setWorkflowScenes((prev) => {
      const n = prev.length + 1;
      const next = [
        ...prev,
        {
          id: createId("import_scene"),
          parsedRef: null,
          title: `Scene ${n}`,
          readAloud: "",
          gmNotes: "",
          npcs: [{ id: createId("import_npc"), name: "", role: "" }],
          saved: false,
        },
      ];
      setActiveSceneIndex(next.length - 1);
      return next;
    });
  }, []);

  const removeScene = useCallback((index) => {
    setWorkflowScenes((prev) => {
      if (prev.length <= 1) return prev;
      const n = prev.length;
      const cur = activeSceneIndexRef.current;
      let nextActive = cur;
      if (cur < index) nextActive = cur;
      else if (cur > index) nextActive = cur - 1;
      else nextActive = Math.min(index, n - 2);
      setActiveSceneIndex(nextActive);
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  const handleSaveScene = useCallback(
    async (index) => {
      const w = workflowScenes[index];
      if (!w) return;
      const title = (w.title || "").trim();
      if (!title) {
        setBuilderError("Scene title is required before saving.");
        return;
      }
      setBuilderError("");
      const campaignId = resolveOrCreateCampaignId(documentName);

      const namedNpcs = w.npcs.filter((n) => (n.name || "").trim());
      const npcIds = [];
      for (const row of namedNpcs) {
        const nm = row.name.trim();
        const role = (row.role || "").trim();
        upsertNpc({
          id: row.id,
          campaignId,
          name: nm,
          summary: role || "",
          role: role || undefined,
          tags: [],
        });
        npcIds.push(row.id);
      }

      const existing = storeScenes.find((s) => s.id === w.id);
      const readAloud = w.readAloud;
      const notes = w.gmNotes;
      const summary =
        readAloud.trim().slice(0, 280) || title;

      upsertScene({
        ...(existing || {}),
        id: w.id,
        campaignId,
        title,
        name: title,
        summary,
        readAloud,
        read_aloud: readAloud,
        notes,
        npcIds,
        codexEntryIds: existing?.codexEntryIds ?? [],
        actionLogIds: existing?.actionLogIds ?? [],
        narrationClipIds: existing?.narrationClipIds ?? [],
        tags: existing?.tags,
      });

      void persistSceneContent(authFetch, title, { readAloud, notes });

      setWorkflowScenes((prev) => prev.map((row, i) => (i === index ? { ...row, saved: true } : row)));
    },
    [authFetch, documentName, resolveOrCreateCampaignId, storeScenes, upsertNpc, upsertScene, workflowScenes]
  );

  const handleGoPrep = useCallback(() => {
    navigate("/prep");
  }, [navigate]);

  return (
    <div
      className={[
        "dm-shell dm-fit prep-shell intake-shell mx-auto p-3 md:p-4",
        step === 2 ? "flex flex-col flex-1 min-h-0 min-w-0" : "",
      ]
        .join(" ")
        .trim()}
    >
      <header className="prep-header intake-header mb-4 shrink-0">
        <div className="header-glow" />
        <div className="relative z-10 text-center">
          <h1 className="font-heading text-[clamp(1.5rem,2vw,2.25rem)] leading-tight text-[#e7c27a]">
            Adventure library
          </h1>
          <p className="font-heading text-sm text-[#d8b36f] mt-1">
            Step {step} of 2 — {step === 1 ? "Upload & parse" : "Build campaign"}
          </p>
        </div>
      </header>

      {requireApiKey && (
        <div className="mb-4 max-w-xl mx-auto shrink-0">
          <label className="block text-xs text-[#9c7a3a] font-heading tracking-wide mb-1">
            API key (required)
          </label>
          <input
            type="password"
            autoComplete="off"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-3 py-2 text-[#e7c27a] text-sm"
            placeholder="X-API-Key"
          />
        </div>
      )}

      {step === 1 && (
        <div className="max-w-xl mx-auto shrink-0">
          <PrepPanel title="Upload adventure docs">
            <p className="intake-hint">
              Drop in session notes, module PDFs, or campaign text, then parse to extract campaign data.
            </p>

            <div className="mb-4">
              <h3
                className="text-[#e7c27a] font-heading text-sm tracking-wide mb-2 border-b border-[#5a3e1b] pb-1"
              >
                Campaign system
              </h3>
              <label className="block text-[#9c7a3a] text-xs font-heading tracking-wide mb-1">
                Rules preset
              </label>
              <select
                value={selectedSystemId}
                onChange={(e) => setSelectedSystemId(normalizeGameSystemId(e.target.value))}
                className="w-full bg-[#1a0f06] text-[#e7c27a] border border-[#5a3e1b] rounded px-2 py-2 font-heading text-sm cursor-pointer"
              >
                {campaignSystems.map((system) => (
                  <option key={system.id} value={system.id}>
                    {system.label}
                  </option>
                ))}
              </select>
              {selectedSystem && (
                <div className="mt-2 border border-[#4f341f] rounded-md bg-[rgba(32,18,8,0.72)] p-3 text-xs text-[#b89a62] leading-relaxed">
                  <div className="text-[#e7c27a] font-heading text-sm mb-1">{selectedSystem.label}</div>
                  <p>{selectedSystem.rules_flavor}</p>
                </div>
              )}
            </div>

            <div
              className="border-2 border-dashed border-[#4f341f] rounded-md p-3 mb-2 transition-colors hover:border-[#9b7440]"
            >
              <label className="intake-file-pick cursor-pointer">
                <Upload size={16} className="inline mr-1" />
                <span>Select files</span>
                <input
                  type="file"
                  multiple
                  accept="text/plain,text/markdown,application/pdf,.txt,.md,.pdf"
                  onChange={onFileChange}
                />
              </label>
              {isExtractingText && (
                <p className="text-[11px] text-[#9c7a3a] font-heading mt-2 mb-0 animate-pulse">
                  Loading document text…
                </p>
              )}
              {extractTextError && !isExtractingText && (
                <p className="text-[11px] text-amber-700/90 mt-2 mb-0">{extractTextError}</p>
              )}
              {!isExtractingText && !extractTextError && documentText && files.length > 0 && (
                <p className="text-[11px] text-[#6b8f6b]/90 mt-2 mb-0">
                  Document text loaded ({documentText.length.toLocaleString()} characters). Continue to Parse when
                  ready.
                </p>
              )}
            </div>

            <div className="flex flex-col gap-2 mt-2">
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn is-active"
                onClick={() => runParse("/adventure/parse")}
                disabled={isParsing}
              >
                {isParsing ? "Parsing…" : "Parse Document"}
              </button>
            </div>

            {parseError && <div className="intake-error mt-2">{parseError}</div>}

            <div className="subhead mt-3">Queued files</div>
            <div className="intake-file-list">
              {files.length ? (
                files.map((f) => (
                  <div key={`${f.name}-${f.size}`} className="intake-file-item">
                    <span>{f.name}</span>
                    <small>{Math.max(1, Math.round(f.size / 1024))} KB</small>
                  </div>
                ))
              ) : (
                <div className="intake-empty">No files selected.</div>
              )}
            </div>
          </PrepPanel>
        </div>
      )}

      {step === 2 && (
        <div className="flex flex-col flex-1 min-h-0 min-w-0 gap-2">
          <div className="flex flex-wrap gap-2 justify-center shrink-0">
            <button type="button" className="nav-glyph-btn intake-parse-btn" onClick={handleStartOver}>
              ← Upload another document
            </button>
            {allScenesSaved && (
              <button type="button" className="nav-glyph-btn intake-parse-btn is-active" onClick={handleGoPrep}>
                Go to Prep →
              </button>
            )}
          </div>

          <div
            className="flex flex-1 min-h-0 min-w-0 gap-0 border border-[#5a3e1b] rounded-lg overflow-hidden bg-[#120a04]"
            style={{ minHeight: "min(70dvh, 720px)", maxHeight: "calc(100dvh - 12rem)" }}
          >
            {/* Left — document paragraphs, search, images */}
            <section className="flex flex-1 min-w-0 min-h-0 flex-col border-r border-[#5a3e1b]">
              <div className="shrink-0 border-b border-[#3a2510] bg-[#1a1008]">
                <div className="px-3 py-2">
                  <h2 className="font-heading text-sm text-[#e7c27a] tracking-wide m-0">Document content</h2>
                </div>
                <div className="px-3 pb-3">
                  <label className="sr-only" htmlFor="library-doc-search">
                    Filter paragraphs by keyword
                  </label>
                  <input
                    id="library-doc-search"
                    type="search"
                    value={docSearchQuery}
                    onChange={(e) => setDocSearchQuery(e.target.value)}
                    placeholder="Search paragraphs…"
                    className="w-full bg-[#130c06] border border-solid border-[#2a1a08] rounded-md px-3 py-2 text-sm text-[#e8d4a8] placeholder:text-[#5c4a38] focus:outline-none focus:ring-1 focus:ring-[#9b7440]"
                  />
                </div>
              </div>
              <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
                {!documentText ? (
                  <p className="text-xs text-[#b89a62] leading-relaxed m-0">
                    No document text yet. Go back to step 1 and select a file — text loads automatically from the first
                    file. You can still use Parse Document to pre-fill scenes on the right.
                  </p>
                ) : (
                  <>
                    {searchLower && visibleParagraphs.length === 0 ? (
                      <p className="text-xs text-[#9b7440] m-0">No paragraphs match your search.</p>
                    ) : null}
                    {visibleParagraphs.map((para, vi) => (
                      <div
                        key={`vp-${vi}-${para.length}-${para.slice(0, 24)}`}
                        className="relative rounded-[6px] border border-solid border-[#2a1a08] bg-[#130c06] pl-3 pr-12 py-2.5"
                      >
                        <button
                          type="button"
                          className="absolute top-2 right-2 p-1.5 rounded border border-[#3a2818] bg-[#1a0f06] text-[#d8b36f] hover:border-[#9b7440] hover:text-[#e7c27a]"
                          title="Copy paragraph"
                          aria-label="Copy paragraph to clipboard"
                          onClick={() => copyParagraphToClipboard(para)}
                        >
                          <Clipboard size={16} strokeWidth={1.75} aria-hidden />
                        </button>
                        <p
                          className="m-0 text-sm text-[#e8d4a8] pr-1"
                          style={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}
                        >
                          {highlightParagraphContent(para, docSearchQuery)}
                        </p>
                      </div>
                    ))}
                  </>
                )}

                {libraryImageUrls.length > 0 ? (
                  <div className="pt-4 border-t border-[#2a1a08] mt-2">
                    <h3 className="font-heading text-xs text-[#e7c27a] tracking-wide uppercase mb-3 m-0">
                      Document images
                    </h3>
                    <div className="grid grid-cols-3 gap-2">
                      {libraryImageUrls.map((url) => (
                        <div
                          key={url}
                          className="relative rounded-[6px] overflow-hidden border border-solid border-[#2a1a08] bg-[#130c06]"
                          draggable
                          onDragStart={(e) => {
                            e.dataTransfer.setData("imageUrl", url);
                            e.dataTransfer.effectAllowed = "copy";
                          }}
                        >
                          <div
                            className="absolute left-1 top-1 z-10 flex items-center justify-center p-1 rounded bg-[#0e0a05]/90 text-[#7a6348] cursor-grab active:cursor-grabbing border border-[#2a1a08]"
                            title="Drag to scene (drop target coming soon)"
                            aria-hidden
                          >
                            <GripVertical size={14} />
                          </div>
                          <button
                            type="button"
                            className="block w-full p-0 m-0 border-0 bg-transparent cursor-pointer leading-none"
                            onClick={() => copyImageUrlWithToast(url)}
                            title="Copy image URL"
                          >
                            <img
                              src={url}
                              alt=""
                              className="w-full h-[100px] object-cover rounded-[6px] block"
                              draggable={false}
                            />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                {imageCopyToast ? (
                  <div
                    className="sticky bottom-0 left-0 right-0 mt-3 py-2 px-3 rounded-md border border-[#5a3e1b] bg-[#1a1008] text-center text-xs font-heading text-[#e7c27a]"
                    role="status"
                  >
                    {imageCopyToast}
                  </div>
                ) : null}
              </div>
            </section>

            {/* Right — scene builder */}
            <section className="flex flex-1 min-w-0 min-h-0 flex-col">
              <div className="shrink-0 px-2 py-2 border-b border-[#3a2510] bg-[#1a1008]">
                <h2 className="font-heading text-sm text-[#e7c27a] tracking-wide px-1 mb-2">Scene builder</h2>
                <div className="flex flex-wrap items-center gap-1 overflow-x-auto pb-1">
                  {workflowScenes.map((s, i) => {
                    const active = i === activeSceneIndex;
                    return (
                      <div key={s.id} className="flex items-center gap-0.5 shrink-0">
                        <button
                          type="button"
                          onClick={() => setActiveSceneIndex(i)}
                          className={[
                            "px-2 py-1 rounded text-xs font-heading border transition-colors",
                            active
                              ? "bg-[#3d2814] border-[#c9a227] text-[#f0d78c]"
                              : "bg-[#1a0f06] border-[#5a3e1b] text-[#c4a574] hover:border-[#8a6236]",
                          ].join(" ")}
                        >
                          {s.title || `Scene ${i + 1}`}
                          {s.saved ? " ✓" : ""}
                        </button>
                        {workflowScenes.length > 1 && (
                          <button
                            type="button"
                            className="text-[#a08060] hover:text-[#e7c27a] px-1 text-xs"
                            aria-label={`Remove scene ${i + 1}`}
                            onClick={() => removeScene(i)}
                          >
                            ×
                          </button>
                        )}
                      </div>
                    );
                  })}
                  <button
                    type="button"
                    className="px-2 py-1 rounded text-xs font-heading border border-dashed border-[#6b5030] text-[#b89a62] hover:border-[#9b7440]"
                    onClick={addWorkflowScene}
                  >
                    + Add scene
                  </button>
                </div>
              </div>

              <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
                {activeWorkflow && (
                  <>
                    <label className="block">
                      <span className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading">
                        Scene title
                      </span>
                      <input
                        type="text"
                        value={activeWorkflow.title}
                        onChange={(e) => updateWorkflowField(activeSceneIndex, { title: e.target.value })}
                        className="mt-1 w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-2 text-sm text-[#e7c27a]"
                      />
                    </label>
                    <label className="block">
                      <span className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading">
                        Read-aloud text
                      </span>
                      <textarea
                        value={activeWorkflow.readAloud}
                        onChange={(e) => updateWorkflowField(activeSceneIndex, { readAloud: e.target.value })}
                        placeholder="Paste from document →"
                        rows={6}
                        className="mt-1 w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-2 text-sm text-[#e7c27a] resize-y min-h-[120px]"
                      />
                    </label>
                    <label className="block">
                      <span className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading">
                        GM notes
                      </span>
                      <textarea
                        value={activeWorkflow.gmNotes}
                        onChange={(e) => updateWorkflowField(activeSceneIndex, { gmNotes: e.target.value })}
                        placeholder="Paste from document →"
                        rows={4}
                        className="mt-1 w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-2 text-sm text-[#e7c27a] resize-y min-h-[80px]"
                      />
                    </label>

                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading mb-2">
                        NPCs
                      </div>
                      <div className="space-y-2">
                        {activeWorkflow.npcs.map((npc) => (
                          <div
                            key={npc.id}
                            className="flex flex-col sm:flex-row gap-2 border border-[#4f341f] rounded p-2 bg-[rgba(32,18,8,0.5)]"
                          >
                            <input
                              type="text"
                              value={npc.name}
                              onChange={(e) => updateNpcRow(activeSceneIndex, npc.id, { name: e.target.value })}
                              placeholder="Name"
                              className="flex-1 min-w-0 bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-1.5 text-sm text-[#e7c27a]"
                            />
                            <input
                              type="text"
                              value={npc.role}
                              onChange={(e) => updateNpcRow(activeSceneIndex, npc.id, { role: e.target.value })}
                              placeholder="Role"
                              className="flex-1 min-w-0 bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-1.5 text-sm text-[#e7c27a]"
                            />
                            {activeWorkflow.npcs.length > 1 && (
                              <button
                                type="button"
                                className="text-xs text-[#a08060] hover:text-[#e7c27a] shrink-0 self-start sm:self-center"
                                onClick={() => removeNpcRow(activeSceneIndex, npc.id)}
                              >
                                Remove
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                      <button
                        type="button"
                        className="mt-2 text-xs font-heading text-[#d8b36f] border border-dashed border-[#6b5030] rounded px-2 py-1 hover:border-[#9b7440]"
                        onClick={() => addNpcRow(activeSceneIndex)}
                      >
                        + Add NPC
                      </button>
                    </div>

                    {builderError && <div className="text-sm text-red-400/90">{builderError}</div>}

                    <button
                      type="button"
                      className="nav-glyph-btn intake-parse-btn is-active w-full sm:w-auto"
                      onClick={() => handleSaveScene(activeSceneIndex)}
                    >
                      Save scene
                    </button>
                  </>
                )}
              </div>
            </section>
          </div>
        </div>
      )}
    </div>
  );
}
