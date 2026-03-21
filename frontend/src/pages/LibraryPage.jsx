import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { GripVertical, Upload } from "lucide-react";
import PrepPanel from "../components/prep/PrepPanel";
import LibraryPdfViewer from "../components/library/LibraryPdfViewer";
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

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** @param {string} line */
function boldSegments(line) {
  const parts = String(line).split(/\*\*/);
  return parts.map((p, i) => (i % 2 === 1 ? `<strong>${escapeHtml(p)}</strong>` : escapeHtml(p))).join("");
}

/** @param {string} input */
function markdownLiteToHtml(input) {
  const lines = String(input || "").split(/\n/);
  const out = [];
  for (const line of lines) {
    if (/^---\s*$/.test(line.trim())) {
      out.push('<hr class="library-md-hr" />');
      continue;
    }
    const hm = /^(#{1,6})\s+(.+)$/.exec(line);
    if (hm) {
      out.push(`<div class="library-md-heading">${escapeHtml(hm[2])}</div>`);
      continue;
    }
    if (!line.trim()) {
      out.push('<div class="library-md-spacer" aria-hidden="true"></div>');
      continue;
    }
    out.push(`<p class="library-md-p">${boldSegments(line)}</p>`);
  }
  return out.join("");
}

/** @param {File | null | undefined} file */
function getDocKind(file) {
  if (!file) return null;
  const name = (file.name || "").toLowerCase();
  const mime = file.type || "";
  if (name.endsWith(".pdf") || mime === "application/pdf") return "pdf";
  if (name.endsWith(".md") || mime === "text/markdown") return "md";
  if (name.endsWith(".txt") || mime === "text/plain") return "txt";
  return "unknown";
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

  /** First selected file — kept for Step 2 viewer (PDF.js / text) */
  const [uploadedFile, setUploadedFile] = useState(null);
  /** URL from POST /adventure/upload-doc (optional; PDF uses local File + ArrayBuffer) */
  const [uploadedDocUrl, setUploadedDocUrl] = useState("");
  const [textFileContent, setTextFileContent] = useState("");
  const [uploadDocLoading, setUploadDocLoading] = useState(false);
  const [uploadDocError, setUploadDocError] = useState("");

  const docScrollRef = useRef(null);
  const leftPaneRef = useRef(null);
  const pdfViewerRef = useRef(null);
  const libraryFileInputRef = useRef(null);
  const selectionReadTimerRef = useRef(null);
  const [leftPaneWidth, setLeftPaneWidth] = useState(0);
  const [pdfNumPages, setPdfNumPages] = useState(0);
  const [pdfPage, setPdfPage] = useState(1);
  const [pdfScale, setPdfScale] = useState(1);
  const [pdfFitScale, setPdfFitScale] = useState(1);
  const [pdfUseFit, setPdfUseFit] = useState(true);
  const [libraryImageUrls, setLibraryImageUrls] = useState([]);
  const [imageCopyToast, setImageCopyToast] = useState("");
  const imageToastTimerRef = useRef(null);
  const [selectionText, setSelectionText] = useState("");
  const [showToolbar, setShowToolbar] = useState(false);
  const [toolbarPos, setToolbarPos] = useState({ x: 0, y: 0 });
  const [sendToast, setSendToast] = useState("");
  const sendToastTimerRef = useRef(null);

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

  const docKind = useMemo(() => getDocKind(uploadedFile), [uploadedFile]);

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
      if (sendToastTimerRef.current) clearTimeout(sendToastTimerRef.current);
      if (selectionReadTimerRef.current) clearTimeout(selectionReadTimerRef.current);
    };
  }, []);

  useEffect(() => {
    const measurePane = () => {
      const w = leftPaneRef.current?.offsetWidth || 0;
      if (w > 0) setLeftPaneWidth(w);
    };
    measurePane();
    window.addEventListener("resize", measurePane);
    return () => window.removeEventListener("resize", measurePane);
  }, []);

  useEffect(() => {
    if (docKind === "pdf" && pdfUseFit) {
      setPdfScale(pdfFitScale);
    }
  }, [docKind, pdfUseFit, pdfFitScale]);

  useEffect(() => {
    if (step === 2 && files.length > 0 && !uploadedFile) {
      setUploadedFile(files[0]);
    }
  }, [step, files, uploadedFile]);

  const onFileChange = useCallback(
    async (e) => {
      const list = Array.from(e.target.files || []);
      setFiles(list);
      setParseError("");
      setUploadDocError("");
      setUploadedDocUrl("");
      setTextFileContent("");
      setPdfNumPages(0);
      setPdfPage(1);
      setPdfScale(1);
      setPdfFitScale(1);
      setPdfUseFit(true);

      if (!list.length) {
        setUploadedFile(null);
        return;
      }

      const first = list[0];
      setUploadedFile(first);

      if (requireApiKey && !apiKey.trim()) {
        setUploadDocError("Enter your API key to upload the document.");
        return;
      }

      setUploadDocLoading(true);
      try {
        const fd = new FormData();
        fd.append("file", first);
        const res = await authFetch("/adventure/upload-doc", { method: "POST", body: fd });
        const raw = await res.text();
        let payload = null;
        try {
          payload = raw ? JSON.parse(raw) : null;
        } catch {
          payload = null;
        }
        if (!res.ok) {
          throw new Error(payload?.detail || raw || `Upload failed (${res.status})`);
        }
        const url = typeof payload?.file_url === "string" ? payload.file_url : "";
        if (url) setUploadedDocUrl(url);

        const name = (first.name || "").toLowerCase();
        const mime = first.type || "";
        const isTextLike =
          /\.(txt|md)$/i.test(name) || mime === "text/plain" || mime === "text/markdown";
        if (isTextLike) {
          const text = await new Promise((resolve, reject) => {
            const fr = new FileReader();
            fr.onload = () => resolve(typeof fr.result === "string" ? fr.result : "");
            fr.onerror = () => reject(new Error("Could not read file"));
            fr.readAsText(first);
          });
          setTextFileContent(text);
        } else {
          setTextFileContent("");
        }
        setUploadDocError("");
      } catch (err) {
        const isUnreachable =
          err?.name === "TypeError" ||
          (typeof err?.message === "string" &&
            (err.message.includes("NetworkError") || err.message.includes("Failed to fetch")));
        setUploadDocError(
          isUnreachable
            ? "Could not reach the API. Start the backend (e.g. python server.py on port 7862), restart npm run dev after proxy changes, and use http://localhost:5173/preview/"
            : err.message || "Upload failed."
        );
      } finally {
        setUploadDocLoading(false);
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

      if (files[0]) setUploadedFile(files[0]);

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
    setStep(1);
    setParseError("");
    setDocumentName("");
    setFiles([]);
    setUploadedFile(null);
    setUploadedDocUrl("");
    setTextFileContent("");
    setUploadDocError("");
    setPdfNumPages(0);
    setPdfPage(1);
    setPdfScale(1);
    setPdfFitScale(1);
    setPdfUseFit(true);
    if (libraryFileInputRef.current) libraryFileInputRef.current.value = "";
    setParsePayload(null);
    setWorkflowScenes([]);
    setActiveSceneIndex(0);
    setBuilderError("");
    setLibraryImageUrls([]);
    setImageCopyToast("");
    setSelectionText("");
    setShowToolbar(false);
    setToolbarPos({ x: 0, y: 0 });
    setSendToast("");
    window.setTimeout(() => {
      if (libraryFileInputRef.current) libraryFileInputRef.current.value = "";
    }, 0);
  };

  const handleClearDocumentData = useCallback(() => {
    setFiles([]);
    setUploadedFile(null);
    setUploadedDocUrl("");
    setTextFileContent("");
    setUploadDocError("");
    setUploadDocLoading(false);
    setPdfNumPages(0);
    setPdfPage(1);
    setPdfScale(1);
    setPdfFitScale(1);
    setPdfUseFit(true);
    setLibraryImageUrls([]);
    setSelectionText("");
    setShowToolbar(false);
    if (typeof window !== "undefined") {
      window.getSelection()?.removeAllRanges();
    }
    window.setTimeout(() => {
      if (libraryFileInputRef.current) libraryFileInputRef.current.value = "";
    }, 0);
  }, []);

  const activeWorkflow = workflowScenes[activeSceneIndex] ?? null;

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

  const clearSelectionToolbar = useCallback(() => {
    if (typeof window !== "undefined") {
      window.getSelection()?.removeAllRanges();
    }
    setSelectionText("");
    setShowToolbar(false);
  }, []);

  const emitSentToast = useCallback((label) => {
    setSendToast(`✓ Sent to ${label}`);
    if (sendToastTimerRef.current) clearTimeout(sendToastTimerRef.current);
    sendToastTimerRef.current = setTimeout(() => {
      setSendToast("");
      sendToastTimerRef.current = null;
    }, 1500);
  }, []);

  const handleDocumentMouseUp = useCallback((e) => {
    if (selectionReadTimerRef.current) clearTimeout(selectionReadTimerRef.current);
    const rect = e.currentTarget.getBoundingClientRect();
    selectionReadTimerRef.current = window.setTimeout(() => {
      const selected = window.getSelection()?.toString().trim() || "";
      if (selected.length > 3) {
        setSelectionText(selected);
        setToolbarPos({ x: rect.left, y: rect.top });
        setShowToolbar(true);
        return;
      }
      setSelectionText("");
      setShowToolbar(false);
    }, 10);
  }, []);

  const upsertSceneFromWorkflow = useCallback(
    (sceneDraft) => {
      if (!sceneDraft) return;
      const existing = storeScenes.find((s) => s.id === sceneDraft.id);
      const campaignId = existing?.campaignId || activeCampaignId || "";
      upsertScene({
        ...(existing || {}),
        id: sceneDraft.id,
        campaignId,
        title: sceneDraft.title || "Scene",
        name: sceneDraft.title || "Scene",
        summary: (sceneDraft.readAloud || sceneDraft.title || "Scene").trim().slice(0, 280) || "Scene",
        readAloud: sceneDraft.readAloud,
        read_aloud: sceneDraft.readAloud,
        notes: sceneDraft.gmNotes,
        npcIds: (sceneDraft.npcs || [])
          .filter((n) => String(n.name || "").trim())
          .map((n) => n.id),
        codexEntryIds: existing?.codexEntryIds ?? [],
        actionLogIds: existing?.actionLogIds ?? [],
        narrationClipIds: existing?.narrationClipIds ?? [],
        tags: existing?.tags,
      });
    },
    [storeScenes, activeCampaignId, upsertScene]
  );

  const sendSelectionToField = useCallback(
    (field) => {
      if (!activeWorkflow || !selectionText.trim()) return;
      const raw = selectionText.trim();
      let next = { ...activeWorkflow, saved: false };

      if (field === "readAloud") {
        next = { ...next, readAloud: raw };
        emitSentToast("Read-aloud");
      } else if (field === "gmNotes") {
        next = { ...next, gmNotes: raw };
        emitSentToast("GM notes");
      } else if (field === "sceneTitle") {
        next = { ...next, title: raw.slice(0, 60) };
        emitSentToast("Scene title");
      } else if (field === "newNpc") {
        const firstLine = raw.split(/\r?\n/)[0]?.trim() || "";
        const npcName = firstLine.split(/\s*[—-]\s*/)[0].trim() || firstLine.slice(0, 60).trim();
        if (npcName) {
          next = {
            ...next,
            npcs: [...(next.npcs || []), { id: createId("import_npc"), name: npcName, role: "" }],
          };
          emitSentToast("New NPC");
        }
      }

      setWorkflowScenes((prev) => prev.map((w, i) => (i === activeSceneIndex ? next : w)));
      upsertSceneFromWorkflow(next);
      clearSelectionToolbar();
    },
    [activeWorkflow, selectionText, activeSceneIndex, upsertSceneFromWorkflow, clearSelectionToolbar, emitSentToast]
  );

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
              Drop in session notes, module PDFs, or campaign text, then parse to extract campaign data. Supports PDF,
              .txt, and .md files.
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
                  ref={libraryFileInputRef}
                  type="file"
                  multiple
                  accept="text/plain,text/markdown,application/pdf,.txt,.md,.pdf"
                  onChange={onFileChange}
                />
              </label>
              {uploadDocLoading && (
                <p className="text-[11px] text-[#9c7a3a] font-heading mt-2 mb-0 animate-pulse">
                  Uploading document…
                </p>
              )}
              {uploadDocError && !uploadDocLoading && (
                <p className="text-[11px] text-amber-700/90 mt-2 mb-0">{uploadDocError}</p>
              )}
              {!uploadDocLoading && !uploadDocError && uploadedFile && (
                <p className="text-[11px] text-[#6b8f6b]/90 mt-2 mb-0">
                  File ready
                  {textFileContent
                    ? ` — ${textFileContent.length.toLocaleString()} characters loaded for text preview.`
                    : uploadedDocUrl
                      ? " — uploaded to server."
                      : "."}{" "}
                  Continue to Parse when ready.
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
            {/* Left — PDF / text viewer, selection toolbar, images */}
            <section ref={leftPaneRef} className="flex flex-1 min-w-0 min-h-0 flex-col border-r border-[#5a3e1b]">
              <div className="shrink-0 border-b border-[#3a2510] bg-[#1a1008]">
                <div className="px-3 py-2 flex flex-wrap items-center gap-2 justify-between">
                  <h2 className="font-heading text-sm text-[#e7c27a] tracking-wide m-0">Document content</h2>
                  <div className="flex items-center gap-2">
                    {uploadedFile && docKind === "pdf" && (
                      <span className="text-[10px] font-heading uppercase tracking-wider px-2 py-0.5 rounded border border-amber-700/50 bg-amber-950/30 text-amber-300">
                        PDF
                      </span>
                    )}
                    {uploadedFile && docKind === "md" && (
                      <span className="text-[10px] font-heading uppercase tracking-wider px-2 py-0.5 rounded border border-blue-700/50 bg-blue-950/30 text-blue-300">
                        Markdown
                      </span>
                    )}
                    {uploadedFile && docKind === "txt" && (
                      <span className="text-[10px] font-heading uppercase tracking-wider px-2 py-0.5 rounded border border-[#4a4a4a] bg-[#1a1a1a]/50 text-[#b0b0b0]">
                        Text
                      </span>
                    )}
                    {uploadedFile && docKind === "unknown" && (
                      <span className="text-[10px] font-heading uppercase tracking-wider px-2 py-0.5 rounded border border-[#4a4a4a] bg-[#1a1a1a]/50 text-[#b0b0b0]">
                        File
                      </span>
                    )}
                    {uploadedFile && (
                      <button
                        type="button"
                        className="text-[10px] font-heading uppercase tracking-wider px-2 py-1 rounded border border-red-800/70 bg-red-950/40 text-red-300 hover:bg-red-900/50"
                        onClick={handleClearDocumentData}
                      >
                        Clear data
                      </button>
                    )}
                  </div>
                </div>
                {uploadedFile && docKind === "pdf" && pdfNumPages > 0 ? (
                  <div className="px-3 pb-2 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-[#5a3e1b] bg-[#130c06] text-[#e7c27a] hover:border-[#9b7440] disabled:opacity-40"
                      disabled={pdfPage <= 1}
                      onClick={() => {
                        const n = Math.max(1, pdfPage - 1);
                        setPdfPage(n);
                        pdfViewerRef.current?.scrollToPage(n);
                      }}
                    >
                      ← Prev
                    </button>
                    <span className="text-xs font-heading text-[#d8b36f]">
                      Page {pdfPage} of {pdfNumPages}
                    </span>
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-[#5a3e1b] bg-[#130c06] text-[#e7c27a] hover:border-[#9b7440]"
                      onClick={() => {
                        setPdfUseFit(false);
                        setPdfScale((s) => Math.max(0.5, Number((s - 0.25).toFixed(2))));
                      }}
                      title="Zoom out"
                    >
                      -
                    </button>
                    <span className="text-xs font-heading text-[#d8b36f] min-w-[3.5rem] text-center">
                      {Math.round((pdfScale || 1) * 100)}%
                    </span>
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-[#5a3e1b] bg-[#130c06] text-[#e7c27a] hover:border-[#9b7440]"
                      onClick={() => {
                        setPdfUseFit(false);
                        setPdfScale((s) => Math.min(3, Number((s + 0.25).toFixed(2))));
                      }}
                      title="Zoom in"
                    >
                      +
                    </button>
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-[#5a3e1b] bg-[#130c06] text-[#e7c27a] hover:border-[#9b7440]"
                      onClick={() => {
                        setPdfUseFit(true);
                        setPdfScale(pdfFitScale);
                      }}
                    >
                      Fit
                    </button>
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-[#5a3e1b] bg-[#130c06] text-[#e7c27a] hover:border-[#9b7440] disabled:opacity-40"
                      disabled={pdfPage >= pdfNumPages}
                      onClick={() => {
                        const n = Math.min(pdfNumPages, pdfPage + 1);
                        setPdfPage(n);
                        pdfViewerRef.current?.scrollToPage(n);
                      }}
                    >
                      Next →
                    </button>
                  </div>
                ) : null}
              </div>
              <div
                ref={docScrollRef}
                className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3"
                onMouseUp={handleDocumentMouseUp}
              >
                {showToolbar && (
                  <div
                    className="sticky top-0 z-20 -mx-3 mb-3 px-3 py-2 bg-[#1a0f06] border-b border-[#2a1a08] flex flex-wrap items-center gap-2"
                    data-toolbar-anchor={toolbarPos.x}
                  >
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-green-700/60 bg-green-950/40 text-green-300 hover:bg-green-900/50"
                      onClick={() => sendSelectionToField("readAloud")}
                    >
                      Read-aloud
                    </button>
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-blue-700/60 bg-blue-950/40 text-blue-300 hover:bg-blue-900/50"
                      onClick={() => sendSelectionToField("gmNotes")}
                    >
                      GM notes
                    </button>
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-red-700/60 bg-red-950/40 text-red-300 hover:bg-red-900/50"
                      onClick={() => sendSelectionToField("newNpc")}
                    >
                      New NPC
                    </button>
                    <button
                      type="button"
                      className="text-xs font-heading px-2 py-1 rounded border border-amber-700/60 bg-amber-950/40 text-amber-300 hover:bg-amber-900/50"
                      onClick={() => sendSelectionToField("sceneTitle")}
                    >
                      Scene title
                    </button>
                  </div>
                )}
                {!uploadedFile ? (
                  <div className="border-2 border-dashed border-[#4f341f] rounded-md p-4 text-center">
                    <p className="text-xs text-[#b89a62] leading-relaxed m-0 mb-3">
                      No file loaded. Select a document to view it here. Supports PDF, .txt, and .md files.
                    </p>
                    <button
                      type="button"
                      className="nav-glyph-btn intake-parse-btn text-sm"
                      onClick={() => libraryFileInputRef.current?.click()}
                    >
                      <Upload size={16} className="inline mr-1" />
                      Choose file
                    </button>
                    <input
                      ref={libraryFileInputRef}
                      type="file"
                      className="hidden"
                      accept="text/plain,text/markdown,application/pdf,.txt,.md,.pdf"
                      onChange={onFileChange}
                    />
                  </div>
                ) : docKind === "pdf" ? (
                  <LibraryPdfViewer
                    ref={pdfViewerRef}
                    file={uploadedFile}
                    scale={pdfScale}
                    containerWidth={leftPaneWidth}
                    scrollRootRef={docScrollRef}
                    onMeta={setPdfNumPages}
                    onVisiblePageChange={setPdfPage}
                    onFitScaleChange={(nextFit) => {
                      const safeFit = Math.min(3, Math.max(0.5, nextFit || 1));
                      setPdfFitScale(safeFit);
                    }}
                  />
                ) : docKind === "md" || docKind === "txt" ? (
                  <div
                    className="library-md-view max-w-none rounded-[6px] border border-solid border-[#2a1a08] bg-[#130c06] p-3"
                    dangerouslySetInnerHTML={{ __html: markdownLiteToHtml(textFileContent) }}
                  />
                ) : (
                  <p className="text-xs text-[#b89a62] leading-relaxed m-0">
                    Unsupported type for preview. Parsing may still work on the right.
                  </p>
                )}

                {libraryImageUrls.length > 0 ? (
                  <div className="pt-4 border-t border-[#2a1a08] mt-2">
                    <h3 className="font-heading text-xs text-[#e7c27a] tracking-wide uppercase mb-3 m-0">
                      Extracted images — drag to scene
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
                {sendToast ? (
                  <div
                    className="sticky bottom-0 left-0 right-0 mt-2 py-2 px-3 rounded-md border border-[#2a1a08] bg-[#130c06] text-center text-xs font-heading text-[#9dd08d]"
                    role="status"
                  >
                    {sendToast}
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
