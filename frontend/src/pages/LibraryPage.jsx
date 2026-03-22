import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload } from "lucide-react";
import PrepPanel from "../components/prep/PrepPanel";
import LibraryPdfViewer from "../components/library/LibraryPdfViewer";
import { createId } from "../lib/utils/ids";
import { clearCampaignData } from "../lib/clearCampaignData";
import { setBackendCampaignId } from "../lib/campaignPersistence";
import { useCampaignContextStore } from "../store/campaignContext";
import {
  DEFAULT_GAME_SYSTEM_ID,
  listGameSystemPlugins,
  normalizeGameSystemId,
  normalizeGameSystemPlugin,
  resolveGameSystemPlugin,
} from "../lib/gameSystemPlugins";
import {
  isStructuredTemplate,
  parseMarkdownTemplate,
  MARKDOWN_ADVENTURE_TEMPLATE,
} from "../lib/parseMarkdownTemplate";

const SCENE_TYPES = ["combat", "social", "exploration", "trap", "travel"];
const TEXT_PAGE_SIZE = 3000;

function getDocKind(file) {
  if (!file) return null;
  const name = (file.name || "").toLowerCase();
  const mime = file.type || "";
  if (name.endsWith(".pdf") || mime === "application/pdf") return "pdf";
  if (name.endsWith(".md") || mime === "text/markdown") return "md";
  if (name.endsWith(".txt") || mime === "text/plain") return "txt";
  return "unknown";
}

function buildReviewItems(payload) {
  const scenes = Array.isArray(payload.scenes) ? payload.scenes : [];
  const rawNpcs = Array.isArray(payload.npcs) ? payload.npcs : [];
  if (scenes.length === 0) return [];

  return scenes
    .map((s) => {
      if (!s || typeof s !== "object") return null;
      const title = String(s.title || s.name || "").trim();
      if (!title) return null;
      const readAloud = String(s.read_aloud || s.readAloud || "").trim();
      const gmNotes = String(s.notes || s.gmNotes || "").trim();
      const type = SCENE_TYPES.includes(String(s.type || "").toLowerCase())
        ? String(s.type).toLowerCase()
        : "exploration";
      const sceneNpcNames = Array.isArray(s.npcs) ? s.npcs : [];
      const npcs = sceneNpcNames.map((ref) => {
        const name = typeof ref === "string" ? ref.trim() : String(ref?.name || "").trim();
        const found = rawNpcs.find(
          (rn) => typeof rn === "object" && String(rn.name || "").trim().toLowerCase() === name.toLowerCase()
        );
        const role = found && typeof found === "object" ? String(found.role || found.profession || "").trim() : "";
        return { id: createId("rv_npc"), name, role };
      });
      const monsters = npcs
        .map((row) =>
          rawNpcs.find(
            (rn) => typeof rn === "object" && String(rn.name || "").trim().toLowerCase() === row.name.toLowerCase()
          )
        )
        .filter((n) => n && (n.role === "villain" || String(n.hp || "").trim() !== "" || String(n.cr || "").trim() !== ""))
        .map((n) => ({
          id: createId("rv_monster"),
          name: String(n.name || "").trim(),
          role: String(n.role || "").trim(),
          hp: String(n.hp || "").trim(),
          ac: n.ac ? String(n.ac) : "",
          cr: String(n.cr || "").trim(),
        }));
      const imageUrl = String(s.image_url ?? s.imageUrl ?? "").trim();
      return { id: createId("rv_scene"), title, readAloud, gmNotes, type, npcs: npcs.length ? npcs : [], monsters, imageUrl, status: "pending" };
    })
    .filter(Boolean);
}

function reviewItemsFromMarkdown(parsed) {
  return parsed.map((s) => ({
    id: createId("rv_scene"),
    title: s.title,
    readAloud: s.readAloud || "",
    gmNotes: s.notes || "",
    type: SCENE_TYPES.includes(s.type) ? s.type : "exploration",
    npcs: (s.npcs || []).map((n) => ({
      id: createId("rv_npc"),
      name: n.name || "",
      role: [n.role, n.personality].filter(Boolean).join(", "),
    })),
    monsters: [],
    imageUrl: "",
    status: "pending",
  }));
}

export default function LibraryPage() {
  const navigate = useNavigate();

  const upsertScene = useCampaignContextStore((s) => s.upsertScene);
  const upsertNpc = useCampaignContextStore((s) => s.upsertNpc);
  const upsertCampaign = useCampaignContextStore((s) => s.upsertCampaign);
  const setActiveCampaign = useCampaignContextStore((s) => s.setActiveCampaign);
  const activeCampaignId = useCampaignContextStore((s) => s.activeCampaignId);
  const campaigns = useCampaignContextStore((s) => s.campaigns);

  const [step, setStep] = useState(1);
  const [files, setFiles] = useState([]);
  const [isParsing, setIsParsing] = useState(false);
  const [parseError, setParseError] = useState("");
  const [documentName, setDocumentName] = useState("");

  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedDocUrl, setUploadedDocUrl] = useState("");
  const [textFileContent, setTextFileContent] = useState("");
  const [uploadDocLoading, setUploadDocLoading] = useState(false);
  const [uploadDocError, setUploadDocError] = useState("");

  const libraryFileInputRef = useRef(null);
  const [isClearingAllAdventure, setIsClearingAllAdventure] = useState(false);
  const [templateSectionOpen, setTemplateSectionOpen] = useState(false);
  const [templateCopyToast, setTemplateCopyToast] = useState("");
  const templateCopyTimerRef = useRef(null);

  /* Step 2 — page-by-page analysis state */
  const pdfViewerRef = useRef(null);
  const pdfDocRef = useRef(null);
  const [pdfNumPages, setPdfNumPages] = useState(0);
  const [pdfPage, setPdfPage] = useState(1);
  const [pdfScale, setPdfScale] = useState(1);
  const [pdfFitScale, setPdfFitScale] = useState(1);
  const leftPaneRef = useRef(null);
  const [leftPaneWidth, setLeftPaneWidth] = useState(0);

  const textPageCount = useMemo(
    () => (textFileContent ? Math.max(1, Math.ceil(textFileContent.length / TEXT_PAGE_SIZE)) : 0),
    [textFileContent]
  );
  const [textPage, setTextPage] = useState(1);

  const [pageAnalysis, setPageAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState("");
  const [editingCard, setEditingCard] = useState(null);
  const [editingText, setEditingText] = useState("");

  const [workflowScenes, setWorkflowScenes] = useState([]);

  /* Step 3 — review state */
  const [reviewItems, setReviewItems] = useState([]);
  const [activeReviewIdx, setActiveReviewIdx] = useState(0);

  const [requireApiKey, setRequireApiKey] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const [campaignSystems, setCampaignSystems] = useState(() => listGameSystemPlugins());
  const [selectedSystemId, setSelectedSystemId] = useState(DEFAULT_GAME_SYSTEM_ID);
  const selectedSystem = useMemo(
    () => resolveGameSystemPlugin(selectedSystemId, campaignSystems),
    [selectedSystemId, campaignSystems]
  );

  const docKind = useMemo(() => getDocKind(uploadedFile), [uploadedFile]);

  useEffect(() => {
    let cancelled = false;
    fetch("/config")
      .then((r) => (r.ok ? r.json() : { require_api_key: false }))
      .then((cfg) => { if (!cancelled) setRequireApiKey(Boolean(cfg?.require_api_key)); })
      .catch(() => { if (!cancelled) setRequireApiKey(false); });
    return () => { cancelled = true; };
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
        if (nextSystems.length > 0) setCampaignSystems(nextSystems);
        const defaultSystemId = normalizeGameSystemId(payload.default_system_id);
        setSelectedSystemId((current) => normalizeGameSystemId(current || defaultSystemId));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [apiKey]);

  useEffect(() => () => { if (templateCopyTimerRef.current) clearTimeout(templateCopyTimerRef.current); }, []);

  useEffect(() => {
    const measure = () => { const w = leftPaneRef.current?.offsetWidth || 0; if (w > 0) setLeftPaneWidth(w); };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  useEffect(() => {
    if (docKind === "pdf" && pdfFitScale > 0) setPdfScale(pdfFitScale);
  }, [docKind, pdfFitScale]);

  const authFetch = useCallback(
    (input, init = {}) => {
      const headers = new Headers(init.headers || {});
      const key = apiKey.trim();
      if (key) headers.set("X-API-Key", key);
      return fetch(input, { ...init, headers });
    },
    [apiKey]
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
      if (campaigns.length === 1) { setActiveCampaign(campaigns[0].id); return campaigns[0].id; }
      const campaignId = createId("campaign");
      upsertCampaign({ id: campaignId, name: name || "Imported Campaign" });
      setActiveCampaign(campaignId);
      return campaignId;
    },
    [activeCampaignId, campaigns, setActiveCampaign, upsertCampaign]
  );

  /* ── File change handler ────────────────────────────────────────── */

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
      setTextPage(1);

      if (!list.length) { setUploadedFile(null); return; }
      const first = list[0];
      setUploadedFile(first);
      setDocumentName((first.name || "").replace(/\.[^.]+$/i, "").trim() || "Imported adventure");

      const name = (first.name || "").toLowerCase();
      const mime = first.type || "";
      const isTextLike = /\.(txt|md)$/i.test(name) || mime === "text/plain" || mime === "text/markdown";

      if (isTextLike) {
        try {
          const textContent = await new Promise((resolve, reject) => {
            const fr = new FileReader();
            fr.onload = () => resolve(typeof fr.result === "string" ? fr.result : "");
            fr.onerror = () => reject(new Error("Could not read file"));
            fr.readAsText(first);
          });
          setTextFileContent(textContent);

          if (isStructuredTemplate(textContent)) {
            const parsedScenes = parseMarkdownTemplate(textContent);
            if (parsedScenes.length > 0) {
              const items = reviewItemsFromMarkdown(parsedScenes);
              setReviewItems(items);
              setActiveReviewIdx(0);
              setStep(3);
              return;
            }
          }
        } catch {
          setTextFileContent("");
          setUploadDocError("Could not read file.");
          return;
        }
      }

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
        try { payload = raw ? JSON.parse(raw) : null; } catch { payload = null; }
        if (!res.ok) throw new Error(payload?.detail || raw || `Upload failed (${res.status})`);
        const url = typeof payload?.file_url === "string" ? payload.file_url : "";
        if (url) setUploadedDocUrl(url);
        setUploadDocError("");
      } catch (err) {
        const isUnreachable = err?.name === "TypeError" || (typeof err?.message === "string" && (err.message.includes("NetworkError") || err.message.includes("Failed to fetch")));
        setUploadDocError(isUnreachable ? "Could not reach the API. Start the backend." : err.message || "Upload failed.");
      } finally {
        setUploadDocLoading(false);
      }
    },
    [authFetch, requireApiKey, apiKey]
  );

  /* ── Full-doc parse (AI Parse / Quick Parse) — goes to loading then review ─ */

  const runParse = useCallback(
    async (endpoint) => {
      if (!files.length) { setParseError("Select at least one file."); return; }
      if (requireApiKey && !apiKey.trim()) { setParseError("Enter your API key."); return; }
      setParseError("");
      setIsParsing(true);
      try {
        const formData = new FormData();
        files.forEach((f) => formData.append("files", f));
        formData.append("campaign_system", selectedSystemId);
        const res = await authFetch(endpoint, { method: "POST", body: formData });
        const raw = await res.text();
        let payload = null;
        try { payload = raw ? JSON.parse(raw) : null; } catch { payload = null; }
        if (!res.ok) {
          const detail = payload?.detail ?? raw ?? `Parse failed (${res.status})`;
          if (payload?._traceback) console.error("Parse error traceback:", payload._traceback);
          throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
        }
        if (!payload) throw new Error("Parse returned no data.");
        if (payload.campaign_id != null) setBackendCampaignId(payload.campaign_id);
        setDocumentName(deriveDocumentName(payload));

        const items = buildReviewItems(payload);
        if (items.length === 0) {
          items.push({ id: createId("rv_scene"), title: "Scene 1", readAloud: "", gmNotes: "", type: "exploration", npcs: [], monsters: [], imageUrl: "", status: "pending" });
        }
        setReviewItems(items);
        setActiveReviewIdx(0);
        setStep(3);
      } catch (err) {
        setParseError(err.message || "Unable to parse documents.");
      } finally {
        setIsParsing(false);
      }
    },
    [files, requireApiKey, apiKey, selectedSystemId, authFetch, deriveDocumentName]
  );

  /* ── Go to page-by-page Step 2 ──────────────────────────────────── */

  const goToPageMode = useCallback(() => {
    if (!files.length) { setParseError("Select a file first."); return; }
    setPageAnalysis(null);
    setAnalyzeError("");
    setWorkflowScenes([]);
    setStep(2);
  }, [files]);

  /* ── Page text extraction ───────────────────────────────────────── */

  const extractCurrentPageText = useCallback(async () => {
    if (docKind === "pdf") {
      const pdfDoc = pdfDocRef.current;
      if (!pdfDoc) return "";
      const page = await pdfDoc.getPage(pdfPage);
      const content = await page.getTextContent();
      return content.items.map((it) => it.str || "").join(" ").trim();
    }
    if (textFileContent) {
      const start = (textPage - 1) * TEXT_PAGE_SIZE;
      return textFileContent.slice(start, start + TEXT_PAGE_SIZE).trim();
    }
    return "";
  }, [docKind, pdfPage, textPage, textFileContent]);

  /* ── Analyze this page ──────────────────────────────────────────── */

  const analyzePage = useCallback(async () => {
    setIsAnalyzing(true);
    setAnalyzeError("");
    setPageAnalysis(null);
    setEditingCard(null);
    try {
      const pageText = await extractCurrentPageText();
      if (!pageText) { setAnalyzeError("No text found on this page."); return; }
      const pageNum = docKind === "pdf" ? pdfPage : textPage;
      const res = await authFetch("/adventure/analyze-page", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ page_text: pageText, page_number: pageNum, campaign_system: selectedSystemId }),
      });
      const data = await res.json();
      if (data.error) { setAnalyzeError(data.error); return; }
      setPageAnalysis(data);
    } catch (err) {
      setAnalyzeError(err.message || "Analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }, [extractCurrentPageText, docKind, pdfPage, textPage, authFetch, selectedSystemId]);

  /* ── Workflow scene management ──────────────────────────────────── */

  const activeScene = workflowScenes[workflowScenes.length - 1] ?? null;

  const acceptSceneTitle = useCallback((title, sceneType) => {
    const newScene = {
      id: createId("wf_scene"),
      title,
      readAloud: "",
      gmNotes: "",
      type: SCENE_TYPES.includes(sceneType) ? sceneType : "exploration",
      npcs: [],
      monsters: [],
    };
    setWorkflowScenes((prev) => [...prev, newScene]);
  }, []);

  const appendToActiveScene = useCallback((field, value) => {
    setWorkflowScenes((prev) => {
      if (prev.length === 0) {
        const s = { id: createId("wf_scene"), title: "Scene 1", readAloud: "", gmNotes: "", type: "exploration", npcs: [], monsters: [] };
        return [{ ...s, [field]: value }];
      }
      return prev.map((s, i) => (i === prev.length - 1 ? { ...s, [field]: s[field] ? s[field] + "\n\n" + value : value } : s));
    });
  }, []);

  const acceptNpc = useCallback((npc) => {
    setWorkflowScenes((prev) => {
      if (prev.length === 0) {
        return [{ id: createId("wf_scene"), title: "Scene 1", readAloud: "", gmNotes: "", type: "exploration", npcs: [npc], monsters: [] }];
      }
      return prev.map((s, i) => (i === prev.length - 1 ? { ...s, npcs: [...s.npcs, npc] } : s));
    });
  }, []);

  const acceptMonster = useCallback((monster) => {
    setWorkflowScenes((prev) => {
      if (prev.length === 0) {
        return [{ id: createId("wf_scene"), title: "Scene 1", readAloud: "", gmNotes: "", type: "exploration", npcs: [], monsters: [monster] }];
      }
      return prev.map((s, i) => (i === prev.length - 1 ? { ...s, monsters: [...s.monsters, monster] } : s));
    });
  }, []);

  const wfTotals = useMemo(() => {
    let npcs = 0, monsters = 0;
    for (const s of workflowScenes) { npcs += s.npcs.length; monsters += s.monsters.length; }
    return { scenes: workflowScenes.length, npcs, monsters };
  }, [workflowScenes]);

  /* ── Accept all & next page ─────────────────────────────────────── */

  const acceptAllAndNext = useCallback(() => {
    if (!pageAnalysis) return;
    const pa = pageAnalysis;
    if (pa.is_new_scene && pa.scene_title) acceptSceneTitle(pa.scene_title, pa.scene_type);
    if (pa.read_aloud) appendToActiveScene("readAloud", pa.read_aloud);
    if (pa.gm_notes) appendToActiveScene("gmNotes", pa.gm_notes);
    for (const npc of pa.npcs || []) {
      if (npc.name) acceptNpc({ id: createId("wf_npc"), ...npc });
    }
    for (const m of pa.monsters || []) {
      if (m.name) acceptMonster({ id: createId("wf_mon"), ...m });
    }
    setPageAnalysis(null);
    setEditingCard(null);
    const maxPage = docKind === "pdf" ? pdfNumPages : textPageCount;
    const currentPage = docKind === "pdf" ? pdfPage : textPage;
    if (currentPage < maxPage) {
      if (docKind === "pdf") setPdfPage((p) => p + 1);
      else setTextPage((p) => p + 1);
    }
  }, [pageAnalysis, acceptSceneTitle, appendToActiveScene, acceptNpc, acceptMonster, docKind, pdfNumPages, textPageCount, pdfPage, textPage]);

  const skipPage = useCallback(() => {
    setPageAnalysis(null);
    setEditingCard(null);
    const maxPage = docKind === "pdf" ? pdfNumPages : textPageCount;
    const currentPage = docKind === "pdf" ? pdfPage : textPage;
    if (currentPage < maxPage) {
      if (docKind === "pdf") setPdfPage((p) => p + 1);
      else setTextPage((p) => p + 1);
    }
  }, [docKind, pdfNumPages, textPageCount, pdfPage, textPage]);

  /* ── Save campaign from workflow ────────────────────────────────── */

  const saveWorkflowAndGoPrep = useCallback(() => {
    const campaignId = resolveOrCreateCampaignId(documentName);
    for (const scene of workflowScenes) {
      const npcIds = [];
      for (const npc of scene.npcs) {
        if (!(npc.name || "").trim()) continue;
        const npcId = npc.id || createId("wf_npc");
        npcIds.push(npcId);
        upsertNpc({ id: npcId, campaignId, name: npc.name.trim(), summary: npc.description || npc.role || "", role: npc.role || undefined, tags: [] });
      }
      upsertScene({
        id: scene.id, campaignId, title: scene.title || "Scene", name: scene.title || "Scene",
        summary: (scene.readAloud || scene.title || "Scene").trim().slice(0, 280) || "Scene",
        readAloud: scene.readAloud, read_aloud: scene.readAloud, notes: scene.gmNotes, type: scene.type,
        npcIds, monsters: scene.monsters || [], codexEntryIds: [], actionLogIds: [], narrationClipIds: [], tags: [],
      });
    }
    navigate("/prep");
  }, [workflowScenes, resolveOrCreateCampaignId, documentName, upsertNpc, upsertScene, navigate]);

  /* ── Reset ──────────────────────────────────────────────────────── */

  const resetLibraryWizardState = useCallback(() => {
    setStep(1);
    setParseError(""); setDocumentName(""); setFiles([]); setUploadedFile(null);
    setUploadedDocUrl(""); setTextFileContent(""); setUploadDocError(""); setUploadDocLoading(false);
    setIsParsing(false);
    if (libraryFileInputRef.current) libraryFileInputRef.current.value = "";
    setReviewItems([]); setActiveReviewIdx(0);
    setTemplateSectionOpen(false); setTemplateCopyToast("");
    setPdfNumPages(0); setPdfPage(1); setPdfScale(1); setPdfFitScale(1);
    setTextPage(1); setPageAnalysis(null); setIsAnalyzing(false); setAnalyzeError("");
    setWorkflowScenes([]); setEditingCard(null); setEditingText("");
    pdfDocRef.current = null;
    window.setTimeout(() => { if (libraryFileInputRef.current) libraryFileInputRef.current.value = ""; }, 0);
  }, []);

  const handleStartOver = resetLibraryWizardState;

  const handleClearAllAdventureData = useCallback(async () => {
    if (isClearingAllAdventure) return;
    if (!window.confirm("Clear all adventure data everywhere? This cannot be undone.")) return;
    setIsClearingAllAdventure(true);
    try {
      await clearCampaignData({ deleteBackendCampaigns: true, xApiKey: apiKey.trim() });
      resetLibraryWizardState();
    } finally {
      setIsClearingAllAdventure(false);
    }
  }, [apiKey, isClearingAllAdventure, resetLibraryWizardState]);

  /* ── Step 3 review helpers ──────────────────────────────────────── */

  const activeReview = reviewItems[activeReviewIdx] ?? null;
  const nextUnreviewed = useCallback((after) => {
    for (let i = after + 1; i < reviewItems.length; i++) { if (reviewItems[i].status === "pending") return i; }
    for (let i = 0; i < after; i++) { if (reviewItems[i].status === "pending") return i; }
    return -1;
  }, [reviewItems]);
  const updateReviewField = useCallback((idx, patch) => { setReviewItems((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r))); }, []);
  const updateReviewNpc = useCallback((si, npcId, patch) => {
    setReviewItems((prev) => prev.map((r, i) => (i === si ? { ...r, npcs: r.npcs.map((n) => (n.id === npcId ? { ...n, ...patch } : n)) } : r)));
  }, []);
  const addReviewNpc = useCallback((si) => {
    setReviewItems((prev) => prev.map((r, i) => (i === si ? { ...r, npcs: [...r.npcs, { id: createId("rv_npc"), name: "", role: "" }] } : r)));
  }, []);
  const removeReviewNpc = useCallback((si, npcId) => {
    setReviewItems((prev) => prev.map((r, i) => (i === si ? { ...r, npcs: r.npcs.filter((n) => n.id !== npcId) } : r)));
  }, []);
  const addBlankScene = useCallback(() => {
    setReviewItems((prev) => [...prev, { id: createId("rv_scene"), title: `Scene ${prev.length + 1}`, readAloud: "", gmNotes: "", type: "exploration", npcs: [], monsters: [], imageUrl: "", status: "pending" }]);
    setActiveReviewIdx(reviewItems.length);
  }, [reviewItems.length]);
  const markScene = useCallback((status) => {
    updateReviewField(activeReviewIdx, { status });
    const next = nextUnreviewed(activeReviewIdx);
    if (next >= 0) setActiveReviewIdx(next);
  }, [activeReviewIdx, nextUnreviewed, updateReviewField]);

  const handleSaveAll = useCallback(() => {
    const campaignId = resolveOrCreateCampaignId(documentName);
    for (const item of reviewItems) {
      if (item.status === "skipped") continue;
      const npcIds = [];
      for (const npc of item.npcs) {
        if (!(npc.name || "").trim()) continue;
        npcIds.push(npc.id);
        upsertNpc({ id: npc.id, campaignId, name: npc.name.trim(), summary: npc.role || "", role: npc.role || undefined, tags: [] });
      }
      upsertScene({
        id: item.id, campaignId, title: item.title || "Scene", name: item.title || "Scene",
        summary: (item.readAloud || item.title || "Scene").trim().slice(0, 280) || "Scene",
        readAloud: item.readAloud, read_aloud: item.readAloud, notes: item.gmNotes, type: item.type, npcIds,
        monsters: item.monsters || [], imageUrl: item.imageUrl || "",
        codexEntryIds: [], actionLogIds: [], narrationClipIds: [], tags: [],
      });
    }
    navigate("/prep");
  }, [reviewItems, resolveOrCreateCampaignId, documentName, upsertNpc, upsertScene, navigate]);

  const allReviewed = reviewItems.length > 0 && reviewItems.every((r) => r.status === "approved" || r.status === "skipped");

  const stepLabel = step === 1 ? "Upload & parse" : step === 2 ? "Page-by-page analysis" : "Review & correct";

  /* ── PDF.js doc ref capture ─────────────────────────────────────── */

  const handlePdfMeta = useCallback((numPages, pdfDoc) => {
    setPdfNumPages(numPages);
    if (pdfDoc) pdfDocRef.current = pdfDoc;
  }, []);

  /* ── Render ─────────────────────────────────────────────────────── */

  return (
    <div className={["dm-shell dm-fit prep-shell intake-shell mx-auto p-3 md:p-4", step >= 2 ? "flex flex-col flex-1 min-h-0 min-w-0" : ""].join(" ").trim()}>
      <header className="prep-header intake-header mb-4 shrink-0">
        <div className="header-glow" />
        <div className="relative z-10 text-center">
          <h1 className="font-heading text-[clamp(1.5rem,2vw,2.25rem)] leading-tight text-[#e7c27a]">Adventure library</h1>
          <p className="font-heading text-sm text-[#d8b36f] mt-1">Step {step} of 3 — {stepLabel}</p>
        </div>
      </header>

      {requireApiKey && (
        <div className="mb-4 max-w-xl mx-auto shrink-0">
          <label className="block text-xs text-[#9c7a3a] font-heading tracking-wide mb-1">API key (required)</label>
          <input type="password" autoComplete="off" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className="w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-3 py-2 text-[#e7c27a] text-sm" placeholder="X-API-Key" />
        </div>
      )}

      {/* ─── Step 1: Upload ─────────────────────────────────────── */}
      {step === 1 && (
        <div className="max-w-xl mx-auto shrink-0">
          <PrepPanel title="Upload adventure docs">
            <p className="intake-hint">Drop in session notes, module PDFs, or campaign text, then parse to extract campaign data.</p>

            <div className="mb-4">
              <h3 className="text-[#e7c27a] font-heading text-sm tracking-wide mb-2 border-b border-[#5a3e1b] pb-1">Campaign system</h3>
              <label className="block text-[#9c7a3a] text-xs font-heading tracking-wide mb-1">Rules preset</label>
              <select value={selectedSystemId} onChange={(e) => setSelectedSystemId(normalizeGameSystemId(e.target.value))} className="w-full bg-[#1a0f06] text-[#e7c27a] border border-[#5a3e1b] rounded px-2 py-2 font-heading text-sm cursor-pointer">
                {campaignSystems.map((system) => (<option key={system.id} value={system.id}>{system.label}</option>))}
              </select>
              {selectedSystem && (
                <div className="mt-2 border border-[#4f341f] rounded-md bg-[rgba(32,18,8,0.72)] p-3 text-xs text-[#b89a62] leading-relaxed">
                  <div className="text-[#e7c27a] font-heading text-sm mb-1">{selectedSystem.label}</div>
                  <p>{selectedSystem.rules_flavor}</p>
                </div>
              )}
            </div>

            <div className="border-2 border-dashed border-[#4f341f] rounded-md p-3 mb-2 transition-colors hover:border-[#9b7440]">
              <label className="intake-file-pick cursor-pointer">
                <Upload size={16} className="inline mr-1" /><span>Select files</span>
                <input ref={libraryFileInputRef} type="file" multiple accept="text/plain,text/markdown,application/pdf,.txt,.md,.pdf" onChange={onFileChange} />
              </label>
              {uploadDocLoading && <p className="text-[11px] text-[#9c7a3a] font-heading mt-2 mb-0 animate-pulse">Uploading document…</p>}
              {uploadDocError && !uploadDocLoading && <p className="text-[11px] text-amber-700/90 mt-2 mb-0">{uploadDocError}</p>}
              {!uploadDocLoading && !uploadDocError && uploadedFile && (
                <p className="text-[11px] text-[#6b8f6b]/90 mt-2 mb-0">File ready{textFileContent ? ` — ${textFileContent.length.toLocaleString()} characters loaded.` : uploadedDocUrl ? " — uploaded to server." : "."}</p>
              )}
            </div>

            <div className="mt-3 border border-[#4f341f] rounded-md overflow-hidden">
              <button type="button" className="w-full px-3 py-2 text-left text-xs font-heading text-[#e7c27a] hover:bg-[#130c06] border-0 bg-transparent flex items-center justify-between gap-2" onClick={() => setTemplateSectionOpen((v) => !v)} aria-expanded={templateSectionOpen}>
                <span>Template format</span>
                <span className="text-[#9c7a3a] shrink-0">{templateSectionOpen ? "▼" : "▶"}</span>
              </button>
              {templateSectionOpen ? (
                <div className="px-3 pb-3 space-y-2 border-t border-[#3a2510] bg-[rgba(18,10,4,0.72)]">
                  <p className="text-[11px] text-[#b89a62] m-0 leading-relaxed">Use this structure in .txt or .md files for instant import.</p>
                  <pre className="m-0 text-[11px] leading-relaxed overflow-x-auto p-3 rounded border border-[#3a2510] bg-[#0e0804] text-[#d8c4a0] font-mono whitespace-pre-wrap">{MARKDOWN_ADVENTURE_TEMPLATE}</pre>
                  <button type="button" className="nav-glyph-btn intake-parse-btn text-xs py-1.5" onClick={() => {
                    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) return;
                    void navigator.clipboard.writeText(MARKDOWN_ADVENTURE_TEMPLATE).then(() => {
                      setTemplateCopyToast("Copied!");
                      if (templateCopyTimerRef.current) clearTimeout(templateCopyTimerRef.current);
                      templateCopyTimerRef.current = setTimeout(() => { setTemplateCopyToast(""); templateCopyTimerRef.current = null; }, 2000);
                    });
                  }}>Copy template</button>
                  {templateCopyToast ? <p className="text-[11px] text-[#6b8f6b] m-0 font-heading">{templateCopyToast}</p> : null}
                </div>
              ) : null}
            </div>

            <div className="flex flex-col gap-2 mt-3">
              <button type="button" className="nav-glyph-btn intake-parse-btn is-active" onClick={() => runParse("/adventure/ai-parse")} disabled={isParsing || !files.length}>
                {isParsing ? "Extracting…" : "AI Parse (Claude)"}
              </button>
              <button type="button" className="nav-glyph-btn intake-parse-btn" onClick={() => runParse("/adventure/parse")} disabled={isParsing || !files.length}>
                Quick Parse (no AI)
              </button>
              <button type="button" className="nav-glyph-btn intake-parse-btn" onClick={goToPageMode} disabled={!files.length}>
                Page-by-page mode
              </button>
            </div>

            {parseError && <div className="intake-error mt-2">{parseError}</div>}

            <div className="mt-4 pt-3 border-t border-[#4f341f]">
              <p className="text-[11px] text-[#9c7a3a] font-heading mb-2 leading-relaxed m-0">Clear all adventure data app-wide.</p>
              <button type="button" className="w-full text-xs font-heading px-3 py-2 rounded border border-red-800/70 bg-red-950/40 text-red-300 hover:bg-red-900/50 disabled:opacity-50" onClick={() => void handleClearAllAdventureData()} disabled={isClearingAllAdventure}>
                {isClearingAllAdventure ? "Clearing…" : "Clear all adventure data"}
              </button>
            </div>

            <div className="subhead mt-3">Queued files</div>
            <div className="intake-file-list">
              {files.length ? files.map((f) => (<div key={`${f.name}-${f.size}`} className="intake-file-item"><span>{f.name}</span><small>{Math.max(1, Math.round(f.size / 1024))} KB</small></div>)) : <div className="intake-empty">No files selected.</div>}
            </div>
          </PrepPanel>
        </div>
      )}

      {/* ─── Step 2: Page-by-page analysis ─────────────────────── */}
      {step === 2 && (
        <div className="flex flex-col flex-1 min-h-0 min-w-0 gap-2">
          <div className="flex flex-wrap gap-2 items-center justify-between shrink-0">
            <button type="button" className="nav-glyph-btn intake-parse-btn" onClick={handleStartOver}>← Back to upload</button>
            <button type="button" className="nav-glyph-btn intake-parse-btn is-active" onClick={saveWorkflowAndGoPrep} disabled={workflowScenes.length === 0}>
              Save campaign & go to Prep →
            </button>
          </div>

          <div className="flex flex-1 min-h-0 min-w-0 gap-0 border border-[#5a3e1b] rounded-lg overflow-hidden bg-[#120a04]" style={{ minHeight: "min(70dvh, 720px)", maxHeight: "calc(100dvh - 12rem)" }}>

            {/* Left — document viewer */}
            <section ref={leftPaneRef} className="flex flex-1 min-w-0 min-h-0 flex-col border-r border-[#5a3e1b]">
              <div className="flex-1 min-h-0 overflow-y-auto overflow-x-auto p-3">
                {docKind === "pdf" && uploadedFile ? (
                  <LibraryPdfViewer
                    ref={pdfViewerRef}
                    file={uploadedFile}
                    currentPage={pdfPage}
                    scale={pdfScale}
                    containerWidth={leftPaneWidth}
                    onMeta={handlePdfMeta}
                    onFitScaleChange={(v) => setPdfFitScale(Math.min(3, Math.max(0.5, v || 1)))}
                  />
                ) : textFileContent ? (
                  <div className="rounded border border-[#2a1a08] bg-[#130c06] p-3 text-sm text-[#e8d4a8] leading-relaxed whitespace-pre-wrap">
                    {textFileContent.slice((textPage - 1) * TEXT_PAGE_SIZE, textPage * TEXT_PAGE_SIZE)}
                  </div>
                ) : (
                  <p className="text-xs text-[#7a6348] italic m-0">No document loaded.</p>
                )}
              </div>
              <div className="shrink-0 px-3 py-2 border-t border-[#3a2510] bg-[#1a1008] flex flex-wrap items-center gap-2 justify-between">
                <div className="flex items-center gap-2">
                  <button type="button" className="text-xs font-heading px-2 py-1 rounded border border-[#5a3e1b] bg-[#130c06] text-[#e7c27a] hover:border-[#9b7440] disabled:opacity-40" disabled={(docKind === "pdf" ? pdfPage : textPage) <= 1}
                    onClick={() => { if (docKind === "pdf") setPdfPage((p) => Math.max(1, p - 1)); else setTextPage((p) => Math.max(1, p - 1)); }}>
                    ← Prev
                  </button>
                  <span className="text-xs font-heading text-[#d8b36f]">
                    Page {docKind === "pdf" ? pdfPage : textPage} of {docKind === "pdf" ? pdfNumPages : textPageCount}
                  </span>
                  <button type="button" className="text-xs font-heading px-2 py-1 rounded border border-[#5a3e1b] bg-[#130c06] text-[#e7c27a] hover:border-[#9b7440] disabled:opacity-40"
                    disabled={(docKind === "pdf" ? pdfPage : textPage) >= (docKind === "pdf" ? pdfNumPages : textPageCount)}
                    onClick={() => { if (docKind === "pdf") setPdfPage((p) => Math.min(pdfNumPages, p + 1)); else setTextPage((p) => Math.min(textPageCount, p + 1)); }}>
                    Next →
                  </button>
                </div>
                <button type="button" className="nav-glyph-btn intake-parse-btn is-active text-sm py-1.5" onClick={analyzePage} disabled={isAnalyzing}>
                  {isAnalyzing ? "Analyzing…" : "Analyze this page ✦"}
                </button>
              </div>
            </section>

            {/* Right — AI results panel */}
            <section className="flex w-[380px] shrink-0 min-h-0 flex-col bg-[#1a1008]">
              <div className="shrink-0 px-3 py-2 border-b border-[#3a2510] flex items-center justify-between">
                <p className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading m-0">
                  {wfTotals.scenes} scenes · {wfTotals.npcs} NPCs · {wfTotals.monsters} monsters collected
                </p>
              </div>

              <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
                {/* Idle state */}
                {!pageAnalysis && !isAnalyzing && !analyzeError && (
                  <div className="flex flex-col items-center justify-center gap-4 py-8 text-center">
                    <img src="/static/img/Spellbook.png" alt="" style={{ height: 60 }} draggable={false} className="opacity-60" />
                    <p className="text-xs text-[#7a6348] m-0 font-heading leading-relaxed">
                      Click "Analyze this page" to extract scenes, NPCs, and read-alouds from the current page.
                    </p>
                  </div>
                )}

                {/* Loading */}
                {isAnalyzing && (
                  <div className="flex flex-col items-center justify-center gap-4 py-8">
                    <p className="font-heading text-sm text-[#e7c27a] m-0 library-loading-dots">
                      Claude is reading page {docKind === "pdf" ? pdfPage : textPage}
                      <span className="ld-dot">.</span><span className="ld-dot">.</span><span className="ld-dot">.</span>
                    </p>
                  </div>
                )}

                {analyzeError && <div className="intake-error text-xs">{analyzeError}</div>}

                {/* Results cards */}
                {pageAnalysis && !isAnalyzing && (
                  <>
                    {pageAnalysis.is_new_scene && pageAnalysis.scene_title ? (
                      <ResultCard
                        headerColor="#c9a227" headerLabel="New scene" title={pageAnalysis.scene_title}
                        onAccept={() => acceptSceneTitle(pageAnalysis.scene_title, pageAnalysis.scene_type)}
                        editingCard={editingCard} cardKey="scene_title" setEditingCard={setEditingCard}
                        editingText={editingText} setEditingText={setEditingText}
                        onAcceptEdited={(t) => acceptSceneTitle(t, pageAnalysis.scene_type)}
                        initialEdit={pageAnalysis.scene_title}
                      />
                    ) : null}

                    {pageAnalysis.read_aloud ? (
                      <ResultCard
                        headerColor="#6b8f6b" headerLabel="Read-aloud text" body={pageAnalysis.read_aloud}
                        onAccept={() => appendToActiveScene("readAloud", pageAnalysis.read_aloud)}
                        editingCard={editingCard} cardKey="read_aloud" setEditingCard={setEditingCard}
                        editingText={editingText} setEditingText={setEditingText}
                        onAcceptEdited={(t) => appendToActiveScene("readAloud", t)}
                        initialEdit={pageAnalysis.read_aloud}
                      />
                    ) : null}

                    {pageAnalysis.gm_notes ? (
                      <ResultCard
                        headerColor="#5a7abf" headerLabel="GM notes" body={pageAnalysis.gm_notes}
                        onAccept={() => appendToActiveScene("gmNotes", pageAnalysis.gm_notes)}
                        editingCard={editingCard} cardKey="gm_notes" setEditingCard={setEditingCard}
                        editingText={editingText} setEditingText={setEditingText}
                        onAcceptEdited={(t) => appendToActiveScene("gmNotes", t)}
                        initialEdit={pageAnalysis.gm_notes}
                      />
                    ) : null}

                    {(pageAnalysis.npcs || []).map((npc, i) => npc.name ? (
                      <ResultCard
                        key={`npc-${i}`}
                        headerColor="#bf5a5a" headerLabel={`NPC — ${npc.name}`}
                        body={[npc.role, npc.description, npc.personality].filter(Boolean).join(" · ")}
                        badges={[npc.hp && `HP ${npc.hp}`, npc.ac && `AC ${npc.ac}`, npc.cr && `CR ${npc.cr}`].filter(Boolean)}
                        onAccept={() => acceptNpc({ id: createId("wf_npc"), ...npc })}
                        editingCard={editingCard} cardKey={`npc-${i}`} setEditingCard={setEditingCard}
                        editingText={editingText} setEditingText={setEditingText}
                        onAcceptEdited={() => acceptNpc({ id: createId("wf_npc"), ...npc })}
                        initialEdit=""
                        hideEdit
                      />
                    ) : null)}

                    {(pageAnalysis.monsters || []).map((m, i) => m.name ? (
                      <ResultCard
                        key={`mon-${i}`}
                        headerColor="#c9a227" headerLabel={`Monster — ${m.name}`}
                        body={m.notes || ""}
                        badges={[m.hp && `HP ${m.hp}`, m.ac && `AC ${m.ac}`, m.cr && `CR ${m.cr}`].filter(Boolean)}
                        onAccept={() => acceptMonster({ id: createId("wf_mon"), ...m })}
                        editingCard={editingCard} cardKey={`mon-${i}`} setEditingCard={setEditingCard}
                        editingText={editingText} setEditingText={setEditingText}
                        onAcceptEdited={() => acceptMonster({ id: createId("wf_mon"), ...m })}
                        initialEdit=""
                        hideEdit
                      />
                    ) : null)}

                    {!pageAnalysis.scene_title && !pageAnalysis.read_aloud && !pageAnalysis.gm_notes && !(pageAnalysis.npcs || []).length && !(pageAnalysis.monsters || []).length && (
                      <p className="text-xs text-[#7a6348] italic m-0">Nothing relevant found on this page.</p>
                    )}
                  </>
                )}
              </div>

              {pageAnalysis && !isAnalyzing && (
                <div className="shrink-0 px-3 py-2 border-t border-[#3a2510] flex flex-col gap-1.5">
                  <button type="button" className="nav-glyph-btn intake-parse-btn is-active w-full text-xs" onClick={acceptAllAndNext}>
                    Accept all & next page →
                  </button>
                  <button type="button" className="text-xs font-heading px-3 py-1.5 rounded border border-[#5a5a5a] bg-[#2a2a2a]/40 text-[#aaa] hover:bg-[#333]/60 w-full" onClick={skipPage}>
                    Skip page →
                  </button>
                </div>
              )}
            </section>
          </div>
        </div>
      )}

      {/* ─── Step 3: Review & correct ───────────────────────────── */}
      {step === 3 && (
        <div className="flex flex-col flex-1 min-h-0 min-w-0 gap-2">
          <div className="flex flex-wrap gap-2 justify-center shrink-0">
            <button type="button" className="nav-glyph-btn intake-parse-btn" onClick={handleStartOver}>← Upload another document</button>
          </div>

          <div className="flex flex-1 min-h-0 min-w-0 gap-0 border border-[#5a3e1b] rounded-lg overflow-hidden bg-[#120a04]" style={{ minHeight: "min(70dvh, 720px)", maxHeight: "calc(100dvh - 12rem)" }}>
            <aside className="shrink-0 w-[220px] flex flex-col border-r border-[#5a3e1b] bg-[#1a1008]">
              <div className="px-3 py-2 border-b border-[#3a2510]">
                <p className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading m-0">{reviewItems.length} scenes found</p>
              </div>
              <div className="flex-1 overflow-y-auto">
                {reviewItems.map((item, i) => {
                  const dotColor = item.status === "approved" ? "#6b8f6b" : item.status === "skipped" ? "#6b6b6b" : "#c9a227";
                  const isActive = i === activeReviewIdx;
                  return (
                    <button key={item.id} type="button" onClick={() => setActiveReviewIdx(i)}
                      className={["w-full text-left px-3 py-2 text-xs border-b border-[#2a1a08] transition-colors flex items-start gap-2", isActive ? "bg-[#3d2814]" : "hover:bg-[#1a0f06]"].join(" ")}
                      style={{ background: isActive ? "#3d2814" : undefined }}>
                      <span className="shrink-0 mt-1 inline-block w-2 h-2 rounded-full" style={{ background: dotColor }} />
                      <span className={isActive ? "text-[#f0d78c] font-heading" : "text-[#c4a574] font-heading"}>{item.title || `Scene ${i + 1}`}</span>
                    </button>
                  );
                })}
              </div>
              <div className="px-3 py-2 border-t border-[#3a2510]">
                <button type="button" className="w-full text-xs font-heading text-[#d8b36f] border border-dashed border-[#6b5030] rounded px-2 py-1 hover:border-[#9b7440]" onClick={addBlankScene}>+ Add scene</button>
              </div>
            </aside>

            <section className="flex flex-1 min-w-0 min-h-0 flex-col">
              <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
                {activeReview ? (
                  <>
                    {activeReview.imageUrl ? (
                      <div className="flex items-center gap-3 p-2 rounded border border-[#3a2510] bg-[#0e0804]">
                        <img src={activeReview.imageUrl} alt="" className="rounded object-contain" style={{ maxHeight: 80 }} draggable={false} />
                        <span className="text-[10px] text-[#7a6348] font-heading">Scene image from document</span>
                      </div>
                    ) : null}
                    <label className="block">
                      <span className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading">Scene title</span>
                      <input type="text" value={activeReview.title} onChange={(e) => updateReviewField(activeReviewIdx, { title: e.target.value })} className="mt-1 w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-2 text-sm text-[#e7c27a]" />
                    </label>
                    <label className="block">
                      <span className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading inline-flex items-center gap-2">
                        Read-aloud text
                        {activeReview.readAloud && <span className="text-[9px] normal-case px-1.5 py-0.5 rounded border border-amber-700/50 bg-amber-950/30 text-amber-300">AI extracted</span>}
                      </span>
                      <textarea value={activeReview.readAloud} onChange={(e) => updateReviewField(activeReviewIdx, { readAloud: e.target.value })} rows={6} className="mt-1 w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-2 text-sm text-[#e7c27a] resize-y min-h-[120px]" />
                    </label>
                    <label className="block">
                      <span className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading">GM notes</span>
                      <textarea value={activeReview.gmNotes} onChange={(e) => updateReviewField(activeReviewIdx, { gmNotes: e.target.value })} rows={4} className="mt-1 w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-2 text-sm text-[#e7c27a] resize-y min-h-[80px]" />
                    </label>
                    <label className="block">
                      <span className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading">Scene type</span>
                      <select value={activeReview.type} onChange={(e) => updateReviewField(activeReviewIdx, { type: e.target.value })} className="mt-1 w-full bg-[#1a0f06] text-[#e7c27a] border border-[#5a3e1b] rounded px-2 py-2 text-sm cursor-pointer">
                        {SCENE_TYPES.map((t) => (<option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>))}
                      </select>
                    </label>
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading mb-2">NPCs</div>
                      <div className="space-y-2">
                        {activeReview.npcs.map((npc) => (
                          <div key={npc.id} className="flex flex-col sm:flex-row gap-2 border border-[#4f341f] rounded p-2 bg-[rgba(32,18,8,0.5)] items-start sm:items-center">
                            <input type="text" value={npc.name} onChange={(e) => updateReviewNpc(activeReviewIdx, npc.id, { name: e.target.value })} placeholder="Name" className="flex-1 min-w-0 bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-1.5 text-sm text-[#e7c27a]" />
                            <input type="text" value={npc.role} onChange={(e) => updateReviewNpc(activeReviewIdx, npc.id, { role: e.target.value })} placeholder="Role" className="flex-1 min-w-0 bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-1.5 text-sm text-[#e7c27a]" />
                            <button type="button" className="text-xs text-[#a08060] hover:text-[#e7c27a] shrink-0" onClick={() => removeReviewNpc(activeReviewIdx, npc.id)} title="Remove NPC">×</button>
                          </div>
                        ))}
                      </div>
                      <button type="button" className="mt-2 text-xs font-heading text-[#d8b36f] border border-dashed border-[#6b5030] rounded px-2 py-1 hover:border-[#9b7440]" onClick={() => addReviewNpc(activeReviewIdx)}>+ Add NPC</button>
                    </div>
                    {activeReview.monsters?.length > 0 ? (
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-[#9c7a3a] font-heading mb-2">Monsters & villains</div>
                        <div className="space-y-1">
                          {activeReview.monsters.map((m) => (
                            <div key={m.id} className="flex flex-wrap items-center gap-2 px-2 py-1.5 rounded border border-[#3a2510] bg-[#0e0804]">
                              <span className="text-sm text-[#e7c27a] font-heading">{m.name}</span>
                              {m.role ? <span className="text-[9px] uppercase px-1.5 py-0.5 rounded border border-red-800/50 bg-red-950/30 text-red-300">{m.role}</span> : null}
                              {m.hp ? <span className="text-[9px] px-1.5 py-0.5 rounded border border-[#5a3e1b] bg-[#1a0f06] text-[#b89a62]">HP {m.hp}</span> : null}
                              {m.ac ? <span className="text-[9px] px-1.5 py-0.5 rounded border border-[#5a3e1b] bg-[#1a0f06] text-[#b89a62]">AC {m.ac}</span> : null}
                              {m.cr ? <span className="text-[9px] px-1.5 py-0.5 rounded border border-[#5a3e1b] bg-[#1a0f06] text-[#b89a62]">CR {m.cr}</span> : null}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    <div className="flex flex-wrap gap-2 pt-2 border-t border-[#3a2510]">
                      <button type="button" className="text-xs font-heading px-3 py-1.5 rounded border border-[#5a5a5a] bg-[#2a2a2a]/40 text-[#aaa] hover:bg-[#333]/60" onClick={() => markScene("skipped")}>Skip scene</button>
                      <button type="button" className="text-xs font-heading px-3 py-1.5 rounded border border-green-700/60 bg-green-950/40 text-green-300 hover:bg-green-900/50" onClick={() => markScene("approved")}>Looks good ✓</button>
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-[#7a6348] italic m-0">Select a scene from the sidebar.</p>
                )}
              </div>
              <div className="shrink-0 px-4 py-3 border-t border-[#3a2510] bg-[#1a1008]">
                <button type="button" className={["nav-glyph-btn intake-parse-btn is-active w-full", allReviewed ? "library-pulse-gold" : ""].join(" ")} onClick={handleSaveAll}>Save all & go to Prep →</button>
              </div>
            </section>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── ResultCard component for Step 2 AI results ───────────────────── */

function ResultCard({ headerColor, headerLabel, title, body, badges, onAccept, editingCard, cardKey, setEditingCard, editingText, setEditingText, onAcceptEdited, initialEdit, hideEdit }) {
  const isEditing = editingCard === cardKey;
  return (
    <div className="rounded-md border border-[#3a2510] bg-[#0e0804] overflow-hidden">
      <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider font-heading" style={{ color: headerColor }}>{headerLabel}</div>
      <div className="px-3 pb-2">
        {title && <p className="text-sm text-[#e7c27a] font-heading m-0 mb-1">{title}</p>}
        {body && !isEditing && <p className="text-xs text-[#d8c4a0] leading-relaxed m-0 mb-2 whitespace-pre-wrap">{body.length > 300 ? body.slice(0, 300) + "…" : body}</p>}
        {badges?.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {badges.map((b, i) => (<span key={i} className="text-[9px] px-1.5 py-0.5 rounded border border-[#5a3e1b] bg-[#1a0f06] text-[#b89a62]">{b}</span>))}
          </div>
        )}
        {isEditing && (
          <div className="space-y-1 mb-2">
            <textarea value={editingText} onChange={(e) => setEditingText(e.target.value)} rows={4} className="w-full bg-[#1a0f06] border border-[#5a3e1b] rounded px-2 py-1.5 text-xs text-[#e7c27a] resize-y" />
            <button type="button" className="text-xs font-heading px-2 py-1 rounded border border-green-700/60 bg-green-950/40 text-green-300 hover:bg-green-900/50"
              onClick={() => { onAcceptEdited(editingText); setEditingCard(null); }}>Done</button>
          </div>
        )}
        {!isEditing && (
          <div className="flex items-center gap-2">
            <button type="button" className="text-[11px] font-heading px-2 py-1 rounded border border-green-700/60 bg-green-950/40 text-green-300 hover:bg-green-900/50" onClick={onAccept}>Accept</button>
            {!hideEdit && (
              <button type="button" className="text-[11px] font-heading px-2 py-1 rounded border border-amber-700/50 bg-amber-950/30 text-amber-300 hover:bg-amber-900/40"
                onClick={() => { setEditingCard(cardKey); setEditingText(initialEdit || body || title || ""); }}>Edit</button>
            )}
            <button type="button" className="text-[11px] font-heading px-2 py-1 rounded border border-[#5a5a5a] bg-[#2a2a2a]/40 text-[#aaa] hover:bg-[#333]/60"
              onClick={() => {}}>Skip</button>
          </div>
        )}
      </div>
    </div>
  );
}
