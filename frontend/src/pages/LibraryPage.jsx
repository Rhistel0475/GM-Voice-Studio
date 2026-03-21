import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, Zap } from "lucide-react";
import PrepPanel from "../components/prep/PrepPanel";
import ExtractionReviewQueue from "../components/intake/ExtractionReviewQueue";
import { parseResultToExtractionBatch } from "../lib/parseResultToExtractionBatch";
import { useExtractionReviewQueueStore } from "../store/extractionReview";
import { setBackendCampaignId } from "../lib/campaignPersistence";
import {
  DEFAULT_GAME_SYSTEM_ID,
  listGameSystemPlugins,
  normalizeGameSystemId,
  normalizeGameSystemPlugin,
  resolveGameSystemPlugin,
} from "../lib/gameSystemPlugins";

export default function ImportPage() {
  const navigate = useNavigate();
  const clearQueue = useExtractionReviewQueueStore((s) => s.clearQueue);
  const enqueueBatch = useExtractionReviewQueueStore((s) => s.enqueueBatch);

  const [step, setStep] = useState(1);
  const [files, setFiles] = useState([]);
  const [isParsing, setIsParsing] = useState(false);
  const [parseError, setParseError] = useState("");
  const [documentName, setDocumentName] = useState("");

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

  const authFetch = useCallback(
    (input, init = {}) => {
      const headers = new Headers(init.headers || {});
      const key = apiKey.trim();
      if (key) headers.set("X-API-Key", key);
      return fetch(input, { ...init, headers });
    },
    [apiKey]
  );

  const onFileChange = (e) => {
    setFiles(Array.from(e.target.files || []));
    setParseError("");
  };

  const deriveDocumentName = (payload) => {
    const fromTitle = typeof payload?.title === "string" ? payload.title.trim() : "";
    if (fromTitle) return fromTitle;
    const first = files[0]?.name;
    if (first) return first.replace(/\.[^.]+$/, "");
    return "Imported adventure";
  };

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

      clearQueue();
      const batch = parseResultToExtractionBatch(payload, docName);
      enqueueBatch(batch);

      setStep(2);
    } catch (err) {
      setParseError(err.message || "Unable to parse documents.");
    } finally {
      setIsParsing(false);
    }
  };

  const handleStartOver = () => {
    clearQueue();
    setStep(1);
    setParseError("");
    setDocumentName("");
  };

  const handleApplied = (result) => {
    if (result?.totalApplied > 0) navigate("/prep");
  };

  return (
    <div className="dm-shell dm-fit prep-shell intake-shell mx-auto p-3 md:p-4">
      <header className="prep-header intake-header mb-4">
        <div className="header-glow" />
        <div className="relative z-10 text-center">
          <h1 className="font-heading text-[clamp(1.5rem,2vw,2.25rem)] leading-tight text-[#e7c27a]">
            Import adventure
          </h1>
          <p className="font-heading text-sm text-[#d8b36f] mt-1">
            Step {step} of 2 — {step === 1 ? "Upload & parse" : "Review extractions"}
          </p>
        </div>
      </header>

      {requireApiKey && (
        <div className="mb-4 max-w-xl mx-auto">
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
        <div className="max-w-xl mx-auto">
          <PrepPanel title="Upload adventure docs">
            <p className="intake-hint">
              Drop in session notes, module PDFs, or campaign text. AI Parse uses Claude to extract full
              campaign data.
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
            </div>

            <div className="flex flex-col gap-2 mt-2">
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn is-active"
                onClick={() => runParse("/adventure/ai-parse")}
                disabled={isParsing}
              >
                <Zap size={14} className="inline mr-1" />
                {isParsing ? "Parsing with AI…" : "AI Parse (Claude)"}
              </button>
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn"
                onClick={() => runParse("/adventure/parse")}
                disabled={isParsing}
              >
                {isParsing ? "Parsing…" : "Quick Parse (fast)"}
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
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2 justify-center">
            <button type="button" className="nav-glyph-btn intake-parse-btn" onClick={handleStartOver}>
              ← Upload another document
            </button>
          </div>
          <ExtractionReviewQueue
            documentName={documentName}
            onApplied={handleApplied}
          />
        </div>
      )}
    </div>
  );
}
