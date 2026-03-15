import { useEffect, useRef, useState } from "react";
import { BookOpenText, Search, Upload, Volume2 } from "lucide-react";
import { FantasyButton } from "../shared";

export default function CampaignBrainPanel({
  campaignId,
  authFetch,
  documents = [],
  onDocumentsChange,
  onNarrateAnswer,
}) {
  const fileInputRef = useRef(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [matchedMetadata, setMatchedMetadata] = useState(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [narrating, setNarrating] = useState(false);

  useEffect(() => {
    setAnswer("");
    setSources([]);
    setMatchedMetadata(null);
    setStatus("");
    setError("");
  }, [campaignId]);

  useEffect(() => {
    if (!campaignId || !authFetch || !onDocumentsChange) return undefined;
    let cancelled = false;
    authFetch(`/api/campaigns/${campaignId}/documents`)
      .then((response) => (response.ok ? response.json() : { documents: [] }))
      .then((payload) => {
        if (!cancelled && Array.isArray(payload?.documents)) {
          onDocumentsChange(payload.documents);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [authFetch, campaignId, onDocumentsChange]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!campaignId) {
      setError("Load a saved campaign before indexing documents.");
      return;
    }
    if (!files.length) return;

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setUploading(true);
    setError("");
    setStatus("");
    try {
      const response = await authFetch(`/api/campaigns/${campaignId}/documents`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || "Document indexing failed.");
      }
      const uploadedDocs = Array.isArray(payload?.documents) ? payload.documents : [];
      if (uploadedDocs.length > 0) {
        onDocumentsChange?.((current = []) => {
          const merged = [...uploadedDocs, ...current];
          const seen = new Set();
          return merged.filter((doc) => {
            const key = String(doc?.id || doc?.filename || doc?.title || "");
            if (!key || seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        });
      }
      setStatus(
        `Indexed ${payload?.total_documents || uploadedDocs.length || files.length} document${(payload?.total_documents || uploadedDocs.length || files.length) === 1 ? "" : "s"}.`
      );
    } catch (err) {
      setError(err?.message || "Document indexing failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async () => {
    if (!campaignId) {
      setError("Load a saved campaign before using Campaign Brain.");
      return;
    }
    if (!question.trim()) {
      setError("Enter a question or search prompt first.");
      return;
    }

    setQuerying(true);
    setError("");
    setStatus("");
    try {
      const response = await authFetch("/campaign/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          campaign_id: campaignId,
          question: question.trim(),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || "Campaign Brain query failed.");
      }
      setAnswer(payload?.answer || "");
      setSources(Array.isArray(payload?.relevant_chunks) ? payload.relevant_chunks : []);
      setMatchedMetadata(payload?.matched_metadata || null);
      if (payload?.documents_indexed) {
        setStatus(`Searched ${payload.documents_indexed} indexed document${payload.documents_indexed === 1 ? "" : "s"}.`);
      }
    } catch (err) {
      setAnswer("");
      setSources([]);
      setMatchedMetadata(null);
      setError(err?.message || "Campaign Brain query failed.");
    } finally {
      setQuerying(false);
    }
  };

  const handleNarrate = async () => {
    if (!answer.trim() || !campaignId || !onNarrateAnswer) return;
    setNarrating(true);
    setError("");
    try {
      await onNarrateAnswer({ campaignId, answer: answer.trim() });
    } catch (err) {
      setError(err?.message || "Answer narration failed.");
    } finally {
      setNarrating(false);
    }
  };

  return (
    <section className="rounded-lg border border-[#6d4b29] bg-[#150d07]/95 p-3 shadow-[0_0_14px_rgba(0,0,0,0.22)]">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.md,.markdown,.txt"
        multiple
        className="hidden"
        onChange={handleFileChange}
      />

      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[var(--gold)] font-heading text-sm">
            <BookOpenText size={14} />
            <span>Campaign Brain</span>
          </div>
          <div className="text-[11px] text-[var(--text-2)] mt-1">
            {documents.length > 0
              ? `${documents.length} indexed source document${documents.length === 1 ? "" : "s"}`
              : "Index PDFs, DOCX files, or markdown notes for lore retrieval."}
          </div>
        </div>
        <FantasyButton
          variant="secondary"
          className="text-[11px] whitespace-nowrap"
          onClick={handleUploadClick}
          disabled={!campaignId || uploading}
        >
          <Upload size={12} className="inline mr-1" />
          {uploading ? "Indexing..." : "Upload Docs"}
        </FantasyButton>
      </div>

      <textarea
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        rows={3}
        className="w-full rounded-md border border-[#6d4b29] bg-[#23160d] px-3 py-2 text-sm text-[var(--text-1)] placeholder:text-[var(--text-2)] focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
        placeholder="Ask about NPC motives, scenes, quests, locations, or lore..."
      />

      <div className="mt-2 flex flex-wrap gap-2">
        <FantasyButton
          variant="primary"
          className="text-xs"
          onClick={handleQuery}
          disabled={!campaignId || querying}
        >
          <Search size={12} className="inline mr-1" />
          {querying ? "Searching..." : "Ask Campaign Brain"}
        </FantasyButton>
        <FantasyButton
          variant="ghost"
          className="text-xs"
          onClick={handleNarrate}
          disabled={!answer.trim() || narrating}
        >
          <Volume2 size={12} className="inline mr-1" />
          {narrating ? "Narrating..." : "Narrate Answer"}
        </FantasyButton>
      </div>

      {status ? <div className="mt-2 text-[11px] text-[#d7bb7a]">{status}</div> : null}
      {error ? <div className="mt-2 text-[11px] text-[#ff9d82]">{error}</div> : null}

      {answer ? (
        <div className="mt-3 rounded-md border border-[#5c4228] bg-[#1b120b] p-3">
          <div className="text-[11px] uppercase tracking-[0.18em] text-[var(--gold)] mb-1">Answer</div>
          <div className="text-sm text-[var(--text-1)] whitespace-pre-wrap max-h-32 overflow-auto">{answer}</div>
        </div>
      ) : null}

      {matchedMetadata && (
        <div className="mt-2 flex flex-wrap gap-2">
          {["npcs", "locations", "quests", "items"].map((key) => {
            const values = Array.isArray(matchedMetadata?.[key]) ? matchedMetadata[key] : [];
            if (!values.length) return null;
            return (
              <div key={key} className="rounded-md border border-[#5c4228] bg-[#1b120b] px-2 py-1 text-[11px] text-[var(--text-2)]">
                <span className="text-[var(--gold)] mr-1">{key}:</span>
                {values.slice(0, 3).join(", ")}
              </div>
            );
          })}
        </div>
      )}

      {sources.length > 0 ? (
        <div className="mt-3 rounded-md border border-[#5c4228] bg-[#120c08] p-2">
          <div className="text-[11px] uppercase tracking-[0.18em] text-[var(--gold)] mb-2">Sources</div>
          <div className="space-y-2 max-h-36 overflow-auto">
            {sources.slice(0, 4).map((source) => (
              <div key={source.chunk_id || `${source.filename}-${source.score}`} className="rounded-md border border-[#4e361f] bg-[#1b120b] px-2 py-2">
                <div className="flex items-center justify-between gap-3 text-[11px] text-[var(--text-2)] mb-1">
                  <span className="truncate">{source.filename || "Campaign Document"}</span>
                  <span>{typeof source.score === "number" ? source.score.toFixed(2) : "--"}</span>
                </div>
                <div className="text-[11px] text-[var(--text-2)] max-h-16 overflow-hidden">
                  {source.text || ""}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
