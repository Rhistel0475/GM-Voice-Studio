import { Crown } from "lucide-react";
import { StatusPill } from "../shared";

const SOURCE_LABELS = { system: "System", cloned: "Cloned", custom: "Custom" };
const TONE_LABELS = {
  warm: "Warm", grim: "Grim", noble: "Noble", mysterious: "Mysterious", rough: "Rough", neutral: "Neutral",
};

/**
 * Voice metadata: name, source, status, accent, tone, tags, description.
 */
export default function VoiceMetadataCard({ voice }) {
  if (!voice) return null;
  const source = voice.source;
  const status = voice.status;
  const accent = voice.accent;
  const tone = voice.tone;
  const tags = voice.tags || [];
  const description = voice.description;
  const updatedAt = voice.updatedAt;
  const featured = Boolean(voice.featured);

  const statusVariant = status === "ready" ? "ready" : status === "training" ? "generating" : status === "failed" ? "recording" : "offline";

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-heading uppercase tracking-wider" style={{ color: "#6b3e10" }}>Details</p>
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill status={statusVariant} />
        {featured && (
          <span
            className="text-xs font-heading border px-2 py-0.5 rounded inline-flex items-center gap-1"
            style={{ color: "#6b3e10", borderColor: "#a17a42", background: "rgba(202,167,75,0.18)" }}
          >
            <Crown size={12} aria-hidden />
            Master Voice
          </span>
        )}
        {source && (
          <span
            className="text-xs font-heading border px-2 py-0.5 rounded"
            style={{ color: "#6b3e10", borderColor: "#a17a42", background: "rgba(202,167,75,0.18)" }}
          >
            {SOURCE_LABELS[source] || source}
          </span>
        )}
      </div>
      {(accent || tone) && (
        <div className="flex flex-wrap gap-2 text-xs" style={{ color: "#3a1e08" }}>
          {accent && <span>Accent: {accent}</span>}
          {tone && <span>Tone: {TONE_LABELS[tone] || tone}</span>}
        </div>
      )}
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {tags.map((t) => (
            <span
              key={t}
              className="text-[10px] px-1.5 py-0.5 rounded border border-[#a17a42]"
              style={{ color: "#3a1e08", background: "rgba(0,0,0,0.06)" }}
            >
              {t}
            </span>
          ))}
        </div>
      )}
      {description && (
        <p className="text-sm leading-relaxed" style={{ color: "#3a1e08" }}>{description}</p>
      )}
      {updatedAt && (
        <p className="text-xs opacity-70" style={{ color: "#3a1e08" }}>Updated {updatedAt}</p>
      )}
    </div>
  );
}
