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

  const statusVariant = status === "ready" ? "ready" : status === "training" ? "generating" : status === "failed" ? "recording" : "offline";

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">Details</p>
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill status={statusVariant} />
        {source && (
          <span className="text-xs text-[var(--ink-1)] border border-[#8c6435] px-2 py-0.5 rounded bg-[rgba(0,0,0,0.08)]">
            {SOURCE_LABELS[source] || source}
          </span>
        )}
      </div>
      {(accent || tone) && (
        <div className="flex flex-wrap gap-2 text-xs text-[var(--ink-1)]">
          {accent && <span>Accent: {accent}</span>}
          {tone && <span>Tone: {TONE_LABELS[tone] || tone}</span>}
        </div>
      )}
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {tags.map((t) => (
            <span
              key={t}
              className="text-[10px] px-1.5 py-0.5 rounded border border-[#8c6435] text-[var(--ink-1)] bg-[rgba(0,0,0,0.06)]"
            >
              {t}
            </span>
          ))}
        </div>
      )}
      {description && (
        <p className="text-sm text-[var(--ink-1)] leading-relaxed">{description}</p>
      )}
      {updatedAt && (
        <p className="text-xs text-[var(--ink-1)] opacity-80">Updated {updatedAt}</p>
      )}
    </div>
  );
}
