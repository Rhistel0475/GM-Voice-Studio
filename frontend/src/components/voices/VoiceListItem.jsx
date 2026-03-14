import { Crown, Play } from "lucide-react";

const SOURCE_LABELS = { system: "System", cloned: "Cloned", custom: "Custom" };

/**
 * Single voice row for list view: name, source/tone pill, one-line description. Matches NPCListItem/CodexResultCard selected style.
 */
export default function VoiceListItem({ voice, selected, onSelect, onPlaySample }) {
  const id = voice?.voice_id || voice?.id;
  const name = voice?.name?.trim() || id || "Unknown";
  const source = voice?.source;
  const tone = voice?.tone;
  const featured = Boolean(voice?.featured);
  const pill = source ? SOURCE_LABELS[source] || source : tone || "";
  const excerpt = voice?.description || (voice?.tags?.length ? voice.tags.slice(0, 3).join(", ") : "");

  const handleRowClick = () => onSelect?.(voice);
  const handlePlay = (e) => {
    e.stopPropagation();
    onPlaySample?.(id);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      className={`w-full text-left border p-2 rounded transition-colors cursor-pointer ${
        selected
          ? "border-[var(--gold)] ring-1 ring-[var(--gold)] bg-[#1a1008]"
          : "border-[#5c3e23] bg-[#1a1008] hover:border-[var(--gold)]"
      }`}
      onClick={handleRowClick}
      onKeyDown={(e) => e.key === "Enter" && handleRowClick()}
    >
      <div className="flex items-center gap-2 flex-wrap">
        {onPlaySample && (
          <button
            type="button"
            className="shrink-0 p-1 rounded text-[var(--text-2)] hover:text-[var(--gold)] hover:bg-[rgba(202,167,75,0.12)]"
            onClick={handlePlay}
            title="Play sample"
            aria-label="Play sample"
          >
            <Play size={14} />
          </button>
        )}
        <span className="font-heading text-[var(--text-1)] text-sm">
          {featured ? <Crown size={13} className="inline mr-1 text-[var(--gold)]" aria-hidden /> : null}
          {name}
        </span>
        {featured && (
          <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded border border-[var(--gold)] text-[var(--gold)] shrink-0">
            Master Voice
          </span>
        )}
        {pill && (
          <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded border border-[#5c3e23] text-[var(--text-2)] shrink-0">
            {pill}
          </span>
        )}
      </div>
      {excerpt && (
        <div className="text-xs text-[var(--text-2)] line-clamp-2 mt-1">
          {excerpt}
        </div>
      )}
    </div>
  );
}
