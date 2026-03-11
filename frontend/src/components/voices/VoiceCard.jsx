import { FantasyButton } from "../shared";
import { Play, UserPlus } from "lucide-react";
import { getVoicePlaceholder } from "../../lib/placeholders";

/**
 * Single voice card for the library grid. Supports voice_id or id; shows status. Optional onSelect for click-to-select.
 */
export default function VoiceCard({ voice, selected, onPlaySample, onAssign, onSelect }) {
  const id = voice?.voice_id || voice?.id;
  const name = voice?.name?.trim() || id || "Unknown";
  const status = voice?.status;
  const source = voice?.source;
  const tone = voice?.tone;
  const placeholderSrc = getVoicePlaceholder(tone);

  const handleCardClick = () => onSelect?.(voice);

  return (
    <div
      role={onSelect ? "button" : undefined}
      tabIndex={onSelect ? 0 : undefined}
      onClick={onSelect ? handleCardClick : undefined}
      onKeyDown={onSelect ? (e) => e.key === "Enter" && handleCardClick() : undefined}
      className={`portrait-card flex flex-col items-center rounded border p-2 transition-colors ${
        selected ? "border-[var(--gold)] bg-[rgba(202,167,75,0.12)] ring-1 ring-[var(--gold)]/40" : "border-[#6f4d2a]"
      } ${onSelect ? "cursor-pointer hover:border-[var(--gold)]/60" : ""}`}
    >
      <div className="portrait-slot w-full overflow-hidden flex items-center justify-center">
        <img
          src={placeholderSrc}
          alt=""
          className="w-full aspect-square object-cover"
        />
      </div>
      <p className="mt-1 text-sm font-heading text-[var(--text-1)] text-center line-clamp-2">{name}</p>
      {(status && status !== "ready") || source ? (
        <span className="text-[10px] uppercase tracking-wide text-[var(--text-2)] mt-0.5">
          {status && status !== "ready" ? status : source}
        </span>
      ) : null}
      <FantasyButton
        variant="secondary"
        className="text-xs mt-1 w-full"
        onClick={(e) => { e.stopPropagation(); onPlaySample?.(id); }}
      >
        <Play size={12} className="inline mr-1" /> Play sample
      </FantasyButton>
      {onAssign && (
        <FantasyButton variant="ghost" className="text-xs mt-1 w-full" onClick={(e) => { e.stopPropagation(); onAssign(voice); }}>
          <UserPlus size={12} className="inline mr-1" /> Assign to NPC
        </FantasyButton>
      )}
    </div>
  );
}
