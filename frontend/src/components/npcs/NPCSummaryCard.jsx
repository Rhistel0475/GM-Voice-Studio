/**
 * Summary block: role, profession, faction, location, summary. Dossier-style typography.
 */
export default function NPCSummaryCard({ profile }) {
  if (!profile) return null;
  const summary = profile.summary || "";
  const role = profile.role || "";
  const profession = profile.profession || "";
  const faction = profile.faction || "";
  const location = profile.location || "";
  const hasMeta = role || profession || faction || location;

  return (
    <div className="border-b border-[#5c3e23] pb-3 last:border-0 last:pb-0">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-1">
        Summary
      </div>
      {hasMeta && (
        <div className="text-xs text-[var(--text-2)] space-y-0.5 mb-2">
          {role && <p><span className="text-[var(--candle-glow)]">Role:</span> {role}</p>}
          {profession && role !== profession && <p><span className="text-[var(--candle-glow)]">Profession:</span> {profession}</p>}
          {faction && <p><span className="text-[var(--candle-glow)]">Faction:</span> {faction}</p>}
          {location && <p><span className="text-[var(--candle-glow)]">Location:</span> {location}</p>}
        </div>
      )}
      {summary ? (
        <p className="text-sm text-[var(--ink-1)] whitespace-pre-wrap">{summary}</p>
      ) : (
        <p className="text-sm text-[var(--text-2)] italic">No summary yet.</p>
      )}
    </div>
  );
}
