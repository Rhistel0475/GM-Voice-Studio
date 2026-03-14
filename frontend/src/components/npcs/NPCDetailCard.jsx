import { useState } from "react";
import { EmptyState } from "../shared";
import { ChevronDown, ChevronRight } from "lucide-react";

/**
 * Details card: traits, goals, quirks; secrets in a collapsed section.
 * Accepts NPCProfile shape or legacy npc with personality.
 */
export default function NPCDetailCard({ npc }) {
  const [secretsOpen, setSecretsOpen] = useState(false);
  if (!npc) {
    return <EmptyState message="Select an NPC to view details." />;
  }
  const traits = npc.personalityTraits || (npc.personality ? [npc.personality] : []);
  const goals = npc.goals || [];
  const quirks = npc.quirks || [];
  const secrets = npc.secrets || [];
  const hasAny = traits.length > 0 || goals.length > 0 || quirks.length > 0 || secrets.length > 0;

  if (!hasAny) {
    return (
      <div className="border-b border-[#5c3e23] pb-3 last:border-0">
        <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-1">
          Details
        </div>
        <p className="text-sm text-[var(--text-2)]">{npc.role && `${npc.role}. `}No details yet.</p>
      </div>
    );
  }

  return (
    <div className="border-b border-[#5c3e23] pb-3 last:border-0">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-1">
        Details
      </div>
      {traits.length > 0 && (
        <div className="mb-2">
          <p className="text-[10px] text-[var(--text-2)] uppercase tracking-wide mb-0.5">Traits</p>
          <p className="text-sm text-[var(--ink-1)]">{traits.join(" · ")}</p>
        </div>
      )}
      {goals.length > 0 && (
        <div className="mb-2">
          <p className="text-[10px] text-[var(--text-2)] uppercase tracking-wide mb-0.5">Goals</p>
          <ul className="text-sm text-[var(--ink-1)] list-disc list-inside space-y-0.5">
            {goals.map((g, i) => (
              <li key={i}>{g}</li>
            ))}
          </ul>
        </div>
      )}
      {quirks.length > 0 && (
        <div className="mb-2">
          <p className="text-[10px] text-[var(--text-2)] uppercase tracking-wide mb-0.5">Quirks</p>
          <p className="text-sm text-[var(--ink-1)]">{quirks.join(" · ")}</p>
        </div>
      )}
      {secrets.length > 0 && (
        <div>
          <button
            type="button"
            className="flex items-center gap-1 text-xs text-[var(--text-2)] hover:text-[var(--gold)]"
            onClick={() => setSecretsOpen(!secretsOpen)}
          >
            {secretsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Secrets ({secrets.length})
          </button>
          {secretsOpen && (
            <ul className="text-sm text-[var(--ink-1)] list-disc list-inside mt-1 space-y-0.5 border-l border-[#5c3e23] pl-3">
              {secrets.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
