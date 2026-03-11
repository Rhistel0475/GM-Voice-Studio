import { UserPlus } from "lucide-react";
import { FantasyButton } from "../shared";

/**
 * Assign voice to NPC: dropdown of NPCs and list of currently assigned.
 */
export default function VoiceAssignmentPanel({
  voice,
  npcOptions = [],
  onAssign,
  onUnassign,
  disabled,
}) {
  const voiceId = voice?.voice_id || voice?.id;
  const assignedNPCs = voice?.assignedNPCs || [];
  const [selectedNpcId, setSelectedNpcId] = React.useState("");

  const handleAssign = () => {
    if (!selectedNpcId || !onAssign) return;
    onAssign(voiceId, selectedNpcId);
    setSelectedNpcId("");
  };

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">Assign to NPC</p>
      <div className="flex gap-2">
        <select
          className="chat-input flex-1 min-w-0 text-sm"
          value={selectedNpcId}
          onChange={(e) => setSelectedNpcId(e.target.value)}
          disabled={disabled}
        >
          <option value="">Select NPC…</option>
          {npcOptions.map((npc) => (
            <option key={npc.id} value={npc.id}>
              {npc.name || npc.id}
            </option>
          ))}
        </select>
        <FantasyButton
          variant="secondary"
          className="shrink-0"
          onClick={handleAssign}
          disabled={!selectedNpcId || disabled}
        >
          <UserPlus size={14} />
        </FantasyButton>
      </div>
      {assignedNPCs.length > 0 && (
        <ul className="text-sm text-[var(--ink-1)] space-y-1">
          <p className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-1">Assigned</p>
          {assignedNPCs.map((npcId) => (
            <li key={npcId} className="flex items-center justify-between gap-2">
              <span>{npcId}</span>
              {onUnassign && (
                <button
                  type="button"
                  className="text-xs text-[var(--gold)] hover:underline"
                  onClick={() => onUnassign(voiceId, npcId)}
                >
                  Remove
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
