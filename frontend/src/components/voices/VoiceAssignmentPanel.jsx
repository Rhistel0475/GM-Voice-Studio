import { useState } from "react";
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
  const [selectedNpcId, setSelectedNpcId] = useState("");

  const handleAssign = () => {
    if (!selectedNpcId || !onAssign) return;
    onAssign(voiceId, selectedNpcId);
    setSelectedNpcId("");
  };

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-heading uppercase tracking-wider" style={{ color: "#6b3e10" }}>Assign to NPC</p>
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
        <ul className="text-sm space-y-1" style={{ color: "#3a1e08" }}>
          <p className="text-xs font-heading uppercase tracking-wider mb-1" style={{ color: "#6b3e10" }}>Assigned</p>
          {assignedNPCs.map((npcId) => {
            const npc = npcOptions.find((n) => n.id === npcId);
            return (
              <li key={npcId} className="flex items-center justify-between gap-2">
                <span>{npc?.name || npcId}</span>
                {onUnassign && (
                  <button
                    type="button"
                    className="text-xs hover:underline"
                    style={{ color: "#7a4010" }}
                    onClick={() => onUnassign(voiceId, npcId)}
                  >
                    Remove
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
