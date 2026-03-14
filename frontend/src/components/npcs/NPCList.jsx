/**
 * Scrollable list of NPCs; header matches CodexResultList. EmptyState when none.
 */
import { NPCWorkshopEmptyState } from "../shared";
import NPCListItem from "./NPCListItem";

export default function NPCList({ filteredNpcs = [], selectedNpc, onSelectNpc }) {
  if (!filteredNpcs.length) {
    return (
      <div className="flex-1 min-h-0 flex items-center justify-center p-4">
        <NPCWorkshopEmptyState />
      </div>
    );
  }
  return (
    <div className="flex flex-col min-h-0 flex-1">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-2 shrink-0">
        Saved ({filteredNpcs.length})
      </div>
      <div className="space-y-1 overflow-y-auto flex-1 min-h-0">
        {filteredNpcs.map((npc) => (
          <NPCListItem
            key={npc.id}
            npc={npc}
            selected={selectedNpc?.id === npc.id}
            onSelect={onSelectNpc}
          />
        ))}
      </div>
    </div>
  );
}
