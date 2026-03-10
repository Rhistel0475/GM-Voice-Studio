import React from "react";

/**
 * Wrapper for the main workspace content (where Live Board, Codex, NPC Workshop, Voice Studio render).
 * Use inside AppShell's main area for consistent padding and min-height.
 */
export default function WorkspaceContainer({ children, className = "" }) {
  return (
    <div
      className={`workspace-container flex flex-col min-h-0 flex-1 min-w-0 gap-4 ${className}`.trim()}
    >
      {children}
    </div>
  );
}
