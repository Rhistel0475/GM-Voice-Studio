import React from "react";
import { BookOpen, Users, Mic, LayoutDashboard } from "lucide-react";
import EmptyState from "./EmptyState";

/**
 * Empty state for Codex when no documents or campaign data exist.
 * Use when the research list is empty (no campaign loaded or no items).
 */
export function CodexEmptyState({ action }) {
  return (
    <EmptyState
      icon={BookOpen}
      message="No research yet. Load a campaign from Library or add documents to build your codex."
      action={action}
    />
  );
}

/**
 * Empty state for NPC Workshop when no NPCs exist.
 * Use when the roster is empty (no campaign or no NPCs parsed).
 */
export function NPCWorkshopEmptyState({ action }) {
  return (
    <EmptyState
      icon={Users}
      message="No NPCs yet. Create one with the generator or load a campaign from Library to import characters."
      action={action}
    />
  );
}

/**
 * Empty state for Voice Studio when no voices exist.
 * Use when the voice library is empty (no cloned or system voices).
 */
export function VoiceStudioEmptyState({ action }) {
  return (
    <EmptyState
      icon={Mic}
      message="No voices yet. Clone a voice from a sample or wait for the server to load the voice library."
      action={action}
    />
  );
}

/**
 * Empty state for Live Board when no session/scene data.
 * Use when there is no active scene or campaign (e.g. before starting a session).
 */
export function LiveBoardEmptyState({ action }) {
  return (
    <EmptyState
      icon={LayoutDashboard}
      message="No active session. Select a campaign and scene to run your live board, or start from the sidebar."
      action={action}
    />
  );
}
