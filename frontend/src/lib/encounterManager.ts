/**
 * Live Encounter Manager — load encounters, track enemies, initiative, and progress on the LiveBoard.
 */

import { create } from "zustand";
import type { ExtractedEncounter } from "../types";

export interface EncounterActor {
  id: string;
  name: string;
  type: "enemy" | "ally" | "neutral";
  initiative?: number;
  currentHp?: number;
  maxHp?: number;
  notes?: string;
}

export interface EncounterState {
  encounterId: string | null;
  encounter: ExtractedEncounter | null;
  actors: EncounterActor[];
  round: number;
  currentActorIndex: number;
  status: "idle" | "active" | "ended";
  startedAt: string | null;
}

const initialState: EncounterState = {
  encounterId: null,
  encounter: null,
  actors: [],
  round: 0,
  currentActorIndex: 0,
  status: "idle",
  startedAt: null,
};

export const useEncounterManagerStore = create<EncounterState & {
  startEncounter: (encounter: ExtractedEncounter, actors?: EncounterActor[]) => void;
  updateEncounterState: (updates: Partial<Pick<EncounterState, "actors" | "round" | "currentActorIndex">>) => void;
  endEncounter: () => void;
}>((set) => ({
  ...initialState,

  startEncounter(encounter, actors) {
    const defaultActors: EncounterActor[] = (encounter.enemyNpcRefs ?? []).map((ref, i) => ({
      id: `enemy-${encounter.id}-${i}`,
      name: ref.refName,
      type: "enemy",
    }));
    set({
      encounterId: encounter.id,
      encounter,
      actors: actors ?? defaultActors,
      round: 1,
      currentActorIndex: 0,
      status: "active",
      startedAt: new Date().toISOString(),
    });
  },

  updateEncounterState(updates) {
    set((state) => {
      if (state.status !== "active") return {};
      return {
        ...updates,
        actors: updates.actors ?? state.actors,
        round: updates.round ?? state.round,
        currentActorIndex: updates.currentActorIndex ?? state.currentActorIndex,
      };
    });
  },

  endEncounter() {
    set({
      ...initialState,
      status: "idle",
    });
  },
}));

/** Convenience: start encounter from store. */
export function startEncounter(encounter: ExtractedEncounter, actors?: EncounterActor[]): void {
  useEncounterManagerStore.getState().startEncounter(encounter, actors);
}

/** Convenience: update state. */
export function updateEncounterState(
  updates: Partial<Pick<EncounterState, "actors" | "round" | "currentActorIndex">>
): void {
  useEncounterManagerStore.getState().updateEncounterState(updates);
}

/** Convenience: end encounter. */
export function endEncounter(): void {
  useEncounterManagerStore.getState().endEncounter();
}
