import { create } from "zustand";
import type {
  Campaign,
  Session,
  Scene,
  Npc,
  Voice,
  CodexEntry,
  ActionLogEvent,
  ActionLogEventType,
  NarrationClip,
} from "../types";
import { createId, nowIso } from "../lib/utils/ids";
import { seedCampaignContext } from "./seedData";

const ACTION_LOG_MAX = 100;

export interface CampaignContextState {
  activeCampaignId: string | null;
  activeSessionId: string | null;
  activeSceneId: string | null;
  campaigns: Campaign[];
  sessions: Session[];
  scenes: Scene[];
  npcs: Npc[];
  voices: Voice[];
  codexEntries: CodexEntry[];
  actionLog: ActionLogEvent[];
  narrationClips: NarrationClip[];
}

const emptyState: CampaignContextState = {
  activeCampaignId: null,
  activeSessionId: null,
  activeSceneId: null,
  campaigns: [],
  sessions: [],
  scenes: [],
  npcs: [],
  voices: [],
  codexEntries: [],
  actionLog: [],
  narrationClips: [],
};

export interface AddActionLogEventInput {
  sessionId?: string;
  sceneId?: string;
  type: ActionLogEventType;
  text: string;
}

export interface AddNarrationClipInput {
  campaignId?: string;
  sessionId?: string;
  sceneId?: string;
  title: string;
  voiceId?: string;
  audioUrl?: string;
  duration?: number;
}

export const useCampaignContextStore = create<CampaignContextState & {
  setActiveCampaign: (id: string | null) => void;
  setActiveSession: (id: string | null) => void;
  setActiveScene: (id: string | null) => void;
  upsertSession: (session: Session) => void;
  assignVoiceToNpc: (npcId: string, voiceId: string) => void;
  unassignVoiceFromNpc: (npcId: string) => void;
  assignNpcToScene: (sceneId: string, npcId: string) => void;
  removeNpcFromScene: (sceneId: string, npcId: string) => void;
  addCodexEntryToScene: (sceneId: string, codexEntryId: string) => void;
  removeCodexEntryFromScene: (sceneId: string, codexEntryId: string) => void;
  attachNarrationClipToScene: (sceneId: string, clipId: string) => void;
  addActionLogEvent: (input: AddActionLogEventInput) => void;
  addNarrationClip: (input: AddNarrationClipInput) => void;
  upsertNpc: (npc: Npc) => void;
  upsertVoice: (voice: Voice) => void;
  upsertCodexEntry: (entry: CodexEntry) => void;
  upsertScene: (scene: Scene) => void;
  upsertCampaign: (campaign: Campaign) => void;
  setVoices: (voices: Voice[]) => void;
  hydrateFromSeed: () => void;
  resetCampaignContext: () => void;
}>((set) => ({
  ...emptyState,

  setActiveCampaign(id) {
    set((state) => {
      const campaign = state.campaigns.find((c) => c.id === id);
      const sessionId = id && campaign?.activeSessionId
        ? campaign.activeSessionId
        : state.sessions.find((s) => s.campaignId === id)?.id ?? null;
      const session = sessionId ? state.sessions.find((s) => s.id === sessionId) : null;
      const sceneId = session?.activeSceneId ?? state.scenes.find((s) => s.campaignId === id)?.id ?? null;
      return {
        activeCampaignId: id,
        activeSessionId: sessionId,
        activeSceneId: sceneId,
      };
    });
  },

  setActiveSession(id) {
    set((state) => {
      const session = id ? state.sessions.find((s) => s.id === id) : null;
      const sceneId = session?.activeSceneId ?? (id ? state.scenes.find((s) => s.sessionId === id)?.id : null) ?? null;
      return {
        activeSessionId: id,
        activeSceneId: sceneId ?? state.activeSceneId,
      };
    });
  },

  setActiveScene(id) {
    set({ activeSceneId: id });
  },

  upsertSession(session) {
    set((state) => {
      const idx = state.sessions.findIndex((s) => s.id === session.id);
      const sessions = idx >= 0
        ? state.sessions.map((s) => (s.id === session.id ? session : s))
        : [...state.sessions, session];
      return { sessions };
    });
  },

  assignVoiceToNpc(npcId, voiceId) {
    set((state) => {
      const npcs = state.npcs.map((n) =>
        n.id === npcId ? { ...n, voiceId } : n
      );
      const voices = state.voices.map((v) => {
        if (v.id === voiceId) {
          const assigned = v.assignedNpcIds.includes(npcId) ? v.assignedNpcIds : [...v.assignedNpcIds, npcId];
          return { ...v, assignedNpcIds: assigned };
        }
        if (v.assignedNpcIds.includes(npcId)) {
          return { ...v, assignedNpcIds: v.assignedNpcIds.filter((id) => id !== npcId) };
        }
        return v;
      });
      return { npcs, voices };
    });
    // TODO: Backend — PATCH npc/voice assignment
  },

  unassignVoiceFromNpc(npcId) {
    set((state) => {
      const npcs = state.npcs.map((n) => (n.id === npcId ? { ...n, voiceId: undefined } : n));
      const voices = state.voices.map((v) => ({
        ...v,
        assignedNpcIds: v.assignedNpcIds.filter((id) => id !== npcId),
      }));
      return { npcs, voices };
    });
  },

  assignNpcToScene(sceneId, npcId) {
    set((state) => {
      const scene = state.scenes.find((s) => s.id === sceneId);
      if (!scene || scene.npcIds.includes(npcId)) return {};
      const scenes = state.scenes.map((s) =>
        s.id === sceneId ? { ...s, npcIds: [...s.npcIds, npcId] } : s
      );
      return { scenes };
    });
  },

  removeNpcFromScene(sceneId, npcId) {
    set((state) => ({
      scenes: state.scenes.map((s) =>
        s.id === sceneId ? { ...s, npcIds: s.npcIds.filter((id) => id !== npcId) } : s
      ),
    }));
  },

  addCodexEntryToScene(sceneId, codexEntryId) {
    set((state) => {
      const scene = state.scenes.find((s) => s.id === sceneId);
      if (!scene) return {};

      const entry = state.codexEntries.find((e) => e.id === codexEntryId);

      const scenes =
        scene.codexEntryIds.includes(codexEntryId)
          ? state.scenes
          : state.scenes.map((s) =>
              s.id === sceneId
                ? { ...s, codexEntryIds: [...s.codexEntryIds, codexEntryId] }
                : s
            );

      const codexEntries =
        entry && !(entry.relatedSceneIds ?? []).includes(sceneId)
          ? state.codexEntries.map((e) =>
              e.id === codexEntryId
                ? { ...e, relatedSceneIds: [...(e.relatedSceneIds ?? []), sceneId] }
                : e
            )
          : state.codexEntries;

      return { scenes, codexEntries };
    });
  },

  removeCodexEntryFromScene(sceneId, codexEntryId) {
    set((state) => {
      const scene = state.scenes.find((s) => s.id === sceneId);
      if (!scene) return {};

      const scenes = state.scenes.map((s) =>
        s.id === sceneId
          ? { ...s, codexEntryIds: s.codexEntryIds.filter((id) => id !== codexEntryId) }
          : s
      );

      const codexEntries = state.codexEntries.map((e) =>
        e.id === codexEntryId && e.relatedSceneIds
          ? { ...e, relatedSceneIds: e.relatedSceneIds.filter((id) => id !== sceneId) }
          : e
      );

      return { scenes, codexEntries };
    });
  },

  attachNarrationClipToScene(sceneId, clipId) {
    set((state) => {
      const scene = state.scenes.find((s) => s.id === sceneId);
      if (!scene) return {};

      const clip = state.narrationClips.find((c) => c.id === clipId);

      const scenes =
        scene.narrationClipIds.includes(clipId)
          ? state.scenes
          : state.scenes.map((s) =>
              s.id === sceneId
                ? { ...s, narrationClipIds: [...s.narrationClipIds, clipId] }
                : s
            );

      const narrationClips =
        clip && clip.sceneId !== sceneId
          ? state.narrationClips.map((c) =>
              c.id === clipId ? { ...c, sceneId } : c
            )
          : state.narrationClips;

      return { scenes, narrationClips };
    });
  },

  addActionLogEvent(input) {
    set((state) => {
      const sessionId = input.sessionId ?? state.activeSessionId ?? "";
      const sceneId = input.sceneId ?? state.activeSceneId ?? undefined;
      const event: ActionLogEvent = {
        id: createId("log"),
        sessionId,
        sceneId,
        type: input.type,
        text: input.text,
        createdAt: nowIso(),
      };
      const actionLog = [...state.actionLog, event].slice(-ACTION_LOG_MAX);
      const nextState: Partial<CampaignContextState> = { actionLog };

      if (sceneId) {
        const scene = state.scenes.find((s) => s.id === sceneId);
        if (scene && !scene.actionLogIds.includes(event.id)) {
          nextState.scenes = state.scenes.map((s) =>
            s.id === sceneId
              ? { ...s, actionLogIds: [...s.actionLogIds, event.id] }
              : s
          );
        }
      }

      return nextState;
    });
    // TODO: Backend — POST /api/session/events or websocket
  },

  addNarrationClip(input) {
    set((state) => {
      const campaignId = input.campaignId ?? state.activeCampaignId ?? "";
      const sessionId = input.sessionId ?? state.activeSessionId ?? undefined;
      const sceneId = input.sceneId ?? state.activeSceneId ?? undefined;
      const clip: NarrationClip = {
        id: createId("clip"),
        campaignId,
        sessionId,
        sceneId,
        title: input.title,
        voiceId: input.voiceId,
        audioUrl: input.audioUrl,
        duration: input.duration,
        createdAt: nowIso(),
      };
      const narrationClips = [...state.narrationClips, clip];
      const nextState: Partial<CampaignContextState> = { narrationClips };
      if (sceneId) {
        const scene = state.scenes.find((s) => s.id === sceneId);
        if (scene && !scene.narrationClipIds.includes(clip.id)) {
          nextState.scenes = state.scenes.map((s) =>
            s.id === sceneId ? { ...s, narrationClipIds: [...s.narrationClipIds, clip.id] } : s
          );
        }
      }
      return nextState;
    });
  },

  upsertNpc(npc) {
    set((state) => {
      const idx = state.npcs.findIndex((n) => n.id === npc.id);
      const npcs = idx >= 0
        ? state.npcs.map((n) => (n.id === npc.id ? npc : n))
        : [...state.npcs, npc];
      return { npcs };
    });
  },

  upsertVoice(voice) {
    set((state) => {
      const idx = state.voices.findIndex((v) => v.id === voice.id);
      const voices = idx >= 0
        ? state.voices.map((v) => (v.id === voice.id ? voice : v))
        : [...state.voices, voice];
      return { voices };
    });
  },

  upsertCodexEntry(entry) {
    set((state) => {
      const idx = state.codexEntries.findIndex((e) => e.id === entry.id);
      const codexEntries = idx >= 0
        ? state.codexEntries.map((e) => (e.id === entry.id ? entry : e))
        : [...state.codexEntries, entry];
      return { codexEntries };
    });
  },

  upsertScene(scene) {
    set((state) => {
      const idx = state.scenes.findIndex((s) => s.id === scene.id);
      const scenes = idx >= 0
        ? state.scenes.map((s) => (s.id === scene.id ? scene : s))
        : [...state.scenes, scene];
      return { scenes };
    });
  },

  upsertCampaign(campaign) {
    set((state) => {
      const idx = state.campaigns.findIndex((c) => c.id === campaign.id);
      const campaigns = idx >= 0
        ? state.campaigns.map((c) => (c.id === campaign.id ? campaign : c))
        : [...state.campaigns, campaign];
      return { campaigns };
    });
  },

  setVoices(voices) {
    set({ voices: Array.isArray(voices) ? voices : [] });
  },

  hydrateFromSeed() {
    set({
      ...seedCampaignContext,
      activeCampaignId: seedCampaignContext.campaigns[0]?.id ?? null,
      activeSessionId: seedCampaignContext.sessions[0]?.id ?? null,
      activeSceneId: seedCampaignContext.sessions[0]?.activeSceneId ?? seedCampaignContext.scenes[0]?.id ?? null,
    });
  },

  resetCampaignContext() {
    set(emptyState);
  },
}));
