/**
 * Multi-campaign management — campaign selector, separate data per campaign.
 * createCampaign, switchCampaign, deleteCampaign.
 */

import { useCampaignContextStore } from "../store/campaignContext";
import type { Campaign } from "../types";
import { createId } from "./utils/ids";

export interface CreateCampaignOptions {
  name: string;
  setting?: string;
}

/**
 * Create a new campaign and add it to the store. Optionally switch to it.
 */
export function createCampaign(
  options: CreateCampaignOptions,
  options2?: { switchTo?: boolean }
): Campaign {
  const campaign: Campaign = {
    id: createId("campaign"),
    name: options.name,
    setting: options.setting,
  };

  useCampaignContextStore.setState((state) => ({
    campaigns: [...state.campaigns, campaign],
    ...(options2?.switchTo !== false
      ? {
          activeCampaignId: campaign.id,
          activeSessionId: null,
          activeSceneId: null,
        }
      : {}),
  }));

  return campaign;
}

/**
 * Switch the active campaign (and clear session/scene if not belonging to the campaign).
 */
export function switchCampaign(campaignId: string | null): void {
  useCampaignContextStore.getState().setActiveCampaign(campaignId);
}

/**
 * Delete a campaign and remove its sessions, scenes, npcs, codex entries, action log entries, and clips from the store.
 * If the deleted campaign was active, clears active ids.
 */
export function deleteCampaign(campaignId: string): void {
  const state = useCampaignContextStore.getState();

  const campaigns = state.campaigns.filter((c) => c.id !== campaignId);
  const sessions = state.sessions.filter((s) => s.campaignId !== campaignId);
  const removedSessionIds = new Set(
    state.sessions.filter((s) => s.campaignId === campaignId).map((s) => s.id)
  );
  const scenes = state.scenes.filter((s) => s.campaignId !== campaignId);
  const npcs = state.npcs.filter((n) => n.campaignId !== campaignId);
  const codexEntries = state.codexEntries.filter((e) => e.campaignId !== campaignId);
  const actionLog = state.actionLog.filter(
    (e) => !e.sessionId || !removedSessionIds.has(e.sessionId)
  );
  const narrationClips = state.narrationClips.filter((c) => c.campaignId !== campaignId);

  useCampaignContextStore.setState({
    campaigns,
    sessions,
    scenes,
    npcs,
    codexEntries,
    actionLog,
    narrationClips,
    ...(state.activeCampaignId === campaignId
      ? { activeCampaignId: null, activeSessionId: null, activeSceneId: null }
      : {}),
  });
}
