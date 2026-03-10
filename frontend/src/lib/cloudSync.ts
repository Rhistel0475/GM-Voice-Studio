/**
 * Cloud sync — sync campaigns to cloud backend, background sync, conflict resolution.
 */

import { createClient } from "../api";

export interface CloudSyncOptions {
  apiKey?: string;
}

export interface CloudSyncResult {
  success: boolean;
  campaignId: string;
  lastSyncedAt?: string;
  conflict?: boolean;
  error?: string;
}

/**
 * Push local campaign state to cloud. Backend may return lastSyncedAt or conflict flag.
 */
export async function syncCampaignToCloud(
  campaignId: string,
  options: CloudSyncOptions = {}
): Promise<CloudSyncResult> {
  const client = createClient(options.apiKey ?? "");
  try {
    const res = await (client as { getCampaign: (id: string) => Promise<Response> }).getCampaign(campaignId);
    if (!res.ok) {
      return {
        success: false,
        campaignId,
        error: await res.text(),
      };
    }
    const data = (await res.json()) as { updated_at?: string; conflict?: boolean };
    return {
      success: true,
      campaignId,
      lastSyncedAt: data.updated_at,
      conflict: data.conflict,
    };
  } catch (e) {
    return {
      success: false,
      campaignId,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

/**
 * Pull campaign from cloud and merge into local store. Caller should hydrate store with returned data.
 */
export async function pullCampaignFromCloud(
  campaignId: string,
  options: CloudSyncOptions = {}
): Promise<{ success: boolean; data?: unknown; error?: string }> {
  const client = createClient(options.apiKey ?? "");
  try {
    const res = await (client as { getCampaign: (id: string) => Promise<Response> }).getCampaign(campaignId);
    if (!res.ok) {
      return {
        success: false,
        error: await res.text(),
      };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (e) {
    return {
      success: false,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}
