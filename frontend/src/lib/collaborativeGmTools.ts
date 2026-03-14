/**
 * Collaborative GM tools — shared campaign editing, roles and permissions, shared world building.
 * inviteCollaborator, assignGmRole. Real-time updates would be wired to a backend/websocket layer.
 */

import { createClient } from "../api";

export type GmRole = "owner" | "co_gm" | "editor" | "viewer";

export interface Collaborator {
  userId: string;
  email: string;
  role: GmRole;
  invitedAt?: string;
}

export interface InviteCollaboratorOptions {
  campaignId: string;
  email: string;
  role?: GmRole;
  apiKey?: string;
}

export interface AssignGmRoleOptions {
  campaignId: string;
  userId: string;
  role: GmRole;
  apiKey?: string;
}

/**
 * Invite a collaborator by email. Backend should send invite and return pending collaborator.
 */
export async function inviteCollaborator(
  options: InviteCollaboratorOptions
): Promise<{ success: boolean; error?: string }> {
  const client = createClient(options.apiKey ?? "");
  const base = (client as { getBaseUrl: () => string }).getBaseUrl?.() ?? "";
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (options.apiKey) headers["X-API-Key"] = options.apiKey;
  try {
    const res = await fetch(
      `${base}/api/campaigns/${options.campaignId}/invite`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          email: options.email,
          role: options.role ?? "editor",
        }),
      }
    );
    if (!res.ok) {
      return { success: false, error: await res.text() };
    }
    return { success: true };
  } catch (e) {
    return {
      success: false,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

/**
 * Assign a GM role to a user (owner only). Backend enforces permissions.
 */
export async function assignGmRole(
  options: AssignGmRoleOptions
): Promise<{ success: boolean; error?: string }> {
  const client = createClient(options.apiKey ?? "");
  const base = (client as { getBaseUrl: () => string }).getBaseUrl?.() ?? "";
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (options.apiKey) headers["X-API-Key"] = options.apiKey;
  try {
    const res = await fetch(
      `${base}/api/campaigns/${options.campaignId}/role`,
      {
        method: "PATCH",
        headers,
        body: JSON.stringify({
          userId: options.userId,
          role: options.role,
        }),
      }
    );
    if (!res.ok) {
      return { success: false, error: await res.text() };
    }
    return { success: true };
  } catch (e) {
    return {
      success: false,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}
