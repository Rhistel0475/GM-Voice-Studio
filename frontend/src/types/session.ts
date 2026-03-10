/**
 * Session status for lifecycle.
 */
export type SessionStatus = "prep" | "active" | "closed";

/**
 * Session model — a play session within a campaign.
 * Backend: session create/join; scene index sync.
 */
export interface Session {
  id: string;
  campaignId: string;
  title: string;
  activeSceneId?: string;
  startedAt?: string;
  status?: SessionStatus;
}
