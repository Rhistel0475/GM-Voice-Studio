/**
 * Campaign model for the shared data layer.
 * Backend: GET/PATCH /api/campaigns/:id
 */
export interface Campaign {
  id: string;
  name: string;
  setting?: string;
  activeSessionId?: string;
}
