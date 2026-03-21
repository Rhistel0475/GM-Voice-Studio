import type { ReactNode } from "react";

/**
 * Mount point for campaign-scoped UI. State lives in useCampaignContextStore (zustand).
 */
export function CampaignProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
