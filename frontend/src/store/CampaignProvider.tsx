import { useEffect, type ReactNode } from "react";
import { useCampaignContextStore } from "./campaignContext";

/**
 * Mount point for campaign-scoped UI. State lives in useCampaignContextStore (zustand).
 * Syncs requireApiKey from GET /config once on mount.
 */
export function CampaignProvider({ children }: { children: ReactNode }) {
  const setRequireApiKey = useCampaignContextStore((s) => s.setRequireApiKey);
  useEffect(() => {
    let cancelled = false;
    fetch("/config")
      .then((r) => (r.ok ? r.json() : { require_api_key: false }))
      .then((cfg) => {
        if (!cancelled) setRequireApiKey(Boolean(cfg?.require_api_key));
      })
      .catch(() => {
        if (!cancelled) setRequireApiKey(false);
      });
    return () => {
      cancelled = true;
    };
  }, [setRequireApiKey]);
  return <>{children}</>;
}
