/**
 * Generate a unique ID with optional prefix (timestamp + random suffix).
 */
export function createId(prefix = ""): string {
  const t = Date.now().toString(36);
  const r = Math.random().toString(36).slice(2, 10);
  return prefix ? `${prefix}_${t}-${r}` : `${t}-${r}`;
}

/**
 * Current time in ISO 8601 format.
 */
export function nowIso(): string {
  return new Date().toISOString();
}
