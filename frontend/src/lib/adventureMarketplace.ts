/**
 * Adventure marketplace foundation — package format, metadata, installable bundles.
 */

export interface AdventurePackageMetadata {
  author: string;
  title: string;
  system: string;
  difficulty?: "beginner" | "intermediate" | "expert";
  version?: string;
  description?: string;
}

export interface AdventurePackage {
  metadata: AdventurePackageMetadata;
  /** Campaign + sessions, scenes, npcs, codex, etc. (same shape as campaign export). */
  campaignData: unknown;
}

const installedPackages: AdventurePackage[] = [];

/**
 * Install an adventure package (merge into local registry). Does not auto-create campaign; caller can use importCampaign with package.campaignData.
 */
export function installAdventurePackage(pkg: AdventurePackage): void {
  const existing = installedPackages.find(
    (p) => p.metadata.title === pkg.metadata.title && p.metadata.author === pkg.metadata.author
  );
  if (existing) {
    const idx = installedPackages.indexOf(existing);
    installedPackages[idx] = pkg;
  } else {
    installedPackages.push(pkg);
  }
}

/**
 * List currently installed adventure packages (from local registry).
 * For "available" marketplace list, this would typically call a backend; we return installed only.
 */
export function listAvailableAdventurePackages(): AdventurePackageMetadata[] {
  return installedPackages.map((p) => p.metadata);
}

/**
 * Get full package by title+author for import.
 */
export function getInstalledPackage(
  title: string,
  author: string
): AdventurePackage | undefined {
  return installedPackages.find(
    (p) => p.metadata.title === title && p.metadata.author === author
  );
}
