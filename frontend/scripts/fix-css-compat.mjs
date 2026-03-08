import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const assetsDir = join(process.cwd(), "..", "static", "frontend", "assets");

let patched = 0;

for (const name of readdirSync(assetsDir)) {
  if (!name.endsWith(".css")) continue;
  const fullPath = join(assetsDir, name);
  if (!statSync(fullPath).isFile()) continue;
  const original = readFileSync(fullPath, "utf8");
  let css = original;

  // Keep only the standard property to silence compat tooling warnings.
  css = css.replaceAll(
    "-webkit-text-size-adjust:100%;text-size-adjust:100%;",
    "text-size-adjust:100%;",
  );
  css = css.replaceAll(
    "-webkit-text-size-adjust:100%;-moz-tab-size:4",
    "text-size-adjust:100%;-moz-tab-size:4",
  );
  css = css.replaceAll("-webkit-text-size-adjust:100%;", "");
  css = css.replaceAll("text-size-adjust:100%;text-size-adjust:100%;", "text-size-adjust:100%;");

  if (css === original) continue;
  writeFileSync(fullPath, css, "utf8");
  patched += 1;
}

console.log(`[fix-css-compat] patched ${patched} CSS asset(s).`);
