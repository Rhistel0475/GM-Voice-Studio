/**
 * Pure regex-based parser for structured adventure markdown templates.
 * No API / AI — safe to call from the client.
 */

export interface ParsedTemplateNpc {
  name: string;
  role: string;
  personality: string;
}

export interface ParsedTemplateScene {
  title: string;
  readAloud: string;
  notes: string;
  npcs: ParsedTemplateNpc[];
  type: string;
  sceneOrder: number;
}

const VALID_TYPES = new Set(["combat", "social", "exploration", "trap", "travel"]);

/** Exact template string for the Library "Copy template" button. */
export const MARKDOWN_ADVENTURE_TEMPLATE = `## Scene title here

**Read-aloud:** The text you read to players out loud.

**GM notes:** Tactical info, DCs, secrets — never read to players.

**NPCs:** Name One (role, personality), Name Two (role, personality)

**Type:** combat

---

## Next scene title

**Read-aloud:** Another read-aloud block.

**GM notes:** More GM-only notes.

**NPCs:** Guard (watchman, suspicious)

**Type:** social
`;

const RE_READ_ALOUD = /\*\*Read\s*[- ]?aloud\s*:\*\*/i;
const RE_GM_NOTES = /\*\*GM\s*notes?\s*:\*\*/i;
const RE_NPCS = /\*\*NPCs?\s*:\*\*/i;
const RE_TYPE = /\*\*Type\s*:\*\*/i;
/** Next structured field label like **GM notes:** (not inline **bold**). */
const RE_NEXT_FIELD = /\*\*[A-Za-z][^*]*:\*\*/;

function normalizeType(raw: string): string {
  const t = String(raw || "")
    .trim()
    .toLowerCase();
  if (VALID_TYPES.has(t)) return t;
  return "exploration";
}

function sliceUntilNextLabel(block: string, labelEndIndex: number): string {
  const after = block.slice(labelEndIndex);
  const next = after.match(RE_NEXT_FIELD);
  const end = next && next.index !== undefined ? next.index : after.length;
  return after.slice(0, end).trim();
}

function extractAfterLabel(block: string, labelRe: RegExp): string {
  const m = labelRe.exec(block);
  if (!m || m.index === undefined) return "";
  const start = m.index + m[0].length;
  return sliceUntilNextLabel(block, start);
}

function parseNpcSegment(segment: string): ParsedTemplateNpc | null {
  const s = segment.trim();
  if (!s) return null;
  const open = s.indexOf("(");
  const close = s.lastIndexOf(")");
  if (open === -1 || close <= open) {
    return { name: s, role: "", personality: "" };
  }
  const name = s.slice(0, open).trim();
  const inner = s.slice(open + 1, close).trim();
  const comma = inner.indexOf(",");
  const role = comma === -1 ? inner.trim() : inner.slice(0, comma).trim();
  const personality = comma === -1 ? "" : inner.slice(comma + 1).trim();
  return { name, role, personality };
}

function parseNpcsField(npcText: string): ParsedTemplateNpc[] {
  const t = npcText.trim();
  if (!t) return [];
  const parts = t.split(/(?<=\))\s*,\s*/);
  const out: ParsedTemplateNpc[] = [];
  for (const p of parts) {
    const npc = parseNpcSegment(p);
    if (npc && npc.name) out.push(npc);
  }
  if (out.length === 0 && t) {
    const single = parseNpcSegment(t);
    if (single && single.name) out.push(single);
  }
  return out;
}

function splitSceneBlocks(text: string): { title: string; body: string }[] {
  const re = /^##\s+(.+)$/gm;
  const matches = [...text.matchAll(re)];
  if (matches.length === 0) return [];
  const scenes: { title: string; body: string }[] = [];
  for (let i = 0; i < matches.length; i++) {
    const title = (matches[i][1] || "").trim();
    const start = (matches[i].index ?? 0) + matches[i][0].length;
    const end = i + 1 < matches.length ? (matches[i + 1].index ?? text.length) : text.length;
    const body = text.slice(start, end).trim();
    if (title) scenes.push({ title, body });
  }
  return scenes;
}

/**
 * Returns true when the document looks like our structured template:
 * at least one ## heading and at least one **Read-aloud:** (or **Read aloud:**) label.
 */
export function isStructuredTemplate(text: string): boolean {
  try {
    if (!text || typeof text !== "string") return false;
    const hasHeading = /^##\s+\S/m.test(text);
    const hasReadAloud = RE_READ_ALOUD.test(text);
    return hasHeading && hasReadAloud;
  } catch {
    return false;
  }
}

/**
 * Parse structured markdown into scene objects. Never throws — returns partial/empty on failure.
 */
export function parseMarkdownTemplate(text: string): ParsedTemplateScene[] {
  try {
    if (!text || typeof text !== "string") return [];
    const blocks = splitSceneBlocks(text);
    if (blocks.length === 0) return [];

    const scenes: ParsedTemplateScene[] = [];
    let order = 0;
    for (const { title, body } of blocks) {
      order += 1;
      const readAloud = extractAfterLabel(body, RE_READ_ALOUD);
      const notes = extractAfterLabel(body, RE_GM_NOTES);
      const npcsRaw = extractAfterLabel(body, RE_NPCS);
      const typeRaw = extractAfterLabel(body, RE_TYPE);
      scenes.push({
        title,
        readAloud,
        notes,
        npcs: parseNpcsField(npcsRaw),
        type: normalizeType(typeRaw),
        sceneOrder: order,
      });
    }
    return scenes;
  } catch {
    return [];
  }
}
