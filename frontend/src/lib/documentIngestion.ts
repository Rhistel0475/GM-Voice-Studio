/**
 * Document ingestion for adventure documents (PDF, DOCX, Markdown).
 * Feeds Codex, NPC Workshop, and scene generation.
 */

import { createClient } from "../api";

export interface DocumentFileMeta {
  name: string;
  pageCount?: number;
  [key: string]: unknown;
}

export interface StructuredSection {
  heading: string;
  level: number;
  body: string;
  startOffset: number;
  contentType?: string;
}

export interface IngestedAdventureResult {
  files: DocumentFileMeta[];
  title: string;
  summary: string;
  npcs: unknown[];
  scenes: unknown[];
  locations: unknown[];
  codex_entries?: unknown[];
  relationships?: unknown[];
  [key: string]: unknown;
}

/**
 * Ingest an adventure document: upload to backend and return parsed result.
 * Supports PDF, DOCX, and Markdown (backend may support .txt, .md, .pdf).
 */
export async function ingestAdventureDocument(
  file: File,
  apiKey?: string
): Promise<IngestedAdventureResult> {
  const client = createClient(apiKey ?? "");
  const formData = new FormData();
  formData.append("files", file);

  const response = await (client as { postAdventureParse: (fd: FormData) => Promise<IngestedAdventureResult> })
    .postAdventureParse(formData);

  return response;
}

const SECTION_HEADING_RE = /^(#{1,6})\s+(.+)$/m;

/**
 * Extract structured sections from document text (client-side).
 * Preserves headings and approximate offsets; use for traceability and section-aware extraction.
 */
export function extractStructuredSections(documentText: string): StructuredSection[] {
  const sections: StructuredSection[] = [];
  const lines = documentText.split(/\r?\n/);
  let currentHeading = "";
  let currentLevel = 0;
  let currentBody: string[] = [];
  let startOffset = 0;
  let offset = 0;

  function flushSection() {
    if (currentHeading || currentBody.length > 0) {
      sections.push({
        heading: currentHeading,
        level: currentLevel,
        body: currentBody.join("\n").trim(),
        startOffset,
      });
    }
  }

  for (const line of lines) {
    const match = line.match(SECTION_HEADING_RE);
    if (match) {
      flushSection();
      currentLevel = match[1].length;
      currentHeading = match[2].trim();
      currentBody = [];
      startOffset = offset;
    } else {
      currentBody.push(line);
    }
    offset += line.length + 1;
  }
  flushSection();

  return sections;
}
