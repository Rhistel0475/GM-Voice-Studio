"""
Campaign document ingestion and retrieval for the Adventure Intelligence Engine.

This service keeps the implementation local and incremental:
- raw files are stored on disk per campaign
- extracted text and chunks are stored in SQLite
- embeddings are stored per chunk when OPENAI_API_KEY is available
- query answering uses Anthropic when available, with safe fallbacks
"""
from __future__ import annotations

import io
import json
import logging
import math
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree

import anthropic
import fitz

from app.core.config import (
    AI_MODEL,
    ANTHROPIC_API_KEY,
    CAMPAIGN_DOCUMENT_STORAGE_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from app.infrastructure.database import SessionLocal
from app.infrastructure.db_models import Campaign, CampaignDocument, CampaignDocumentChunk

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_NPC_RE = re.compile(
    r"\b(?:Lord|Lady|Captain|Sir|Dame|Queen|King|Prince|Princess|Master|Mistress|Archmage|High Priest|Father|Mother)\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"
)
_LOCATION_RE = re.compile(
    r"\b(?:City|Forest|Keep|Tower|Temple|Inn|Village|Castle|Cavern|Ruins|Pass|Harbor|Hall|Court)\s+"
    r"of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+"
    r"(?:Keep|Tower|Temple|Inn|Village|Castle|Cavern|Ruins|Pass|Harbor|Hall|Court)\b"
)
_ITEM_HINT_RE = re.compile(r"\b(?:Sword|Blade|Amulet|Key|Scroll|Orb|Crown|Ring|Staff|Dagger|Lantern)\b", re.IGNORECASE)
_QUEST_HINT_RE = re.compile(r"\b(?:quest|mission|objective|task|bounty|goal|must|seek|recover|rescue|escort)\b", re.IGNORECASE)

_ANTHROPIC_CLIENT: Optional[anthropic.Anthropic] = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in str(text or "").replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    blank_pending = False
    for line in lines:
        if not line:
            if cleaned:
                blank_pending = True
            continue
        if blank_pending:
            cleaned.append("")
            blank_pending = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _safe_filename(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", filename or "document").strip("-") or "document"


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(_TOKEN_RE.findall(text)) * 1.3))


def _dedupe_keep_order(values: list[str], limit: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _extract_text_from_pdf(raw_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc!s}") from exc

    pages: list[str] = []
    for page in doc:
        text = (page.get_text("text") or "").strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_text_from_docx(raw_bytes: bytes) -> str:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw_bytes))
        xml_bytes = archive.read("word/document.xml")
    except Exception as exc:
        raise RuntimeError(f"DOCX extraction failed: {exc!s}") from exc

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise RuntimeError(f"DOCX XML parse failed: {exc!s}") from exc

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        runs = [node.text or "" for node in para.findall(".//w:t", ns)]
        text = "".join(runs).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _extract_text(filename: str, raw_bytes: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(raw_bytes)
    if suffix == ".docx":
        return _extract_text_from_docx(raw_bytes)
    try:
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"Text extraction failed: {exc!s}") from exc


def _chunk_text(text: str, target_words: int = 650, overlap_words: int = 120) -> list[dict[str, Any]]:
    words = text.split()
    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(len(words), start + target_words)
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append({
                "index": index,
                "text": chunk_text,
                "token_count": _estimate_tokens(chunk_text),
            })
            index += 1
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _heuristic_metadata(text: str, filename: str) -> dict[str, Any]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]

    npc_candidates = _dedupe_keep_order(_NPC_RE.findall(text), limit=10)
    location_candidates = _dedupe_keep_order(_LOCATION_RE.findall(text), limit=10)

    quests: list[str] = []
    items: list[str] = []
    for line in lines:
        if _QUEST_HINT_RE.search(line):
            quests.append(line[:140])
        if _ITEM_HINT_RE.search(line):
            items.append(line[:100])

    summary_source = paragraphs[0] if paragraphs else text[:400]
    summary = summary_source[:320].strip()

    return {
        "summary": summary or f"Ingested campaign document: {filename}",
        "npcs": npc_candidates,
        "locations": location_candidates,
        "quests": _dedupe_keep_order(quests, limit=8),
        "items": _dedupe_keep_order(items, limit=8),
    }


def _get_anthropic_client() -> anthropic.Anthropic:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
        _ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _ANTHROPIC_CLIENT


def _parse_json_payload(raw_text: str) -> Any:
    raw = (raw_text or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
    return json.loads(raw.strip())


def _extract_metadata_with_ai(text: str, filename: str) -> dict[str, Any]:
    baseline = _heuristic_metadata(text, filename)
    if not ANTHROPIC_API_KEY:
        return baseline

    prompt = (
        "Extract compact campaign-document metadata and return ONLY valid JSON.\n"
        "JSON shape:\n"
        '{"summary":"two-sentence summary","npcs":["..."],"locations":["..."],"quests":["..."],"items":["..."]}\n'
        "Rules:\n"
        "- Keep arrays short: max 8 per field.\n"
        "- Use exact names or short phrases.\n"
        "- If a field is unavailable, return [].\n\n"
        f"Filename: {filename}\n"
        "Document excerpt:\n"
        f"{text[:12000]}"
    )
    try:
        response = _get_anthropic_client().messages.create(
            model=AI_MODEL,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _parse_json_payload(response.content[0].text)
        if not isinstance(payload, dict):
            return baseline
    except Exception as exc:
        logging.warning("Campaign Brain metadata extraction fell back to heuristics: %s", exc)
        return baseline

    merged = {
        "summary": str(payload.get("summary") or baseline["summary"]).strip() or baseline["summary"],
        "npcs": _dedupe_keep_order(list(payload.get("npcs") or []) + baseline["npcs"], limit=10),
        "locations": _dedupe_keep_order(list(payload.get("locations") or []) + baseline["locations"], limit=10),
        "quests": _dedupe_keep_order(list(payload.get("quests") or []) + baseline["quests"], limit=8),
        "items": _dedupe_keep_order(list(payload.get("items") or []) + baseline["items"], limit=8),
    }
    return merged


def _embed_texts(texts: list[str]) -> list[Optional[list[float]]]:
    if not texts:
        return []
    if not OPENAI_API_KEY:
        return [None for _ in texts]

    payload = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "GM-Voice-Studio/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("OpenAI embeddings response was malformed.")
        embeddings = [row.get("embedding") for row in rows]
        if len(embeddings) != len(texts):
            raise RuntimeError("OpenAI embeddings response length mismatch.")
        return embeddings
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logging.warning("Campaign Brain embeddings failed: %s %s", exc.code, body[:400])
    except Exception as exc:
        logging.warning("Campaign Brain embeddings failed: %s", exc)
    return [None for _ in texts]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_overlap_score(question: str, text: str) -> float:
    q_terms = set(_TOKEN_RE.findall(question.lower()))
    t_terms = set(_TOKEN_RE.findall(text.lower()))
    if not q_terms or not t_terms:
        return 0.0
    overlap = len(q_terms & t_terms)
    return overlap / max(1, len(q_terms))


def _document_payload(document: CampaignDocument) -> dict[str, Any]:
    try:
        metadata = json.loads(document.metadata_json or "{}")
        if not isinstance(metadata, dict):
            metadata = {}
    except json.JSONDecodeError:
        metadata = {}
    return {
        "id": document.id,
        "campaign_id": document.campaign_id,
        "title": document.filename,
        "filename": document.filename,
        "file_type": document.file_type,
        "mime_type": document.mime_type,
        "summary": document.summary or metadata.get("summary", ""),
        "chunk_count": document.chunk_count,
        "created_at": document.created_at,
        "metadata": metadata,
    }


def list_campaign_documents(campaign_id: int) -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        docs = (
            db.query(CampaignDocument)
            .filter(CampaignDocument.campaign_id == campaign_id)
            .order_by(CampaignDocument.created_at.desc(), CampaignDocument.id.desc())
            .all()
        )
        return [_document_payload(doc) for doc in docs]
    finally:
        db.close()


def ingest_campaign_documents(campaign_id: int, uploads: list[dict[str, Any]]) -> dict[str, Any]:
    if not uploads:
        raise ValueError("Upload at least one campaign document.")

    storage_root = Path(CAMPAIGN_DOCUMENT_STORAGE_PATH)
    storage_root.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if campaign is None:
            raise FileNotFoundError("Campaign not found")

        campaign_dir = storage_root / str(campaign_id)
        campaign_dir.mkdir(parents=True, exist_ok=True)

        ingested_docs: list[dict[str, Any]] = []
        total_chunks = 0
        any_embeddings = False

        for upload in uploads:
            filename = str(upload.get("filename") or "").strip() or "document"
            suffix = Path(filename).suffix.lower()
            if suffix not in _ALLOWED_SUFFIXES:
                raise ValueError(f"Unsupported file type for {filename}. Upload PDF, DOCX, or Markdown.")

            raw_bytes = upload.get("content") or b""
            if not raw_bytes:
                raise ValueError(f"{filename} was empty.")

            extracted_text = _normalize_whitespace(_extract_text(filename, raw_bytes))
            if not extracted_text:
                raise ValueError(f"{filename} did not contain readable text.")

            metadata = _extract_metadata_with_ai(extracted_text, filename)
            chunks = _chunk_text(extracted_text)
            if not chunks:
                raise ValueError(f"{filename} did not produce searchable chunks.")

            embeddings = _embed_texts([chunk["text"] for chunk in chunks])
            if any(embedding for embedding in embeddings):
                any_embeddings = True

            stored_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{_safe_filename(filename)}"
            stored_path = campaign_dir / stored_name
            stored_path.write_bytes(raw_bytes)

            document = CampaignDocument(
                campaign_id=campaign_id,
                filename=filename,
                file_type=suffix.lstrip("."),
                mime_type=str(upload.get("content_type") or "").strip(),
                storage_path=str(stored_path),
                raw_text=extracted_text,
                summary=str(metadata.get("summary") or "").strip(),
                metadata_json=json.dumps(metadata, ensure_ascii=False),
                chunk_count=len(chunks),
                created_at=_utcnow_iso(),
            )
            db.add(document)
            db.flush()

            for chunk, embedding in zip(chunks, embeddings):
                db.add(
                    CampaignDocumentChunk(
                        document_id=document.id,
                        campaign_id=campaign_id,
                        chunk_index=int(chunk["index"]),
                        token_count=int(chunk["token_count"]),
                        text=str(chunk["text"]),
                        metadata_json=json.dumps(
                            {
                                "filename": filename,
                                "document_summary": document.summary,
                                "document_metadata": metadata,
                            },
                            ensure_ascii=False,
                        ),
                        embedding_json=json.dumps(embedding) if embedding else None,
                        created_at=_utcnow_iso(),
                    )
                )

            total_chunks += len(chunks)
            ingested_docs.append(_document_payload(document))

        db.commit()
        return {
            "campaign_id": campaign_id,
            "documents": ingested_docs,
            "total_documents": len(ingested_docs),
            "total_chunks": total_chunks,
            "embedding_backend": "openai" if any_embeddings else "lexical-fallback",
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _fallback_answer(question: str, relevant_chunks: list[dict[str, Any]]) -> str:
    if not relevant_chunks:
        return "No relevant campaign context was found for that question."
    excerpts = []
    for chunk in relevant_chunks[:3]:
        text = str(chunk.get("text") or "").strip()
        filename = str(chunk.get("filename") or "Campaign Document").strip()
        if text:
            excerpts.append(f"{filename}: {text[:280]}")
    if not excerpts:
        return "No relevant campaign context was found for that question."
    return f"Best matching campaign notes for '{question}':\n\n" + "\n\n".join(excerpts)


def _summarize_answer(
    *,
    campaign_title: str,
    question: str,
    relevant_chunks: list[dict[str, Any]],
) -> str:
    if not relevant_chunks:
        return "No relevant campaign context was found for that question."
    if not ANTHROPIC_API_KEY:
        return _fallback_answer(question, relevant_chunks)

    context_blocks = []
    for idx, chunk in enumerate(relevant_chunks[:6], start=1):
        context_blocks.append(
            f"[Chunk {idx} | {chunk.get('filename', 'Campaign Document')} | score={chunk.get('score', 0):.3f}]\n"
            f"{chunk.get('text', '')}"
        )

    prompt = (
        "You are Campaign Brain for a tabletop RPG GM.\n"
        "Answer the question using ONLY the supplied campaign document excerpts.\n"
        "If the answer is uncertain, say what is clear from the excerpts and name the gap.\n"
        "Keep the answer concise but useful for live play.\n\n"
        f"Campaign: {campaign_title or 'Untitled Campaign'}\n"
        f"Question: {question}\n\n"
        "Document excerpts:\n"
        + "\n\n".join(context_blocks)
    )

    try:
        response = _get_anthropic_client().messages.create(
            model=AI_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip()
        return answer or _fallback_answer(question, relevant_chunks)
    except Exception as exc:
        logging.warning("Campaign Brain summary fell back to excerpts: %s", exc)
        return _fallback_answer(question, relevant_chunks)


def query_campaign_documents(campaign_id: int, question: str, top_k: int = 5) -> dict[str, Any]:
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        raise ValueError("Question is required.")

    db = SessionLocal()
    try:
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if campaign is None:
            raise FileNotFoundError("Campaign not found")

        rows = (
            db.query(CampaignDocumentChunk, CampaignDocument)
            .join(CampaignDocument, CampaignDocument.id == CampaignDocumentChunk.document_id)
            .filter(CampaignDocumentChunk.campaign_id == campaign_id)
            .order_by(CampaignDocument.id.desc(), CampaignDocumentChunk.chunk_index.asc())
            .all()
        )
        if not rows:
            raise RuntimeError("No campaign documents have been indexed yet.")

        query_embedding = _embed_texts([cleaned_question])[0]
        scored: list[dict[str, Any]] = []

        for chunk_row, document_row in rows:
            lexical_score = _keyword_overlap_score(cleaned_question, chunk_row.text or "")
            semantic_score = 0.0
            if query_embedding and chunk_row.embedding_json:
                try:
                    semantic_score = _cosine_similarity(query_embedding, json.loads(chunk_row.embedding_json))
                except Exception:
                    semantic_score = 0.0

            score = (semantic_score * 0.85 + lexical_score * 0.15) if query_embedding else lexical_score
            if score <= 0 and len(rows) > top_k * 2:
                continue

            try:
                chunk_metadata = json.loads(chunk_row.metadata_json or "{}")
                if not isinstance(chunk_metadata, dict):
                    chunk_metadata = {}
            except json.JSONDecodeError:
                chunk_metadata = {}

            scored.append(
                {
                    "chunk_id": chunk_row.id,
                    "document_id": document_row.id,
                    "filename": document_row.filename,
                    "score": score,
                    "text": chunk_row.text,
                    "token_count": chunk_row.token_count,
                    "metadata": chunk_metadata,
                }
            )

        if not scored:
            scored = [
                {
                    "chunk_id": chunk_row.id,
                    "document_id": document_row.id,
                    "filename": document_row.filename,
                    "score": 0.0,
                    "text": chunk_row.text,
                    "token_count": chunk_row.token_count,
                    "metadata": json.loads(chunk_row.metadata_json or "{}") if chunk_row.metadata_json else {},
                }
                for chunk_row, document_row in rows[:top_k]
            ]

        relevant_chunks = sorted(scored, key=lambda item: item["score"], reverse=True)[: max(1, min(top_k, 8))]

        matched_npcs: list[str] = []
        matched_locations: list[str] = []
        matched_quests: list[str] = []
        matched_items: list[str] = []
        for chunk in relevant_chunks:
            doc_meta = chunk.get("metadata", {}).get("document_metadata", {})
            if isinstance(doc_meta, dict):
                matched_npcs.extend(doc_meta.get("npcs") or [])
                matched_locations.extend(doc_meta.get("locations") or [])
                matched_quests.extend(doc_meta.get("quests") or [])
                matched_items.extend(doc_meta.get("items") or [])

        answer = _summarize_answer(
            campaign_title=campaign.title,
            question=cleaned_question,
            relevant_chunks=relevant_chunks,
        )

        return {
            "campaign_id": campaign_id,
            "question": cleaned_question,
            "answer": answer,
            "relevant_chunks": relevant_chunks,
            "matched_metadata": {
                "npcs": _dedupe_keep_order(matched_npcs, limit=10),
                "locations": _dedupe_keep_order(matched_locations, limit=10),
                "quests": _dedupe_keep_order(matched_quests, limit=8),
                "items": _dedupe_keep_order(matched_items, limit=8),
            },
            "documents_indexed": db.query(CampaignDocument).filter(CampaignDocument.campaign_id == campaign_id).count(),
        }
    finally:
        db.close()
