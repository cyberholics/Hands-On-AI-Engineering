"""
Ingestion pipeline: load, section, chunk, embed, write.

Supported sources:
  .pdf          research papers, manuals, reports
  .txt / .md    notes, articles, plain text
  .vtt / .srt   video transcripts (timestamps stripped, text kept)

Every chunk is tagged with doc_id, doc_title, source_type, section and
chunk_index before it is written to the shared Actian collection. That
metadata is what lets the Writer agent cite a specific document and section
instead of vaguely gesturing at "the sources".
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from research_assistant.config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".vtt", ".srt"}

SOURCE_TYPE_BY_SUFFIX = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "text",
    ".vtt": "transcript",
    ".srt": "transcript",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            # Page markers double as section anchors for scanned-in papers.
            pages.append(f"[page {page_number}]\n{text}")
    return "\n\n".join(pages)


_TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?[.,]\d{1,3}\s*-->.*$")
_CUE_INDEX_RE = re.compile(r"^\d+$")


def _read_transcript(path: Path) -> str:
    """Strip WEBVTT/SRT cue numbers and timestamps, keep the spoken text."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT":
            continue
        if _TIMESTAMP_RE.match(line) or _CUE_INDEX_RE.match(line):
            continue
        if lines and lines[-1] == line:  # transcripts repeat rolling captions
            continue
        lines.append(line)
    return " ".join(lines)


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".vtt", ".srt"}:
        return _read_transcript(path)
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Sectioning and chunking
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^\s*(?:#{1,4}\s+(?P<md>.+)|(?P<page>\[page \d+\])|(?P<caps>[A-Z][A-Za-z0-9 ,&/-]{2,60}))\s*$"
)

# Headings we actually care about in papers and manuals.
_KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "related work", "method",
    "methods", "methodology", "approach", "experiments", "evaluation",
    "results", "discussion", "limitations", "conclusion", "conclusions",
    "future work", "references", "appendix",
}


def _detect_section(line: str, current: str) -> str:
    match = _HEADING_RE.match(line)
    if not match:
        return current
    if match.group("md"):
        return match.group("md").strip()
    if match.group("page"):
        return match.group("page").strip("[]")
    caps = (match.group("caps") or "").strip()
    if caps.lower().strip(":. ") in _KNOWN_SECTIONS:
        return caps.strip(":. ")
    return current


def _split_with_sections(text: str) -> list[tuple[str, str]]:
    """Return (section, paragraph) pairs, carrying the last seen heading down."""
    blocks: list[tuple[str, str]] = []
    section = "body"
    buffer: list[str] = []

    for line in text.splitlines():
        new_section = _detect_section(line, section)
        if new_section != section:
            if buffer:
                blocks.append((section, "\n".join(buffer).strip()))
                buffer = []
            section = new_section
            continue
        buffer.append(line)

    if buffer:
        blocks.append((section, "\n".join(buffer).strip()))
    return [(s, p) for s, p in blocks if p]


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Character-window chunking that prefers to break on sentence ends."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        window = text[start:end]

        # Prefer a clean sentence boundary in the last quarter of the window.
        if end < len(text):
            pivot = max(
                window.rfind(". "), window.rfind(".\n"),
                window.rfind("! "), window.rfind("? "),
            )
            if pivot > size * 0.6:
                end = start + pivot + 1
                window = text[start:end]

        cleaned = window.strip()
        if cleaned:
            chunks.append(cleaned)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chunks(
    path: Path,
    doc_title: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Turn one file into a list of tagged chunk dicts ready for embedding."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    raw = load_text(path)
    if not raw.strip():
        return []

    doc_id = hashlib.sha1(path.name.encode()).hexdigest()[:12]
    title = doc_title or path.stem.replace("_", " ").replace("-", " ").title()
    source_type = SOURCE_TYPE_BY_SUFFIX[suffix]

    chunks: list[dict] = []
    index = 0
    for section, paragraph in _split_with_sections(raw):
        for piece in _chunk_text(paragraph, chunk_size, chunk_overlap):
            chunks.append({
                "text": piece,
                "doc_id": doc_id,
                "doc_title": title,
                "source_type": source_type,
                "section": section,
                "chunk_index": index,
                "filename": path.name,
            })
            index += 1

    return chunks


def ingest_path(
    client,
    embedder,
    path: Path,
) -> dict:
    """Ingest a single file into the shared collection. Returns a summary."""
    from research_assistant.vectorstore import upsert_chunks

    chunks = build_chunks(path)
    written = upsert_chunks(client, chunks, embedder)
    return {
        "filename": path.name,
        "doc_title": chunks[0]["doc_title"] if chunks else path.stem,
        "chunks": written,
    }


def ingest_directory(client, embedder, directory: Path) -> list[dict]:
    """Ingest every supported file in a directory. Returns per-file summaries."""
    results = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            results.append(ingest_path(client, embedder, path))
    return results
