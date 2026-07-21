"""
Retrieval tools.

Agents never read documents directly. They call these two tools, which are
the only path to the shared Actian collection. Keeping it this way is what
makes the system auditable: every piece of evidence in a final answer can be
traced back to a logged tool call.

  doc_search(query)          semantic search across all indexed material
  get_section(doc_id, name)  pull a named section from one document
  memory_search(query)       recall findings from earlier sessions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from research_assistant.config import MEMORY_TOP_K, MIN_SCORE, TOP_K
from research_assistant.vectorstore import (
    scroll_documents,
    search_documents,
    search_memory,
)


@dataclass
class Evidence:
    """One retrieved chunk, normalised for prompting and citation."""

    text: str
    doc_id: str
    doc_title: str
    section: str
    source_type: str
    score: float
    origin: str = "documents"  # documents | memory
    chunk_index: int = 0

    def citation(self) -> str:
        if self.origin == "memory":
            return f"[memory: {self.doc_title}]"
        if self.section and self.section != "body":
            return f"[{self.doc_title} — {self.section}]"
        return f"[{self.doc_title}]"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    tool: str
    query: str
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.evidence)


def _hit_to_evidence(hit, origin: str) -> Evidence:
    payload = getattr(hit, "payload", {}) or {}
    if origin == "memory":
        text = payload.get("finding", "")
        title = payload.get("query", "earlier finding")
        section = "long-term memory"
    else:
        text = payload.get("text", "")
        title = payload.get("doc_title", payload.get("doc_id", "unknown"))
        section = payload.get("section", "body")

    return Evidence(
        text=text,
        doc_id=payload.get("doc_id", payload.get("session_id", "")),
        doc_title=title,
        section=section,
        source_type=payload.get("source_type", origin),
        score=round(float(getattr(hit, "score", 0.0)), 3),
        origin=origin,
        chunk_index=int(payload.get("chunk_index", 0)),
    )


class ResearchTools:
    """Bundle of retrieval tools bound to one client and embedder."""

    def __init__(self, client, embedder, logger=None) -> None:
        self.client = client
        self.embedder = embedder
        self.logger = logger

    def _log(self, tool: str, query: str, result_count: int) -> None:
        if self.logger:
            self.logger.log(
                "tool_call",
                tool=tool,
                query=query[:200],
                results=result_count,
            )

    # -- semantic search ---------------------------------------------------

    def doc_search(self, query: str, top_k: int = TOP_K) -> ToolResult:
        """Semantic search across every ingested document."""
        vector = self.embedder.embed_query(query)
        hits = search_documents(self.client, vector, top_k=top_k)
        evidence = [_hit_to_evidence(h, "documents") for h in hits]
        evidence = [e for e in evidence if e.score >= MIN_SCORE and e.text.strip()]
        self._log("doc_search", query, len(evidence))
        return ToolResult(tool="doc_search", query=query, evidence=evidence)

    # -- metadata lookup ---------------------------------------------------

    def get_section(self, doc_id: str, section: str, limit: int = 6) -> ToolResult:
        """
        Pull chunks belonging to a named section of one document.

        Filters on payload metadata rather than similarity, so it is the right
        tool when the agent already knows exactly where to look, for example
        "the limitations section of paper B".
        """
        wanted = section.lower().strip()
        matches: list[Evidence] = []

        for point in scroll_documents(self.client):
            payload = getattr(point, "payload", {}) or {}
            if doc_id and payload.get("doc_id") != doc_id:
                continue
            if wanted and wanted not in str(payload.get("section", "")).lower():
                continue
            matches.append(
                Evidence(
                    text=payload.get("text", ""),
                    doc_id=payload.get("doc_id", ""),
                    doc_title=payload.get("doc_title", ""),
                    section=payload.get("section", "body"),
                    source_type=payload.get("source_type", "documents"),
                    score=1.0,
                    origin="documents",
                    chunk_index=int(payload.get("chunk_index", 0)),
                )
            )

        matches.sort(key=lambda e: e.chunk_index)
        matches = matches[:limit]
        self._log("get_section", f"{doc_id}:{section}", len(matches))
        return ToolResult(
            tool="get_section", query=f"{doc_id}:{section}", evidence=matches
        )

    # -- memory ------------------------------------------------------------

    def memory_search(self, query: str, top_k: int = MEMORY_TOP_K) -> ToolResult:
        """Recall findings written during earlier sessions."""
        vector = self.embedder.embed_query(query)
        hits = search_memory(self.client, vector, top_k=top_k)
        evidence = [_hit_to_evidence(h, "memory") for h in hits]
        evidence = [e for e in evidence if e.score >= MIN_SCORE and e.text.strip()]
        self._log("memory_search", query, len(evidence))
        return ToolResult(tool="memory_search", query=query, evidence=evidence)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_evidence(evidence: list[Evidence], max_chars: int = 6000) -> str:
    """Render evidence into a numbered, citable block for the prompt."""
    if not evidence:
        return "No evidence retrieved."

    lines: list[str] = []
    budget = max_chars
    for i, item in enumerate(evidence, start=1):
        snippet = item.text.strip()
        if len(snippet) > budget:
            snippet = snippet[: max(budget, 0)].rstrip() + "..."
        if not snippet:
            break
        lines.append(
            f"[{i}] {item.citation()} (relevance {item.score})\n{snippet}"
        )
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n".join(lines)


def dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Drop duplicate chunks that several queries pulled back, keep best score."""
    best: dict[tuple, Evidence] = {}
    for item in evidence:
        key = (item.doc_id, item.chunk_index, item.origin)
        if key not in best or item.score > best[key].score:
            best[key] = item
    return sorted(best.values(), key=lambda e: e.score, reverse=True)
