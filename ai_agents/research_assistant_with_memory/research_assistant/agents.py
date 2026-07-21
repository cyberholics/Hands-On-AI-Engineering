"""
The four specialist agents.

  Planner   decides what to do: how to decompose the question into retrieval
            queries, and whether prior session memory is worth pulling.
  Research  executes those queries through the tools and synthesises notes.
  Writer    turns notes into a cited answer.
  Critic    scores the answer and decides whether it goes back for another pass.

Each agent is a plain function over state. Keeping them free of graph wiring
means they can be unit tested on their own, and the graph in graph.py stays
readable.
"""

from __future__ import annotations

from research_assistant.config import (
    CRITIC_PASS_THRESHOLD,
    MEMORY_TOP_K,
    TEMPERATURE_PROSE,
    TOP_K,
)
from research_assistant.tools import (
    Evidence,
    ResearchTools,
    dedupe_evidence,
    format_evidence,
)

MAX_SUBQUERIES = 4


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the Planner in a research team.

Break the user's question into focused search queries that will retrieve the \
right passages from a document collection. Think about what evidence is needed, \
not how to answer.

Return JSON only:
{
  "intent": "single_doc" | "comparison" | "synthesis" | "followup" | "chitchat",
  "sub_queries": ["...", "..."],
  "use_memory": true | false,
  "reasoning": "one sentence"
}

Rules:
- 1 to 4 sub_queries. Each must stand alone as a search query.
- For comparisons, write one sub_query per item being compared.
- Set use_memory true only if the question refers to earlier discussion.
- Use "chitchat" for greetings, thanks, small talk, or meta questions about the \
assistant itself ("who are you", "what can you do", "how does this work"). These \
are not requests for information from the document collection, so sub_queries \
can be left empty.
- No prose outside the JSON."""


def planner_agent(query: str, conversation: str, llm, logger) -> dict:
    """Produce a retrieval plan for this question."""
    fallback = {
        "intent": "synthesis",
        "sub_queries": [query],
        "use_memory": False,
        "reasoning": "Planner fallback: searching the question verbatim.",
    }

    plan = llm.complete_json(
        system=PLANNER_SYSTEM,
        user=f"Conversation so far:\n{conversation}\n\nQuestion: {query}",
        fallback=fallback,
    )

    # Normalise whatever the model returned into something the graph can trust.
    sub_queries = plan.get("sub_queries") or []
    if isinstance(sub_queries, str):
        sub_queries = [sub_queries]
    sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()]
    if not sub_queries:
        sub_queries = [query]

    plan["sub_queries"] = sub_queries[:MAX_SUBQUERIES]
    plan["intent"] = plan.get("intent", "synthesis")
    plan["use_memory"] = bool(plan.get("use_memory", False))
    plan["reasoning"] = str(plan.get("reasoning", ""))[:300]

    logger.log(
        "planner",
        intent=plan["intent"],
        sub_queries=len(plan["sub_queries"]),
        use_memory=plan["use_memory"],
    )
    return plan


# ---------------------------------------------------------------------------
# Chitchat
# ---------------------------------------------------------------------------

CHITCHAT_SYSTEM = """You are the Multi-Agent Research Assistant, replying to a \
greeting, thanks, or a meta question about yourself rather than a real research \
question.

You are a research assistant over the documents in this workspace's library \
(papers, manuals, transcripts, notes), not a general-purpose chatbot. You only \
answer from documents that have actually been added to that library, never from \
your own training knowledge or the web, and a Critic checks every research \
answer for that before it is shown. The library isn't fixed: more documents can \
be added anytime from the Library tab.

Rules:
- Two or three sentences maximum.
- Friendly and direct, no filler openings.
- If asked what you can do or who you are, say so plainly, mention you answer
  from the document library specifically, and suggest asking a question about
  what's in it (or adding more via the Library tab if it's empty).
- Never fabricate document titles or content you have not been shown."""


def chitchat_agent(query: str, conversation: str, llm, logger) -> str:
    """Answer a greeting or meta question directly, skipping retrieval entirely."""
    reply = llm.complete(
        system=CHITCHAT_SYSTEM,
        user=f"Conversation so far:\n{conversation}\n\nMessage: {query}",
        temperature=TEMPERATURE_PROSE,
        max_tokens=150,
    )
    logger.log("chitchat", reply_chars=len(reply))
    return reply


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM = """You are the Research agent.

You are given passages retrieved from a document collection. Write research \
notes that the Writer will use to compose an answer.

Rules:
- Only state what the passages support. Never add outside knowledge.
- Attribute each note to its source using the bracketed labels shown.
- If the passages do not cover part of the question, say so explicitly.
- Bullet points. No introduction, no conclusion."""


def research_agent(
    query: str,
    plan: dict,
    tools: ResearchTools,
    llm,
    logger,
    revision_note: str = "",
) -> tuple[str, list[Evidence]]:
    """Run the plan's queries through the tools and synthesise notes."""
    collected: list[Evidence] = []

    for sub_query in plan["sub_queries"]:
        result = tools.doc_search(sub_query, top_k=TOP_K)
        collected.extend(result.evidence)

    if plan.get("use_memory"):
        recalled = tools.memory_search(query, top_k=MEMORY_TOP_K)
        collected.extend(recalled.evidence)

    evidence = dedupe_evidence(collected)

    if not evidence:
        logger.log("research", evidence=0, note="no evidence retrieved")
        return (
            "No relevant passages were found in the indexed documents for this "
            "question.",
            [],
        )

    extra = f"\n\nThe Critic asked for another pass: {revision_note}" if revision_note else ""
    notes = llm.complete(
        system=RESEARCH_SYSTEM,
        user=(
            f"Question: {query}\n\n"
            f"Retrieved passages:\n{format_evidence(evidence)}{extra}\n\n"
            "Write the research notes."
        ),
        temperature=0.2,
        max_tokens=900,
    )

    logger.log("research", evidence=len(evidence), notes_chars=len(notes))
    return notes, evidence


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You are the Writer agent.

Compose the final answer from the research notes. You may not introduce facts \
that are absent from the notes.

Format:
- Open with a direct 1-2 sentence answer to the question.
- Then supporting detail under short markdown headings where it helps.
- Cite sources inline using the bracketed labels from the notes.
- If the notes flag a gap, state the limitation plainly rather than papering \
over it.
- No filler openings like "Certainly" or "In this response"."""


def writer_agent(
    query: str,
    notes: str,
    conversation: str,
    llm,
    logger,
    revision_note: str = "",
) -> str:
    """Turn research notes into a cited answer."""
    extra = f"\n\nAddress this Critic feedback in the rewrite: {revision_note}" if revision_note else ""

    answer = llm.complete(
        system=WRITER_SYSTEM,
        user=(
            f"Conversation so far:\n{conversation}\n\n"
            f"Question: {query}\n\n"
            f"Research notes:\n{notes}{extra}\n\n"
            "Write the answer."
        ),
        temperature=TEMPERATURE_PROSE,
        max_tokens=1200,
    )

    logger.log("writer", answer_chars=len(answer), revision=bool(revision_note))
    return answer


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are the Critic. You grade another agent's answer \
strictly against the evidence it was given.

Return JSON only:
{
  "correctness": 1-5,
  "completeness": 1-5,
  "clarity": 1-5,
  "grounded": true | false,
  "comments": "two sentences maximum",
  "revision_request": "what to fix, or empty string if the answer passes"
}

Scoring guide:
- correctness: are the claims supported by the evidence? Unsupported claims \
score 2 or below.
- completeness: does it address every part of the question?
- clarity: is it direct and well organised?
- grounded: false if any claim cannot be traced to the evidence.

Be harsh. A confident answer built on thin evidence is a failure, not a pass."""


def critic_agent(query: str, answer: str, evidence_block: str, llm, logger) -> dict:
    """Score the answer and decide whether another pass is warranted."""
    fallback = {
        "correctness": 3,
        "completeness": 3,
        "clarity": 3,
        "grounded": True,
        "comments": "Critic unavailable; answer passed through ungraded.",
        "revision_request": "",
    }

    verdict = llm.complete_json(
        system=CRITIC_SYSTEM,
        user=(
            f"Question: {query}\n\n"
            f"Evidence provided to the writer:\n{evidence_block}\n\n"
            f"Answer to grade:\n{answer}\n\n"
            "Grade it."
        ),
        fallback=fallback,
    )

    def _score(key: str) -> float:
        try:
            value = float(verdict.get(key, 3))
        except (TypeError, ValueError):
            value = 3.0
        return max(1.0, min(5.0, value))

    scores = {
        "correctness": _score("correctness"),
        "completeness": _score("completeness"),
        "clarity": _score("clarity"),
    }
    average = round(sum(scores.values()) / 3, 2)

    result = {
        **scores,
        "average": average,
        "grounded": bool(verdict.get("grounded", True)),
        "comments": str(verdict.get("comments", ""))[:500],
        "revision_request": str(verdict.get("revision_request", ""))[:300],
        "passed": average >= CRITIC_PASS_THRESHOLD and bool(verdict.get("grounded", True)),
    }

    logger.log(
        "critic",
        average=average,
        grounded=result["grounded"],
        passed=result["passed"],
    )
    return result
