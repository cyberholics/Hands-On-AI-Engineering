"""
LangGraph state machine wiring the four agents together.

               [greeting/meta?] -> chitchat -----------------------> END
              /
    plan ----+
              \\
               [research question] -> research -> write -> critique -> [pass?] -> persist -> END
                                            ^                     |
                                            +------- revise ------+

The conditional edge out of `plan` sends greetings and meta questions ("hi",
"who are you") straight to a lightweight reply with no retrieval, no Critic,
and nothing written to memory, instead of forcing them through a four-agent
pipeline over a document collection they were never asking about.

The conditional edge out of `critique` is the reason the research branch is a
graph and not a sequential pipeline. When the Critic rejects an answer,
control goes back to `research` with the revision request attached, so the
second pass retrieves against the specific gap rather than repeating the
first attempt.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from research_assistant.agents import (
    chitchat_agent,
    critic_agent,
    planner_agent,
    research_agent,
    writer_agent,
)
from research_assistant.config import MAX_REVISIONS
from research_assistant.tools import format_evidence
from research_assistant.vectorstore import store_memory


class ResearchState(TypedDict, total=False):
    # Inputs
    query: str
    conversation: str
    session_id: str

    # Working state
    plan: dict
    notes: str
    evidence: list
    answer: str
    critique: dict
    revision_note: str
    revisions: int

    # Injected runtime dependencies (client, embedder, llm, tools, logger).
    # Passed through state so nodes stay pure functions with no globals.
    _deps: dict[str, Any]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _plan_node(state: ResearchState) -> dict:
    deps = state["_deps"]
    plan = planner_agent(
        query=state["query"],
        conversation=state.get("conversation", ""),
        llm=deps["llm"],
        logger=deps["logger"],
    )
    return {"plan": plan, "revisions": state.get("revisions", 0)}


def _chitchat_node(state: ResearchState) -> dict:
    """
    Answer a greeting or meta question directly, no retrieval or grading.

    Routed here instead of `research` when the Planner classifies the message
    as chitchat, so "hi" does not trigger a multi-minute four-agent pipeline
    over a document collection it was never actually asking about.
    """
    deps = state["_deps"]
    answer = chitchat_agent(
        query=state["query"],
        conversation=state.get("conversation", ""),
        llm=deps["llm"],
        logger=deps["logger"],
    )
    return {"answer": answer, "evidence": [], "notes": ""}


def _research_node(state: ResearchState) -> dict:
    deps = state["_deps"]
    notes, evidence = research_agent(
        query=state["query"],
        plan=state["plan"],
        tools=deps["tools"],
        llm=deps["llm"],
        logger=deps["logger"],
        revision_note=state.get("revision_note", ""),
    )
    return {"notes": notes, "evidence": evidence}


def _write_node(state: ResearchState) -> dict:
    deps = state["_deps"]
    answer = writer_agent(
        query=state["query"],
        notes=state.get("notes", ""),
        conversation=state.get("conversation", ""),
        llm=deps["llm"],
        logger=deps["logger"],
        revision_note=state.get("revision_note", ""),
    )
    return {"answer": answer}


def _critique_node(state: ResearchState) -> dict:
    deps = state["_deps"]
    evidence = state.get("evidence") or []
    critique = critic_agent(
        query=state["query"],
        answer=state.get("answer", ""),
        evidence_block=format_evidence(evidence, max_chars=3500),
        llm=deps["llm"],
        logger=deps["logger"],
    )
    return {"critique": critique}


def _persist_node(state: ResearchState) -> dict:
    """
    Write the finding into long-term memory.

    Only answers that cleared the Critic are stored. Persisting a rejected
    answer would poison future retrievals, which is the failure mode that makes
    naive "agent memory" get worse over time instead of better.
    """
    deps = state["_deps"]
    critique = state.get("critique", {})
    logger = deps["logger"]

    if not deps.get("persist_memory", True):
        logger.log("memory_write", stored=False, reason="persistence disabled")
        return {}

    if not critique.get("passed"):
        logger.log("memory_write", stored=False, reason="did not pass critic")
        return {}

    evidence = state.get("evidence") or []
    doc_ids = sorted({e.doc_id for e in evidence if getattr(e, "doc_id", "")})

    try:
        store_memory(
            client=deps["client"],
            embedder=deps["embedder"],
            query=state["query"],
            finding=state.get("answer", ""),
            session_id=state.get("session_id", ""),
            score=critique.get("average", 0.0),
            doc_ids=doc_ids,
        )
        logger.log("memory_write", stored=True, docs=len(doc_ids))
    except Exception as exc:  # memory is a nice-to-have, never fail the run
        logger.log("memory_write", stored=False, error=str(exc)[:200])

    return {}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def _route_after_plan(state: ResearchState) -> str:
    """Skip retrieval entirely for greetings and meta questions."""
    if state.get("plan", {}).get("intent") == "chitchat":
        return "chitchat"
    return "research"


def _should_revise(state: ResearchState) -> str:
    """Route out of the Critic: another research pass, or persist and finish."""
    critique = state.get("critique", {})
    revisions = state.get("revisions", 0)

    if critique.get("passed"):
        return "persist"
    if revisions >= MAX_REVISIONS:
        state["_deps"]["logger"].log(
            "revision_limit", revisions=revisions, note="accepting best effort"
        )
        return "persist"
    return "revise"


def _revise_node(state: ResearchState) -> dict:
    """Bump the counter and carry the Critic's request into the next pass."""
    critique = state.get("critique", {})
    revisions = state.get("revisions", 0) + 1
    note = critique.get("revision_request") or critique.get("comments", "")

    state["_deps"]["logger"].log("revise", attempt=revisions, request=note[:160])
    return {"revisions": revisions, "revision_note": note}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Compile the research graph. Built once, reused for every question."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan", _plan_node)
    graph.add_node("chitchat", _chitchat_node)
    graph.add_node("research", _research_node)
    graph.add_node("write", _write_node)
    graph.add_node("critique", _critique_node)
    graph.add_node("revise", _revise_node)
    graph.add_node("persist", _persist_node)

    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"chitchat": "chitchat", "research": "research"},
    )
    graph.add_edge("research", "write")
    graph.add_edge("write", "critique")

    graph.add_conditional_edges(
        "critique",
        _should_revise,
        {"revise": "revise", "persist": "persist"},
    )

    # The loop: a rejected answer goes back for targeted retrieval.
    graph.add_edge("revise", "research")
    graph.add_edge("persist", END)
    graph.add_edge("chitchat", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_query(
    query: str,
    session,
    client,
    embedder,
    llm,
    tools,
    logger,
    compiled_graph=None,
    persist_memory: bool = True,
) -> dict:
    """
    Answer one question end to end.

    Returns the final state: answer, critique, evidence, plan and revision count.
    """
    graph = compiled_graph or build_graph()

    initial: ResearchState = {
        "query": query,
        "conversation": session.conversation_context(),
        "session_id": session.session_id,
        "revisions": 0,
        "revision_note": "",
        "_deps": {
            "client": client,
            "embedder": embedder,
            "llm": llm,
            "tools": tools,
            "logger": logger,
            "persist_memory": persist_memory,
        },
    }

    final = graph.invoke(initial)

    critique = final.get("critique", {})
    session.add_turn(
        query=query,
        answer=final.get("answer", ""),
        score=critique.get("average", 0.0),
    )

    return {
        "answer": final.get("answer", ""),
        "critique": critique,
        "evidence": final.get("evidence", []),
        "plan": final.get("plan", {}),
        "notes": final.get("notes", ""),
        "revisions": final.get("revisions", 0),
    }
