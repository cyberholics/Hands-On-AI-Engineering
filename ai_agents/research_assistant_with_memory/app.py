"""
Streamlit interface.

Four tabs:
  Ask        run a question through the graph and see the answer, the Critic's
             scores, the evidence it used, and the full event log
  Library    what is indexed, and upload more sources
  Evaluate   run the eval suite and see scores in a table
  Memory     inspect and reset long-term memory

Run with:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from research_assistant.config import (
    DOCUMENTS_COLLECTION,
    EMBEDDING_MODEL,
    MEMORY_COLLECTION,
    OLLAMA_MODEL,
    VECTORAI_URL,
)
from research_assistant.embedder import Embedder
from research_assistant.evaluation import DEFAULT_QUESTIONS, run_evaluation
from research_assistant.graph import build_graph, run_query
from research_assistant.ingest import SUPPORTED_SUFFIXES, ingest_path
from research_assistant.llm import LLM
from research_assistant.memory import SessionState
from research_assistant.observability import RunLogger
from research_assistant.tools import ResearchTools
from research_assistant.vectorstore import (
    get_client,
    get_collection_counts,
    list_documents,
    reset_memory,
    setup_collections,
)

st.set_page_config(page_title="Research Assistant", page_icon="🔬", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="Connecting to VectorAI DB...")
def load_client(dim: int):
    client = get_client()
    setup_collections(client, dim=dim)
    return client


@st.cache_resource(show_spinner="Connecting to Ollama...")
def load_llm() -> LLM:
    return LLM()


@st.cache_resource(show_spinner="Compiling agent graph...")
def load_graph():
    return build_graph()


def get_session() -> SessionState:
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
    return st.session_state.session


def clear_chat() -> None:
    """Reset both the agent session and the chat transcript shown in the UI."""
    session.reset()
    st.session_state.chat_results = []


def build_suggestions(documents: list[dict]) -> dict[str, str]:
    """Turn what's actually indexed into a few clickable example questions."""
    if not documents:
        return {}

    titles = [d["doc_title"] for d in documents]
    suggestions: dict[str, str] = {
        f":material/lightbulb: Main contribution of \"{titles[0]}\"": (
            f'What is the main contribution of "{titles[0]}"?'
        ),
    }
    if len(titles) >= 2:
        suggestions[f":material/compare_arrows: Compare \"{titles[0]}\" and \"{titles[1]}\""] = (
            f'Compare "{titles[0]}" and "{titles[1]}" on their approach and limitations.'
        )
    suggestions[":material/warning: Limitations across all sources"] = (
        "What are the main limitations across these documents?"
    )
    return suggestions


def render_turn(result: dict, log_rows: list[dict], log_ms: int) -> None:
    """Render one assistant turn: answer, Critic scores, and detail expanders."""
    critique = result["critique"]
    plan = result["plan"]

    st.markdown(result["answer"] or "_No answer produced._")

    if plan.get("intent") == "chitchat":
        # No retrieval happened, so there is nothing for the Critic to grade
        # against, showing a scorecard here would just be misleading.
        with st.expander(f"Event log  ·  {log_ms} ms total"):
            st.dataframe(log_rows, width="stretch", hide_index=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correctness", f"{critique.get('correctness', 0):.1f}/5")
    c2.metric("Completeness", f"{critique.get('completeness', 0):.1f}/5")
    c3.metric("Clarity", f"{critique.get('clarity', 0):.1f}/5")
    c4.metric("Overall", f"{critique.get('average', 0):.2f}/5")

    status_cols = st.columns(3)
    status_cols[0].write(
        "**Verdict:** " + ("passed" if critique.get("passed") else "did not pass")
    )
    status_cols[1].write(
        "**Grounded:** " + ("yes" if critique.get("grounded") else "no")
    )
    status_cols[2].write(f"**Revisions:** {result['revisions']}")

    if critique.get("comments"):
        st.info(critique["comments"])

    with st.expander(f"Plan  ·  intent: {plan.get('intent', 'n/a')}"):
        st.write(plan.get("reasoning", ""))
        for sub_query in plan.get("sub_queries", []):
            st.markdown(f"- `{sub_query}`")
        st.caption(f"Used long-term memory: {plan.get('use_memory', False)}")

    evidence = result["evidence"]
    with st.expander(f"Evidence  ·  {len(evidence)} passages"):
        for i, item in enumerate(evidence, start=1):
            st.markdown(
                f"**[{i}] {item.citation()}** · relevance `{item.score}` · "
                f"origin `{item.origin}`"
            )
            st.caption(item.text[:600] + ("..." if len(item.text) > 600 else ""))
            st.divider()

    with st.expander("Research notes"):
        st.markdown(result["notes"] or "_none_")

    with st.expander(f"Event log  ·  {log_ms} ms total"):
        st.dataframe(log_rows, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

st.title("Multi-Agent Research Assistant")
st.caption(
    "Planner, Research, Writer and Critic agents over a shared Actian VectorAI DB "
    "memory layer. Local inference via Ollama."
)

try:
    embedder = load_embedder()
    client = load_client(embedder.dimension)
    llm = load_llm()
    graph = load_graph()
except Exception as exc:
    st.error(f"Startup failed: {exc}")
    st.info(
        "Check that VectorAI DB is running (`docker compose up -d`) and that "
        "Ollama is running (`ollama serve`)."
    )
    st.stop()

session = get_session()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Stack")
    st.text(f"Vector store : {VECTORAI_URL}")
    st.text(f"Embeddings   : {EMBEDDING_MODEL.split('/')[-1]}")
    st.text(f"LLM          : {OLLAMA_MODEL}")

    ok, message = llm.check()
    (st.success if ok else st.warning)(message)

    st.divider()
    st.subheader("Collections")
    counts = get_collection_counts(client)
    documents = list_documents(client)
    st.metric("Document chunks", counts.get(DOCUMENTS_COLLECTION, 0))
    st.metric("Memory entries", counts.get(MEMORY_COLLECTION, 0))

    st.divider()
    st.subheader("Session")
    st.text(f"id      : {session.session_id}")
    st.text(f"turns   : {len(session.turns)}")
    if session.summary:
        with st.expander("Running summary"):
            st.write(session.summary)
    if st.button("New session", width="stretch"):
        clear_chat()
        st.rerun()


ask_tab, library_tab, eval_tab, memory_tab = st.tabs(
    ["Ask", "Library", "Evaluate", "Memory"]
)


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

with ask_tab:
    st.session_state.setdefault("chat_results", [])
    st.session_state.setdefault("suggestion_nonce", 0)

    if counts.get(DOCUMENTS_COLLECTION, 0) == 0:
        st.warning("No documents indexed yet. Add sources in the Library tab first.")

    header = st.container(horizontal=True, vertical_alignment="center")
    with header:
        st.caption("Planner → Research → Writer → Critic runs on every question.")
        if st.session_state.chat_results and st.button(
            "Clear chat", icon=":material/delete_sweep:"
        ):
            clear_chat()
            st.rerun()

    # Replay prior turns as chat bubbles.
    for turn in st.session_state.chat_results:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant", avatar=":material/travel_explore:"):
            render_turn(turn["result"], turn["log_rows"], turn["log_ms"])

    suggestions = build_suggestions(documents)
    selected = None
    if suggestions:
        # Keyed on a nonce so a consumed pick can be reset to unselected on the
        # next run without mutating an already-instantiated widget's state.
        selected = st.pills(
            "Suggested questions",
            list(suggestions.keys()),
            label_visibility="collapsed",
            key=f"ask_suggestion_pick_{st.session_state.suggestion_nonce}",
        )

    prompt = st.chat_input("Ask a question about your documents...")
    if selected and not prompt:
        prompt = suggestions[selected]

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        logger = RunLogger()
        tools = ResearchTools(client, embedder, logger=logger)

        with st.chat_message("assistant", avatar=":material/travel_explore:"):
            with st.spinner("Planner → Research → Writer → Critic..."):
                try:
                    result = run_query(
                        query=prompt,
                        session=session,
                        client=client,
                        embedder=embedder,
                        llm=llm,
                        tools=tools,
                        logger=logger,
                        compiled_graph=graph,
                    )
                except Exception as exc:
                    st.error(f"Run failed: {exc}")
                    st.stop()

            render_turn(result, logger.rows(), logger.total_ms)

        st.session_state.chat_results.append(
            {
                "question": prompt,
                "result": result,
                "log_rows": logger.rows(),
                "log_ms": logger.total_ms,
            }
        )

        if selected:
            # Retire this pills widget so the next run renders a fresh,
            # unselected one instead of replaying the same suggestion forever.
            st.session_state.suggestion_nonce += 1

        # Compress older history between turns, off the critical path.
        if session.needs_compression():
            try:
                session.compress(llm)
            except Exception:
                pass

        st.rerun()


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

with library_tab:
    st.subheader("Indexed documents")

    if documents:
        st.dataframe(documents, width="stretch", hide_index=True)
    else:
        st.info("Nothing indexed yet.")

    st.divider()
    st.subheader("Add sources")
    st.caption("Supported: " + ", ".join(sorted(SUPPORTED_SUFFIXES)))

    uploads = st.file_uploader(
        "Upload documents",
        type=[s.lstrip(".") for s in sorted(SUPPORTED_SUFFIXES)],
        accept_multiple_files=True,
    )

    if uploads and st.button("Ingest", type="primary"):
        progress = st.progress(0.0)
        for index, upload in enumerate(uploads, start=1):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / upload.name
                path.write_bytes(upload.getbuffer())
                try:
                    summary = ingest_path(client, embedder, path)
                    st.write(
                        f"**{summary['doc_title']}** · {summary['chunks']} chunks"
                    )
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")
            progress.progress(index / len(uploads))
        st.success("Ingestion complete.")
        st.rerun()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

with eval_tab:
    st.subheader("Evaluation suite")
    st.caption(
        "Each question runs through the full graph in a fresh session. "
        "Memory writes are disabled so the run cannot contaminate itself."
    )

    questions_text = st.text_area(
        "Questions (one per line)",
        value="\n".join(DEFAULT_QUESTIONS),
        height=140,
    )

    if st.button("Run evaluation", type="primary"):
        questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
        if not questions:
            st.warning("Add at least one question.")
        else:
            status = st.empty()
            progress = st.progress(0.0)

            def on_progress(index: int, total: int, question: str) -> None:
                status.write(f"[{index}/{total}] {question}")
                progress.progress(index / total)

            tools = ResearchTools(client, embedder)
            try:
                report = run_evaluation(
                    questions=questions,
                    client=client,
                    embedder=embedder,
                    llm=llm,
                    tools=tools,
                    progress=on_progress,
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()

            status.empty()
            progress.empty()

            aggregate = report.aggregate()
            if aggregate:
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Mean overall", f"{aggregate['mean_overall']:.2f}/5")
                a2.metric("Pass rate", f"{aggregate['pass_rate']:.0%}")
                a3.metric("Grounded", f"{aggregate['grounded_rate']:.0%}")
                a4.metric("Mean revisions", aggregate["mean_revisions"])

            st.dataframe(report.as_table(), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

with memory_tab:
    st.subheader("Long-term memory")
    st.caption(
        "Findings written back after an answer clears the Critic. These are "
        "retrieved on later questions, which is what makes the assistant "
        "improve across sessions."
    )

    memory_count = get_collection_counts(client).get(MEMORY_COLLECTION, 0)
    st.metric("Stored findings", memory_count)

    probe = st.text_input("Search memory", placeholder="retrieval methods")
    if probe.strip():
        tools = ResearchTools(client, embedder)
        recalled = tools.memory_search(probe.strip(), top_k=5)
        if recalled.evidence:
            for item in recalled.evidence:
                st.markdown(f"**{item.doc_title}** · relevance `{item.score}`")
                st.caption(item.text[:500])
                st.divider()
        else:
            st.info("No matching memory entries.")

    st.divider()
    if st.button("Reset memory", help="Clears findings. Documents are untouched."):
        reset_memory(client)
        st.success("Memory cleared.")
        st.rerun()
