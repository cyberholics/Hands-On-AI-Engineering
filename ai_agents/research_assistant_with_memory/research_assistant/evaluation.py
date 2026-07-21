"""
Automated evaluation.

Runs a fixed set of questions through the full graph and collects the Critic's
scores into a table. This is the difference between "the demo worked once" and
knowing whether a prompt change made the system better or worse.

Each question runs in a fresh session so results are independent, and memory
writes are disabled during evaluation to avoid the run contaminating itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from research_assistant.graph import build_graph, run_query
from research_assistant.memory import SessionState
from research_assistant.observability import RunLogger

DEFAULT_QUESTIONS = [
    "What is the main contribution of the retrieval-augmented generation paper?",
    "Compare the retrieval methods described across the indexed documents.",
    "What limitations do the authors acknowledge in their own methods?",
    "What evaluation metrics are used, and are they justified?",
]


@dataclass
class EvalRow:
    question: str
    correctness: float
    completeness: float
    clarity: float
    average: float
    grounded: bool
    passed: bool
    revisions: int
    evidence_count: int
    elapsed_ms: int
    comments: str = ""

    def to_row(self) -> dict:
        return {
            "question": self.question,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "average": self.average,
            "grounded": self.grounded,
            "passed": self.passed,
            "revisions": self.revisions,
            "evidence": self.evidence_count,
            "ms": self.elapsed_ms,
            "comments": self.comments,
        }


@dataclass
class EvalReport:
    rows: list[EvalRow] = field(default_factory=list)

    def add(self, row: EvalRow) -> None:
        self.rows.append(row)

    def as_table(self) -> list[dict]:
        return [r.to_row() for r in self.rows]

    def aggregate(self) -> dict:
        if not self.rows:
            return {}
        n = len(self.rows)
        return {
            "questions": n,
            "mean_correctness": round(sum(r.correctness for r in self.rows) / n, 2),
            "mean_completeness": round(sum(r.completeness for r in self.rows) / n, 2),
            "mean_clarity": round(sum(r.clarity for r in self.rows) / n, 2),
            "mean_overall": round(sum(r.average for r in self.rows) / n, 2),
            "pass_rate": round(sum(1 for r in self.rows if r.passed) / n, 2),
            "grounded_rate": round(sum(1 for r in self.rows if r.grounded) / n, 2),
            "mean_revisions": round(sum(r.revisions for r in self.rows) / n, 2),
        }


def run_evaluation(
    questions: list[str],
    client,
    embedder,
    llm,
    tools,
    progress=None,
) -> EvalReport:
    """
    Execute the evaluation suite.

    `progress` is an optional callback(index, total, question) so a UI can show
    where it is without this module importing Streamlit.
    """
    report = EvalReport()
    graph = build_graph()
    total = len(questions)

    for index, question in enumerate(questions, start=1):
        if progress:
            progress(index, total, question)

        logger = RunLogger()
        session = SessionState()

        result = run_query(
            query=question,
            session=session,
            client=client,
            embedder=embedder,
            llm=llm,
            tools=tools,
            logger=logger,
            compiled_graph=graph,
            persist_memory=False,
        )

        critique = result.get("critique", {})
        report.add(
            EvalRow(
                question=question,
                correctness=critique.get("correctness", 0.0),
                completeness=critique.get("completeness", 0.0),
                clarity=critique.get("clarity", 0.0),
                average=critique.get("average", 0.0),
                grounded=bool(critique.get("grounded", False)),
                passed=bool(critique.get("passed", False)),
                revisions=result.get("revisions", 0),
                evidence_count=len(result.get("evidence", [])),
                elapsed_ms=logger.total_ms,
                comments=critique.get("comments", ""),
            )
        )

    return report
