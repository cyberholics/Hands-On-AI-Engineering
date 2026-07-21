"""
Session state and context engineering.

Two layers of memory:

  Short-term  the current conversation, held in SessionState. Recent turns are
              kept verbatim; older turns are compressed into a running summary
              so the prompt stays small without losing the thread.

  Long-term   findings written back into the Actian memory collection after a
              run passes the Critic. These survive process restarts and are
              retrieved by the Research agent on later questions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# Turns kept verbatim before older history gets compressed.
VERBATIM_TURNS = 4


@dataclass
class Turn:
    query: str
    answer: str
    score: float


@dataclass
class SessionState:
    """Everything that persists within one conversation."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    preferences: list[str] = field(default_factory=list)

    def add_turn(self, query: str, answer: str, score: float) -> None:
        self.turns.append(Turn(query=query, answer=answer, score=score))

    # -- context engineering ----------------------------------------------

    def recent_turns(self, n: int = VERBATIM_TURNS) -> list[Turn]:
        return self.turns[-n:]

    def needs_compression(self) -> bool:
        return len(self.turns) > VERBATIM_TURNS

    def stale_turns(self, n: int = VERBATIM_TURNS) -> list[Turn]:
        return self.turns[:-n] if self.needs_compression() else []

    def conversation_context(self, max_chars: int = 1500) -> str:
        """
        Render conversation history for a prompt: the running summary first,
        then the most recent turns verbatim.
        """
        parts: list[str] = []
        if self.summary:
            parts.append(f"Earlier in this session: {self.summary}")

        for turn in self.recent_turns():
            answer = turn.answer.strip().replace("\n", " ")
            if len(answer) > 300:
                answer = answer[:300].rstrip() + "..."
            parts.append(f"Q: {turn.query}\nA: {answer}")

        if self.preferences:
            parts.append("User preferences: " + "; ".join(self.preferences))

        context = "\n\n".join(parts)
        return context[:max_chars] if context else "This is the first question."

    def compress(self, llm) -> None:
        """
        Fold older turns into the running summary.

        Called after a turn completes, so the cost lands between questions
        rather than inside the critical path of an answer.
        """
        stale = self.stale_turns()
        if not stale:
            return

        transcript = "\n".join(
            f"Q: {t.query}\nA: {t.answer[:300]}" for t in stale
        )
        prior = f"Existing summary: {self.summary}\n\n" if self.summary else ""

        self.summary = llm.complete(
            system=(
                "You compress research conversations. Produce a factual summary "
                "in 3 sentences or fewer covering what was asked and what was "
                "concluded. No preamble."
            ),
            user=f"{prior}New exchanges to fold in:\n{transcript}",
            temperature=0.2,
            max_tokens=220,
        ).strip()

        # Older turns are now represented by the summary.
        self.turns = self.turns[-VERBATIM_TURNS:]

    def note_preference(self, preference: str) -> None:
        cleaned = preference.strip()
        if cleaned and cleaned not in self.preferences:
            self.preferences.append(cleaned)

    def reset(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.preferences.clear()
        self.session_id = uuid.uuid4().hex[:12]
