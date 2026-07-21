"""
Local inference through Ollama.

One thin wrapper shared by all four agents. Nothing leaves the machine.

`complete_json` asks Ollama for structured output and repairs the common
failure mode where a small model wraps its JSON in prose or a code fence.
"""

from __future__ import annotations

import json
import re

from ollama import Client

from research_assistant.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TEMPERATURE_PROSE,
    TEMPERATURE_STRUCTURED,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> None:
        self.model = model
        self.host = host
        self._client = Client(host=host)

    # -- health ------------------------------------------------------------

    def check(self) -> tuple[bool, str]:
        """Verify Ollama is reachable and the configured model is pulled."""
        try:
            listed = self._client.list()
        except Exception as exc:
            return False, f"Cannot reach Ollama at {self.host}: {exc}"

        # Client.list() returns a ListResponse (Pydantic model), not a dict.
        names = {m.model for m in listed.models if m.model}
        # Ollama reports "llama3.2:3b"; tolerate a bare "llama3.2" config too.
        if self.model in names or any(n.split(":")[0] == self.model for n in names):
            return True, f"Ollama ready with {self.model}"
        return False, (
            f"Model '{self.model}' not found on {self.host}. "
            f"Run: ollama pull {self.model}"
        )

    # -- prose -------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = TEMPERATURE_PROSE,
        max_tokens: int = 1200,
    ) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                think=False,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
        except Exception as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc

        # Client.chat() returns a ChatResponse (Pydantic model); response.message
        # is a Message object, not a dict.
        return (response.message.content or "").strip()

    # -- structured --------------------------------------------------------

    def complete_json(
        self,
        system: str,
        user: str,
        fallback: dict,
        temperature: float = TEMPERATURE_STRUCTURED,
        max_tokens: int = 800,
    ) -> dict:
        """
        Ask for JSON and always return a dict.

        Small local models drift out of strict JSON often enough that a
        fallback is not optional. Returning `fallback` keeps the graph moving
        instead of crashing a run halfway through.
        """
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format="json",
                think=False,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            raw = (response.message.content or "").strip()
        except Exception:
            return dict(fallback)

        parsed = _parse_json(raw)
        return parsed if isinstance(parsed, dict) else dict(fallback)


def _parse_json(raw: str):
    """Best-effort JSON extraction from a possibly chatty response."""
    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_RE.search(raw)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
