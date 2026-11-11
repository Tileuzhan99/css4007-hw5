"""A scripted stand-in for the real model, so the tests need no API key.

It is a fake, not a mock of the Anthropic API: it implements only what
`HelpdeskApp` uses - `bind_tools()` and `invoke()` - and returns the tool calls
its script tells it to. That lets CI check the graph, the tools, the loop limit
and the memory wiring without spending a token. The Part 2 demonstration runs
against the real model and is kept separate (see README).
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import AIMessage


class FakeToolModel:
    """Replays a script of turns.

    Each script entry is either:
      {"text": "..."}                               - a plain answer, ends the turn
      {"tools": [{"name": ..., "args": {...}}, ...]} - a round of tool requests

    When the script runs out, the model answers with `fallback_text`.
    """

    def __init__(
        self,
        script: Iterable[dict[str, Any]] | None = None,
        *,
        fallback_text: str = "Done.",
        name: str = "fake-model",
    ) -> None:
        self.script = list(script or [])
        self.fallback_text = fallback_text
        self.model = name
        self.bound_tools: list[Any] = []
        self.requests: list[list[Any]] = []
        self._next_call_id = 0

    # HelpdeskApp calls this once, at construction time.
    def bind_tools(self, tool_list: list[Any]) -> "FakeToolModel":
        self.bound_tools = list(tool_list)
        return self

    def invoke(self, messages: list[Any], **_: Any) -> AIMessage:
        self.requests.append(list(messages))
        step = self.script.pop(0) if self.script else {"text": self.fallback_text}

        if "tools" in step:
            calls = []
            for requested in step["tools"]:
                self._next_call_id += 1
                calls.append(
                    {
                        "name": requested["name"],
                        "args": requested.get("args", {}),
                        "id": f"fake_call_{self._next_call_id}",
                        "type": "tool_call",
                    }
                )
            return AIMessage(content=step.get("text", ""), tool_calls=calls)

        return AIMessage(content=step.get("text", self.fallback_text))

    # -- helpers for tests -------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return [getattr(t, "name", str(t)) for t in self.bound_tools]

    def last_system_prompt(self) -> str:
        """The system message of the most recent request, as plain text."""
        if not self.requests:
            return ""
        first = self.requests[-1][0]
        content = getattr(first, "content", "")
        return content if isinstance(content, str) else str(content)


class AlwaysToolsModel(FakeToolModel):
    """Never stops asking for a tool - used to prove the round limit works."""

    def invoke(self, messages: list[Any], **_: Any) -> AIMessage:
        self.requests.append(list(messages))
        self._next_call_id += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "find_instructions",
                    "args": {"topic": "wifi"},
                    "id": f"fake_call_{self._next_call_id}",
                    "type": "tool_call",
                }
            ],
        )
