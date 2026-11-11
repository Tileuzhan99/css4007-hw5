"""Parts 2 and 3 - the model decides, the graph executes, memory persists.

    START -> load_memory -> call_model -> (tool calls?) -> run_tools -> call_model
                                       \\-> END                     \\-> loop_limit -> END

`call_model` only *requests* tools; `run_tools` is the only place a tool
actually runs. The round limit lives in `route_after_model`, so a model that
keeps asking for tools is stopped by the graph, not by the prompt.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import config, memory, model as model_module, tools

SYSTEM_PROMPT = """You are the assistant for a *practice* university IT helpdesk.
Everything you can reach is fictional local demo data. You never touch real
university systems and you never send a ticket anywhere; prepare_ticket only
writes a local draft.

You can help with Wi-Fi problems and password/account-recovery problems.

Tools and how to use them:
- find_instructions: use it before giving steps, so the steps come from the
  stored guide rather than from memory.
- check_service_status: use it whenever the user asks whether something is
  down, or before concluding that a service is fine. The status is three-valued.
  "operational" means working, "outage" means a known outage, and "unknown"
  means the status source itself was unavailable - in that case say you could
  not determine the status, do not claim the service is operating, and do not
  invent a repair time.
- prepare_ticket: use it only when the user wants a ticket and you have both
  the problem description and the steps they already tried, in their words. If
  the tool reports missing_information, ask the user for exactly what is
  missing. Never invent troubleshooting steps the user did not mention, and
  never put a password, recovery code or API key in a ticket.

If the user made the ticket conditional ("prepare one if Wi-Fi is working"),
check the status first and honour the condition: if the condition does not
hold, say so and explain that you did not prepare the ticket.

Facts you did not get from a tool, from this conversation, or from the user
memory below are not available to you. Say so instead of guessing.

Keep replies short and plain. Say where a fact came from (the stored guide, the
status snapshot, this conversation, or the saved user preference)."""


class AgentState(TypedDict):
    """State for the Part 2/3 graph. Persisted per thread_id by the checkpointer."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_memory: dict[str, str]
    tool_rounds: int
    loop_limit_reached: bool


def default_model(**kwargs: Any):
    """The real tool-capable model for whichever provider is configured."""
    return model_module.build_chat_model(**kwargs)


class HelpdeskApp:
    """One running helpdesk session: one user, one thread, one status snapshot.

    Holds the compiled graph, the checkpointer and the session context the tools
    read. `send()` runs one turn and returns the reply plus that turn's trace.
    """

    def __init__(
        self,
        *,
        user_id: str,
        thread_id: str,
        status_profile: str = config.DEFAULT_STATUS_PROFILE,
        model: Any = None,
        checkpointer: Any = None,
        user_store: memory.UserMemoryStore | None = None,
        max_tool_rounds: int = config.MAX_TOOL_ROUNDS,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.session = tools.SessionContext(
            user_id=user_id, thread_id=thread_id, status_profile=status_profile
        )
        self.max_tool_rounds = max_tool_rounds
        self._turn_trace: list[dict[str, Any]] = []
        self._user_store = user_store or memory.UserMemoryStore()
        self._owns_user_store = user_store is None
        if model is None:
            model = (model_factory or default_model)()
        self._model = model.bind_tools(tools.LANGCHAIN_TOOLS)
        self._checkpointer = checkpointer
        self._graph = self._build()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._owns_user_store:
            self._user_store.close()

    def __enter__(self) -> "HelpdeskApp":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- tracing -----------------------------------------------------------

    def _log(self, node: str, **fields: Any) -> None:
        self._turn_trace.append({"node": node, **fields})

    # -- nodes -------------------------------------------------------------

    def _load_memory(self, state: AgentState) -> dict[str, Any]:
        """Read user memory, and learn a preference if this message states one.

        Runs first on every turn. The user identifier comes from the session,
        never from the message, so a user cannot ask to load someone else's
        memory.
        """
        user_id = self.session.user_id
        latest = ""
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                latest = str(message.content)
                break

        learned = memory.detect_language_preference(latest)
        if learned:
            self._user_store.set_language(user_id, learned)

        remembered = self._user_store.all_for_user(user_id)
        self._log(
            "load_memory",
            user_id=user_id,
            thread_id=self.session.thread_id,
            learned_preference=learned,
            user_memory=remembered,
            source="user memory (local-storage/user_memory.sqlite)",
            conversation_messages_before_turn=len(state["messages"]) - 1,
        )
        # tool_rounds is reset here, not carried over from the previous turn.
        return {"user_memory": remembered, "tool_rounds": 0, "loop_limit_reached": False}

    def _system_message(self, state: AgentState) -> SystemMessage:
        remembered = state.get("user_memory") or {}
        if remembered:
            lines = "\n".join(f"- {key}: {value}" for key, value in sorted(remembered.items()))
            block = (
                "Saved user memory for this user (from earlier conversations; "
                f"source: user memory store):\n{lines}\n"
                "If reply_language is set, write your reply in that language."
            )
        else:
            block = (
                "Saved user memory for this user: nothing recorded yet. If asked "
                "about a preference or an earlier problem, say that nothing is "
                "recorded for this user and offer to note it."
            )
        return SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n{block}\n\n"
                f"Status snapshot selected by the application: "
                f"{self.session.status_profile}. You cannot change it."
            )
        )

    def _call_model(self, state: AgentState) -> dict[str, Any]:
        """Ask the model what to do. It may answer, or request tools."""
        request = [self._system_message(state), *state["messages"]]
        response = self._model.invoke(request)
        requested = [
            {"name": call["name"], "args": call["args"]}
            for call in getattr(response, "tool_calls", []) or []
        ]
        self._log(
            "call_model",
            model=model_module.model_name(),
            provider=model_module.provider(),
            messages_sent=len(request),
            tool_calls_requested=requested,
            text=_text_of(response) if not requested else "",
        )
        return {"messages": [response]}

    def _run_tools(self, state: AgentState) -> dict[str, Any]:
        """Execute every tool the model asked for, in one round."""
        last = state["messages"][-1]
        results: list[ToolMessage] = []
        with tools.use_session(self.session):
            for call in last.tool_calls:
                name = call["name"]
                implementation = tools.TOOLS_BY_NAME.get(name)
                if implementation is None:
                    payload: Any = {
                        "ok": False,
                        "error": "unknown_tool",
                        "message": f"{name} is not a tool of this helpdesk.",
                    }
                else:
                    try:
                        payload = implementation.invoke(call["args"])
                    except Exception as exc:  # surfaced to the model, not raised
                        payload = {
                            "ok": False,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                results.append(
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id=call["id"],
                        name=name,
                    )
                )
                self._log("run_tools", tool=name, args=call["args"], result=payload)
        return {"messages": results, "tool_rounds": state["tool_rounds"] + 1}

    def _loop_limit(self, state: AgentState) -> dict[str, Any]:
        """Stop a model that will not stop asking for tools.

        The pending tool calls still get results - an error saying the limit was
        reached - because a tool_use block left unanswered would make the next
        request to the API invalid.
        """
        last = state["messages"][-1]
        closing: list[AnyMessage] = []
        for call in getattr(last, "tool_calls", []) or []:
            closing.append(
                ToolMessage(
                    content=json.dumps(
                        {
                            "ok": False,
                            "error": "tool_round_limit_reached",
                            "limit": self.max_tool_rounds,
                        }
                    ),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )
        closing.append(
            AIMessage(
                content=(
                    f"I stopped after {self.max_tool_rounds} rounds of tool calls "
                    "without reaching an answer. Nothing was sent. Please tell me "
                    "the problem and the steps you already tried, and I will try again."
                )
            )
        )
        self._log("loop_limit", limit=self.max_tool_rounds, tool_rounds=state["tool_rounds"])
        return {"messages": closing, "loop_limit_reached": True}

    def _route_after_model(self, state: AgentState) -> Literal["run_tools", "loop_limit", "__end__"]:
        """The conditional edge that makes this part different from Part 1.

        Part 1 routed on a category our own code computed. Here the branch is
        decided by what the model's reply contains: tool calls mean another
        round, no tool calls mean the turn is over. The round counter is the
        only thing that can override the model.
        """
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if state["tool_rounds"] >= self.max_tool_rounds:
            return "loop_limit"
        return "run_tools"

    # -- graph -------------------------------------------------------------

    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_memory", self._load_memory)
        builder.add_node("call_model", self._call_model)
        builder.add_node("run_tools", self._run_tools)
        builder.add_node("loop_limit", self._loop_limit)

        builder.add_edge(START, "load_memory")
        builder.add_edge("load_memory", "call_model")
        builder.add_conditional_edges(
            "call_model", self._route_after_model, ["run_tools", "loop_limit", END]
        )
        builder.add_edge("run_tools", "call_model")
        builder.add_edge("loop_limit", END)

        return builder.compile(checkpointer=self._checkpointer)

    # -- public API --------------------------------------------------------

    def send(self, message: str) -> dict[str, Any]:
        """Run one turn. Returns the reply, the trace and what the tools did."""
        self._turn_trace = []
        self.session.last_status_check = None
        tools_before = len(self.session.tool_calls)
        tickets_before = len(self.session.tickets_written)

        graph_config = {"configurable": {"thread_id": self.session.thread_id}}
        with tools.use_session(self.session):
            final = self._graph.invoke({"messages": [HumanMessage(message)]}, graph_config)

        reply = ""
        for item in reversed(final["messages"]):
            if isinstance(item, AIMessage) and _text_of(item):
                reply = _text_of(item)
                break

        return {
            "user_id": self.session.user_id,
            "thread_id": self.session.thread_id,
            "status_profile": self.session.status_profile,
            "message": message,
            "reply": reply,
            "trace": list(self._turn_trace),
            "tool_calls": self.session.tool_calls[tools_before:],
            "tickets": self.session.tickets_written[tickets_before:],
            "user_memory": final.get("user_memory", {}),
            "tool_rounds": final.get("tool_rounds", 0),
            "loop_limit_reached": bool(final.get("loop_limit_reached")),
            "conversation_length": len(final["messages"]),
        }


def _text_of(message: Any) -> str:
    """Pull plain text out of a message whose content may be a block list."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p).strip()
