"""Part 1 - a helpdesk graph with no model in it.

START -> read_request -> conditional edge -> one help node -> END

read_request only *classifies*; the conditional edge only *chooses*; the help
nodes only *answer*. Which node runs is decided by plain Python
(route_request), so every route in this part is reproducible without an API key.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from . import data_access

# Category values produced by read_request.
CATEGORY_WIFI = "wifi"
CATEGORY_PASSWORD = "password"
CATEGORY_OTHER = "other"
CATEGORY_BOTH = "both"


class Part1State(TypedDict):
    """The whole state of the Part 1 graph.

    message  - the user's original message; never overwritten.
    category - written by read_request, read by the conditional edge.
    reply     - written by exactly one help node.
    trace     - append-only log of what each node read and wrote.
    """

    message: str
    category: str
    reply: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def _trace(node: str, reads: dict[str, Any], writes: dict[str, Any]) -> dict[str, Any]:
    return {"node": node, "read": reads, "wrote": writes}


def classify(message: str, *, with_clarification: bool) -> str:
    """The Part 1 routing rules, as a pure function.

    Original rules: wifi wins over password, because it is tested first.
    With clarification: a message naming both topics is its own category.
    """
    topics = data_access.topics_in_message(message)
    has_wifi = CATEGORY_WIFI in topics
    has_password = CATEGORY_PASSWORD in topics

    if with_clarification and has_wifi and has_password:
        return CATEGORY_BOTH
    if has_wifi:
        return CATEGORY_WIFI
    if has_password:
        return CATEGORY_PASSWORD
    return CATEGORY_OTHER


def _instructions_reply(topic: str) -> tuple[str, str]:
    """Render the stored instructions for a topic. Returns (reply, instruction_id)."""
    guide = data_access.instructions_for_topic(topic)
    if guide is None:  # pragma: no cover - both supported topics exist in the fixture
        raise KeyError(f"no instructions for topic {topic!r}")
    steps = "\n".join(f"  {i}. {step}" for i, step in enumerate(guide["steps"], start=1))
    reply = f"{guide['title']} ({guide['instruction_id']}):\n{steps}"
    return reply, guide["instruction_id"]


def build_part1_graph(*, with_clarification: bool = False):
    """Compile the Part 1 graph.

    with_clarification=False is the graph S04 uses; True is the graph S05 uses.
    """

    def read_request(state: Part1State) -> dict[str, Any]:
        category = classify(state["message"], with_clarification=with_clarification)
        return {
            "category": category,
            "trace": [
                _trace(
                    "read_request",
                    {"message": state["message"]},
                    {"category": category},
                )
            ],
        }

    def wifi_help(state: Part1State) -> dict[str, Any]:
        reply, instruction_id = _instructions_reply(CATEGORY_WIFI)
        return {
            "reply": reply,
            "trace": [
                _trace(
                    "wifi_help",
                    {"category": state["category"], "source": "data/helpdesk.json"},
                    {"reply": reply, "instruction_id": instruction_id},
                )
            ],
        }

    def password_help(state: Part1State) -> dict[str, Any]:
        reply, instruction_id = _instructions_reply(CATEGORY_PASSWORD)
        return {
            "reply": reply,
            "trace": [
                _trace(
                    "password_help",
                    {"category": state["category"], "source": "data/helpdesk.json"},
                    {"reply": reply, "instruction_id": instruction_id},
                )
            ],
        }

    def other_help(state: Part1State) -> dict[str, Any]:
        scope = ", ".join(data_access.helpdesk_data()["scope"])
        reply = (
            "I can only help with Wi-Fi or password problems here. "
            "Does your problem concern Wi-Fi or a password? "
            f"(This demo helpdesk covers: {scope}.)"
        )
        return {
            "reply": reply,
            "trace": [
                _trace(
                    "other_help",
                    {"category": state["category"], "source": "data/helpdesk.json"},
                    {"reply": reply},
                )
            ],
        }

    def clarify_problem(state: Part1State) -> dict[str, Any]:
        reply = (
            "Your message mentions both topics. Is the problem connecting to "
            "Wi-Fi, or recovering your account password? Tell me which one you "
            "want to solve first."
        )
        return {
            "reply": reply,
            "trace": [
                _trace(
                    "clarify_problem",
                    {"category": state["category"], "message": state["message"]},
                    {"reply": reply},
                )
            ],
        }

    def route_request(
        state: Part1State,
    ) -> Literal["wifi_help", "password_help", "other_help", "clarify_problem"]:
        """The conditional edge. It reads state and returns a node name only.

        It writes nothing: compare with the nodes above, which return updates.
        """
        return {
            CATEGORY_WIFI: "wifi_help",
            CATEGORY_PASSWORD: "password_help",
            CATEGORY_BOTH: "clarify_problem",
            CATEGORY_OTHER: "other_help",
        }[state["category"]]

    builder = StateGraph(Part1State)
    builder.add_node("read_request", read_request)
    builder.add_node("wifi_help", wifi_help)
    builder.add_node("password_help", password_help)
    builder.add_node("other_help", other_help)

    targets = ["wifi_help", "password_help", "other_help"]
    if with_clarification:
        builder.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")

    builder.add_edge(START, "read_request")
    builder.add_conditional_edges("read_request", route_request, targets)
    for node in targets:
        builder.add_edge(node, END)

    return builder.compile()


def run_once(message: str, *, with_clarification: bool = False) -> Part1State:
    """Convenience wrapper used by the CLI, the scenario runner and the tests."""
    graph = build_part1_graph(with_clarification=with_clarification)
    return graph.invoke({"message": message, "category": "", "reply": "", "trace": []})
