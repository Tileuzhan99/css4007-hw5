"""Part 1 routing.

What this would catch: a broken conditional edge, a routing rule that stopped
ignoring letter case, the two-topic precedence silently flipping, or
clarify_problem being wired in such a way that it never runs.
"""

from __future__ import annotations

import pytest

from helpdesk import part1_graph

BRIEF_MESSAGES = [
    ("I cannot connect to Wi-Fi.", "wifi_help", "clarify-unchanged"),
    ("I forgot my PASSWORD.", "password_help", "clarify-unchanged"),
    ("My laptop screen is broken.", "other_help", "clarify-unchanged"),
    ("My Wi-Fi password does not work.", "wifi_help", "clarify_problem"),
]


def route(message: str, *, with_clarification: bool = False) -> list[str]:
    state = part1_graph.run_once(message, with_clarification=with_clarification)
    return [entry["node"] for entry in state["trace"]]


@pytest.mark.parametrize("message,expected,_unused", BRIEF_MESSAGES)
def test_original_graph_routes_as_the_rules_say(message, expected, _unused):
    assert route(message) == ["read_request", expected]


@pytest.mark.parametrize("message,original,changed", BRIEF_MESSAGES)
def test_changed_graph_only_differs_for_two_topics(message, original, changed):
    nodes = route(message, with_clarification=True)
    expected = original if changed == "clarify-unchanged" else changed
    assert nodes == ["read_request", expected]


@pytest.mark.parametrize(
    "message",
    ["WIFI is down", "wi-fi will not connect", "Wi-Fi trouble", "wIfI"],
)
def test_wifi_aliases_and_case_are_ignored(message):
    assert part1_graph.classify(message, with_clarification=False) == "wifi"


@pytest.mark.parametrize("message", ["PASSWORD", "password", "PassWord reset"])
def test_password_case_is_ignored(message):
    assert part1_graph.classify(message, with_clarification=False) == "password"


def test_wifi_wins_over_password_in_the_original_rules():
    """The brief's precedence: wifi is tested first, so it wins."""
    assert part1_graph.classify("My Wi-Fi password does not work.", with_clarification=False) == "wifi"


def test_two_topics_become_their_own_category_after_the_change():
    assert part1_graph.classify("My Wi-Fi password does not work.", with_clarification=True) == "both"


def test_help_nodes_answer_from_the_fixture_not_from_prose():
    state = part1_graph.run_once("Wi-Fi help please")
    assert "GUIDE-WIFI" in state["reply"]
    assert "Campus-Demo" in state["reply"]


def test_other_help_asks_which_topic():
    reply = part1_graph.run_once("My laptop screen is broken.")["reply"].lower()
    assert "wi-fi" in reply and "password" in reply


def test_clarify_problem_asks_which_one_first():
    reply = part1_graph.run_once(
        "My Wi-Fi password does not work.", with_clarification=True
    )["reply"].lower()
    assert "wi-fi" in reply and "password" in reply
    assert "first" in reply


def test_original_graph_has_no_clarify_node():
    """A message with both topics cannot reach clarify_problem in the first graph."""
    graph = part1_graph.build_part1_graph(with_clarification=False)
    assert "clarify_problem" not in graph.get_graph().nodes


def test_state_carries_the_original_message_through():
    state = part1_graph.run_once("I cannot connect to Wi-Fi.")
    assert state["message"] == "I cannot connect to Wi-Fi."
    assert state["category"] == "wifi"
    assert state["reply"]
