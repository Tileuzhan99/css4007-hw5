"""The three tools, called directly - no model, no graph.

What this would catch: a tool that stopped reading the fixtures, a status
lookup that collapsed "unknown" into operational or outage, a password topic
that stopped mapping to account_recovery, a ticket that invented missing
details, a credential reaching a saved draft, or a tool reading the snapshot
from its arguments instead of from the session.
"""

from __future__ import annotations

import json

import pytest

from helpdesk import config, tools


# -- find_instructions ------------------------------------------------------


def test_find_instructions_returns_the_stored_wifi_guide(session):
    result = tools.find_instructions("wifi")
    assert result["ok"] is True
    assert result["instruction_id"] == "GUIDE-WIFI"
    assert result["steps"][0].startswith("Check that Wi-Fi is enabled")
    assert result["source"] == "data/helpdesk.json"


def test_find_instructions_returns_the_stored_password_guide(session):
    result = tools.find_instructions("password")
    assert result["instruction_id"] == "GUIDE-PASSWORD"


@pytest.mark.parametrize("topic", ["Wi-Fi", "WIFI", " wifi "])
def test_find_instructions_accepts_aliases_and_case(topic, session):
    assert tools.find_instructions(topic)["topic"] == "wifi"


def test_find_instructions_refuses_an_unsupported_topic(session):
    result = tools.find_instructions("printer")
    assert result["ok"] is False
    assert result["error"] == "unknown_topic"
    assert result["supported_topics"] == ["password", "wifi"]


def test_find_instructions_works_without_a_session():
    """Each tool must work when called directly, outside the agent."""
    assert tools.find_instructions("wifi")["ok"] is True


# -- check_service_status ---------------------------------------------------


def test_status_normal_profile_reports_operational(session):
    result = tools.check_service_status("wifi")
    assert (result["status"], result["is_operational"]) == ("operational", True)


def test_status_outage_profile_reports_outage(session):
    session.status_profile = "wifi_outage"
    result = tools.check_service_status("wifi")
    assert (result["status"], result["is_operational"]) == ("outage", False)


def test_unknown_status_is_neither_true_nor_false(session):
    """status_unavailable must stay a third answer, not become an all-clear."""
    session.status_profile = "status_unavailable"
    result = tools.check_service_status("wifi")
    assert result["status"] == "unknown"
    assert result["is_operational"] is None


def test_password_topic_maps_to_account_recovery(session):
    session.status_profile = "recovery_outage"
    result = tools.check_service_status("password")
    assert result["service"] == "account_recovery"
    assert result["status"] == "outage"


def test_recovery_outage_leaves_wifi_operational(session):
    session.status_profile = "recovery_outage"
    assert tools.check_service_status("wifi")["status"] == "operational"


def test_status_snapshot_comes_from_the_session_not_the_caller(session):
    """There is no snapshot argument, so no model output can select one."""
    import inspect

    assert list(inspect.signature(tools.check_service_status).parameters) == ["service"]
    session.status_profile = "wifi_outage"
    assert tools.check_service_status("wifi")["status_profile"] == "wifi_outage"


def test_status_refuses_an_unknown_service(session):
    result = tools.check_service_status("printer")
    assert result["ok"] is False
    assert result["error"] == "unknown_service"


def test_status_needs_a_session():
    with pytest.raises(RuntimeError):
        tools.check_service_status("wifi")


# -- prepare_ticket ---------------------------------------------------------


def test_ticket_refuses_when_details_are_missing(session):
    result = tools.prepare_ticket(summary="", steps_tried=[])
    assert result["ok"] is False
    assert result["error"] == "missing_information"
    assert result["missing"] == ["problem description", "steps already tried"]


def test_ticket_refuses_when_only_steps_are_missing(session):
    result = tools.prepare_ticket(summary="Wi-Fi does not connect", steps_tried=[])
    assert result["missing"] == ["steps already tried"]


def test_ticket_draft_is_written_and_readable(session):
    result = tools.prepare_ticket(
        summary="Campus-Demo will not connect",
        steps_tried=["enabled Wi-Fi", "forgot the saved network"],
        category="wifi",
    )
    assert result["ok"] is True
    ticket = result["ticket"]
    assert ticket["status"] == "draft"
    assert ticket["user_id"] == "demo-user-a"
    assert ticket["thread_id"] == "test-thread"
    assert ticket["category"] == "wifi"
    assert ticket["reference_ids"] == ["GUIDE-WIFI"]
    assert ticket["steps_tried"] == ["enabled Wi-Fi", "forgot the saved network"]

    saved = json.loads(
        (config.tickets_dir() / f"{ticket['ticket_id']}.json").read_text(encoding="utf-8")
    )
    assert saved == ticket


def test_ticket_uses_only_the_steps_it_was_given(session):
    result = tools.prepare_ticket(
        summary="Wi-Fi fails", steps_tried=["reconnected once"], category="wifi"
    )
    assert result["ticket"]["steps_tried"] == ["reconnected once"]


def test_ticket_records_the_status_that_was_actually_checked(session):
    session.status_profile = "wifi_outage"
    tools.check_service_status("wifi")
    result = tools.prepare_ticket(
        summary="Wi-Fi fails", steps_tried=["reconnected"], category="wifi"
    )
    assert result["ticket"]["service_status"] == "outage"
    assert result["ticket"]["status_profile"] == "wifi_outage"


def test_ticket_says_not_checked_when_no_status_lookup_happened(session):
    result = tools.prepare_ticket(
        summary="Wi-Fi fails", steps_tried=["reconnected"], category="wifi"
    )
    assert result["ticket"]["service_status"] == "not_checked"


def test_ticket_is_a_draft_only(session):
    result = tools.prepare_ticket(
        summary="Wi-Fi fails", steps_tried=["reconnected"], category="wifi"
    )
    assert result["delivery"] == "local draft only"
    assert "Nothing has been sent" in result["message"]


@pytest.mark.parametrize(
    "text",
    [
        "my password is hunter2",
        "the recovery code is 884120",
        "api_key: sk-demo-123",
    ],
)
def test_credentials_are_redacted_before_being_stored(text, session):
    assert "[redacted]" in tools.redact_secrets(text)


def test_a_credential_never_reaches_a_saved_draft(session):
    result = tools.prepare_ticket(
        summary=f"Recovery fails and my password is hunter2",
        steps_tried=["opened the sign-in screen"],
        category="password",
    )
    ticket = result["ticket"]
    assert "hunter2" not in json.dumps(ticket)
    assert "[redacted]" in ticket["summary"]


def test_ticket_ids_are_distinct(session):
    first = tools.prepare_ticket(summary="a", steps_tried=["x"], category="wifi")
    second = tools.prepare_ticket(summary="b", steps_tried=["y"], category="wifi")
    assert first["ticket"]["ticket_id"] != second["ticket"]["ticket_id"]


def test_ticket_identity_comes_from_the_session(session):
    """A ticket cannot be filed against another user by passing arguments."""
    import inspect

    parameters = list(inspect.signature(tools.prepare_ticket).parameters)
    assert "user_id" not in parameters and "thread_id" not in parameters


# -- the LangChain wrappers -------------------------------------------------


def test_the_model_sees_exactly_three_tools():
    assert [t.name for t in tools.LANGCHAIN_TOOLS] == [
        "find_instructions",
        "check_service_status",
        "prepare_ticket",
    ]


def test_wrappers_call_the_same_implementation(session):
    direct = tools.find_instructions("wifi")
    through_tool = tools.TOOLS_BY_NAME["find_instructions"].invoke({"topic": "wifi"})
    assert direct == through_tool


def test_tool_descriptions_tell_the_model_about_unknown_status():
    description = tools.TOOLS_BY_NAME["check_service_status"].description
    assert "unknown" in description
    assert "never invent a repair time" in description.lower()
