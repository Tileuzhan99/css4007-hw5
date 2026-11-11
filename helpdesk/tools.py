"""The three reusable helpdesk tools, plus the session context they read.

Every tool is a plain Python function first (`find_instructions`,
`check_service_status`, `prepare_ticket`) and a LangChain tool second
(`LANGCHAIN_TOOLS`). The plain functions are what the tests call directly, with
no model and no graph involved.

What the model may NOT choose
-----------------------------
The status snapshot, the user identifier and the thread identifier come from
`SessionContext`, which the application sets before it invokes the graph. They
are not tool arguments, so no model output can make `prepare_ticket` write a
ticket for another user or make `check_service_status` read a different
snapshot.
"""

from __future__ import annotations

import contextvars
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from langchain_core.tools import tool

from . import config, data_access

# --------------------------------------------------------------------------
# Session context
# --------------------------------------------------------------------------


@dataclass
class SessionContext:
    """Facts about the running session that the model is not allowed to set."""

    user_id: str
    thread_id: str
    status_profile: str = config.DEFAULT_STATUS_PROFILE
    # Filled in by check_service_status so prepare_ticket can record whether a
    # status was actually looked up in this turn, instead of guessing one.
    last_status_check: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tickets_written: list[str] = field(default_factory=list)


_SESSION: contextvars.ContextVar = contextvars.ContextVar("helpdesk_session", default=None)


@contextmanager
def use_session(ctx: SessionContext) -> Iterator[SessionContext]:
    token = _SESSION.set(ctx)
    try:
        yield ctx
    finally:
        _SESSION.reset(token)


def current_session() -> SessionContext:
    ctx = _SESSION.get()
    if ctx is None:
        raise RuntimeError(
            "no active helpdesk session; wrap the call in helpdesk.tools.use_session()"
        )
    return ctx


def _record(name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Log one tool call on the session so the trace can show it."""
    ctx = _SESSION.get()
    if ctx is not None:
        ctx.tool_calls.append({"tool": name, "args": args, "result": result})
    return result


# --------------------------------------------------------------------------
# Secret hygiene
# --------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(?:my\s+)?password\s+(?:is|was|=|:)\s*\S+"),
    re.compile(r"(?i)\b(?:recovery|verification|otp|2fa)\s+code\s+(?:is|was|=|:)\s*\S+"),
    re.compile(r"(?i)\bapi[_ -]?key\s*(?:is|=|:)\s*\S+"),
]


def redact_secrets(text: str) -> str:
    """Drop anything that looks like a credential before it reaches a ticket.

    data/helpdesk.json ticket_policy forbids storing passwords, API keys and
    recovery codes. Talking *about* a password is fine; stating one is not.
    """
    cleaned = text
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[redacted]", cleaned)
    return cleaned


# --------------------------------------------------------------------------
# Tool 1 - find_instructions
# --------------------------------------------------------------------------


def find_instructions(topic: str) -> dict[str, Any]:
    """Look up the stored instructions for a helpdesk topic.

    Pure retrieval: the same topic always returns the same fixture content.
    """
    normalised = (topic or "").strip().lower()
    for canonical, aliases in data_access.topic_aliases().items():
        if normalised == canonical or normalised in [a.lower() for a in aliases]:
            normalised = canonical
            break

    guide = data_access.instructions_for_topic(normalised)
    if guide is None:
        result = {
            "ok": False,
            "error": "unknown_topic",
            "requested_topic": topic,
            "supported_topics": sorted(data_access.helpdesk_data()["instructions"]),
            "message": "This demo helpdesk only has Wi-Fi and password instructions.",
        }
        return _record("find_instructions", {"topic": topic}, result)

    result = {
        "ok": True,
        "topic": normalised,
        "title": guide["title"],
        "steps": list(guide["steps"]),
        "instruction_id": guide["instruction_id"],
        "source": "data/helpdesk.json",
    }
    return _record("find_instructions", {"topic": topic}, result)


# --------------------------------------------------------------------------
# Tool 2 - check_service_status
# --------------------------------------------------------------------------


def check_service_status(service: str) -> dict[str, Any]:
    """Read the selected snapshot for one service.

    `service` may be a service name (wifi, account_recovery) or a topic name
    (wifi, password); a topic is mapped through topic_services. The snapshot
    itself comes from the session, never from the caller.
    """
    ctx = current_session()
    requested = (service or "").strip().lower()
    mapped = data_access.service_for_topic(requested) or requested

    try:
        entry = data_access.service_status(ctx.status_profile, mapped)
    except KeyError as exc:
        result = {"ok": False, "error": "unknown_profile", "message": str(exc)}
        return _record("check_service_status", {"service": service}, result)

    if entry is None:
        result = {
            "ok": False,
            "error": "unknown_service",
            "requested_service": service,
            "known_services": sorted(
                data_access.service_status_data()["profiles"][ctx.status_profile]
            ),
        }
        return _record("check_service_status", {"service": service}, result)

    status = entry["status"]
    if status == "operational":
        is_operational: bool | None = True
    elif status == "outage":
        is_operational = False
    else:
        # Explicitly three-valued: an unavailable source is not an outage and
        # not an all-clear. The model must not collapse unknown into either.
        is_operational = None

    result = {
        "ok": True,
        "service": mapped,
        "status": status,
        "is_operational": is_operational,
        "message": entry["message"],
        "status_profile": ctx.status_profile,
        "source": "data/service_status.json",
    }
    ctx.last_status_check = result
    return _record("check_service_status", {"service": service}, result)


# --------------------------------------------------------------------------
# Tool 3 - prepare_ticket
# --------------------------------------------------------------------------


def prepare_ticket(
    summary: str,
    steps_tried: list[str] | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Write a local ticket draft. Nothing is sent anywhere.

    Refuses rather than inventing anything: a missing summary or an empty
    steps_tried list comes back as `missing_information`, which is what makes
    the assistant ask the user instead of filling the gap itself.
    """
    ctx = current_session()
    args = {"summary": summary, "steps_tried": steps_tried, "category": category}

    missing = []
    if not (summary or "").strip():
        missing.append("problem description")
    if not steps_tried:
        missing.append("steps already tried")
    if missing:
        result = {
            "ok": False,
            "error": "missing_information",
            "missing": missing,
            "message": (
                "I cannot draft a ticket yet. Ask the user for: "
                + ", ".join(missing)
                + ". Do not invent these details."
            ),
            "policy": data_access.ticket_policy()["unknown_information"],
        }
        return _record("prepare_ticket", args, result)

    topic = (category or "").strip().lower() or None
    if topic not in data_access.helpdesk_data()["instructions"]:
        topics = data_access.topics_in_message(summary)
        topic = topics[0] if topics else None

    status_check = ctx.last_status_check
    ticket = data_access.ticket_template()
    ticket.update(
        {
            "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "status": "draft",
            "user_id": ctx.user_id,
            "thread_id": ctx.thread_id,
            "category": topic,
            "summary": redact_secrets(summary.strip()),
            "steps_tried": [redact_secrets(str(s).strip()) for s in steps_tried],
            # "not_checked" is a real answer: S09 drafts a ticket without a
            # status lookup and the draft has to say so.
            "service_status": status_check["status"] if status_check else "not_checked",
            "status_profile": ctx.status_profile,
            "reference_ids": [],
        }
    )
    guide = data_access.instructions_for_topic(topic) if topic else None
    if guide:
        ticket["reference_ids"] = [guide["instruction_id"]]

    path = _write_ticket(ticket)
    ctx.tickets_written.append(str(path))

    result = {
        "ok": True,
        "delivery": data_access.ticket_policy()["delivery"],
        "ticket": ticket,
        "path": _display_path(path),
        "message": "Draft saved locally. Nothing has been sent to a real helpdesk.",
    }
    return _record("prepare_ticket", args, result)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_ticket(ticket: dict[str, Any]) -> Path:
    directory = config.tickets_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ticket['ticket_id']}.json"
    path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# LangChain wrappers - the only thing the model sees
# --------------------------------------------------------------------------


@tool("find_instructions")
def find_instructions_tool(topic: str) -> dict[str, Any]:
    """Look up the official demo helpdesk instructions for a topic.

    Use this before advising the user, so the steps come from the guide rather
    than from memory. Valid topics: "wifi", "password".
    """
    return find_instructions(topic)


@tool("check_service_status")
def check_service_status_tool(service: str) -> dict[str, Any]:
    """Check whether a demo service is operating or has a known outage.

    Pass "wifi" for Wi-Fi problems and "account_recovery" (or "password") for
    sign-in and password-recovery problems. The reply has a three-valued
    status: "operational", "outage", or "unknown" when the status source itself
    is unavailable. "unknown" does not mean the service is working - never
    report it as operating, and never invent a repair time.
    """
    return check_service_status(service)


@tool("prepare_ticket")
def prepare_ticket_tool(
    summary: str,
    steps_tried: list[str],
    category: str | None = None,
) -> dict[str, Any]:
    """Save a local support-ticket draft. Nothing is sent to a real helpdesk.

    `summary` is the user's problem in their own terms and `steps_tried` is the
    list of things the user said they already did. Use only what the user
    actually told you: if either is missing, the tool refuses and you must ask
    the user for it. Never put a password, recovery code or API key in a ticket.
    `category` is "wifi" or "password".
    """
    return prepare_ticket(summary=summary, steps_tried=steps_tried, category=category)


LANGCHAIN_TOOLS = [find_instructions_tool, check_service_status_tool, prepare_ticket_tool]

TOOLS_BY_NAME = {t.name: t for t in LANGCHAIN_TOOLS}
