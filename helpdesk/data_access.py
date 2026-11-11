"""Read-only access to the supplied fixtures in data/.

Everything here is *retrieval*: the same question always reads the same file and
returns the same answer. Nothing in this module remembers anything about a user
or a conversation - that is helpdesk.memory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from . import config


@lru_cache(maxsize=None)
def _load(path_str: str) -> Any:
    with open(path_str, encoding="utf-8") as handle:
        return json.load(handle)


def helpdesk_data() -> dict[str, Any]:
    return _load(str(config.HELPDESK_FILE))


def service_status_data() -> dict[str, Any]:
    return _load(str(config.SERVICE_STATUS_FILE))


def ticket_template() -> dict[str, Any]:
    return json.loads(json.dumps(_load(str(config.TICKET_TEMPLATE_FILE))["ticket"]))


def scenarios() -> list[dict[str, Any]]:
    return _load(str(config.SCENARIOS_FILE))["scenarios"]


def topic_aliases() -> dict[str, list[str]]:
    return helpdesk_data()["topic_aliases"]


def service_for_topic(topic: str) -> str | None:
    """wifi -> wifi, password -> account_recovery."""
    return helpdesk_data()["topic_services"].get(topic)


def instructions_for_topic(topic: str) -> dict[str, Any] | None:
    return helpdesk_data()["instructions"].get(topic)


def ticket_policy() -> dict[str, Any]:
    return helpdesk_data()["ticket_policy"]


def status_profile_names() -> list[str]:
    return sorted(service_status_data()["profiles"])


def service_status(profile: str, service: str) -> dict[str, Any] | None:
    profiles = service_status_data()["profiles"]
    if profile not in profiles:
        raise KeyError(
            f"unknown status profile {profile!r}; choose one of {status_profile_names()}"
        )
    return profiles[profile].get(service)


def topics_in_message(message: str) -> list[str]:
    """Which supported topics the message mentions, in a stable order.

    Case-insensitive, driven by topic_aliases in data/helpdesk.json rather than
    by literals in the code.
    """
    lowered = message.lower()
    found = []
    for topic, aliases in topic_aliases().items():
        if any(alias.lower() in lowered for alias in aliases):
            found.append(topic)
    return found
