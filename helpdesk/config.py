"""Paths and settings shared by every part of the helpdesk."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"

HELPDESK_FILE = DATA_DIR / "helpdesk.json"
SERVICE_STATUS_FILE = DATA_DIR / "service_status.json"
DEMO_USERS_FILE = DATA_DIR / "demo_users.json"
TICKET_TEMPLATE_FILE = DATA_DIR / "ticket_template.json"
SCENARIOS_FILE = DATA_DIR / "scenarios.json"

DEFAULT_STATUS_PROFILE = "normal"

# The model may ask for at most this many rounds of tool calls in one turn.
# Enforced in helpdesk.agent_graph.route_after_model, not in the prompt.
MAX_TOOL_ROUNDS = 4


def storage_dir() -> Path:
    """Where conversation checkpoints, user memory and ticket drafts are written.

    Read from HELPDESK_STORAGE_DIR so a test or a scenario run can point at a
    throwaway directory instead of the one the terminal application uses.
    """
    raw = os.environ.get("HELPDESK_STORAGE_DIR", "local-storage")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def checkpoint_db() -> Path:
    return storage_dir() / "conversations.sqlite"


def user_memory_db() -> Path:
    return storage_dir() / "user_memory.sqlite"


def tickets_dir() -> Path:
    return storage_dir() / "tickets"
