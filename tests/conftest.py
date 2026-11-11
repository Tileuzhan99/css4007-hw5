"""Test fixtures.

Every test gets its own storage directory, so no test can see another test's
checkpoints, user preferences or ticket drafts - and none of them touch the
local-storage directory the terminal application uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point HELPDESK_STORAGE_DIR at a throwaway directory for this test."""
    storage = tmp_path / "storage"
    monkeypatch.setenv("HELPDESK_STORAGE_DIR", str(storage))
    # No API key should ever be needed by the test suite; make that explicit so
    # a stray real call fails loudly instead of quietly billing someone.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return storage


@pytest.fixture
def session(monkeypatch):
    """An active SessionContext, as the application sets up before a graph run."""
    from helpdesk import tools

    ctx = tools.SessionContext(
        user_id="demo-user-a", thread_id="test-thread", status_profile="normal"
    )
    with tools.use_session(ctx):
        yield ctx
