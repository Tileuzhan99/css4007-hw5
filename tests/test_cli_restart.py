"""The command line, driven as real subprocesses.

Everything else in the suite runs inside one Python process, so it can prove the
checkpointer round-trips through SQLite but not that a *restarted program* picks
the conversation back up. These tests launch `python app.py` as separate
processes against the same storage directory, which is what a `restart_process`
event in data/scenarios.json actually means.

They use `--fake`, so no API key and no network are involved. The fake model
gives no helpdesk advice; it reports how many user messages the process can see
and which user memory it loaded, which is exactly what needs checking here.

What this would catch: a checkpointer that only ever lived in memory, state
written under a key a restart does not look up, a preference stored per thread
instead of per user, user memory bleeding across `--user`, or `--thread` not
actually separating conversations.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PREFERENCE_MESSAGE = (
    "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi. "
    "I enabled it and tried reconnecting, but it still fails."
)


def run_app(*args: str) -> subprocess.CompletedProcess:
    """Launch app.py as its own process, inheriting HELPDESK_STORAGE_DIR."""
    completed = subprocess.run(
        [sys.executable, "app.py", *args],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"app.py {' '.join(args)} exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def messages_seen(stdout: str) -> int:
    match = re.search(r"human_messages_seen=(\d+)", stdout)
    assert match, f"the fake model printed no counter:\n{stdout}"
    return int(match.group(1))


def memory_seen(stdout: str) -> str:
    match = re.search(r"user_memory=(\{.*?\})", stdout)
    assert match, f"the fake model printed no memory:\n{stdout}"
    return match.group(1)


# -- a conversation survives the process exiting ---------------------------


def test_a_restarted_process_resumes_the_same_conversation():
    first = run_app(
        "--fake", "--user", "demo-user-a", "--thread", "restart-a",
        "--message", PREFERENCE_MESSAGE,
    )
    assert messages_seen(first.stdout) == 1

    # A genuinely new process. Nothing is passed between them but the storage.
    second = run_app(
        "--fake", "--user", "demo-user-a", "--thread", "restart-a",
        "--message", "Which problem and steps did we discuss?",
    )
    assert messages_seen(second.stdout) == 2
    assert "tried reconnecting" not in "Which problem and steps did we discuss?"
    assert "history_before_turn=2" in second.stdout


def test_the_second_process_was_not_given_the_old_transcript():
    """The only thing the second launch receives on the command line is the new
    message; the history has to come from the checkpoint."""
    run_app(
        "--fake", "--user", "demo-user-a", "--thread", "noreplay",
        "--message", PREFERENCE_MESSAGE,
    )
    second = run_app(
        "--fake", "--user", "demo-user-a", "--thread", "noreplay",
        "--message", "And now?",
    )
    assert messages_seen(second.stdout) == 2


# -- a preference crosses threads, a conversation does not -----------------


def test_a_new_thread_keeps_the_preference_but_not_the_history():
    run_app(
        "--fake", "--user", "demo-user-a", "--thread", "preference-a1",
        "--message", PREFERENCE_MESSAGE,
    )
    other_thread = run_app(
        "--fake", "--user", "demo-user-a", "--thread", "preference-a2",
        "--message", "Which language did I ask you to use?",
    )
    assert messages_seen(other_thread.stdout) == 1  # fresh conversation
    assert "Kazakh" in memory_seen(other_thread.stdout)  # same user memory


# -- users do not share anything -------------------------------------------


def test_another_user_gets_neither_the_preference_nor_the_history():
    run_app(
        "--fake", "--user", "demo-user-a", "--thread", "isolation-a",
        "--message", PREFERENCE_MESSAGE,
    )
    other_user = run_app(
        "--fake", "--user", "demo-user-b", "--thread", "isolation-b",
        "--message", "Have I told you a preferred language or a previous problem?",
    )
    assert messages_seen(other_user.stdout) == 1
    assert memory_seen(other_user.stdout) == "{}"
    assert "Kazakh" not in other_user.stdout


# -- the reset switch, from the outside ------------------------------------


def test_reset_memory_makes_the_next_process_start_empty():
    run_app(
        "--fake", "--user", "demo-user-a", "--thread", "wiped",
        "--message", PREFERENCE_MESSAGE,
    )
    run_app("--reset-memory")

    after = run_app(
        "--fake", "--user", "demo-user-a", "--thread", "wiped",
        "--message", "Do you remember me?",
    )
    assert messages_seen(after.stdout) == 1
    assert memory_seen(after.stdout) == "{}"


# -- Part 1 through the CLI ------------------------------------------------


def test_part1_cli_prints_the_route_and_needs_no_key():
    completed = run_app("--graph", "part1", "--message", "I cannot connect to Wi-Fi.")
    assert "START -> read_request -> wifi_help -> END" in completed.stdout
    assert "GUIDE-WIFI" in completed.stdout


def test_part1_clarify_cli_takes_the_other_branch():
    completed = run_app(
        "--graph", "part1-clarify", "--message", "My Wi-Fi password does not work."
    )
    assert "START -> read_request -> clarify_problem -> END" in completed.stdout


def test_an_unknown_profile_is_rejected():
    completed = subprocess.run(
        [sys.executable, "app.py", "--profile", "sunny", "--message", "hi"],
        cwd=str(REPO_ROOT), env=os.environ.copy(), capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2
    assert "unknown --profile" in completed.stderr


def test_a_saved_trace_records_that_the_run_was_fake():
    import json

    trace = REPO_ROOT / "outputs" / "_tmp-fake-trace.json"
    try:
        run_app(
            "--fake", "--user", "demo-user-a", "--thread", "traced",
            "--message", "hello", "--trace", str(trace),
        )
        payload = json.loads(trace.read_text(encoding="utf-8"))
        assert payload["model"]["pipeline_check_only"] is True
        assert payload["model"]["provider"] == "fake"
    finally:
        trace.unlink(missing_ok=True)
