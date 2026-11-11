"""User memory and conversation checkpoints.

What this would catch: a preference that stopped being learned from the
message, a preference stored per-thread instead of per-user (so a new
conversation would forget it), a query that leaked one user's memory to
another, a store that kept everything in RAM and so lost it on restart, or a
reset switch that wiped the wrong thing.
"""

from __future__ import annotations

import sqlite3

import pytest

from helpdesk import config, memory


# -- learning the preference from a message ---------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Please remember that I prefer replies in Kazakh.", "Kazakh"),
        ("please remember that i prefer replies in kazakh", "Kazakh"),
        ("Could you answer in Russian from now on?", "Russian"),
        ("I prefer replies in Kazakh. My issue is Wi-Fi.", "Kazakh"),
        ("Please reply to me in English.", "English"),
    ],
)
def test_preference_is_learned_from_what_the_user_wrote(message, expected):
    assert memory.detect_language_preference(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "I cannot connect to Wi-Fi.",
        "Prepare a support ticket.",
        "Please check whether Wi-Fi is down.",
        "",
    ],
)
def test_no_preference_is_invented_from_an_ordinary_message(message):
    assert memory.detect_language_preference(message) is None


def test_preference_is_not_read_from_the_fixture_file():
    """data/demo_users.json supplies identifiers only, never a preference."""
    from helpdesk import data_access

    users = data_access._load(str(config.DEMO_USERS_FILE))["users"]
    assert [u["user_id"] for u in users] == ["demo-user-a", "demo-user-b"]
    assert all(set(u) == {"user_id"} for u in users)


# -- the user memory store --------------------------------------------------


def test_preference_round_trips():
    with memory.UserMemoryStore() as store:
        store.set_language("demo-user-a", "Kazakh")
        assert store.language("demo-user-a") == "Kazakh"


def test_nothing_is_remembered_for_an_unseen_user():
    with memory.UserMemoryStore() as store:
        assert store.language("demo-user-b") is None
        assert store.all_for_user("demo-user-b") == {}


def test_one_users_memory_is_not_visible_to_another():
    with memory.UserMemoryStore() as store:
        store.set_language("demo-user-a", "Kazakh")
        assert store.all_for_user("demo-user-a") == {"reply_language": "Kazakh"}
        assert store.all_for_user("demo-user-b") == {}


def test_preference_survives_closing_the_store():
    """Stands in for a process restart: a new connection to the same file."""
    with memory.UserMemoryStore() as store:
        store.set_language("demo-user-a", "Kazakh")
    with memory.UserMemoryStore() as reopened:
        assert reopened.language("demo-user-a") == "Kazakh"


def test_a_later_preference_replaces_the_earlier_one():
    with memory.UserMemoryStore() as store:
        store.set_language("demo-user-a", "Kazakh")
        store.set_language("demo-user-a", "English")
        assert store.all_for_user("demo-user-a") == {"reply_language": "English"}


def test_user_memory_and_conversations_are_separate_files():
    assert config.user_memory_db() != config.checkpoint_db()


# -- conversation checkpoints ----------------------------------------------


def test_checkpoints_are_written_to_disk():
    with memory.conversation_checkpointer():
        pass
    assert config.checkpoint_db().exists()


def test_a_checkpoint_is_readable_through_a_new_connection():
    """What makes resuming after a restart possible at all."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    with memory.conversation_checkpointer() as saver:
        saver.put(
            {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}},
            {
                "v": 1,
                "id": "c1",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": {"note": "hello"},
                "channel_versions": {},
                "versions_seen": {},
            },
            {"source": "update", "step": 0},
            {},
        )

    connection = sqlite3.connect(str(config.checkpoint_db()))
    try:
        reopened = SqliteSaver(connection)
        stored = reopened.get({"configurable": {"thread_id": "t1", "checkpoint_ns": ""}})
        assert stored is not None
        assert stored["channel_values"]["note"] == "hello"
    finally:
        connection.close()


# -- the reset switch -------------------------------------------------------


def test_reset_removes_both_kinds_of_memory():
    with memory.UserMemoryStore() as store:
        store.set_language("demo-user-a", "Kazakh")
    with memory.conversation_checkpointer():
        pass
    assert config.storage_dir().exists()

    removed = memory.reset_all_memory()

    assert removed == [str(config.storage_dir())]
    assert not config.storage_dir().exists()
    with memory.UserMemoryStore() as store:
        assert store.language("demo-user-a") is None
