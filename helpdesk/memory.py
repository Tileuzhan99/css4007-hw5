"""The two kinds of memory, kept deliberately separate.

Conversation memory
    The message history and the facts gathered in *one* conversation. Stored by
    LangGraph's SqliteSaver checkpointer and keyed by `thread_id`.

User memory
    A preference that should outlive the conversation it was stated in - here,
    the language the user asked for. Stored in our own SQLite table and keyed by
    `user_id`.

A `thread_id` names one conversation; a `user_id` names a person. One user has
many threads, so the preference is shared across that user's threads while the
history is not. Nothing here is keyed by both, and no model output chooses
either key - the application supplies them (see helpdesk.tools.SessionContext).
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from . import config

LANGUAGE_PREFERENCE_KEY = "reply_language"

# Only these language names are recognised. A whitelist keeps "prepare a ticket
# in Building A" from being stored as a language; the cost is that an
# unlisted language is simply not learned. See SUBMISSION.md "Choices and
# limitations".
_KNOWN_LANGUAGES = [
    "Kazakh",
    "Russian",
    "English",
    "Turkish",
    "Ukrainian",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Portuguese",
    "Polish",
    "Arabic",
    "Chinese",
    "Japanese",
    "Korean",
]

_PREFERENCE_WORDS = re.compile(
    r"(?i)\b(prefer|preference|remember|reply|replies|respond|answer|speak|write|use)\b"
)
_LANGUAGE_PATTERN = re.compile(
    r"(?i)\b(?:in|to|into)\s+(" + "|".join(_KNOWN_LANGUAGES) + r")\b"
)


def detect_language_preference(message: str) -> str | None:
    """Learn the reply language from what the user actually wrote.

    Returns the canonical language name, or None when the message states no
    preference. The preference is never read from data/demo_users.json and
    never hard-coded per user.
    """
    if not message or not _PREFERENCE_WORDS.search(message):
        return None
    match = _LANGUAGE_PATTERN.search(message)
    if not match:
        return None
    wanted = match.group(1).lower()
    for language in _KNOWN_LANGUAGES:
        if language.lower() == wanted:
            return language
    return None  # pragma: no cover - the pattern only matches known languages


class UserMemoryStore:
    """Persistent per-user key/value memory in SQLite.

    Separate file from the conversation checkpoints on purpose: the two have
    different lifetimes and different keys, and deleting one should not take the
    other with it.
    """

    def __init__(self, path: Any = None) -> None:
        self.path = config.user_memory_db() if path is None else path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id    TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "UserMemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def set(self, user_id: str, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO user_memory (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value,
                                                    updated_at = excluded.updated_at
            """,
            (user_id, key, value, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self._conn.commit()

    def get(self, user_id: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM user_memory WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return row[0] if row else None

    def all_for_user(self, user_id: str) -> dict[str, str]:
        """Everything remembered about one user - and only that user.

        The WHERE clause is the isolation between demo-user-a and demo-user-b.
        """
        rows = self._conn.execute(
            "SELECT key, value FROM user_memory WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {key: value for key, value in rows}

    def set_language(self, user_id: str, language: str) -> None:
        self.set(user_id, LANGUAGE_PREFERENCE_KEY, language)

    def language(self, user_id: str) -> str | None:
        return self.get(user_id, LANGUAGE_PREFERENCE_KEY)


@contextmanager
def conversation_checkpointer() -> Iterator[SqliteSaver]:
    """LangGraph checkpointer for conversation state, keyed by thread_id.

    A plain SQLite file, so the state is still there after the process exits -
    which is what S11 and S12 check.
    """
    config.checkpoint_db().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.checkpoint_db()), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def reset_all_memory() -> list[str]:
    """Delete every stored conversation and user preference.

    This is the documented "start again" switch: `python app.py --reset-memory`.
    Returns the paths it removed.
    """
    removed = []
    directory = config.storage_dir()
    if directory.exists():
        shutil.rmtree(directory)
        removed.append(str(directory))
    return removed
