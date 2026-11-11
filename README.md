# HW5 — University IT helpdesk with LangGraph

A terminal helpdesk assistant for a **fictional** practice university. It routes
Wi-Fi and password questions through a LangGraph graph, lets a model use three
local tools, and remembers both the conversation and the user's language
preference across restarts.

Nothing here touches a real university system and no ticket is ever sent: a
"ticket" is a JSON draft written to a local directory.

- The assignment brief is in [ASSIGNMENT.md](ASSIGNMENT.md).
- My answers, drawings, predictions and run results are in [SUBMISSION.md](SUBMISSION.md).
- Saved execution logs are in [outputs/](outputs/).

## What is where

| Path | What it is |
| --- | --- |
| `helpdesk/part1_graph.py` | Part 1 — the routing graph, no model involved |
| `helpdesk/tools.py` | Part 2 — `find_instructions`, `check_service_status`, `prepare_ticket` |
| `helpdesk/agent_graph.py` | Parts 2–3 — the model/tool loop, the round limit, memory wiring |
| `helpdesk/memory.py` | Part 3 — conversation checkpointer and the user-preference store |
| `helpdesk/model.py` | Which chat model to talk to (Anthropic or OpenAI) |
| `helpdesk/fake_model.py` | Scripted stand-in model, so tests need no API key |
| `app.py` | The terminal application |
| `run_scenarios.py` | Runs S01–S14 from `data/scenarios.json` and saves logs |
| `tests/` | Routing, tools, memory and agent tests |
| `.github/workflows/ci.yml` | CI — runs the tests on every push |

## Requirements

- **Python 3.12.6** (what I used; any 3.12 should work — `from __future__ import
  annotations` plus `X | None` syntax means 3.10 is the floor).
- The exact package versions are in [requirements.txt](requirements.txt); the
  full resolved tree is in [requirements-lock.txt](requirements-lock.txt).

## Install

PowerShell (Windows):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configure a model key

**Part 1 needs no key at all.** Parts 2 and 3 need one tool-capable model.

```powershell
.\set_key.ps1 google     # or: .\set_key.ps1 openai / anthropic
```

`set_key.ps1` asks for the key with hidden input, writes it to `.env` and then
checks that the key and model work. The key is never printed. To do it by hand
instead, `Copy-Item .env.example .env` and edit the file. Pick a provider by setting `MODEL_PROVIDER`, or leave it blank
and whichever key is present is used (Anthropic is tried first):

| `MODEL_PROVIDER` | Key variable | Default `MODEL_NAME` |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-5.6-luna` |
| `google_genai` | `GOOGLE_API_KEY` | `gemini-3.5-flash` |

`MODEL_NAME` is optional; set it to use a different model from that provider.
Only the adapter for the provider you configure is ever imported, so you do not
need keys for the others.

Two things to know if you use a Gemini key on the free tier, because both cost
me time:

- The free tier caps generate-content requests at **20 per day, per model**.
  Running all nine model scenarios needs more than that, so either use a key
  with billing enabled, run them in batches on different days, or change
  `MODEL_NAME` between batches (the cap is counted per model). My own results
  were produced in two batches on two models, and every trace records which.
- `langchain-google-genai` calls API version `v1`. Some models are published
  only on `v1beta` and answer a `v1` request with 404. Set
  `MODEL_API_VERSION=v1beta` for those.

A provider error comes back as one actionable line rather than a traceback:
quota, unknown model and bad credentials are recognised in
`helpdesk/model.py:explain_error()`, and the program exits 3.
The model and provider actually used are recorded in every saved trace, and
[SUBMISSION.md](SUBMISSION.md) states the one my results came from.

The key is read from the environment or from `.env`, which is git-ignored.
No key is ever written to a log, a trace or a ticket.

## Run it

### Part 1 — routing only, no model

```bash
python app.py --graph part1 --message "I cannot connect to Wi-Fi."
python app.py --graph part1 --message "I forgot my PASSWORD."
python app.py --graph part1 --message "My laptop screen is broken."
python app.py --graph part1 --message "My Wi-Fi password does not work."
```

The same four messages against the changed graph, which adds `clarify_problem`:

```bash
python app.py --graph part1-clarify --message "My Wi-Fi password does not work."
```

Each run prints the route, what every node wrote, and the state after each node.

### Parts 2 and 3 — model, tools and memory

One conversation, two messages, trace saved to a file:

```bash
python app.py --user demo-user-a --thread s06 --profile normal --message "Show me the Wi-Fi troubleshooting steps." --message "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem." --trace outputs/my-run.json
```

Interactive conversation (empty line to leave):

```bash
python app.py --user demo-user-a --thread chat --profile normal
```

### The flags

| Flag | Meaning |
| --- | --- |
| `--user` | Invented user identifier, e.g. `demo-user-a`. Keys **user memory**. |
| `--thread` | Conversation identifier. Keys **conversation memory**. |
| `--profile` | Status snapshot: `normal`, `wifi_outage`, `recovery_outage`, `status_unavailable` |
| `--graph` | `agent` (default), `part1`, `part1-clarify` |
| `--message` | One message; repeat it for several turns in the same process |
| `--fake` | Run the graph with a fake model instead of a real one — no key, no network. It gives no helpdesk advice, only reports what the process can see, for checking the graph/storage/restart wiring. |
| `--trace PATH` | Write the whole run as JSON |
| `--reset-memory` | Delete all stored conversations and preferences, then exit |
| `--show-storage` | Print the storage paths and exit |

To resume a conversation after closing the program, run it again with the **same
`--user` and `--thread`**. To start a new conversation for the same user (so the
preference carries over but the history does not), change only `--thread`.

### Run the supplied scenarios

```bash
python run_scenarios.py              # all fourteen (S06-S14 need a key)
python run_scenarios.py --part 1     # S01-S05, no key needed
python run_scenarios.py --only S07   # just one
python run_scenarios.py --dry-run    # print the commands, run nothing
```

Each scenario writes `outputs/<ID>.log.md` (commands and console output) and one
`outputs/<ID>-run<N>.json` trace per process launch. A `restart_process` event
in `data/scenarios.json` is honoured literally: the runner starts a **new
`python app.py` process** against the same storage, so resuming is proved rather
than asserted.

## Where memory is stored, and how to clear it

Everything persistent lives in one directory, `local-storage/` by default
(`HELPDESK_STORAGE_DIR` overrides it). It is git-ignored.

| Path | What |
| --- | --- |
| `local-storage/conversations.sqlite` | LangGraph checkpoints — conversation state per `thread_id` |
| `local-storage/user_memory.sqlite` | User memory — the saved preference per `user_id` |
| `local-storage/tickets/*.json` | Local ticket drafts |

Clear it all:

```bash
python app.py --reset-memory
```

Or delete the directory by hand (`Remove-Item -Recurse local-storage` /
`rm -rf local-storage`). Two separate files on purpose: deleting conversations
should not have to delete a user's preference.

Ticket drafts produced by scenario runs are copied into `outputs/tickets/` so
they can be committed and read; they contain invented demo data only.

## Tests

```bash
python -m pytest -v
```

102 tests, no API key, no network. Each test gets its own throwaway storage
directory, and the fixture deletes `ANTHROPIC_API_KEY` from the environment so a
test that started making real calls would fail loudly instead of billing anyone.

| File | What it covers | What it would catch |
| --- | --- | --- |
| `tests/test_part1_routing.py` | Part 1 routing, both graphs | A broken conditional edge, routing that stopped ignoring case, the two-topic precedence flipping, `clarify_problem` wired so it never runs |
| `tests/test_tools.py` | The three tools called directly | A tool that stopped reading the fixtures, `unknown` status collapsing into operational or outage, `password` no longer mapping to `account_recovery`, a ticket inventing missing details, a credential reaching a saved draft |
| `tests/test_memory.py` | Preference learning, the store, checkpoints, reset | A preference not learned from the message, memory keyed per thread instead of per user, one user's memory leaking to another, a store that lost everything on restart |
| `tests/test_agent_fake.py` | The model/tool graph via `fake_model` | A graph that answers without running the tool, a tool result never returning to the model, a missing round limit, user memory not reaching the model, a conversation that does not resume, threads or users sharing state |
| `tests/test_cli_restart.py` | `app.py` launched as **real separate processes** against one storage directory | A checkpointer that only ever lived in memory, state written under a key a restart does not look up, a preference stored per thread instead of per user, memory bleeding across `--user`, `--thread` not separating conversations |

`tests/test_cli_restart.py` is the one that proves a *restart* works: the other
tests share a single Python process, so they can show the checkpointer
round-trips through SQLite but not that a relaunched program picks the
conversation back up.

The real-model demonstration is deliberately separate: it lives in
`run_scenarios.py` and in the committed `outputs/` logs, not in the test suite.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request: Python 3.12, `pip install -r requirements.txt`, `pytest -v`, then
the Part 1 scenarios. No secret is configured for CI, which is the point — the
suite must pass without one.

## Notes on safety and privacy

- `.env`, `local-storage/` and `*.sqlite` are git-ignored; no key, live memory
  database or personal information is committed.
- Only invented identifiers (`demo-user-a`, `demo-user-b`) appear in saved runs.
- `prepare_ticket` redacts anything shaped like a password, recovery code or API
  key before writing a draft, and refuses to invent details it was not given.
- The model never chooses which user's memory to load or which status snapshot
  to read — the application supplies both.
