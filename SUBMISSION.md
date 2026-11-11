# HW5 — My LangGraph helpdesk

**Repository:** https://github.com/Tileuzhan99/css4007-hw5
**Python:** 3.12.6 (Windows 10). Package versions: [requirements.txt](requirements.txt),
full tree in [requirements-lock.txt](requirements-lock.txt).
**Setup and run commands:** [README.md](README.md). The assignment brief is kept
as [ASSIGNMENT.md](ASSIGNMENT.md).

**AI assistance:** Claude Code (Claude Opus 5) helped me design the graphs,
write the code and tests, and draft the text of this report from my own run logs
in `outputs/`. Every number, route and reply quoted below comes from a run I
made myself with the commands shown; nothing in the tables is copied from the
illustrative dialogues in SCENARIOS.md.

---

## Part 1 — How the graph works

### My graph and state

The first graph, with no model anywhere in it:

```text
            ┌──────────────┐
  START ───▶│ read_request │          writes: category
            └──────┬───────┘
                   │
           route_request(state)        ← conditional edge: reads state,
                   │                     returns a node name, writes nothing
      ┌────────────┼────────────┐
      ▼            ▼            ▼
┌───────────┐ ┌──────────────┐ ┌────────────┐
│ wifi_help │ │password_help │ │ other_help │   each writes: reply
└─────┬─────┘ └──────┬───────┘ └─────┬──────┘
      └──────────────┼───────────────┘
                     ▼
                    END
```

State (`helpdesk/part1_graph.py`, `Part1State`):

| Field | Holds | Written by | Read by |
| --- | --- | --- | --- |
| `message` | the user's original message, never overwritten | the caller | `read_request`, `clarify_problem` |
| `category` | `wifi` / `password` / `other` / `both` | `read_request` | `route_request`, every help node |
| `reply` | the answer text | exactly one help node | the caller |
| `trace` | append-only list of what each node read and wrote | every node | the log writer |

`trace` is `Annotated[list, operator.add]`, so each node's entry is **appended**
rather than replacing the list. That is the one field with a reducer; the others
are plain overwrites.

**1. What does each node read, and what does it change?**

| Node | Reads | Changes |
| --- | --- | --- |
| `read_request` | `message` | `category` (and appends to `trace`) |
| `wifi_help` | `category`, plus `data/helpdesk.json` | `reply` |
| `password_help` | `category`, plus `data/helpdesk.json` | `reply` |
| `other_help` | `category`, plus the `scope` list in `data/helpdesk.json` | `reply` |
| `clarify_problem` | `category`, `message` | `reply` |

No node writes a field another node also writes, so there is no ordering
question between them. The help nodes do not read `reply`, and `read_request`
does not read `category` — each one only needs what the table says.

**2. What is the difference between a node returning a state update and a
conditional edge choosing the next node?**

A node returns a **dict of changes** that LangGraph merges into the state; it
has no say in where control goes next. A conditional edge is a function that
returns **a node name** and merges nothing — it only selects. In my code
`read_request` returns `{"category": ..., "trace": [...]}`, while
`route_request` returns the string `"wifi_help"`. They are deliberately
separate: computing the category is something I can unit-test as a pure function
(`classify()`), and the edge is then a lookup table from category to node.

**3. Does the next node receive only the previous node's update, or the current
graph state?**

The whole current state. `wifi_help` receives `message`, `category`, `reply` and
`trace` — not just the `{"category": ...}` that `read_request` returned. That is
why `clarify_problem` can quote the original `message` even though the node
before it only wrote `category`. The saved traces show this directly: each
`state_after` block in `outputs/S0*.log.md` contains every field, not just the
one that changed.

**4. What do `START`, `END` and compiling mean here?**

`START` and `END` are not nodes I wrote; they are the graph's two markers.
`add_edge(START, "read_request")` says which node receives the input state, and
an edge to `END` says a path is finished — with four nodes pointing at `END`,
whichever one ran ends the run. Compiling (`builder.compile()`) turns the node
and edge declarations into a runnable object: it is where a typo in a node name
or an unreachable node is caught, and it is what gives me `.invoke()`. I build
the graph once and invoke it per message.

### Predictions before running

Predicted from the rules in the brief (wifi tested first, then password,
otherwise other; case ignored) before running anything. Commands and full
traces: `outputs/S01.log.md` … `outputs/S05.log.md`.

| Message | Predicted route | Actual route | What I learned |
| --- | --- | --- | --- |
| I cannot connect to Wi-Fi. | `read_request → wifi_help` | `read_request → wifi_help` (S01) | Matched. `category=wifi`. |
| I forgot my PASSWORD. | `read_request → password_help` | `read_request → password_help` (S02) | Matched. Upper case made no difference — `classify()` lowercases before matching, so this passes only because the comparison is explicit, not because Python is forgiving. |
| My laptop screen is broken. | `read_request → other_help` | `read_request → other_help` (S03) | Matched. `other_help` asks which of the two topics it is, instead of pretending to help. |
| My Wi-Fi password does not work. | `read_request → wifi_help` | `read_request → wifi_help` (S04) | Matched, and this is the interesting one — see below. |

**What happens when a message mentions both topics, and which rule causes it.**
The message contains `wi-fi` *and* `password`, so both conditions are true. The
rule that decides is the **order of the tests**: `classify()` checks the Wi-Fi
aliases first and returns as soon as one matches, so `password` is never
examined. The user asked about a *password*, and the graph sent them Wi-Fi
instructions — the rule is not wrong, it is just making a silent choice on the
user's behalf. Nothing in the trace marks that a second topic was present.

### My change to the graph

`clarify_problem` is a new node, and the two-topic case becomes its own
category instead of being absorbed by `wifi`:

```text
            ┌──────────────┐
  START ───▶│ read_request │          category ∈ {wifi, password, other, both}
            └──────┬───────┘
                   │
           route_request(state)
      ┌────────┬───┴────┬──────────────┐
      ▼        ▼        ▼              ▼
┌──────────┐ ┌────────────┐ ┌──────────┐ ┌─────────────────┐
│wifi_help │ │password_   │ │other_help│ │ clarify_problem │  ← new
│          │ │help        │ │          │ │                 │
└────┬─────┘ └─────┬──────┘ └────┬─────┘ └────────┬────────┘
     └─────────────┴─────────────┴────────────────┘
                              ▼
                             END
```

The change is in two places only: `classify()` gains a first test for
"both topics present" returning `both`, and `route_request` maps `both` to
`clarify_problem`. The existing nodes are untouched.

New trace, same message, from `outputs/S05.log.md`:

```text
you> My Wi-Fi password does not work.
route: START -> read_request -> clarify_problem -> END
  [read_request] wrote {"category": "both"}
      state after: {"message": "My Wi-Fi password does not work.", "category": "both", "reply": ""}
  [clarify_problem] wrote {"reply": "Your message mentions both topics. Is the problem connecting to Wi-Fi, or recovering your account password? Tell me which one you want to solve first."}
helpdesk> Your message mentions both topics. Is the problem connecting to Wi-Fi, or recovering your account password? Tell me which one you want to solve first.
```

Compare with S04, the same message on the original graph:
`read_request → wifi_help`, `category=wifi`. The difference is visible in the
**route and in the state**, not only in the final sentence: a different node
ran, and `category` holds a value the first graph cannot produce. S04 and S05
are run with `--graph part1` and `--graph part1-clarify` respectively, which is
what `data/README.md` asks for.

A test pins this down so the two graphs cannot quietly converge:
`test_original_graph_has_no_clarify_node` asserts `clarify_problem` is absent
from the compiled first graph.

### What LangGraph controls, and what my own functions decide

**LangGraph controls** the mechanics: it holds the state object and applies each
node's returned dict to it (using the reducer for `trace` and overwriting the
rest), it passes the full current state to the next node, it follows the edges
from `START` until a node reaches `END`, and at compile time it validates that
the nodes and edge targets exist.

**My Python decides everything about the content**: `classify()` decides the
category from the message text, `route_request()` decides which node name
corresponds to that category, and each help node decides what the reply says and
reads it out of `data/helpdesk.json`. LangGraph never looks at the words in the
message.

So the split is: *which node runs next* is LangGraph executing a decision my
function made; *what that decision is* is entirely mine. There is no model in
this part — the same message always takes the same route, which is why all of
Part 1 runs in CI with no API key.

---

## Part 2 — Model and tools

### The three tools

All three are plain functions in `helpdesk/tools.py` and work when called
directly, with no agent and no graph — that is what `tests/test_tools.py` does
(`test_find_instructions_works_without_a_session` calls one with no session at
all). `LANGCHAIN_TOOLS` wraps the same functions for the model.

| Tool | Arguments the model supplies | Reads | Returns |
| --- | --- | --- | --- |
| `find_instructions` | `topic` | `data/helpdesk.json` | title, steps, `instruction_id` |
| `check_service_status` | `service` | the selected snapshot in `data/service_status.json` | `status`, `is_operational`, message, which profile |
| `prepare_ticket` | `summary`, `steps_tried`, `category` | `data/ticket_template.json`, the policy, the last status check | the draft and the file it was written to |

**What the model is not allowed to choose.** The status snapshot, the user
identifier and the thread identifier are **not tool arguments**. They live in
`SessionContext`, which `app.py` sets before invoking the graph. Two tests
assert this by inspecting the signatures
(`test_status_snapshot_comes_from_the_session_not_the_caller`,
`test_ticket_identity_comes_from_the_session`), so a future refactor that
promoted them to arguments would fail CI. This is what `data/README.md` means by
"Do not let a model choose which user's stored data to load".

**Three-valued status.** `check_service_status` returns `is_operational` as
`True` / `False` / `None` for `operational` / `outage` / `unknown`. An
unavailable status source is a third answer, not an all-clear, and the tool
description says so in the words the model sees.

**A ticket is a draft.** `prepare_ticket` writes JSON to
`local-storage/tickets/<id>.json` and returns `delivery: "local draft only"`.
Nothing is sent. If `summary` or `steps_tried` is missing it returns
`missing_information` listing exactly what it needs, which is what makes the
assistant ask instead of inventing (S09). Anything shaped like a password,
recovery code or API key is redacted before the draft is written.

### The execution loop

```text
  START ──▶ load_memory ──▶ call_model ──┐
                               ▲         │  route_after_model(state)
                               │         │
                               │    ┌────┴─────────────────┬──────────────┐
                               │    │ reply has tool_calls │  no tool_    │
                               │    │ and rounds < limit   │  calls       │
                               │    ▼                      │              ▼
                               └── run_tools               │             END
                                                           │
                               rounds ≥ limit ─────────────┴──▶ loop_limit ──▶ END
```

- `call_model` sends the system prompt, the conversation and the tool schemas,
  and gets back either text or `tool_calls`. It **requests**; it never executes.
- `run_tools` is the only place a tool function runs. It executes every call in
  the round (so two parallel calls both run), appends one `ToolMessage` per call,
  and increments `tool_rounds`.
- Control returns to `call_model` with the results in the conversation, and the
  model either asks for more tools or answers.

**1. How does the graph decide between running a tool and ending with an
answer?** `route_after_model` looks at the message `call_model` just produced.
No `tool_calls` on it → return `END`, the turn is over. Tool calls present →
`run_tools`. It is a conditional edge, exactly like Part 1's, but the value it
branches on comes from the model's output instead of from my `classify()`.

**2. Which component requests a tool, and which executes it?** The **model**
requests: its reply contains `tool_use` blocks that LangChain exposes as
`.tool_calls`. **My program** executes, in `run_tools`, by looking the name up in
`TOOLS_BY_NAME` and calling it. A name that is not in that dict is answered with
an `unknown_tool` error instead of raising
(`test_an_unknown_tool_name_does_not_crash_the_graph`), so a hallucinated tool
name cannot stop the conversation or reach any code.

**3. What changes in the state after a tool result returns?** Two things. A
`ToolMessage` per call is appended to `messages` (the `add_messages` reducer
appends rather than replaces), carrying the tool's JSON result and the
`tool_call_id` that pairs it with the request. And `tool_rounds` increases by
one. Nothing else: the tool result is not copied into any other field, so the
model's next request sees it as part of the conversation, which is the only way
it can use it.

**4. What makes this different from Part 1's fixed rules?** In Part 1 I could
read the route off the message before running anything — the rules are in my
code and the same input always takes the same path. Here the number of nodes
visited depends on what the model decides, and the same message can take a
different path on a different day. The *shape* of what can happen is still mine:
the model can only pick from three tools, with arguments matching their schemas,
and only for as many rounds as `route_after_model` allows. I moved from
specifying the path to specifying the space of allowed paths.

**Where the loop limit is enforced.** In `route_after_model`, not in the prompt:

```python
if not getattr(last, "tool_calls", None):
    return END
if state["tool_rounds"] >= self.max_tool_rounds:
    return "loop_limit"
return "run_tools"
```

`MAX_TOOL_ROUNDS = 4` in `helpdesk/config.py`. The prompt is not asked to
restrain the model, because a prompt is advice; this is the graph refusing to
build another edge. `loop_limit` also answers every pending tool call with a
`tool_round_limit_reached` error before it stops — an unanswered `tool_use`
block would make the *next* request to the API invalid, which is the kind of
bug that only shows up on the turn after the one that caused it.
`test_the_round_limit_stops_a_model_that_never_stops` drives this with a fake
model that requests a tool forever and asserts it stops after exactly the
configured number of rounds.

### Model and provider

<!-- FILL AFTER THE MODEL RUNS: provider, exact model id, and the date. -->

### Direct calls to each tool, with no agent

<!-- FILL AFTER THE MODEL RUNS (the direct-call evidence is already in
     tests/test_tools.py; quote a short transcript here too). -->

### One run that uses more than one tool

<!-- FILL AFTER THE MODEL RUNS: S06 - instructions, then status, then ticket. -->

### The same conditional request against three snapshots

<!-- FILL AFTER THE MODEL RUNS: S06 (normal), S07 (wifi_outage),
     S08 (status_unavailable), S14 (recovery_outage). -->

---

## Part 3 — Memory

### Two kinds of memory, two stores

| | Conversation memory | User memory |
| --- | --- | --- |
| What | the messages of **one** conversation, and so the problem and the steps already tried | a preference that should outlive the conversation it was stated in — here, the reply language |
| Keyed by | `thread_id` | `user_id` |
| Mechanism | LangGraph `SqliteSaver` checkpointer | my own SQLite table in `helpdesk/memory.py` |
| File | `local-storage/conversations.sqlite` | `local-storage/user_memory.sqlite` |
| Lifetime | as long as that conversation is interesting | until the user changes it |

They are separate files on purpose. Clearing one should not have to clear the
other, and nothing is keyed by the pair — if the preference were stored per
`(user_id, thread_id)` it would be forgotten by every new conversation, which is
exactly what S12 checks.

**What a `thread_id` identifies, and how it differs from a user identifier.**
A `thread_id` names **one conversation**. It is what LangGraph puts in
`config={"configurable": {"thread_id": ...}}`, and it is the key the
checkpointer saves and loads state under; two runs with the same `thread_id`
continue each other, two runs with different ones do not share a single message.
A `user_id` names **a person**. One user has many threads, so the relationship
is one-to-many: the preference attaches to the person and is visible in all of
their conversations, while the history attaches to the conversation and is not.
Neither identifier ever comes from the message text — both come from `--user`
and `--thread` on the command line, so a user who writes "I am demo-user-a"
while running as `demo-user-b` loads nothing
(`test_the_user_identifier_is_not_taken_from_the_message`).

**How the preference is learned.** Not hard-coded, and not read from
`data/demo_users.json` (which supplies identifiers only — a test asserts the
file has no preference field in it). The `load_memory` node runs first on every
turn and passes the newest user message to
`memory.detect_language_preference()`, which recognises a preference statement
and a language name from a whitelist. If it finds one, it writes it to the user
store immediately; then it reads back everything stored for that user and puts
it in the system prompt for this turn. The trace entry for `load_memory` records
`learned_preference` and the `user_memory` it loaded, so each log says where the
language came from.

**Why the instructions are retrieval and the language is memory.**
Looking up Wi-Fi instructions is **retrieval**: the content lives in
`data/helpdesk.json`, it is the same for every user and every conversation, I
did not learn it from anyone, and the same query always returns the same answer.
It is a lookup in a reference file, and if I delete all memory it still works.
Recalling the chosen language is **memory**: it exists only because this
particular user said it in an earlier conversation, it is different per user, it
is written at the moment it is observed, and deleting the store destroys it —
there is nowhere to look it up again. The practical difference is what happens
on a fresh machine: retrieval still answers, memory is empty until someone
speaks.

### Demonstrations

<!-- FILL AFTER THE MODEL RUNS: S10-S13. -->

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | | | |
| Close, restart and resume | | | |
| Same user, new conversation | | | |
| Different user | | | |

---

## Part 4 — Running and checking my work

**Setup steps and run commands:** [README.md](README.md) — virtual environment,
`pip install -r requirements.txt`, how to configure a key for either provider,
every flag of `app.py`, and the scenario runner. Python 3.12.6 is stated there
and here.

**Dependency file:** [requirements.txt](requirements.txt) pins the exact
versions this code was developed and run against — `langgraph==1.2.20`,
`langgraph-checkpoint-sqlite==3.1.3`, `langchain-core==1.7.7`,
`langchain-anthropic==1.7.10`, `langchain-openai==1.7.4`,
`python-dotenv==1.2.4`, `pytest==9.2.0`. The full resolved tree is in
[requirements-lock.txt](requirements-lock.txt), so a reader never has to guess
which LangGraph version works with this code.

**Where memory is saved and how to clear it:** everything persistent is under
`local-storage/` (overridable with `HELPDESK_STORAGE_DIR`), which is
git-ignored: `conversations.sqlite` for conversation checkpoints,
`user_memory.sqlite` for preferences, `tickets/` for drafts. Clear it with
`python app.py --reset-memory`, or delete the directory. `python app.py
--show-storage` prints the exact paths. Ticket drafts from scenario runs are
copied into `outputs/tickets/` so they can be committed and read.

**Storage choice.** SQLite for both, which is why memory survives a restart at
all. For conversation state I used `langgraph-checkpoint-sqlite` rather than
`MemorySaver`: `MemorySaver` would pass a single-process test and fail S11,
because the state would never reach a file. For the preference I wrote a small
SQLite table instead of reusing the checkpointer, because the checkpointer's key
is the thread and the preference must not be — the two need different keys, not
different rows.

### Tests, and what each would catch

`python -m pytest -v` — **100 tests, no API key, no network.** Every test gets a
throwaway storage directory, and the fixture deletes `ANTHROPIC_API_KEY` from the
environment so a test that started making real calls would fail loudly instead
of billing anyone.

| Test file | What it checks | What it would catch if the code broke |
| --- | --- | --- |
| `tests/test_part1_routing.py` | both Part 1 graphs, all four brief messages, aliases and case, the help nodes' content | a broken conditional edge; routing that stopped ignoring case; the two-topic precedence silently flipping; `clarify_problem` wired so it can never run; a help node answering from prose instead of `helpdesk.json` |
| `tests/test_tools.py` | each tool called directly, all four snapshots, the refusal path, redaction, the signatures | a tool that stopped reading the fixtures; `unknown` status collapsing into operational or outage; `password` no longer mapping to `account_recovery`; a ticket inventing missing details; a credential reaching a saved draft; the snapshot or user id becoming a model-supplied argument |
| `tests/test_memory.py` | preference detection and its false positives, the store, isolation, reopening, reset | a preference no longer learned from the message; memory keyed per thread instead of per user (so every new conversation forgets it); one user's memory visible to another; a store that kept everything in RAM and lost it on restart; a reset that wiped the wrong file |
| `tests/test_agent_fake.py` | the Part 2/3 graph driven by `helpdesk/fake_model.py` | a graph that answers without ever running the requested tool; a tool result that never returns to the model; a missing round limit (an endless loop in production); user memory not reaching the model; a conversation that does not resume from its checkpoint; threads or users sharing state they should not |
| `tests/test_cli_restart.py` | `app.py` launched as **real separate processes** against one storage directory | a checkpointer that only ever lived in memory; state written under a key a restart does not look up; a preference stored per thread instead of per user; memory bleeding across `--user`; `--thread` not separating conversations; the CLI rejecting a valid snapshot name |

**Which runs use a fake model.** All of them. `helpdesk/fake_model.py` is a
scripted stand-in implementing only what the graph uses — `bind_tools()` and
`invoke()` — and returning the tool calls its script specifies. That is what
lets CI check the routing, the tools, the round limit and the memory wiring for
free. There are three fakes, each for one job: `FakeToolModel` replays a script
of turns; `AlwaysToolsModel` never stops asking for a tool, which is how the
round limit is proved; and `EchoStateModel` reports what a process can see,
which is how a restart is proved from the outside. `app.py --fake` uses the
last one and labels its own output as a pipeline check, so a fake run can never
be mistaken for a result.

**Why a subprocess test exists at all.** `tests/test_agent_fake.py` opens and
closes the checkpointer around separate `HelpdeskApp` objects, which proves the
state goes through SQLite — but both halves share one Python process, so a bug
that only appears when the interpreter actually exits would survive it.
`tests/test_cli_restart.py` launches `python app.py` twice as separate
processes with nothing passed between them but the storage directory, and
asserts the second launch sees two user messages where only one was given to it.
That is the same thing `restart_process` means in `data/scenarios.json`.

**The real-model demonstration is kept separate** — it is `run_scenarios.py`
plus the committed `outputs/` logs, not part of the test suite. The test suite
must pass with no key; the demonstration must use a real one.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request: Ubuntu, Python 3.12, `pip install -r requirements.txt`,
`pytest -v`, then `python run_scenarios.py --part 1` with an assertion that a
trace file was produced. **No secret is configured for the workflow**, which is
the point rather than an omission: if a test or the Part 1 path ever started
needing a model key, CI would go red.

Runs: https://github.com/Tileuzhan99/css4007-hw5/actions

### Commits

One commit per stage, each saying what changed and why — Part 1, the tools,
memory, the model-and-tool loop with the app, the scenario runner, then the
setup/CI work. `git log --oneline` in the repository.

---

## Choices and limitations

**What I chose, and why it fits.**

- **LangGraph's `StateGraph` for Part 1 too**, rather than only for the agent.
  It makes the Part 1 route inspectable and testable with no key, and it meant
  the step from fixed rules to a model loop was a change of *what the branch
  reads*, not a rewrite.
- **Plain functions under the LangChain tools.** The tools are testable without
  a model, which is most of what CI checks, and the model-facing wrapper is a
  thin layer whose docstring is the description the model sees.
- **`SessionContext` rather than tool arguments** for the snapshot, user and
  thread. It makes "the model cannot choose whose data to load" a property of
  the signatures, not of the prompt.
- **Two SQLite stores**, for the reasons in Part 4.
- **A provider-agnostic `helpdesk/model.py`.** The graph needs only an object
  with `bind_tools()` and `invoke()`, so the Anthropic and OpenAI adapters are
  interchangeable and the fake model drops into the same hole.

**What still does not work reliably.**

- **Preference detection is a whitelist and a regex.** A language outside
  `_KNOWN_LANGUAGES`, or a preference phrased without a language name
  ("answer like last time"), is simply not learned. I chose the false-negative
  direction deliberately — a looser pattern stored "in Building A" as a
  language — but it is a limitation, not a solved problem. A more honest design
  would let the model propose the preference through a fourth tool and have the
  application decide whether to store it.
- **Only one preference key is used.** The store is a general key/value table,
  but nothing except `reply_language` is ever written to it.
- **The reply language is a prompt instruction, not a guarantee.** The system
  prompt asks for the saved language; nothing verifies the reply is in it.
- **`prepare_ticket`'s redaction is pattern-based.** It catches "my password is
  X" and similar shapes, not a credential pasted bare on its own line.
- **The round limit bounds tool rounds, not cost.** Four rounds of large tool
  results is still a large request; there is no token budget.
- **Conversation state grows without bound.** Every turn resends the whole
  thread, with no summarising or trimming, so a very long conversation would
  eventually hit the context window.
- **`other_help` cannot help.** By design it only asks which of the two topics
  it is; a genuinely out-of-scope problem (S03's broken screen) gets no route to
  anything useful, which is the right behaviour for this fixture but not a real
  helpdesk.

---

## Supplied scenarios — my actual results

Run with `python run_scenarios.py` (see README). Each scenario's command,
console output and trace are in `outputs/`. S01–S05 need no API key.

| Scenario | Command or manual steps | Log file | What happened / what I changed |
| --- | --- | --- | --- |
| S01 — Simple Wi-Fi route | `python app.py --graph part1 --user demo-user-a --thread s01 --profile normal --message "I cannot connect to Wi-Fi."` | `outputs/S01.log.md`, `outputs/S01-run1.json` | `category=wifi`, route `read_request → wifi_help`. Reply is GUIDE-WIFI read out of `helpdesk.json`. Matched my prediction; nothing changed. |
| S02 — Case-insensitive password route | same, `--thread s02 --message "I forgot my PASSWORD."` | `outputs/S02.log.md`, `outputs/S02-run1.json` | `category=password`, route `read_request → password_help`, GUIDE-PASSWORD. Upper case made no difference. Matched. |
| S03 — Unsupported request | same, `--thread s03 --message "My laptop screen is broken."` | `outputs/S03.log.md`, `outputs/S03-run1.json` | `category=other`, route `read_request → other_help`. Asks whether it is Wi-Fi or a password rather than attempting an answer. Matched. |
| S04 — Two topics before the graph change | `--graph part1 --thread s04 --message "My Wi-Fi password does not work."` | `outputs/S04.log.md`, `outputs/S04-run1.json` | `category=wifi`, route `read_request → wifi_help`: the Wi-Fi test runs first and returns, so `password` is never checked. The user asked about a password and got Wi-Fi steps. Matched my prediction — and it is the reason for the change below. |
| S05 — Two topics after adding clarification | `--graph part1-clarify --thread s05 --message "My Wi-Fi password does not work."` | `outputs/S05.log.md`, `outputs/S05-run1.json` | `category=both`, route `read_request → clarify_problem`. Asks which problem to solve first. Different node *and* a category the first graph cannot produce — the behaviour changed, not just the wording. |
| S06 — Instructions then local ticket | | | |
| S07 — The same request during a Wi-Fi outage | | | |
| S08 — Status source unavailable | | | |
| S09 — Missing ticket details | | | |
| S10 — Use the current conversation | | | |
| S11 — Persist and recall after restart | | | |
| S12 — Preference across threads without replay | | | |
| S13 — Do not share memory between users | | | |
| S14 — Password recovery outage | | | |
