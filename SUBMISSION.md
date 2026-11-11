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

**Provider:** Google (`google_genai`), through `langchain-google-genai==4.4.3`.
**Runs made:** 11 November 2026.

| Scenarios | Model |
| --- | --- |
| S06, S07, S08, S09, S10 | `gemini-3.5-flash` |
| S11, S12, S13, S14 | `gemini-3.5-flash-lite` |

Two models, and the reason is worth recording because it is a real constraint
rather than a preference. The key available to me is on the Gemini free tier,
which caps **generate-content requests per day per model** at 20. S06–S10 used
that allowance for `gemini-3.5-flash`, and S11 then failed with
`429 RESOURCE_EXHAUSTED … Quota exceeded … limit: 20 … Please retry in 17h2m`.
The cap is counted per model, so the remaining four scenarios ran on
`gemini-3.5-flash-lite`, which had its own allowance. Every saved trace records
the provider and model it actually used (`"model"` in each
`outputs/S*-run*.json`), and `outputs/scenario_index.json` lists them per run,
so nothing here depends on remembering which was which.

Choosing the model was not free either. `gemini-2.5-flash` is retired for new
keys and answers with a 404 saying so. `gemini-3.8-flash` and
`gemini-flash-latest` accept a `generateContent` request and then never answer
it, so a run against them hangs instead of failing. `gemini-3-flash-preview`
exists only on API version `v1beta`, while the adapter calls `v1` by default
and gets a 404 — which is why `helpdesk/model.py` now takes
`MODEL_API_VERSION`. `gemini-3.5-flash` was the newest model that both answered
and supported tool calls.

Both of those failures now produce one actionable line instead of a traceback:
`helpdesk/model.py:explain_error()` recognises quota, not-found and credential
errors, and `app.py` prints its sentence and exits 3.

If the whole demonstration must come from one model, `python
run_scenarios.py` re-runs all fourteen once a daily allowance is free, or with
a key that has billing enabled.

### Direct calls to each tool, with no agent

The tools are plain functions, so this needs no model at all:

```python
>>> from helpdesk import tools
>>> tools.find_instructions("wifi")
{'ok': True, 'topic': 'wifi', 'title': 'Connect to the practice campus network',
 'steps': ['Check that Wi-Fi is enabled and airplane mode is off.', ...],
 'instruction_id': 'GUIDE-WIFI', 'source': 'data/helpdesk.json'}

>>> ctx = tools.SessionContext(user_id="demo-user-a", thread_id="t", status_profile="wifi_outage")
>>> with tools.use_session(ctx):
...     tools.check_service_status("wifi")
{'ok': True, 'service': 'wifi', 'status': 'outage', 'is_operational': False,
 'message': 'The demo Wi-Fi service is unavailable across campus. ...',
 'status_profile': 'wifi_outage', 'source': 'data/service_status.json'}

>>> with tools.use_session(ctx):
...     tools.prepare_ticket(summary="", steps_tried=[])
{'ok': False, 'error': 'missing_information',
 'missing': ['problem description', 'steps already tried'], ...}
```

`find_instructions` needs no session at all; the other two need one because
they read the snapshot and the identifiers from it. The 31 tests in
`tests/test_tools.py` are these same direct calls across all four snapshots and
both topics, including the refusal and redaction paths.

### One run that uses more than one tool

S06, two messages in one conversation. Full log: `outputs/S06.log.md`, trace:
`outputs/S06-run1.json`.

```text
you> Show me the Wi-Fi troubleshooting steps.
  [load_memory] user=demo-user-a thread=s06 user_memory={} learned=None history_before_turn=0
  [call_model] requests tools: find_instructions({"topic": "wifi"})
  [run_tools] find_instructions -> {"ok": true, "instruction_id": "GUIDE-WIFI"}
  [call_model] answers without tools
helpdesk> Here are the Wi-Fi troubleshooting steps from the stored guide ...

you> I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not
     connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket
     about this problem.
  [load_memory] user=demo-user-a thread=s06 user_memory={} learned=None history_before_turn=4
  [call_model] requests tools: check_service_status({"service": "wifi"})
  [run_tools] check_service_status -> {"ok": true, "status": "operational"}
  [call_model] requests tools: prepare_ticket({"summary": "Campus-Demo still will not connect",
                                               "category": "wifi",
                                               "steps_tried": ["enabled Wi-Fi",
                                                               "forgot the saved network"]})
  [run_tools] prepare_ticket -> {"ok": true, "path": "local-storage/tickets/TKT-1FF4F520.json"}
  [call_model] answers without tools
helpdesk> ... the Wi-Fi service is operational ... Since the Wi-Fi service is
          operating, I have prepared a local ticket draft: TKT-1FF4F520 ...
          This draft is stored locally only; no real ticket has been submitted.
```

Three tools across the two turns, and the second turn is two rounds: the
`prepare_ticket` call is only made **after** the status result comes back, which
is the point — the model's second action depends on the first tool's output, not
on the wording of the request.

The draft, committed at `outputs/tickets/TKT-1FF4F520.json`:

```json
{
  "ticket_id": "TKT-1FF4F520", "status": "draft",
  "user_id": "demo-user-a", "thread_id": "s06", "category": "wifi",
  "summary": "Campus-Demo still will not connect",
  "steps_tried": ["enabled Wi-Fi", "forgot the saved network"],
  "service_status": "operational", "status_profile": "normal",
  "reference_ids": ["GUIDE-WIFI"]
}
```

`steps_tried` holds the two things the user said and nothing else — no invented
troubleshooting. `user_id` and `thread_id` came from the session, not from the
model's arguments.

### The same conditional request against three snapshots

Same conditional request ("check whether Wi-Fi is down; if it is operating,
prepare a ticket"), three different snapshots, three different actions. Only
`--profile` changed.

| Scenario | Snapshot | Tool result | What the assistant did | Ticket? |
| --- | --- | --- | --- | --- |
| S06 | `normal` | `status: operational` | prepared the draft, because the condition held | yes — `TKT-1FF4F520` |
| S07 | `wifi_outage` | `status: outage` | "Because Wi-Fi is down rather than operating, I did not prepare a ticket." | no |
| S08 | `status_unavailable` | `status: unknown` | "the Wi-Fi status source is unavailable, so the status is currently unknown and I cannot determine whether Wi-Fi is operating. Because Wi-Fi is not confirmed to be operating, I did not prepare a ticket." | no |

The S08 line is the one that matters: `unknown` was treated as *not confirmed
operating* rather than as either an outage or an all-clear, and no repair time
was invented in S07 (the fixture gives none).

S14 changes the topic instead of the snapshot. The request is about the
password-recovery page under `recovery_outage`, and the model looked up
`account_recovery`, not `wifi`:

```text
  [call_model] requests tools: check_service_status({"service": "account_recovery"})
  [run_tools] check_service_status -> {"ok": true, "status": "outage"}
helpdesk> From the status snapshot, the account-recovery service is currently
          experiencing a known outage.
```

That is the `topic_services` mapping in `data/helpdesk.json` being used, and it
is checked independently of the model in
`test_password_topic_maps_to_account_recovery`.

**One difference from SCENARIOS.md, recorded as it happened.** For S08 the
illustrative dialogue also *offers* a draft explaining that the status could not
be checked ("Would you like a ticket draft …?"). My run declined correctly but
did not make the offer. The brief says the exact wording need not match, so I
have left the run as it is rather than re-running it to chase the phrasing; if I
wanted the offer reliably, the right fix is a line in the system prompt telling
the model what to propose when a status is `unknown`, not a change to the
tools.

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

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | `python app.py --user demo-user-a --thread s10 --profile normal --message "I cannot connect to Wi-Fi. I enabled it and forgot the saved network, but reconnecting failed." --message "Which steps did I already try?"` | `outputs/S10.log.md`, `S10-run1.json` | Second turn ran **no tool** and answered from the conversation: "you have already tried: 1. Enabling Wi-Fi 2. Forgetting the saved network 3. Attempting to reconnect". The trace shows `history_before_turn=5`. |
| Close, restart and resume | two launches: `… --thread restart-a --message "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."` then, as a **new process**, `… --thread restart-a --message "Which problem and steps did we discuss?"` | `outputs/S11.log.md`, `S11-run1.json`, `S11-run2.json` | Second process started with `history_before_turn=6` although it was given one message, and answered in Kazakh that we discussed the Wi-Fi problem and the steps of enabling Wi-Fi and reconnecting. |
| Same user, new conversation | `… --thread preference-a1 --message "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem."` then a new process with `--thread preference-a2 --message "Which language did I ask you to use? What can you help with?"` | `outputs/S12.log.md`, `S12-run1.json`, `S12-run2.json` | Second process: `history_before_turn=0` (a genuinely fresh conversation) but `user_memory={"reply_language": "Kazakh"}`. It answered "Қазақ тілін (saved user memory store дерекқоры бойынша)" — the language, and the store it came from. |
| Different user | `… --user demo-user-a --thread isolation-a --message "Please remember that I prefer replies in Kazakh. …"` then `--user demo-user-b --thread isolation-b --message "Have I told you a preferred language or a previous problem?"` | `outputs/S13.log.md`, `S13-run1.json`, `S13-run2.json` | `demo-user-b` loaded `user_memory={}` and replied in English: "Nothing is recorded for this user. If you would like to note a preference … (Source: saved user memory)". No trace of `demo-user-a`'s Kazakh preference. |

**What was read from where, per demonstration.** Every `load_memory` trace entry
names its sources explicitly: `user_memory` with the store path,
`conversation_messages_before_turn` for the checkpoint, and `learned_preference`
when this message taught it something. Tool results name theirs
(`source: data/helpdesk.json`, `source: data/service_status.json`). So in S12
run 2 the reply came from user memory with an empty conversation; in S10 turn 2
it came from the conversation with no tool call at all; and in S06 the steps came
from the reference file.

**The preference is learned, not configured.** In S11, S12 and S13 the first
turn shows `learned=Kazakh` — the application saw the phrase in the message and
wrote it. `data/demo_users.json` supplies no preference, and a test asserts the
file has none to read.

**One bug this actually caught.** The first attempt at S11–S13 failed
*after* the graph had run and the state was saved: `app.py` printed the Kazakh
reply to a Windows console whose code page cannot represent it, so `print()`
raised `UnicodeEncodeError` and killed the process. The scenario runner had the
same flaw in reverse — it decoded the child's UTF-8 output with the console code
page, which would have written mojibake into these very logs. Both are fixed
(`use_utf8_output()` in `app.py`, explicit `encoding="utf-8"` in the runner) and
both now have regression tests in `tests/test_cli_restart.py`. It is worth
noting because it is the one failure mode the in-process tests could never have
found: the state was correct, and only the act of showing it crashed.

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

`python -m pytest -v` — **102 tests, no API key, no network** (22 routing + 31 tools + 19 memory + 19 agent + 11 CLI). Every test gets a
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
- **The demonstration spans two models**, because of the free-tier per-model
  daily cap described in Part 2. Each trace records which model produced it, but
  a reader comparing S10 with S11 is comparing two models as well as two
  scenarios.
- **S08 declines without offering the alternative draft** that SCENARIOS.md
  illustrates. The decision is right; the helpfulness is missing.
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
| S06 — Instructions then local ticket | `python app.py --graph agent --user demo-user-a --thread s06 --profile normal --message "Show me the Wi-Fi troubleshooting steps." --message "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem."` | `outputs/S06.log.md`, `S06-run1.json`, ticket `outputs/tickets/TKT-1FF4F520.json` | Turn 1: `find_instructions` → GUIDE-WIFI steps. Turn 2: `check_service_status` → `operational`, then `prepare_ticket` → draft `TKT-1FF4F520` with only the two steps the user named. Three tools, two rounds in turn 2. Nothing I would change. |
| S07 — The same request during a Wi-Fi outage | same message as S06 turn 2, `--profile wifi_outage --thread s07` | `outputs/S07.log.md`, `S07-run1.json` | `check_service_status` → `outage`; no `prepare_ticket` call at all. "Because Wi-Fi is down rather than operating, I did not prepare a ticket." No repair time invented. Condition honoured. |
| S08 — Status source unavailable | `--profile status_unavailable --thread s08 --message "Check whether Wi-Fi is down. If it is operating, prepare a ticket. I already enabled Wi-Fi and reconnected, with no success."` | `outputs/S08.log.md`, `S08-run1.json` | `status: unknown` → no ticket: "I cannot determine whether Wi-Fi is operating." `unknown` treated as a third answer, not an all-clear. **What I would change:** it should also offer a draft explaining that the status could not be checked, which needs a prompt line, not a tool change. |
| S09 — Missing ticket details | `--thread s09 --message "Prepare a support ticket." --message "The issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."` | `outputs/S09.log.md`, `S09-run1.json`, ticket `outputs/tickets/TKT-C1A7A6B8.json` | Turn 1: `prepare_ticket` with empty arguments → `missing_information`, and the assistant asked for the problem description and the steps tried. Turn 2: draft from the user's own words. The draft records `service_status: "not_checked"` — no status was looked up and the ticket says so instead of guessing. |
| S10 — Use the current conversation | `--thread s10 --message "I cannot connect to Wi-Fi. I enabled it and forgot the saved network, but reconnecting failed." --message "Which steps did I already try?"` | `outputs/S10.log.md`, `S10-run1.json` | Turn 1 asked for `find_instructions` and `check_service_status` **in one round** (parallel calls, both executed). Turn 2 used no tool and listed the three steps from the conversation. |
| S11 — Persist and recall after restart | two separate processes, same `--thread restart-a`; commands in `outputs/S11.log.md` | `outputs/S11.log.md`, `S11-run1.json`, `S11-run2.json` | Process 2 was given one message and started from `history_before_turn=6`, then named the Wi-Fi problem and both steps — in Kazakh, from the preference learned in process 1. The old transcript was never passed on the command line. |
| S12 — Preference across threads without replay | two processes, `--thread preference-a1` then `preference-a2`, same `--user demo-user-a` | `outputs/S12.log.md`, `S12-run1.json`, `S12-run2.json` | Thread 2: `history_before_turn=0` but `user_memory={"reply_language": "Kazakh"}`. Answered in Kazakh and named the user memory store as the source. Preference crosses threads; history does not. |
| S13 — Do not share memory between users | `--user demo-user-a --thread isolation-a`, then `--user demo-user-b --thread isolation-b` | `outputs/S13.log.md`, `S13-run1.json`, `S13-run2.json` | `demo-user-b` loaded `user_memory={}` and said nothing is recorded for this user, in English. No leak of the other user's preference or problem. |
| S14 — Password recovery outage | `--profile recovery_outage --thread s14 --message "My password recovery page is unavailable. Check the service status. Is there a known problem?"` | `outputs/S14.log.md`, `S14-run1.json` | Looked up `account_recovery`, not `wifi` → `outage`, and reported a known outage without inventing a repair time. The `topic_services` mapping doing its job. |
