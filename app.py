"""Terminal entry point for the HW5 helpdesk.

Examples (PowerShell; see README.md for the full list):

    # Part 1, the original routing graph
    python app.py --graph part1 --message "I cannot connect to Wi-Fi."

    # Part 1, the graph with clarify_problem
    python app.py --graph part1-clarify --message "My Wi-Fi password does not work."

    # Parts 2-3: model, tools and memory, two messages in one conversation
    python app.py --user demo-user-a --thread s06 --profile normal \
        --message "Show me the Wi-Fi troubleshooting steps." \
        --message "I enabled Wi-Fi and forgot the saved network, but Campus-Demo
                   still will not connect. Check whether Wi-Fi is down. If it is
                   operating, prepare a ticket about this problem." \
        --trace outputs/S06.json

    # Interactive conversation (Ctrl-C or an empty line to leave)
    python app.py --user demo-user-a --thread chat

    # Forget every conversation and preference
    python app.py --reset-memory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from helpdesk import config, data_access, memory, model as model_module, part1_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="University IT helpdesk demo (CSS-4007 HW5, LangGraph).",
    )
    parser.add_argument("--user", default="demo-user-a", help="invented user identifier")
    parser.add_argument("--thread", default="default", help="conversation identifier")
    parser.add_argument(
        "--profile",
        default=config.DEFAULT_STATUS_PROFILE,
        help="service-status snapshot from data/service_status.json",
    )
    parser.add_argument(
        "--graph",
        choices=["agent", "part1", "part1-clarify"],
        default="agent",
        help="which graph to run (default: agent = Parts 2 and 3)",
    )
    parser.add_argument(
        "--message",
        action="append",
        default=[],
        metavar="TEXT",
        help="one message; repeat for several turns in the same conversation",
    )
    parser.add_argument("--trace", metavar="PATH", help="write the run as JSON to this file")
    parser.add_argument(
        "--fake",
        action="store_true",
        help=(
            "run the agent graph with a scripted fake model instead of a real one. "
            "It gives no helpdesk advice; it only reports what the process can see, "
            "so the graph, storage and restart wiring can be checked without a key."
        ),
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="delete all conversation checkpoints and user preferences, then exit",
    )
    parser.add_argument(
        "--show-storage", action="store_true", help="print the storage paths and exit"
    )
    return parser


def _normalise(message: str) -> str:
    return " ".join(message.split())


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------


def run_part1(messages: list[str], *, with_clarification: bool) -> list[dict[str, Any]]:
    turns = []
    for message in messages:
        state = part1_graph.run_once(message, with_clarification=with_clarification)
        snapshot = {"message": message, "category": "", "reply": ""}
        steps = []
        for entry in state["trace"]:
            snapshot = {**snapshot, **entry["wrote"]}
            steps.append({**entry, "state_after": dict(snapshot)})
        turns.append(
            {
                "graph": "part1-clarify" if with_clarification else "part1",
                "message": message,
                "route": [entry["node"] for entry in state["trace"]],
                "category": state["category"],
                "reply": state["reply"],
                "trace": steps,
            }
        )
    return turns


def print_part1(turn: dict[str, Any]) -> None:
    print(f"\nyou> {turn['message']}")
    print(f"route: {' -> '.join(['START', *turn['route'], 'END'])}")
    for step in turn["trace"]:
        print(f"  [{step['node']}] wrote {json.dumps(step['wrote'], ensure_ascii=False)}")
        print(f"      state after: {json.dumps(step['state_after'], ensure_ascii=False)}")
    print(f"helpdesk> {turn['reply']}")


# ---------------------------------------------------------------------------
# Parts 2 and 3
# ---------------------------------------------------------------------------


def print_agent_turn(turn: dict[str, Any]) -> None:
    print(f"\nyou> {turn['message']}")
    for step in turn["trace"]:
        node = step["node"]
        if node == "load_memory":
            print(
                f"  [load_memory] user={step['user_id']} thread={step['thread_id']} "
                f"user_memory={json.dumps(step['user_memory'], ensure_ascii=False)} "
                f"learned={step['learned_preference']} "
                f"history_before_turn={step['conversation_messages_before_turn']}"
            )
        elif node == "call_model":
            requested = step["tool_calls_requested"]
            if requested:
                names = ", ".join(
                    f"{call['name']}({json.dumps(call['args'], ensure_ascii=False)})"
                    for call in requested
                )
                print(f"  [call_model] requests tools: {names}")
            else:
                print("  [call_model] answers without tools")
        elif node == "run_tools":
            result = step["result"]
            summary = {
                key: result[key]
                for key in ("ok", "status", "instruction_id", "error", "path")
                if key in result
            }
            print(f"  [run_tools] {step['tool']} -> {json.dumps(summary, ensure_ascii=False)}")
        elif node == "loop_limit":
            print(f"  [loop_limit] stopped after {step['limit']} tool rounds")
    if turn["tickets"]:
        for path in turn["tickets"]:
            print(f"  ticket draft: {path}")
    if turn["loop_limit_reached"]:
        print("  note: the tool-round limit was reached")
    print(f"helpdesk> {turn['reply']}")


def run_agent(
    messages: list[str],
    *,
    user_id: str,
    thread_id: str,
    status_profile: str,
    interactive: bool,
    fake: bool = False,
) -> list[dict[str, Any]]:
    from helpdesk.agent_graph import HelpdeskApp

    model = None
    if fake:
        from helpdesk.fake_model import EchoStateModel

        model = EchoStateModel()
        print("NOTE: --fake is a pipeline check with a fake model, not a real run.")

    turns: list[dict[str, Any]] = []
    with memory.conversation_checkpointer() as checkpointer:
        with HelpdeskApp(
            user_id=user_id,
            thread_id=thread_id,
            status_profile=status_profile,
            checkpointer=checkpointer,
            model=model,
        ) as app:
            for message in messages:
                turn = app.send(message)
                print_agent_turn(turn)
                turns.append(turn)

            if interactive:
                print(
                    f"\nConversation {thread_id!r} for {user_id!r}, snapshot "
                    f"{status_profile!r}. Empty line to leave."
                )
                while True:
                    try:
                        text = input("\nyou> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        break
                    if not text:
                        break
                    turn = app.send(text)
                    print_agent_turn(turn)
                    turns.append(turn)
    return turns


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.reset_memory:
        removed = memory.reset_all_memory()
        print("removed:", ", ".join(removed) if removed else "nothing to remove")
        return 0

    if args.show_storage:
        print(f"storage dir       : {config.storage_dir()}")
        print(f"conversations     : {config.checkpoint_db()}")
        print(f"user memory       : {config.user_memory_db()}")
        print(f"ticket drafts     : {config.tickets_dir()}")
        return 0

    if args.profile not in data_access.status_profile_names():
        print(
            f"unknown --profile {args.profile!r}; "
            f"choose one of {data_access.status_profile_names()}",
            file=sys.stderr,
        )
        return 2

    messages = [_normalise(m) for m in args.message]

    if args.graph in ("part1", "part1-clarify"):
        if not messages:
            print("Part 1 needs at least one --message", file=sys.stderr)
            return 2
        turns = run_part1(messages, with_clarification=args.graph == "part1-clarify")
        for turn in turns:
            print_part1(turn)
    else:
        turns = run_agent(
            messages,
            user_id=args.user,
            thread_id=args.thread,
            status_profile=args.profile,
            interactive=not messages,
            fake=args.fake,
        )

    if args.trace:
        path = Path(args.trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": args.graph,
            "user_id": args.user,
            "thread_id": args.thread,
            "status_profile": args.profile,
            "model": (
                {"provider": "fake", "model": "EchoStateModel", "pipeline_check_only": True}
                if args.fake
                else model_module.describe()
            )
            if args.graph == "agent"
            else None,
            "turns": turns,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ntrace written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
