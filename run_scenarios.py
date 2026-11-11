"""Run the supplied demonstrations S01-S14 and save one log per scenario.

Each scenario is split into *segments*. A new segment starts at a
`restart_process` event, and also whenever the user or the thread changes
(S13 talks to two users). Every segment is a separate `python app.py ...`
subprocess, so a `restart_process` really is a new process reading the same
storage - not another call inside this one.

    python run_scenarios.py                 # all fourteen
    python run_scenarios.py --only S01 S04  # just these
    python run_scenarios.py --part 1        # the scenarios that need no API key
    python run_scenarios.py --dry-run       # print the commands only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from helpdesk import config, data_access

# S04 must use the original Part 1 graph and S05 the changed one
# (data/README.md); the other Part 1 scenarios may use either, and we use the
# original so S01-S04 are one consistent set.
PART1_GRAPH = {"S05": "part1-clarify"}


def graph_for(scenario: dict[str, Any]) -> str:
    if scenario["part"] == 1:
        return PART1_GRAPH.get(scenario["id"], "part1")
    return "agent"


def segments(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Group the events into one subprocess run each."""
    result: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    restart_pending = False

    for event in scenario["events"]:
        if event["action"] == "restart_process":
            restart_pending = True
            current = None
            continue
        if event["action"] != "message":
            raise ValueError(f"unsupported event action {event['action']!r}")

        key = (event["user_id"], event["thread_id"])
        if current is None or current["key"] != key:
            current = {
                "key": key,
                "user_id": event["user_id"],
                "thread_id": event["thread_id"],
                "messages": [],
                "after_restart": restart_pending,
            }
            restart_pending = False
            result.append(current)
        current["messages"].append(" ".join(event["message"].split()))

    return result


def command_for(
    scenario: dict[str, Any], segment: dict[str, Any], trace_path: Path
) -> list[str]:
    command = [
        sys.executable,
        "app.py",
        "--graph",
        graph_for(scenario),
        "--user",
        segment["user_id"],
        "--thread",
        segment["thread_id"],
        "--profile",
        scenario["status_profile"],
    ]
    for message in segment["messages"]:
        command += ["--message", message]
    command += ["--trace", trace_path.relative_to(config.REPO_ROOT).as_posix()]
    return command


def pretty(command: list[str]) -> str:
    """The same command, in a form a reader can paste into a shell."""
    parts = []
    for item in command:
        if item == sys.executable:
            parts.append("python")
        elif " " in item:
            parts.append(f'"{item}"')
        else:
            parts.append(item)
    return " ".join(parts)


def reset_storage() -> None:
    directory = config.storage_dir()
    if directory.exists():
        shutil.rmtree(directory)


def collect_tickets(destination: Path) -> list[str]:
    source = config.tickets_dir()
    if not source.exists():
        return []
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.glob("*.json")):
        target = destination / path.name
        if not target.exists():
            shutil.copy2(path, target)
            copied.append(target.name)
    return copied


def run_scenario(scenario: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    scenario_id = scenario["id"]
    outputs = config.OUTPUTS_DIR
    outputs.mkdir(parents=True, exist_ok=True)

    if scenario["reset_before"] and not dry_run:
        reset_storage()

    record: dict[str, Any] = {
        "scenario": scenario_id,
        "title": scenario["title"],
        "part": scenario["part"],
        "status_profile": scenario["status_profile"],
        "graph": graph_for(scenario),
        "reset_before": scenario["reset_before"],
        "segments": [],
    }
    console: list[str] = [
        f"# {scenario_id} - {scenario['title']}",
        f"part: {scenario['part']}   graph: {record['graph']}   "
        f"snapshot: {scenario['status_profile']}   reset_before: {scenario['reset_before']}",
        "",
    ]

    for index, segment in enumerate(segments(scenario), start=1):
        trace_path = outputs / f"{scenario_id}-run{index}.json"
        command = command_for(scenario, segment, trace_path)
        label = (
            f"## run {index}"
            + (" (new process, same storage)" if segment["after_restart"] else "")
            + f" - user {segment['user_id']}, thread {segment['thread_id']}"
        )
        console += [label, "", "```", pretty(command), "```", ""]
        print(f"  {label}")
        print(f"    {pretty(command)}")

        entry = {
            "run": index,
            "after_restart": segment["after_restart"],
            "user_id": segment["user_id"],
            "thread_id": segment["thread_id"],
            "messages": segment["messages"],
            "command": pretty(command),
            "trace_file": trace_path.name,
        }

        if not dry_run:
            completed = subprocess.run(
                command,
                cwd=str(config.REPO_ROOT),
                capture_output=True,
                text=True,
                # app.py writes UTF-8. Without this, Python decodes the child's
                # output with the console code page and a Kazakh reply lands in
                # the log as mojibake.
                encoding="utf-8",
                errors="replace",
            )
            console += [completed.stdout.rstrip(), ""]
            if completed.stderr.strip():
                console += ["stderr:", completed.stderr.rstrip(), ""]
            entry["exit_code"] = completed.returncode
            if completed.returncode != 0:
                print(f"    FAILED (exit {completed.returncode})")
                print(completed.stderr.strip()[:2000])

        record["segments"].append(entry)

    if not dry_run:
        tickets = collect_tickets(outputs / "tickets")
        record["tickets"] = tickets
        if tickets:
            console += ["ticket drafts copied to outputs/tickets/: " + ", ".join(tickets), ""]
        (config.OUTPUTS_DIR / f"{scenario_id}.log.md").write_text(
            "\n".join(console) + "\n", encoding="utf-8"
        )

    return record


INDEX_NAME = "scenario_index.json"


def write_index(records: list[dict[str, Any]]) -> Path:
    """Merge these records into the index instead of replacing it.

    Scenarios are often run in batches - the Part 1 ones need no key, and a
    free-tier daily quota can force the rest into more than one go - so an
    index that only ever held the newest batch would always be incomplete.
    """
    index_path = config.OUTPUTS_DIR / INDEX_NAME
    existing: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        try:
            for entry in json.loads(index_path.read_text(encoding="utf-8")):
                existing[entry["scenario"]] = entry
        except (json.JSONDecodeError, KeyError, TypeError):
            existing = {}

    for record in records:
        existing[record["scenario"]] = record

    ordered = [existing[s["id"]] for s in data_access.scenarios() if s["id"] in existing]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    return index_path


def reindex() -> list[dict[str, Any]]:
    """Rebuild index entries from the traces already saved in outputs/.

    Runs no scenario and calls no model. Entries rebuilt this way are marked
    `rebuilt_from_trace`, because the exit code of the original process is not
    observed here - only the trace it left behind.
    """
    rebuilt = []
    for scenario in data_access.scenarios():
        entries = []
        for index, segment in enumerate(segments(scenario), start=1):
            trace_path = config.OUTPUTS_DIR / f"{scenario['id']}-run{index}.json"
            if not trace_path.exists():
                continue
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            entries.append(
                {
                    "run": index,
                    "after_restart": segment["after_restart"],
                    "user_id": segment["user_id"],
                    "thread_id": segment["thread_id"],
                    "messages": segment["messages"],
                    "command": pretty(command_for(scenario, segment, trace_path)),
                    "trace_file": trace_path.name,
                    "model": trace.get("model"),
                    "rebuilt_from_trace": True,
                }
            )
        if entries:
            rebuilt.append(
                {
                    "scenario": scenario["id"],
                    "title": scenario["title"],
                    "part": scenario["part"],
                    "status_profile": scenario["status_profile"],
                    "graph": graph_for(scenario),
                    "reset_before": scenario["reset_before"],
                    "segments": entries,
                }
            )
    return rebuilt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the supplied HW5 scenarios.")
    parser.add_argument("--only", nargs="*", metavar="ID", help="scenario ids, e.g. S01 S06")
    parser.add_argument("--part", type=int, choices=[1, 2, 3], help="only this part")
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild scenario_index.json from the traces already in outputs/, running nothing",
    )
    args = parser.parse_args(argv)

    if args.reindex:
        records = reindex()
        index_path = write_index(records)
        print(f"rebuilt {len(records)} scenarios into {index_path.relative_to(config.REPO_ROOT)}")
        return 0

    wanted = data_access.scenarios()
    if args.only:
        requested = {item.upper() for item in args.only}
        wanted = [s for s in wanted if s["id"] in requested]
    if args.part:
        wanted = [s for s in wanted if s["part"] == args.part]
    if not wanted:
        print("no scenarios selected", file=sys.stderr)
        return 2

    summary = []
    for scenario in wanted:
        print(f"\n=== {scenario['id']} - {scenario['title']} ===")
        summary.append(run_scenario(scenario, dry_run=args.dry_run))

    if not args.dry_run:
        index_path = write_index(summary)
        print(f"\nsummary written to {index_path.relative_to(config.REPO_ROOT)}")

        failed = [
            f"{record['scenario']} run {seg['run']}"
            for record in summary
            for seg in record["segments"]
            if seg.get("exit_code") not in (0, None)
        ]
        if failed:
            print("FAILED: " + ", ".join(failed), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
