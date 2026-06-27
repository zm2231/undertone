from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

from pydantic import ValidationError

from undertone_audio.commands import connectors as connector_commands
from undertone_audio.commands import core as core_commands
from undertone_audio.commands import fingerprints as fingerprint_commands
from undertone_audio.commands import ops as ops_commands
from undertone_audio.commands import sources as source_commands
from undertone_audio.commands import webhook as webhook_commands


_COMMAND_GROUPS: list[tuple[str, list[str]]] = [
    ("Ingest audio", ["run-wav", "finalize-json"]),
    ("Reprocess", ["reenrich"]),
    ("Browse & inspect", ["list", "load", "search", "stats"]),
    (
        "Speakers",
        [
            "fingerprints",
            "fingerprint-label",
            "relabel",
            "resolve-names",
            "fingerprint-export",
            "fingerprint-import",
            "fingerprint-merge",
            "fingerprint-adopt-model",
        ],
    ),
    ("Maintenance", ["delete"]),
    ("Webhook", ["emit-ready", "webhook-preview"]),
    (
        "Sources",
        [
            "youtube-ingest",
            "connector-list",
            "connector-ingest",
            "podcast-list",
            "podcast-ingest",
            "quill-list",
            "quill-ingest",
            "meet-discover",
            "meet-ingest",
            "sources",
        ],
    ),
    ("Diagnostics", ["doctor", "models", "schema", "install-skills"]),
]


class _OverviewHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):
            return ""
        return super()._format_action(action)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _configure_logging(args)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except (OSError, RuntimeError, sqlite3.Error, ValidationError, ValueError) as exc:
        if getattr(args, "progress", "off") == "json":
            print(
                json.dumps(
                    {"event": "error", "error": str(exc), "error_type": type(exc).__name__},
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
        else:
            print(f"undertone: {exc}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="undertone",
        description="Local audio transcript producer: transcription, diarization, "
        "speaker fingerprints, and audio-derived enrichment.",
        formatter_class=_OverviewHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite DB path. Defaults to UNDERTONE_DB_PATH or ./undertone.db.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Enable INFO logs.")
    verbosity.add_argument("--quiet", action="store_true", help="Only print errors and requested output.")
    parser.set_defaults(func=None)
    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    core_commands.register(subcommands)
    webhook_commands.register(subcommands)
    fingerprint_commands.register(subcommands)
    ops_commands.register(subcommands)
    source_commands.register(subcommands)
    connector_commands.register(subcommands)

    parser.epilog = _overview_text(subcommands)
    return parser


def _overview_text(subcommands: argparse._SubParsersAction) -> str:
    help_by_name = {action.dest: (action.help or "") for action in subcommands._choices_actions}
    names = list(help_by_name)
    width = max((len(name) for name in names), default=0)

    grouped: set[str] = set()
    blocks: list[str] = ["Commands:"]
    for title, group in _COMMAND_GROUPS:
        present = [name for name in group if name in help_by_name]
        if not present:
            continue
        lines = [f"  {title}:"]
        for name in present:
            grouped.add(name)
            lines.append(f"    {name.ljust(width)}  {help_by_name.get(name, '')}")
        blocks.append("\n".join(lines))

    ungrouped = [name for name in names if name not in grouped]
    if ungrouped:
        lines = ["  Other:"]
        for name in ungrouped:
            lines.append(f"    {name.ljust(width)}  {help_by_name.get(name, '')}")
        blocks.append("\n".join(lines))

    blocks.append(
        "Run 'undertone <command> -h' for options on a command.\n"
        "For agents: add --json for machine-readable output."
    )
    return "\n\n".join(blocks)


def _configure_logging(args: argparse.Namespace) -> None:
    if getattr(args, "quiet", False):
        level = logging.ERROR
    elif getattr(args, "verbose", False):
        level = logging.INFO
    elif getattr(args, "progress", "off") == "json":
        level = logging.ERROR
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s", force=True)


if __name__ == "__main__":
    raise SystemExit(main())
