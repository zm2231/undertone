from __future__ import annotations

import argparse
import json

from undertone_audio.commands.common import config_for_args
from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
from undertone_audio.storage import TranscriptStore


def register(subcommands: argparse._SubParsersAction) -> None:
    fingerprints = subcommands.add_parser("fingerprints", help="List voice fingerprints.")
    fingerprints.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    fingerprints.add_argument("--unnamed", action="store_true", help="Only show fingerprints without labels.")
    fingerprints.add_argument("--excerpts", action="store_true", help="Show sample transcript lines.")
    fingerprints.add_argument("--limit-excerpts", type=int, default=3)
    fingerprints.set_defaults(func=fingerprints_cmd)

    label = subcommands.add_parser("fingerprint-label", help="Set a display name for a voice fingerprint.")
    label.add_argument("fingerprint_id")
    label.add_argument("display_name")
    label.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    label.set_defaults(func=fingerprint_label_cmd)


def fingerprints_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        rows = [
            {"fingerprint_id": fingerprint_id, "display_name": display_name, "sample_count": count}
            for fingerprint_id, display_name, count in SpeakerFingerprintStore(config.db_path).list_all()
            if not args.unnamed or not display_name
        ]
        excerpts = (
            store.fingerprint_excerpts(
                unnamed_only=args.unnamed,
                limit_per_fingerprint=args.limit_excerpts,
            )
            if args.excerpts
            else {}
        )
        if args.json:
            for row in rows:
                row["excerpts"] = excerpts.get(row["fingerprint_id"], [])
            print(json.dumps(rows, separators=(",", ":")))
        else:
            print(_render_fingerprints(rows, excerpts))
    finally:
        store.close()
    return 0


def fingerprint_label_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    store.close()
    SpeakerFingerprintStore(config.db_path).label(args.fingerprint_id, args.display_name)
    payload = {"fingerprint_id": args.fingerprint_id, "display_name": args.display_name}
    if args.json:
        print(json.dumps(payload, separators=(",", ":")))
    else:
        print(f"Labeled {args.fingerprint_id} as {args.display_name}")
    return 0


def _render_fingerprints(rows: list[dict], excerpts: dict[str, list[dict]]) -> str:
    if not rows:
        return "No voice fingerprints found."
    lines = ["Voice fingerprints"]
    for row in rows:
        name = row["display_name"] or "(unnamed)"
        lines.append(f"  {row['fingerprint_id']}  {name}  samples={row['sample_count']}")
        for excerpt in excerpts.get(row["fingerprint_id"], []):
            ts = _timestamp(excerpt["start_ms"])
            text = excerpt["text"].replace("\n", " ")
            lines.append(
                f"    - {excerpt['transcript_id']} {ts} {excerpt['speaker_id']}: {text}"
            )
    return "\n".join(lines)


def _timestamp(ms: int) -> str:
    minutes, seconds = divmod(ms // 1000, 60)
    return f"{minutes:02d}:{seconds:02d}"
