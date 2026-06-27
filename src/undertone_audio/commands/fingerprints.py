from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from undertone_audio.commands.common import config_for_args
from undertone_audio.diarization.fingerprint import SpeakerFingerprintStore
from undertone_audio.pipeline import effective_fingerprint_embedding_model
from undertone_audio.storage import TranscriptStore
from undertone_audio.storage.sqlite import _normalize_fingerprint_rows


def register(subcommands: argparse._SubParsersAction) -> None:
    fingerprints = subcommands.add_parser("fingerprints", help="List voice fingerprints.")
    fingerprints.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    fingerprints.add_argument("--format", choices=["text", "json"], default="text")
    fingerprints.add_argument("--unnamed", action="store_true", help="Only show fingerprints without labels.")
    fingerprints.add_argument("--excerpts", action="store_true", help="Show sample transcript lines.")
    fingerprints.add_argument("--limit-excerpts", type=int, default=3)
    fingerprints.set_defaults(func=fingerprints_cmd)

    label = subcommands.add_parser("fingerprint-label", help="Set a display name for a voice fingerprint.")
    label.add_argument("fingerprint_id")
    label.add_argument("display_name")
    label.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    label.set_defaults(func=fingerprint_label_cmd)

    relabel = subcommands.add_parser(
        "relabel",
        aliases=["resolve-names"],
        help="Re-stamp saved speaker display names from the voice fingerprint DB.",
    )
    relabel.add_argument("transcript_id", nargs="?", help="Transcript to relabel. Omit with --all.")
    relabel.add_argument("--all", action="store_true", help="Relabel every saved transcript.")
    relabel.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    relabel.set_defaults(func=relabel_cmd)

    export = subcommands.add_parser("fingerprint-export", help="Export voice fingerprints as JSON.")
    export.add_argument("--output", type=Path)
    export.set_defaults(func=fingerprint_export_cmd)

    import_cmd = subcommands.add_parser("fingerprint-import", help="Import voice fingerprints from JSON.")
    import_cmd.add_argument("json_path", type=Path)
    import_cmd.add_argument("--replace", action="store_true", help="Replace existing fingerprint rows.")
    import_cmd.add_argument("--dry-run", action="store_true", help="Print the import plan without writing.")
    import_cmd.add_argument("--yes", action="store_true", help="Confirm writes.")
    import_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    import_cmd.set_defaults(func=fingerprint_import_cmd)

    merge = subcommands.add_parser("fingerprint-merge", help="Merge one voice fingerprint into another.")
    merge.add_argument("source_fingerprint_id")
    merge.add_argument("target_fingerprint_id")
    merge.add_argument("--dry-run", action="store_true", help="Print the merge plan without writing.")
    merge.add_argument("--yes", action="store_true", help="Confirm writes.")
    merge.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    merge.set_defaults(func=fingerprint_merge_cmd)

    adopt = subcommands.add_parser(
        "fingerprint-adopt-model",
        help="Assert an embedding model for legacy voice fingerprints.",
    )
    adopt.add_argument("--embedding-model", help="Model to assert. Defaults to the active model.")
    adopt.add_argument(
        "--all",
        action="store_true",
        help="Adopt every fingerprint row, not only legacy rows with no model.",
    )
    adopt.add_argument(
        "--allow-non-current",
        action="store_true",
        help="Allow asserting a model different from the active configuration.",
    )
    adopt.add_argument("--dry-run", action="store_true", help="Print the adoption plan without writing.")
    adopt.add_argument("--yes", action="store_true", help="Confirm writes.")
    adopt.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    adopt.set_defaults(func=fingerprint_adopt_model_cmd)


def fingerprints_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        rows = [
            {
                "fingerprint_id": fingerprint_id,
                "display_name": display_name,
                "sample_count": count,
                "embedding_model": embedding_model,
            }
            for fingerprint_id, display_name, count, embedding_model in SpeakerFingerprintStore(
                config.db_path
            ).list_all()
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
        if args.json or args.format == "json":
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


def relabel_cmd(args: argparse.Namespace) -> int:
    if args.all and args.transcript_id:
        raise ValueError("relabel accepts either a transcript id or --all, not both")
    if not args.all and not args.transcript_id:
        raise ValueError("relabel requires a transcript id or --all")
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        payload = store.relabel_speakers(None if args.all else args.transcript_id)
    finally:
        store.close()
    if args.json:
        print(json.dumps(payload, separators=(",", ":")))
    else:
        target = "all transcripts" if args.all else args.transcript_id
        print(f"Relabeled {payload['speakers_updated']} speaker rows in {target}")
    return 0


def fingerprint_export_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    store = TranscriptStore(config.db_path)
    try:
        payload = {
            "schema_version": "1",
            "exported_at": _now(),
            "fingerprints": store.export_fingerprints(),
        }
    finally:
        store.close()
    body = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n")
    else:
        print(body)
    return 0


def fingerprint_import_cmd(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.yes:
        raise ValueError("fingerprint-import requires --yes or --dry-run")
    payload = json.loads(args.json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("fingerprint-import JSON must be an object with a fingerprints array")
    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != "1":
        raise ValueError("fingerprint-import JSON schema_version must be 1")
    fingerprints = payload.get("fingerprints", [])
    if not isinstance(fingerprints, list):
        raise ValueError("fingerprint-import JSON field fingerprints must be an array")
    _normalize_fingerprint_rows(fingerprints)
    config = config_for_args(args)
    with _open_fingerprint_store(config.db_path, dry_run=args.dry_run) as store:
        if args.dry_run:
            plan = store.fingerprint_import_plan(fingerprints, replace=args.replace)
        else:
            plan = store.fingerprint_import_plan(fingerprints, replace=args.replace)
            if _fingerprint_import_will_write(plan):
                backup = _backup_db(config.db_path)
                plan = store.import_fingerprints(fingerprints, replace=args.replace)
                plan["backup_path"] = str(backup)
            else:
                plan["dry_run"] = False
    _print_plan(plan, args.json)
    return 0


def fingerprint_merge_cmd(args: argparse.Namespace) -> int:
    if args.source_fingerprint_id == args.target_fingerprint_id:
        raise ValueError("source and target fingerprint ids must differ")
    if not args.dry_run and not args.yes:
        raise ValueError("fingerprint-merge requires --yes or --dry-run")
    config = config_for_args(args)
    with _open_fingerprint_store(config.db_path, dry_run=args.dry_run) as store:
        if args.dry_run:
            plan = store.fingerprint_merge_plan(
                args.source_fingerprint_id,
                args.target_fingerprint_id,
            )
        else:
            store.fingerprint_merge_plan(
                args.source_fingerprint_id,
                args.target_fingerprint_id,
            )
            backup = _backup_db(config.db_path)
            plan = store.merge_fingerprints(
                args.source_fingerprint_id,
                args.target_fingerprint_id,
            )
            plan["backup_path"] = str(backup)
    _print_plan(plan, args.json)
    return 0


def fingerprint_adopt_model_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    active_model = effective_fingerprint_embedding_model(config)
    target_model = args.embedding_model or active_model
    if target_model != active_model and not args.allow_non_current:
        raise ValueError(
            "fingerprint-adopt-model is a provenance assertion, not a vector conversion; "
            f"target model {target_model!r} differs from active model {active_model!r}. "
            "Pass --allow-non-current only if the stored vectors were produced by that model."
        )
    if not args.dry_run and not args.yes:
        raise ValueError("fingerprint-adopt-model requires --yes or --dry-run")
    with _open_fingerprint_store(config.db_path, dry_run=args.dry_run) as store:
        if args.dry_run:
            plan = store.fingerprint_adopt_model_plan(target_model, only_legacy=not args.all)
        else:
            plan = store.fingerprint_adopt_model_plan(target_model, only_legacy=not args.all)
            if plan["fingerprints_to_update"]:
                backup = _backup_db(config.db_path)
                plan = store.adopt_fingerprint_model(target_model, only_legacy=not args.all)
                plan["backup_path"] = str(backup)
            else:
                plan["dry_run"] = False
    plan["active_embedding_model"] = active_model
    plan["warning"] = (
        "This asserts provenance for existing vectors; it does not convert embeddings "
        "between model spaces."
    )
    _print_plan(plan, args.json)
    return 0


@contextmanager
def _open_fingerprint_store(db_path: Path, *, dry_run: bool) -> Iterator[TranscriptStore]:
    if not dry_run:
        store = TranscriptStore(db_path)
        try:
            yield store
        finally:
            store.close()
        return

    temp_path = _migrated_dry_run_copy(db_path)
    store = TranscriptStore(temp_path, migrate=False, read_only=True)
    try:
        yield store
    finally:
        store.close()
        temp_path.unlink(missing_ok=True)


def _migrated_dry_run_copy(db_path: Path) -> Path:
    if not db_path.exists():
        raise ValueError(f"fingerprint dry-run requires an existing database: {db_path}")
    fd, temp_name = tempfile.mkstemp(prefix=f"{db_path.name}.dry-run.", suffix=".db")
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source:
            with sqlite3.connect(temp_path) as dest:
                source.backup(dest)
        TranscriptStore(temp_path).close()
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _render_fingerprints(rows: list[dict], excerpts: dict[str, list[dict]]) -> str:
    if not rows:
        return "No voice fingerprints found."
    lines = ["Voice fingerprints"]
    for row in rows:
        name = row["display_name"] or "(unnamed)"
        model = row["embedding_model"] or "(legacy model unknown)"
        lines.append(
            f"  {row['fingerprint_id']}  {name}  samples={row['sample_count']}  model={model}"
        )
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


def _backup_db(db_path: Path) -> Path:
    backup = _reserve_backup_path(db_path)
    try:
        with sqlite3.connect(db_path) as source, sqlite3.connect(backup) as dest:
            source.backup(dest)
    except Exception:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _reserve_backup_path(db_path: Path) -> Path:
    stem = f"{db_path.name}.{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}.bak"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f".{attempt}"
        backup = db_path.with_name(f"{stem}{suffix}")
        try:
            fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        else:
            os.close(fd)
            return backup
    raise RuntimeError(f"could not reserve a unique backup path for {db_path}")


def _fingerprint_import_will_write(plan: dict) -> bool:
    return bool(plan.get("to_insert") or plan.get("to_replace"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _print_plan(plan: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan, separators=(",", ":"), default=str))
        return
    lines = [plan["operation"]]
    for key, value in plan.items():
        if key in {"operation", "excerpts", "to_insert", "to_replace", "skipped_existing"}:
            continue
        lines.append(f"  {key}: {value}")
    if "to_insert" in plan:
        lines.append(f"  insert: {len(plan['to_insert'])}")
        lines.append(f"  replace: {len(plan['to_replace'])}")
        lines.append(f"  skip existing: {len(plan['skipped_existing'])}")
    if plan.get("excerpts"):
        lines.append("  excerpts:")
        for excerpt in plan["excerpts"]:
            lines.append(f"    - {excerpt['transcript_id']} {excerpt['speaker_id']}: {excerpt['text']}")
    print("\n".join(lines))
