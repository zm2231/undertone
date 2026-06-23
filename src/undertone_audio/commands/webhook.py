from __future__ import annotations

import argparse
import json
import sys

from undertone_audio.commands.common import config_for_args, db_path
from undertone_audio.storage import TranscriptStore
from undertone_audio.webhooks import (
    encode_payload,
    emit_transcript_ready,
    readiness_payload,
    should_emit_reason,
    signature_header,
)


def register(subcommands: argparse._SubParsersAction) -> None:
    emit = subcommands.add_parser("emit-ready", help="Emit readiness for a saved transcript.")
    emit.add_argument("transcript_id")
    emit.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    emit.set_defaults(func=emit_ready_cmd)

    preview = subcommands.add_parser("webhook-preview", help="Preview readiness webhook payload/signature.")
    preview.add_argument("transcript_id")
    preview.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    preview.set_defaults(func=webhook_preview_cmd)


def emit_ready_cmd(args: argparse.Namespace) -> int:
    store = TranscriptStore(db_path(args))
    try:
        transcript = store.load(args.transcript_id)
        if transcript is None:
            print(f"undertone: transcript not found: {args.transcript_id}", file=sys.stderr)
            return 1
        config = config_for_args(args)
        emitted = emit_transcript_ready(transcript, store.db_path, config)
        _should_emit, reason = should_emit_reason(transcript, config)
        payload = {"transcript_id": args.transcript_id, "emitted": emitted, "reason": reason}
        print(json.dumps(payload, separators=(",", ":")) if args.json else _render_emit(payload))
        return 0
    finally:
        store.close()


def webhook_preview_cmd(args: argparse.Namespace) -> int:
    config = config_for_args(args)
    store = TranscriptStore(db_path(args))
    try:
        transcript = store.load(args.transcript_id)
        if transcript is None:
            print(f"undertone: transcript not found: {args.transcript_id}", file=sys.stderr)
            return 1
        payload = readiness_payload(transcript, store.db_path)
        body = encode_payload(payload)
        should_emit, reason = should_emit_reason(transcript, config)
        preview = {
            "transcript_id": args.transcript_id,
            "should_emit": should_emit,
            "reason": reason,
            "payload": payload,
            "signature_header": signature_header(body, config.webhook_secret)
            if config.webhook_secret
            else None,
            "webhook_url": config.webhook_url,
        }
        print(json.dumps(preview, separators=(",", ":")) if args.json else _render_preview(preview))
        return 0
    finally:
        store.close()


def _render_emit(payload: dict) -> str:
    status = "emitted" if payload["emitted"] else "not emitted"
    return f"{payload['transcript_id']}: {status} ({payload['reason']})"


def _render_preview(preview: dict) -> str:
    payload = preview["payload"]
    lines = [
        "Webhook preview",
        f"  transcript: {preview['transcript_id']}",
        f"  should emit: {'yes' if preview['should_emit'] else 'no'} ({preview['reason']})",
        f"  url: {preview['webhook_url'] or '-'}",
        f"  signature: {preview['signature_header'] or '-'}",
        "  payload:",
        f"    event: {payload['event']}",
        f"    transcript_id: {payload['transcript_id']}",
        f"    source: {payload['source']}",
        f"    recorded_at: {payload['recorded_at']}",
        f"    store_ref: {payload['store_ref']}",
    ]
    return "\n".join(lines)
