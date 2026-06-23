from __future__ import annotations

import re
from typing import Any

PRIVATE_METADATA_TERMS = {
    "attendee",
    "attendees",
    "client",
    "clients",
    "contact",
    "contacts",
    "crm",
    "owner",
    "owners",
    "person",
    "people",
    "project",
    "projects",
    "scope",
    "workspace",
}


def sanitize_source_metadata(source_metadata: dict | None) -> dict:
    if not source_metadata:
        return {}
    return {
        key: _sanitize_metadata_value(value)
        for key, value in source_metadata.items()
        if not _is_private_metadata_key(key)
    }


def _sanitize_metadata_value(value: Any):
    if isinstance(value, dict):
        return sanitize_source_metadata(value)
    if isinstance(value, list):
        return [_sanitize_metadata_value(item) for item in value]
    return value


def _is_private_metadata_key(key: str) -> bool:
    parts = _metadata_key_parts(key)
    return any(part in PRIVATE_METADATA_TERMS for part in parts)


def _metadata_key_parts(key: str) -> set[str]:
    split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", split_camel).lower()
    return {part for part in normalized.split("_") if part}
