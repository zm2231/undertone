from __future__ import annotations

import shutil
from pathlib import Path

from undertone_audio.connectors.base import default_download_dir
from undertone_audio.sources.meet import meet_auth_check
from undertone_audio.sources.quill import DEFAULT_MEETINGS_DIR, DEFAULT_QUILL_DB


def source_statuses(
    *,
    check_meet: bool = False,
    quill_db: Path = DEFAULT_QUILL_DB,
    quill_meetings_dir: Path = DEFAULT_MEETINGS_DIR,
) -> list[dict]:
    return [
        youtube_status(),
        podcast_status(),
        meet_status(check_auth=check_meet),
        quill_status(db_path=quill_db, meetings_dir=quill_meetings_dir),
    ]


def youtube_status() -> dict:
    path = shutil.which("yt-dlp")
    if path:
        return {
            "source": "youtube",
            "state": "ready",
            "detail": f"yt-dlp: {path}",
            "fix": None,
        }
    return {
        "source": "youtube",
        "state": "missing-dependency",
        "detail": "yt-dlp not found on PATH",
        "fix": "Install connectors with `pip install -e '.[connectors]'` or pass --yt-dlp-bin.",
    }


def podcast_status() -> dict:
    download_dir = default_download_dir() / "podcasts"
    return {
        "source": "podcast",
        "state": "ready",
        "detail": f"download cache: {download_dir}",
        "fix": None,
    }


def meet_status(*, check_auth: bool = False) -> dict:
    if not check_auth:
        return {
            "source": "meet",
            "state": "not-checked",
            "detail": "Google auth not checked",
            "fix": "Run `undertone sources --check-meet` or `undertone doctor --check-meet`.",
        }
    check = meet_auth_check()
    if check["ok"]:
        project = check.get("project")
        detail = f"ADC ok{f' project={project}' if project else ''}"
        return {"source": "meet", "state": "ready", "detail": detail, "fix": None}
    return {
        "source": "meet",
        "state": "needs-auth",
        "detail": check["error"],
        "fix": check["fix"],
    }


def quill_status(
    *,
    db_path: Path = DEFAULT_QUILL_DB,
    meetings_dir: Path = DEFAULT_MEETINGS_DIR,
) -> dict:
    db_exists = Path(db_path).expanduser().exists()
    meetings_exists = Path(meetings_dir).expanduser().exists()
    if db_exists and meetings_exists:
        return {
            "source": "quill",
            "state": "ready",
            "detail": f"db: {db_path}",
            "fix": None,
        }
    missing = []
    if not db_exists:
        missing.append(f"db not found: {db_path}")
    if not meetings_exists:
        missing.append(f"recordings dir not found: {meetings_dir}")
    return {
        "source": "quill",
        "state": "not-found",
        "detail": "; ".join(missing),
        "fix": "Install/sign in to Quill, or pass --quill-db and --meetings-dir.",
    }
