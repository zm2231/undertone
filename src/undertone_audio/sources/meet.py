from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from undertone_audio.audio import AudioPreprocessor
from undertone_audio.engines.base import RawTranscript
from undertone_audio.schema import Segment, Speaker, Word

MEET_BASE = "https://meet.googleapis.com/v2"
DRIVE_DOWNLOAD_BASE = "https://www.googleapis.com/drive/v3/files"
MEET_ADC_COMMAND = (
    "gcloud auth application-default login "
    "--scopes=\"https://www.googleapis.com/auth/meetings.space.readonly,"
    "https://www.googleapis.com/auth/drive.meet.readonly\""
)


@dataclass(frozen=True)
class MeetAudioSelection:
    audio_path: Path | None
    source_kind: str
    source_metadata: dict[str, Any]
    raw_fallback: RawTranscript | None = None


class MeetSource:
    """Google Meet source adapter with local-audio-first precedence."""

    def __init__(
        self,
        download_dir: Path | None = None,
        preprocessor: AudioPreprocessor | None = None,
        google_account: str | None = None,
        adc_file: Path | None = None,
    ):
        self.download_dir = Path(download_dir or Path.home() / ".cache" / "undertone" / "meet")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.google_account = google_account
        self.adc_file = _resolve_adc_file(google_account, adc_file)

    def select(
        self,
        conference_record: str,
        *,
        local_audio: Path | None = None,
        allow_text_fallback: bool = True,
    ) -> MeetAudioSelection:
        if local_audio is not None:
            normalized = self.preprocessor.normalize(local_audio)
            return MeetAudioSelection(
                audio_path=normalized,
                source_kind="local-audio",
                source_metadata={
                    "source": "google-meet",
                    "conference_record": conference_record,
                    "audio_source": str(local_audio),
                    "audio_priority": "explicit-local",
                },
            )

        recording = self.best_recording(conference_record)
        if recording is not None:
            drive_file_id = recording.get("driveDestination", {}).get("file")
            if drive_file_id:
                mp4_path = self.download_dir / f"{drive_file_id}.mp4"
                if not mp4_path.exists() or mp4_path.stat().st_size == 0:
                    download_recording_mp4(
                        drive_file_id,
                        mp4_path,
                        google_account=self.google_account,
                        adc_file=self.adc_file,
                    )
                wav_path = self.preprocessor.normalize(mp4_path)
                return MeetAudioSelection(
                    audio_path=wav_path,
                    source_kind="meet-recording",
                    source_metadata={
                        "source": "google-meet",
                        "conference_record": conference_record,
                        "recording_name": recording.get("name"),
                        "drive_file_id": drive_file_id,
                        "export_uri": recording.get("driveDestination", {}).get("exportUri"),
                        "audio_priority": "meet-recording",
                    },
                )

        if not allow_text_fallback:
            raise ValueError(f"No local or downloadable Meet recording for {conference_record}")
        raw, metadata = parse_meet_record(
            conference_record,
            google_account=self.google_account,
            adc_file=self.adc_file,
        )
        metadata["audio_priority"] = "meet-text-fallback"
        return MeetAudioSelection(
            audio_path=None,
            source_kind="meet-text-fallback",
            source_metadata=metadata,
            raw_fallback=raw,
        )

    def best_recording(self, conference_record: str) -> dict | None:
        ready = [
            recording
            for recording in list_recordings(
                conference_record,
                google_account=self.google_account,
                adc_file=self.adc_file,
            )
            if recording.get("state") in {"ENDED", "FILE_GENERATED"}
            and recording.get("driveDestination", {}).get("file")
        ]
        if not ready:
            return None
        return sorted(ready, key=lambda item: item.get("endTime") or item.get("startTime") or "")[-1]


def _get_token(
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> str:
    try:
        import google.auth
        import google.auth.exceptions
        import google.auth.transport.requests
    except ImportError as exc:
        raise RuntimeError(
            "Google Meet ingestion requires optional Google auth dependencies. "
            "Install Undertone with `pip install -e '.[meet]'`."
        ) from exc
    try:
        resolved_adc = _resolve_adc_file(google_account, adc_file)
        if resolved_adc:
            creds, _ = google.auth.load_credentials_from_file(str(resolved_adc))
        else:
            creds, _ = google.auth.default()
        if not creds.valid:
            creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except Exception as exc:
        raise RuntimeError(_meet_auth_error(str(exc))) from exc


def meet_auth_check(
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> dict:
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        return {
            "ok": False,
            "error": "Google auth dependencies are not installed.",
            "fix": "Install Undertone with `pip install -e '.[meet]'`.",
        }
    try:
        resolved_adc = _resolve_adc_file(google_account, adc_file)
        if resolved_adc:
            creds, project = google.auth.load_credentials_from_file(str(resolved_adc))
        else:
            creds, project = google.auth.default()
        if not creds.valid:
            creds.refresh(google.auth.transport.requests.Request())
        return {"ok": True, "project": project}
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "fix": _meet_auth_fix(str(exc)),
        }


def _meet_auth_error(detail: str) -> str:
    detail = detail.rstrip(".")
    return (
        f"Google Meet auth is not ready: {detail}. "
        f"{_meet_auth_fix(detail)} "
        "Check with `undertone doctor --check-meet`."
    )


def _meet_auth_fix(detail: str) -> str:
    if "Reauthentication is needed" in detail or "reauth" in detail.lower():
        return f"Run `{MEET_ADC_COMMAND}` to reauthenticate, or pass --adc-file."
    if "default credentials" in detail.lower() or "could not automatically determine" in detail.lower():
        return f"Run `{MEET_ADC_COMMAND}`, or pass --adc-file."
    return f"Install `.[meet]`, run `{MEET_ADC_COMMAND}`, or pass --adc-file."


def _headers(
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> dict[str, str]:
    return {"Authorization": f"Bearer {_get_token(google_account=google_account, adc_file=adc_file)}"}


def _resolve_adc_file(google_account: str | None, adc_file: Path | None) -> Path | None:
    if adc_file:
        return Path(adc_file).expanduser()
    if not google_account:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.@-]+", "_", google_account)
    candidates = [
        Path.home() / ".config" / "undertone" / "google" / f"{safe}.json",
        Path.home() / ".config" / "gcloud" / f"application_default_credentials.{safe}.json",
        Path.home() / ".config" / "gcloud" / safe / "application_default_credentials.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def _ms(ts: str | None) -> int:
    if not ts:
        return 0
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)


def get_conference_record(
    conference_record: str,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> dict:
    response = requests.get(
        f"{MEET_BASE}/{conference_record}",
        headers=_headers(google_account=google_account, adc_file=adc_file),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def list_recordings(
    conference_record: str,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> list[dict]:
    response = requests.get(
        f"{MEET_BASE}/{conference_record}/recordings",
        headers=_headers(google_account=google_account, adc_file=adc_file),
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("recordings", [])


def list_recent_conferences(
    limit: int = 25,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> list[dict]:
    out: list[dict] = []
    url = f"{MEET_BASE}/conferenceRecords?pageSize=50"
    while url and len(out) < limit:
        response = requests.get(
            url,
            headers=_headers(google_account=google_account, adc_file=adc_file),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        out.extend(data.get("conferenceRecords", []))
        token = data.get("nextPageToken")
        url = f"{MEET_BASE}/conferenceRecords?pageSize=50&pageToken={token}" if token else None
    return out[:limit]


def download_recording_mp4(
    drive_file_id: str,
    dest: Path,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> Path:
    response = requests.get(
        f"{DRIVE_DOWNLOAD_BASE}/{drive_file_id}?alt=media",
        headers=_headers(google_account=google_account, adc_file=adc_file),
        timeout=600,
        stream=True,
    )
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                fp.write(chunk)
    return dest


def extract_audio_from_mp4(mp4_path: Path, out_path: Path | None = None) -> Path:
    out_path = out_path or mp4_path.with_suffix(".wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def list_transcripts(
    conference_record: str,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> list[dict]:
    response = requests.get(
        f"{MEET_BASE}/{conference_record}/transcripts",
        headers=_headers(google_account=google_account, adc_file=adc_file),
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("transcripts", [])


def list_entries(
    transcript_name: str,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> list[dict]:
    entries = []
    url = f"{MEET_BASE}/{transcript_name}/entries?pageSize=100"
    while url:
        response = requests.get(
            url,
            headers=_headers(google_account=google_account, adc_file=adc_file),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        entries.extend(data.get("transcriptEntries", []))
        token = data.get("nextPageToken")
        url = f"{MEET_BASE}/{transcript_name}/entries?pageSize=100&pageToken={token}" if token else None
    return entries


def resolve_participants(
    conference_record: str,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> dict[str, dict]:
    response = requests.get(
        f"{MEET_BASE}/{conference_record}/participants?pageSize=100",
        headers=_headers(google_account=google_account, adc_file=adc_file),
        timeout=30,
    )
    response.raise_for_status()
    result: dict[str, dict] = {}
    for participant in response.json().get("participants", []):
        name = participant["name"]
        signed = participant.get("signedinUser")
        anon = participant.get("anonymousUser")
        phone = participant.get("phoneUser")
        if signed:
            result[name] = {"display_name": signed.get("displayName")}
        elif anon:
            result[name] = {"display_name": anon.get("displayName")}
        elif phone:
            result[name] = {"display_name": phone.get("displayName")}
    return result


def parse_meet_record(
    conference_record: str,
    *,
    google_account: str | None = None,
    adc_file: Path | None = None,
) -> tuple[RawTranscript, dict[str, Any]]:
    record = get_conference_record(conference_record, google_account=google_account, adc_file=adc_file)
    transcripts = list_transcripts(conference_record, google_account=google_account, adc_file=adc_file)
    if not transcripts:
        raise ValueError(f"No transcripts available for {conference_record}")
    transcript = next((item for item in transcripts if item.get("state") == "FILE_GENERATED"), transcripts[0])
    participants = resolve_participants(conference_record, google_account=google_account, adc_file=adc_file)
    speaker_map = {
        participant_name: Speaker(
            speaker_id=f"S{index + 1}",
            display_name=data.get("display_name"),
        )
        for index, (participant_name, data) in enumerate(participants.items())
    }

    entries = sorted(
        list_entries(transcript["name"], google_account=google_account, adc_file=adc_file),
        key=lambda item: item.get("startTime", ""),
    )
    base_epoch_ms = _ms(record.get("startTime")) or (_ms(entries[0].get("startTime")) if entries else 0)
    unknown_speakers: dict[str, Speaker] = {}
    segments: list[Segment] = []
    for index, entry in enumerate(entries):
        text = entry.get("text", "").strip()
        if not text:
            continue
        participant_ref = entry.get("participant", "")
        speaker = speaker_map.get(participant_ref)
        if speaker is None:
            speaker = unknown_speakers.setdefault(
                participant_ref or f"unknown-{index}",
                Speaker(speaker_id=f"S_unk_{len(unknown_speakers)}"),
            )
        start_ms = max(_ms(entry.get("startTime")) - base_epoch_ms, 0)
        end_ms = max(_ms(entry.get("endTime")) - base_epoch_ms, start_ms)
        words = _even_words(text, start_ms, end_ms)
        segments.append(
            Segment(
                segment_id=f"meet-{index + 1}",
                speaker_id=speaker.speaker_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                words=words,
            )
        )

    raw = RawTranscript(
        duration_ms=max((segment.end_ms for segment in segments), default=0),
        language="en",
        speakers=[*speaker_map.values(), *unknown_speakers.values()],
        segments=segments,
        engine="meet-text-fallback",
    )
    metadata = {
        "source": "google-meet",
        "conference_record": conference_record,
        "transcript_name": transcript["name"],
        "space": record.get("space"),
        "start_time": record.get("startTime"),
        "end_time": record.get("endTime"),
        "docs_url": transcript.get("docsDestination", {}).get("exportUri"),
        "google_account": google_account,
    }
    return raw, {key: value for key, value in metadata.items() if value}


def _even_words(text: str, start_ms: int, end_ms: int) -> list[Word]:
    tokens = text.split()
    if not tokens or end_ms <= start_ms:
        return []
    duration = end_ms - start_ms
    per_word = duration / len(tokens)
    return [
        Word(
            text=token,
            start_ms=start_ms + int(index * per_word),
            end_ms=start_ms + int((index + 1) * per_word),
        )
        for index, token in enumerate(tokens)
    ]
