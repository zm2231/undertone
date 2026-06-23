from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from pathlib import Path

from undertone_audio.schema import Segment

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 600.0
SILENCE_THRESHOLD_DB = -25.0
MIN_DIP_DB = 2.0
MIN_PEAK_DISTANCE_S = 0.08
INTENSITY_FRAME_S = 0.01
ARTICULATION_RATE_CEILING = 9.0


def compute_speaker_voice_metrics(
    audio_path: Path | None,
    segments: list[Segment],
    *,
    required: bool = True,
) -> dict[str, dict[str, float]]:
    if audio_path is None:
        if required:
            raise RuntimeError("voice metrics required but no audio_path was provided")
        return {}

    try:
        import numpy as np
        import parselmouth
    except ImportError as exc:
        if required:
            raise RuntimeError(
                f"voice metrics required (audio_path={audio_path}) but a dependency is "
                f"missing: {exc}. Install Undertone[voice] or disable voice metrics."
            ) from exc
        log.warning("voice metrics skipped; dependency missing: %s", exc)
        return {}

    samples, sample_rate = _load_audio(audio_path, np)
    if samples is None:
        if required:
            raise RuntimeError(f"voice metrics required but audio could not be loaded: {audio_path}")
        return {}

    spans_by_speaker: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for segment in segments:
        spans_by_speaker[segment.speaker_id].append((segment.start_ms, segment.end_ms))

    results: dict[str, dict[str, float]] = {}
    eligible = 0
    failures = []
    for speaker, spans in spans_by_speaker.items():
        speaker_audio = _concatenate_spans(samples, sample_rate, spans)
        if len(speaker_audio) < sample_rate * 0.5:
            continue
        eligible += 1
        try:
            results[speaker] = _analyze_speaker(speaker_audio, sample_rate, np, parselmouth)
        except Exception as exc:
            failures.append((speaker, exc))
            log.warning("voice metrics failed for speaker %s: %s", speaker, exc)

    if required and not eligible:
        raise RuntimeError(f"voice metrics required but no eligible speaker audio found in {audio_path}")
    if failures and required:
        failed_speakers = ", ".join(speaker for speaker, _exc in failures)
        raise RuntimeError(f"voice metrics failed for required speakers: {failed_speakers}")
    if eligible and not results and required:
        raise RuntimeError(
            f"voice metrics failed for all {eligible} eligible speakers in {audio_path}"
        )
    return results


def _load_audio(audio_path: Path, np) -> tuple[object | None, int]:
    try:
        import soundfile as sf

        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sample_rate != TARGET_SAMPLE_RATE:
            samples = _resample(samples, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE
        return samples, sample_rate
    except Exception:
        pass

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-f",
                "f32le",
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        samples = np.frombuffer(proc.stdout, dtype="float32")
        return samples, TARGET_SAMPLE_RATE
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.warning("voice metrics audio load failed: %s", exc)
        return None, 0


def _concatenate_spans(samples, sample_rate: int, spans: list[tuple[int, int]]):
    import numpy as np

    chunks = []
    for start_ms, end_ms in spans:
        start = int(start_ms / 1000 * sample_rate)
        end = int(end_ms / 1000 * sample_rate)
        if 0 <= start < end <= len(samples):
            chunks.append(samples[start:end])
    if not chunks:
        return np.array([], dtype="float32")
    return np.concatenate(chunks)


def _resample(samples, source_rate: int, target_rate: int):
    import numpy as np

    if source_rate == target_rate:
        return samples
    ratio = target_rate / source_rate
    target_count = int(len(samples) * ratio)
    source_idx = np.linspace(0, len(samples) - 1, target_count)
    return np.interp(source_idx, np.arange(len(samples)), samples).astype("float32")


def _analyze_speaker(samples, sample_rate: int, np, parselmouth) -> dict[str, float]:
    sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(
        time_step=0.01,
        pitch_floor=PITCH_FLOOR_HZ,
        pitch_ceiling=PITCH_CEILING_HZ,
    )
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0]

    f0_mean = float(np.mean(voiced)) if len(voiced) else 0.0
    f0_stdev = float(np.std(voiced)) if len(voiced) else 0.0
    voiced_duration = float(len(voiced)) * 0.01
    syllable_count = _count_syllables(sound, np)
    articulation_rate = syllable_count / voiced_duration if voiced_duration > 0 else 0.0
    if articulation_rate > ARTICULATION_RATE_CEILING:
        articulation_rate = ARTICULATION_RATE_CEILING

    jitter = _safe_query(
        lambda: parselmouth.praat.call(
            sound.to_point_process_cc(
                pitch_floor=PITCH_FLOOR_HZ,
                pitch_ceiling=PITCH_CEILING_HZ,
            ),
            "Get jitter (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
        )
    )
    shimmer = _safe_query(
        lambda: parselmouth.praat.call(
            [
                sound,
                sound.to_point_process_cc(
                    pitch_floor=PITCH_FLOOR_HZ,
                    pitch_ceiling=PITCH_CEILING_HZ,
                ),
            ],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
    )

    return {
        "f0_mean_hz": f0_mean,
        "f0_stdev_hz": f0_stdev,
        "articulation_rate": articulation_rate,
        "jitter_local": jitter,
        "shimmer_local": shimmer,
        "voiced_duration_s": voiced_duration,
    }


def _count_syllables(sound, np) -> int:
    intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR_HZ, time_step=INTENSITY_FRAME_S)
    values = intensity.values.T.flatten()
    if len(values) == 0:
        return 0

    voiced = values[values > 0]
    if voiced.size == 0:
        return 0
    threshold = float(np.median(voiced)) + SILENCE_THRESHOLD_DB
    min_frames_between_peaks = max(1, int(MIN_PEAK_DISTANCE_S / INTENSITY_FRAME_S))

    peaks = 0
    last_peak_frame = -10**9
    last_peak_db = -1e9
    pending_peak_frame: int | None = None
    pending_peak_db = -1e9
    state = "below"

    for frame, db in enumerate(values):
        if db < threshold:
            if pending_peak_frame is not None and (
                pending_peak_db - db >= MIN_DIP_DB
                and pending_peak_frame - last_peak_frame >= min_frames_between_peaks
            ):
                peaks += 1
                last_peak_frame = pending_peak_frame
                last_peak_db = pending_peak_db
            pending_peak_frame = None
            pending_peak_db = -1e9
            state = "below"
            continue

        if state == "below":
            state = "climbing"
            pending_peak_frame = frame
            pending_peak_db = db
            continue

        if db > pending_peak_db:
            pending_peak_frame = frame
            pending_peak_db = db
            state = "climbing"
        elif pending_peak_db - db >= MIN_DIP_DB and pending_peak_frame is not None:
            if (
                pending_peak_frame - last_peak_frame >= min_frames_between_peaks
                and pending_peak_db > last_peak_db - MIN_DIP_DB
            ):
                peaks += 1
                last_peak_frame = pending_peak_frame
                last_peak_db = pending_peak_db
            pending_peak_frame = frame
            pending_peak_db = db
            state = "climbing"
        else:
            state = "descending"

    if pending_peak_frame is not None and (
        pending_peak_db - last_peak_db > -MIN_DIP_DB
        and pending_peak_frame - last_peak_frame >= min_frames_between_peaks
    ):
        peaks += 1

    return peaks


def _safe_query(fn) -> float:
    try:
        return float(fn())
    except Exception:
        return 0.0
