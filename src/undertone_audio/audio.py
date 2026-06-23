from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
DEFAULT_CACHE = Path.home() / ".cache" / "undertone" / "audio"

COMMON_AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".opus",
    ".ogg",
    ".oga",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
    ".aiff",
    ".aif",
    ".wma",
    ".amr",
}


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    codec: str
    bit_rate: int | None = None


class AudioPreprocessor:
    """Normalize audio/video containers to canonical 16 kHz mono WAV."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE,
        target_sr: int = TARGET_SAMPLE_RATE,
        target_channels: int = TARGET_CHANNELS,
    ):
        self.cache_dir = Path(cache_dir)
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._verify_ffmpeg()

    @staticmethod
    def _verify_ffmpeg() -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found on PATH. Install with `brew install ffmpeg`.")
        if not shutil.which("ffprobe"):
            raise RuntimeError("ffprobe not found on PATH. It ships with ffmpeg.")

    def probe(self, audio_path: Path) -> AudioInfo:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bit_rate:format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=0",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ValueError(f"ffprobe failed for {audio_path}: {result.stderr.strip()[:200]}")

        info: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key.strip()] = value.strip()

        return AudioInfo(
            path=audio_path,
            duration_s=float(info.get("duration", 0) or 0),
            sample_rate=int(info.get("sample_rate", 0) or 0),
            channels=int(info.get("channels", 0) or 0),
            codec=info.get("codec_name", "unknown"),
            bit_rate=int(info["bit_rate"]) if info.get("bit_rate", "").isdigit() else None,
        )

    def normalize(self, audio_path: Path) -> Path:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ValueError(f"audio file not found: {audio_path}")
        info = self.probe(audio_path)
        if (
            audio_path.suffix.lower() == ".wav"
            and info.sample_rate == self.target_sr
            and info.channels == self.target_channels
        ):
            return audio_path

        out_path = self.cache_dir / f"{self._cache_key(audio_path, info)}.wav"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                str(self.target_channels),
                "-ar",
                str(self.target_sr),
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        return out_path

    def mix_to_wav(self, audio_paths: list[Path], label: str) -> Path:
        if not audio_paths:
            raise ValueError("audio_paths must not be empty")
        normalized = [self.normalize(path) for path in audio_paths]
        if len(normalized) == 1:
            return normalized[0]
        out_path = self.cache_dir / f"{self._mix_key(normalized, label)}.wav"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        cmd = ["ffmpeg", "-y", "-v", "error"]
        for path in normalized:
            cmd.extend(["-i", str(path)])
        inputs = "".join(f"[{index}:a]" for index in range(len(normalized)))
        cmd.extend(
            [
                "-filter_complex",
                f"{inputs}amix=inputs={len(normalized)}:duration=longest:normalize=0[aout]",
                "-map",
                "[aout]",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ]
        )
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path

    def _cache_key(self, audio_path: Path, info: AudioInfo) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(f"{self.target_sr}|{self.target_channels}|".encode())
        size = audio_path.stat().st_size
        if size <= 256 * 1024 * 1024:
            with audio_path.open("rb") as fp:
                for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            stat = audio_path.stat()
            digest.update(str(audio_path.resolve()).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
        return digest.hexdigest()

    def _mix_key(self, audio_paths: list[Path], label: str) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(label.encode())
        for path in audio_paths:
            stat = path.stat()
            digest.update(str(path.resolve()).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
        return digest.hexdigest()
