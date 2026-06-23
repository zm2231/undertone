from undertone_audio.config import Config, load as load_config
from undertone_audio.engines.base import RawTranscript, TranscriptionEngine
from undertone_audio.engines.fluidaudio_cli import FluidAudioCLIEngine, FluidAudioModelSelection
from undertone_audio.engines.fluidaudio_hybrid import FluidAudioHybridEngine


def create_engine(name: str | None = None, config: Config | None = None) -> TranscriptionEngine:
    cfg = config or load_config()
    engine_name = name or cfg.default_engine
    model_selection = FluidAudioModelSelection(
        asr_model=cfg.asr_model,
        diarization_model=cfg.diarization_model,
        vad_model=cfg.vad_model,
        embedding_model=cfg.embedding_model,
    )
    if engine_name == "fluidaudio-cli":
        return FluidAudioCLIEngine(
            cli_path=cfg.fluidaudio_cli,
            clustering_threshold=cfg.clustering_threshold,
            model_selection=model_selection,
        )
    if engine_name == "fluidaudio-hybrid":
        return FluidAudioHybridEngine(
            cli_path=cfg.fluidaudio_cli,
            clustering_threshold=cfg.clustering_threshold,
            model_selection=model_selection,
        )
    raise ValueError(
        f"unknown Undertone engine {engine_name!r}; expected fluidaudio-hybrid or fluidaudio-cli"
    )


__all__ = [
    "FluidAudioCLIEngine",
    "FluidAudioHybridEngine",
    "RawTranscript",
    "TranscriptionEngine",
    "create_engine",
]
