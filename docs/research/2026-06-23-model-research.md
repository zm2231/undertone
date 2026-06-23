# Local Mac Meeting-Audio Model Research (mid-2026)

Status: started 2026-06-23. This document is being appended incrementally as each pipeline stage is researched.

Scope: local-only, Mac-native or Mac-runnable meeting-audio pipeline for Apple Silicon Mac mini. No cloud inference.

## 1. ASR / transcription

### Recommendation

Recommended local-Mac ASR for mid-2026: **NVIDIA Parakeet TDT 0.6B v3 via FluidAudio Core ML**, with **WhisperKit / Whisper large-v3-turbo Core ML** as the best Whisper-family fallback.

This is a change from the older "Whisper or Quill plus FluidAudio" framing. Current FluidAudio itself has moved toward a Parakeet-based ASR stack on Apple devices: its public README describes local Apple-device ASR using **Parakeet TDT v3 0.6B** and other TDT/CTC models, with inference offloaded to the Apple Neural Engine. Its Hugging Face Core ML model card for `FluidInference/parakeet-tdt-0.6b-v3-coreml` says it is on-device, supports 25 European languages, and reports about **110x real-time factor on M4 Pro** for batch ASR, roughly 1 minute of audio in 0.5 seconds.

### Why this is best for meeting audio on an M-series Mac mini

- **Accuracy and timestamping:** NVIDIA's `parakeet-tdt-0.6b-v3` model card describes a 600M-parameter FastConformer-TDT ASR model with punctuation, capitalization, word-level and segment-level timestamps, and long-audio support. Word timestamps matter for aligning ASR with diarization and prosody windows.
- **Mac-native execution:** The FluidInference Core ML conversion is the most Mac-native path found: Core ML, Apple Silicon, ANE/CPU execution, no cloud after model download. `parakeet-mlx` also exists and is useful for Python/MLX integration, but the Core ML/FluidAudio path is more directly Apple-platform-native.
- **Meeting suitability:** For English and supported European languages, Parakeet v3 is now the local throughput/quality leader. It is not itself a diarizer, so meeting quality still depends on a separate diarization pass. For heavy accents and multilingual meetings outside the supported language set, Whisper large-v3-turbo remains the safer fallback because Whisper-family models remain broader and battle-tested.
- **Apple-native fallback:** WhisperKit is still highly credible. The July 2025 WhisperKit paper claims an optimized on-device real-time ASR system for Apple devices, benchmarking against cloud and open-source systems and reporting 2.2% WER and 0.46s latency in its setup. WhisperKit has a Homebrew CLI and Swift package path, so it remains the pragmatic fallback when Parakeet language coverage, behavior, or integration is not enough.

### Comparison to alternatives

- **FluidAudio current:** If the current project is using an older FluidAudio CLI with pyannote-derived diarization plus Whisper/Quill assumptions, the ASR side should be checked. Public FluidAudio in mid-2026 already advertises Parakeet TDT v3 Core ML ASR. If the installed CLI is older and still using a Whisper/Quill path, upgrading FluidAudio or calling the Parakeet Core ML model is a clear upgrade.
- **whisper.cpp:** Excellent C/C++ portability and simple local deployment; Metal acceleration exists in the ecosystem. For Apple-specific best quality/speed in 2026, WhisperKit/Core ML or MLX Whisper is usually a better Apple-native route.
- **faster-whisper:** Strong on CUDA and CPU, good Python ergonomics; less compelling on Apple Silicon than Core ML/MLX-native paths.
- **WhisperKit:** Best Apple-native Whisper path. Use when multilingual robustness, known Whisper behavior, or Swift integration matters more than Parakeet throughput.
- **mlx-whisper:** Good Python-native Apple Silicon route. Useful if the pipeline is Python-first and avoiding Swift/Core ML wrappers matters.
- **Apple Speech / SpeechAnalyzer:** Apple introduced SpeechAnalyzer at WWDC 2025 for on-device long-form transcription across Apple platforms. It is compelling for app UX and OS-integrated transcription, but it is proprietary/opaque, less benchmark-transparent than open models, and not a full meeting pipeline. It may be useful as a secondary recognizer or UX mode, not the most controllable research stack.
- **Distil-Whisper / Moonshine:** Useful smaller/streaming models, but not the best accuracy target for multi-speaker meetings on a Mac mini in 2026.
- **Parakeet MLX / parakeet.cpp:** Promising for Python/C++ local runners. For a Mac-native app or CLI, FluidAudio's Core ML conversion is the cleaner production target; MLX is attractive for experimentation.

### Integration path

- Swift/Core ML path: add FluidAudio as a Swift package or use its CLI/library if already present; select the Parakeet TDT v3 Core ML ASR model; normalize audio to 16 kHz mono; keep word/segment timestamps.
- Python path: use `parakeet-mlx` for Apple Silicon MLX if the pipeline is Python-first, or shell out to a FluidAudio CLI if the existing system already wraps it.
- Fallback path: use `whisperkit-cli` / WhisperKit Core ML large-v3-turbo for languages/edge cases where Parakeet is not reliable enough.

### Upgrade verdict

**Clear upgrade if current ASR is older Whisper/Quill or non-Core-ML Whisper. Marginal/no upgrade if the installed FluidAudio CLI is already using Parakeet TDT v3 Core ML.** The highest-value action is to verify the installed `fluidaudio` CLI model/version and ensure it emits word-level timestamps in the current pipeline.

### ASR sources checked

- NVIDIA `parakeet-tdt-0.6b-v3` model card, published around 2025-08, accessed 2026-06-23: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- FluidInference `parakeet-tdt-0.6b-v3-coreml` model card, published around 2026-03/2026-05, accessed 2026-06-23: https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml
- FluidAudio GitHub README, accessed 2026-06-23: https://github.com/FluidInference/FluidAudio
- WhisperKit paper, arXiv 2025-07-14, accessed 2026-06-23: https://arxiv.org/abs/2507.10860
- WhisperKit / Argmax OSS Swift repo, accessed 2026-06-23: https://github.com/argmaxinc/argmax-oss-swift
- Apple WWDC 2025 SpeechAnalyzer session, accessed 2026-06-23: https://developer.apple.com/videos/play/wwdc2025/277/
- `parakeet-mlx` GitHub repo, accessed 2026-06-23: https://github.com/senstella/parakeet-mlx

## 2. Diarization

### Recommendation

Recommended local-Mac diarization for mid-2026: **pyannote `speaker-diarization-community-1` for accuracy when Python/PyTorch is acceptable; FluidAudio's Core ML pyannote-derived diarization path for a Mac-native Swift/ANE pipeline.**

For a live/streaming product, keep **NVIDIA Streaming Sortformer v2.1** on the watchlist, but do not make it the default for a Mac mini meeting pipeline unless the meetings are known to fit its constraints and an ONNX/CPU/Core ML path has been validated locally.

### Why

- **Best open offline baseline:** pyannote Community-1 is the strongest open-source pyannote release found in current public evidence. Its Hugging Face model card says it is "much better" than legacy `speaker-diarization-3.1` and reports a benchmark table last updated 2025-09 with lower DER than 3.1 across AISHELL-4, AliMeeting, AMI, AVA-AVD, CALLHOME, and other datasets, with automatic processing and no collar / no overlap skipping.
- **Meeting-friendly output:** Community-1 explicitly improves speaker assignment and counting and offers "exclusive speaker diarization," which simplifies timestamp reconciliation with ASR tokens. That matters more in a meeting transcript pipeline than a small segmentation-only gain.
- **Mac-native option:** FluidInference's `speaker-diarization-coreml` model card says it is based on pyannote models and optimized for the Apple Neural Engine, with 16 kHz mono input and speaker segments/timestamps output. That is the most Mac-native diarization path found.
- **Overlap:** Sortformer is the main open model family to watch for native overlap-aware end-to-end diarization. NVIDIA's 2025 Streaming Sortformer work claims state-of-the-art streaming performance on DIHARD III and CALLHOME up to four speakers, and the model card is current enough for mid-2026 consideration. The practical issue is deployment: NVIDIA's official path is NeMo/Riva/CUDA-centered, and the streaming model is constrained to up to four speakers. Meetings may exceed that, and Mac GPU/ANE support is not first-class.

### Comparison to current FluidAudio / pyannote-derived setup

- **If current FluidAudio uses pyannote-derived Core ML diarization:** It is still a reasonable Mac-native choice. The likely upgrade is not "replace FluidAudio" but verify the exact FluidAudio model version, ensure it has the latest Community-1-like speaker assignment/counting improvements if available, and tune thresholds on meeting data.
- **If current setup is pyannote 3.1 via Python:** Upgrade to `pyannote/speaker-diarization-community-1` for offline batch quality unless licensing/user-condition constraints block it.
- **If current setup uses clustering plus speaker-merge thresholds:** Keep this layer. Public 2026 evidence still shows diarization is unsolved and domain-sensitive. Threshold tuning, minimum cluster-size behavior, and post-hoc speaker merge/split policies are still important for meeting audio.

### Alternatives

- **pyannote 3.1:** Still widely used and robust, but superseded by Community-1 for open-source pyannote diarization quality.
- **NVIDIA NeMo MSDD / Sortformer:** Strong on NVIDIA GPUs. Sortformer is attractive for streaming and overlap, but Mac-native acceleration is not first-class; ONNX ports exist in the community, but they need local validation before becoming the production default.
- **Diart:** Useful for online diarization frameworks and low-latency experiments; usually built on pyannote segmentation/embedding rather than being the current highest-accuracy batch answer.
- **WhisperX:** Good integration wrapper for Whisper + alignment + pyannote diarization, but it is not a better diarization model by itself.
- **SpeakerKit / SDBench direction:** SDBench 2025 reports a SpeakerKit system built on Pyannote v3 that is 9.6x faster with comparable error rates. This is a useful design direction for Mac optimization, but not yet the obvious off-the-shelf default compared with Community-1 / FluidAudio Core ML.

### Mac runtime and rough perf

- **FluidAudio Core ML:** Best Apple-platform fit; ANE/CPU, Swift, likely real-time or faster depending on segmentation stride and audio length. Use this when the app is Swift/macOS-first.
- **pyannote Community-1 Python:** Runs locally via PyTorch. On Apple Silicon it can run CPU/MPS depending on model/operator support, but expect tuning and package friction. Best when accuracy and Python control matter more than fully native packaging.
- **Sortformer:** Officially CUDA/NeMo/Riva-centered. Mac execution requires ONNX/Python experimentation and should be treated as research until benchmarked on the target Mac mini.

### Upgrade verdict

**Clear upgrade from pyannote 3.1 to Community-1 for Python batch diarization. Marginal/conditional upgrade over current FluidAudio Core ML if FluidAudio has already incorporated recent pyannote-derived improvements.** For local Mac production, the recommendation is to keep FluidAudio for native execution and use Community-1 as the quality reference/regression benchmark.

### Diarization sources checked

- pyannote Community-1 model card, benchmark last updated 2025-09, accessed 2026-06-23: https://huggingface.co/pyannote/speaker-diarization-community-1
- pyannote Community-1 announcement/blog, accessed 2026-06-23: https://www.pyannote.ai/blog/community-1
- pyannote.audio GitHub README, accessed 2026-06-23: https://github.com/pyannote/pyannote-audio
- FluidInference `speaker-diarization-coreml` model card, published around 2026-06, accessed 2026-06-23: https://huggingface.co/FluidInference/speaker-diarization-coreml
- NVIDIA Streaming Sortformer v2.1 model card, published around 2025-07/2025-08, accessed 2026-06-23: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- NVIDIA Streaming Sortformer blog, published around 2025-08, accessed 2026-06-23: https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- Streaming Sortformer paper, arXiv 2025-07, accessed 2026-06-23: https://arxiv.org/html/2507.18446v1
- SDBench paper, arXiv 2025-07, accessed 2026-06-23: https://arxiv.org/abs/2507.16136

## 3. Speaker embeddings / voice fingerprint

### Recommendation

Recommended best discriminative speaker-identity model for mid-2026: **3D-Speaker ERes2Net-large or ERes2NetV2** for cross-recording identity matching, with **WeSpeaker ResNet34-LM / pyannote wrapper / FluidAudio embedding** as the best Mac-native production baseline.

This stage should be treated separately from diarization. Diarization answers "who spoke when in this recording"; the embedding/fingerprint stage answers "is this the same person across meetings?" For that second question, benchmarked speaker-verification EER matters more than diarization DER.

### Why

- **Best public EER among practical open models found:** 3D-Speaker's benchmark table reports VoxCeleb1-O EER of **0.52% for ERes2Net-large**, **0.61% for ERes2NetV2**, **0.65% for CAM++**, **0.84% for ERes2Net-base**, and **0.86% for ECAPA-TDNN**. It also reports harder CNCeleb and 3D-Speaker dataset results, where ERes2Net-large / ERes2NetV2 remain strong.
- **Cross-recording use case:** ERes2NetV2 is explicitly designed for robust short-duration speaker verification. That matters because diarization segments in meetings are often short, noisy, and clipped.
- **Mac practicality:** 3D-Speaker is PyTorch-first, not Apple-native. It should run on CPU/MPS with the usual PyTorch caveats, but the cleanest production Mac integration would require ONNX/Core ML conversion and a local calibration benchmark. For immediate Mac-native use, WeSpeaker/FluidAudio embeddings are more practical.
- **Current FluidAudio baseline:** FluidAudio's docs list a Pyannote CoreML diarization pipeline as segmentation + WeSpeaker embeddings. CocoaPods also advertises speaker embedding extraction for voice comparison/clustering. That makes the current WeSpeaker/pyannote-derived embedding stack reasonable, but not necessarily the most discriminative model available.

### Model comparison

- **3D-Speaker ERes2Net-large / ERes2NetV2:** Best open verification numbers found. Use for the highest-confidence cross-meeting dedup if you can tolerate PyTorch or can invest in conversion. Likely the best upgrade target.
- **3D-Speaker CAM++:** Strong and small; FluidInference also has a CAM++ Core ML conversion, but the surfaced Core ML model appears Mandarin/AISHELL-oriented, so it is not my default for English meeting identity unless locally benchmarked.
- **WeSpeaker ResNet34-LM:** Mature, production-oriented, and already used in pyannote/FluidAudio contexts. The MLX community model card describes a ResNet34 VoxCeleb model with **256-dimensional embeddings**, 6.6M parameters, and Apple Silicon MLX optimization. This is the most practical Apple-native embedding path found.
- **pyannote/wespeaker-voxceleb-resnet34-LM:** A pyannote.audio wrapper around WeSpeaker ResNet34-LM; suitable if staying in pyannote. Discussions/model cards indicate 256-d output for this model, while older pyannote embeddings may be 512-d.
- **NVIDIA TitaNet-large:** Good older baseline, 192-dimensional embeddings, reported around 0.66-0.68% EER on VoxCeleb1 cleaned/original depending source. It is still competitive and easy through NeMo, but not a clear 2026 upgrade over 3D-Speaker ERes2Net for pure verification.
- **ECAPA-TDNN / SpeechBrain:** Still widely used and easy, but older and not the top open EER result in the current comparison.
- **WavLM-based embeddings:** Strong research direction and useful in diarization systems, but not the most straightforward Mac production embedding extractor found for the specific "stable voiceprint vector + cosine similarity" requirement.

### Mac runtime and integration path

- **Highest quality path:** Add a separate speaker-fingerprint extractor using 3D-Speaker ERes2NetV2 or ERes2Net-large. Run it on clean single-speaker spans from diarization, average/L2-normalize embeddings per speaker per meeting, and store calibrated centroids. Use cosine scoring plus duration/quality gates. Benchmark thresholds on known same/different speaker pairs from the user's recordings.
- **Mac-native path:** Keep FluidAudio/WeSpeaker embeddings or use an MLX WeSpeaker ResNet34-LM model for Apple Silicon. This is the better first production path if Swift/ANE/MLX deployment matters more than squeezing out the last verification points.
- **Important implementation detail:** Do not embed overlapped speech, backchannels, or very short clips. Use VAD + diarization confidence + minimum voiced duration. Keep per-segment embeddings and an aggregate centroid; this allows later re-clustering when thresholds improve.

### Upgrade verdict

**Clear quality upgrade opportunity if the project uses generic pyannote/WeSpeaker embeddings for durable cross-recording identity and can afford a new extractor.** Use 3D-Speaker ERes2NetV2/large as the new benchmark candidate. **Marginal/pragmatic upgrade if Mac-native packaging is the top constraint;** in that case, keep WeSpeaker/FluidAudio and focus on calibration, segment filtering, and threshold evaluation.

### Speaker embedding sources checked

- 3D-Speaker GitHub benchmark table, accessed 2026-06-23: https://github.com/modelscope/3D-Speaker
- 3D-Speaker paper, ICASSP 2025 / arXiv 2024-03, accessed 2026-06-23: https://arxiv.org/html/2403.19971v3
- ERes2NetV2 paper, Interspeech 2024, accessed 2026-06-23: https://www.isca-archive.org/interspeech_2024/chen24l_interspeech.pdf
- WeSpeaker pretrained models docs, accessed 2026-06-23: https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
- WeSpeaker paper, arXiv 2022-10 / Speech Communication 2024, accessed 2026-06-23: https://arxiv.org/abs/2210.17016
- pyannote WeSpeaker wrapper model card, accessed 2026-06-23: https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
- MLX community WeSpeaker model card, accessed 2026-06-23: https://huggingface.co/mlx-community/wespeaker-voxceleb-resnet34-LM
- NVIDIA TitaNet-large model card, accessed 2026-06-23: https://huggingface.co/nvidia/speakerverification_en_titanet_large
- FluidAudio model docs, accessed 2026-06-23: https://docs.fluidinference.com/reference/models

## 4. Voice / prosody metrics

### Recommendation

Recommended mid-2026 local-Mac prosody stack: **keep Parselmouth/Praat for canonical F0, jitter, shimmer, HNR, and formant-style phonetic measures; add openSMILE Python with eGeMAPSv02 for standardized meeting analytics features.**

This is an additive upgrade, not a replacement. Parselmouth remains the right tool when the metric name itself is a Praat/phonetics metric and reproducibility against prior phonetic literature matters. openSMILE is the better default for broader, standardized prosody/voice-quality feature vectors used in affective computing and behavioral analytics.

### Why

- **Parselmouth/Praat remains valid:** Parselmouth is a Pythonic interface to Praat and is still the standard route for exact Praat-style F0, jitter, shimmer, HNR, intensity, formants, and point-process measures. If existing analytics depend on Praat definitions, replacing it risks silent metric drift.
- **openSMILE is more complete for analytics:** openSMILE is an open-source speech/music feature extraction toolkit widely applied in automatic expression recognition and affective computing. The Python wrapper exposes standard sets including GeMAPS and **eGeMAPSv02**, and its docs recommend the latest eGeMAPS variant unless backward compatibility is needed.
- **2025 evidence warns against mixing tools casually:** Interspeech 2025 work comparing OpenSMILE, Praat, and Librosa found toolkit-dependent variation. F0 percentiles correlated highly across tools, but other measures diverged. That supports a stable, versioned feature policy rather than swapping extractors blindly.
- **Meeting analytics need time aggregation:** For talk-time, speaking rate, interruption/crosstalk indicators, pitch/loudness dynamics, and sentiment-adjacent prosody, openSMILE LLDs + functionals over diarized speaker turns are more useful than isolated jitter/shimmer alone.

### Alternatives

- **librosa:** Good general audio/music DSP toolkit. Useful for spectral features, RMS, tempo-like analysis, and custom pipelines, but less speech-specific and less standardized than openSMILE/Praat for voice-quality analytics.
- **DisVoice:** Useful research toolkit for glottal, phonation, articulation, prosody, and pathological/paralinguistic features. Worth evaluating for clinical-style speech analytics, but not the cleanest production default for meeting analytics.
- **torchaudio:** Good lower-level PyTorch audio feature plumbing, not a replacement for canonical Praat/openSMILE feature definitions.
- **SpeechBrain voice-analysis examples:** Useful reference code, including comparison with OpenSMILE for jitter/shimmer-like analysis, but not the primary production feature standard.
- **Neural prosody/emotion models:** Useful for downstream classifiers, but not a replacement for interpretable, auditable meeting metrics. If used, they should sit downstream of transcript/diarization/acoustic features and be clearly labeled as model predictions, not raw prosody.

### Mac runtime and integration path

- **Parselmouth:** CPU-local Python package; no special Apple Silicon acceleration needed. Run per diarized speaker segment with careful duration and voicing thresholds.
- **openSMILE Python:** CPU-local Python package with precompiled binaries and standard feature sets. Extract eGeMAPSv02 at LowLevelDescriptors for time series and Functionals for per-turn/per-speaker summaries.
- **Pipeline detail:** Compute prosody only on clean single-speaker spans from diarization. Store extractor name, version, parameter settings, sample rate, voiced-frame thresholds, and aggregation windows. This is essential because acoustic feature values vary by toolkit and parameterization.

### Upgrade verdict

**Marginal upgrade if current needs are only F0/jitter/shimmer and the system already uses Parselmouth correctly. Clear upgrade if the product wants meeting analytics beyond phonetic voice-quality metrics.** Add openSMILE eGeMAPSv02 rather than replacing Parselmouth.

### Prosody sources checked

- Parselmouth PyPI / docs, accessed 2026-06-23: https://pypi.org/project/praat-parselmouth/ and https://parselmouth.readthedocs.io/en/latest/
- Parselmouth paper, Computer Speech & Language 2018, accessed 2026-06-23: https://www.mpi.nl/publications/item2627915/introducing-parselmouth-python-interface-praat
- openSMILE Python GitHub README, accessed 2026-06-23: https://github.com/audeering/opensmile-python
- openSMILE 3.0 overview, accessed 2026-06-23: https://www.audeering.com/research/opensmile/
- openSMILE Python changelog / v2.6.0 release, accessed 2026-06-23: https://audeering.github.io/opensmile-python/changelog.html and https://github.com/audeering/opensmile-python/releases
- Comparative evaluation of OpenSMILE, Praat, and Librosa, Interspeech 2025 / arXiv 2025-06, accessed 2026-06-23: https://arxiv.org/html/2506.01129v1
- DisVoice docs/GitHub, accessed 2026-06-23: https://disvoice.readthedocs.io/ and https://github.com/jcvasquezc/disvoice

## Final recommended local-Mac stack

### Best quality / practical Mac stack

1. **ASR:** FluidAudio / Core ML **Parakeet TDT 0.6B v3** for English and supported European-language meetings. Keep **WhisperKit large-v3-turbo** as fallback for unsupported languages, edge-case accents, or when Whisper behavior is preferred.
2. **Diarization:** FluidAudio Core ML pyannote-derived diarization for native Swift/ANE execution, benchmarked against **pyannote `speaker-diarization-community-1`** as the quality reference. If the pipeline is Python-first and batch/offline, use Community-1 directly.
3. **Speaker fingerprint:** For immediate production, keep **WeSpeaker ResNet34-LM / FluidAudio embeddings** with strict segment-quality filtering and calibrated thresholds. For best cross-meeting identity quality, evaluate **3D-Speaker ERes2NetV2 or ERes2Net-large** as a separate extractor and convert/optimize only after local threshold tests show a real gain.
4. **Prosody:** Keep **Parselmouth/Praat** for canonical F0/jitter/shimmer/HNR/formants; add **openSMILE eGeMAPSv02** for standardized per-turn and per-speaker meeting analytics.

### Upgrade ranking vs current FluidAudio / pyannote / Parselmouth context

1. **ASR model/version verification and Parakeet TDT v3 adoption:** highest impact if the current CLI is not already using Parakeet v3 Core ML. This directly improves transcript quality, speed, punctuation, and timestamp quality.
2. **Diarization move from pyannote 3.1 to Community-1 or latest FluidAudio Core ML diarization:** high impact if current diarization is older pyannote 3.1. Biggest likely gains are speaker counting and speaker assignment.
3. **Speaker fingerprint calibration and possible 3D-Speaker ERes2Net evaluation:** high impact for cross-recording identity, but only after a local benchmark. The extractor may improve EER, but bad segment selection/thresholds can erase the gain.
4. **Add openSMILE eGeMAPSv02 beside Parselmouth:** medium impact. It expands analytics coverage and standardization, but does not replace the core transcript/diarization quality path.
5. **Apple SpeechAnalyzer:** useful product/UX fallback, but not the main research stack because it is opaque and not a full diarized meeting pipeline.
6. **Sortformer:** watchlist for streaming/overlap. Do not default to it on Mac until the target Mac mini has a validated ONNX/Core ML/MLX route and the 4-speaker limitation is acceptable.

### Concrete next checks in this repo/pipeline

- Run `fluidaudio --version` or inspect the installed package to confirm which ASR model and diarization model it actually uses.
- Confirm whether `fluidaudio` emits word-level timestamps; if not, expose them from Parakeet/WhisperKit or add a forced-alignment step.
- Build a small local evaluation set: 5-10 meetings with known speakers, accents, overlap, and at least two repeat speakers across meetings.
- Score ASR WER-ish manually on representative spans, diarization DER/confusion manually on a few clips, and speaker fingerprint ROC/EER or at least same/different cosine distributions.
- Store model/version/threshold metadata with every transcript so future upgrades are auditable.

Status: complete as of 2026-06-23.
