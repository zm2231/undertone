# Changelog

## 0.2.1

### Upgrade notes

- **Match diagnostics are null on old transcripts until re-enriched.** Speakers now carry a nullable `match` object; transcripts ingested before this release report `match=null` until `reenrich` or a fresh ingest. A re-enrich matches against today's centroids, not the state at original ingest.
- **Content dedupe needs `fpcalc` to skip anything.** Audio-fingerprint duplicate skipping is the only silent gate, and it is off unless Chromaprint `fpcalc` is on PATH (`doctor --check-fpcalc`). Text similarity never drops an ingest.

### Added

- Web media resolver: `connector-resolve` and `web-ingest` turn an article/page URL into ranked candidate recordings, then ingest the chosen one through the local pipeline. The `web` connector is force-only and never auto-steals URLs from other connectors.
- `ConnectorCandidate` schema and `schema connector-candidate`.
- Per-speaker fingerprint match diagnostics: `match.kind` (`strong`, `margin`, `new`, `no_enroll`, `name_match`, `preassigned`, `no_embedding`) plus diagnostic similarity values, in `load`, `raw-json`, `jsonl`, and `csv`. Absent comparison scalars are `null`, not fabricated zeroes.
- Content dedupe: audio-fingerprint skip across different source ids, advisory text simhash for review, `--allow-duplicate` to override, and `doctor --check-fpcalc`.
- `doctor --check-yt-dlp` / `--yt-dlp-bin` and a stale-yt-dlp warning.

### Changed

- Speaker output gains the additive `match` field; `Speaker.embedding` is a supported output contract.
- `youtube-ingest` and `--connector youtube` reject non-YouTube hosts instead of silently treating them as YouTube; use `web-ingest` for article pages.

## 0.2.0

### Upgrade notes (read before upgrading an existing database)

- **Existing fingerprint libraries go dormant until adopted.** Fingerprints are now namespaced by embedding model. Voiceprints created before this release carry no model tag, so they are dormant for normal matching until you adopt them. After upgrading, run `undertone --db ./undertone.db fingerprint-adopt-model --dry-run` then `--yes`. Dormant and incompatible counts are shown in `doctor`, `models`, and `--progress json` warnings. Only adopt when the stored vectors were produced by the active model; if the embedder changed, rerun the source audio instead.
- **`fingerprints` lists active prints by default.** It previously listed every print. Use `--status all` (or `--status discarded`) to see retired or legacy prints. JSON output now includes `status` and `discard_reason`.
- **Discarding a fingerprint starts that speaker over.** When a discarded speaker reappears, ingest mints a fresh, unnamed fingerprint; label it again as needed.

### Added

- Speaker fingerprint corrective actions: `fingerprint-discard`, `fingerprint-restore`, and `fingerprint-destroy`, with `--dry-run`/`--yes` and a timestamped `.bak` on writes.
- Embedding-model namespacing for fingerprints, with dormant-legacy handling and `fingerprint-adopt-model`.
- Fingerprint `fingerprint-export` / `fingerprint-import` / `fingerprint-merge`, and `relabel` / `resolve-names` for re-stamping saved transcript names without re-running ASR.
- Connector plugin model under the `undertone.connectors` entry point, with built-in YouTube and podcast connectors and direct audio URL support.
- `--progress json` events on stderr for long-running ingest paths.
- `asr_confidence` and `diarization_quality` fields in `raw-json` and `jsonl` exports.
- A single `undertone` skill (router plus references), distributable as a Claude marketplace plugin or via `undertone install-skills` for Claude and Codex.

### Changed

- `fingerprints` default status filter, fingerprint name-resolution, and matcher now exclude discarded prints. `fingerprint-merge` and `fingerprint-adopt-model` refuse discarded prints.
- `models`, `doctor`, and `stats` report active and discarded fingerprint counts.
