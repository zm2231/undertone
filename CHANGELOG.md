# Changelog

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
