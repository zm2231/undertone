# Undertone Speaker Fingerprints

Voice fingerprints give a speaker a durable, cross-recording identity. Undertone mints them from speaker embeddings, matches new speakers against the stored library by cosine similarity, and lets you name them once so they self-name on later recordings.

## Inspect

```bash
undertone --db ./undertone.db fingerprints
undertone --db ./undertone.db fingerprints --format json
undertone --db ./undertone.db fingerprints --unnamed --excerpts
```

`fingerprints` lists stored voiceprints with display name, sample count, and embedding model. `--unnamed` filters to the ones still needing a label; `--excerpts` shows sample transcript lines so you can recognize the voice. Add `--json` (or `--format json`) for machine-readable output.

## Label and relabel

```bash
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
undertone --db ./undertone.db relabel meeting-1
undertone --db ./undertone.db relabel --all
```

`fingerprint-label` sets the display name on a voiceprint. `relabel` re-stamps saved transcript speaker names from the current fingerprint DB without re-running ASR, diarization, or enrichment, so it is the cheap "label then fix the back catalog" path. Pass a single transcript id or `--all`; passing both an id and `--all` is rejected.

## Duration gates

Fingerprinting is duration-gated on every engine, not just pyannote. A speaker below the enroll threshold of total talk time does not mint a durable cross-recording fingerprint, and a sample below the update threshold is not folded into a stored centroid. This is deliberate: short, noisy speakers are the main source of garbage identities. A brief speaker still appears in the transcript with per-recording diarization; it just does not get a stable durable identity.

## Over-segmentation and merge

If diarization splits one person into two voiceprints, tune it at ingest or merge after the fact:

```bash
undertone --db ./undertone.db run-wav ./ep.wav --expected-speaker-count 1
undertone --db ./undertone.db run-wav ./ep.wav --fingerprint-similarity-threshold 0.78
undertone --db ./undertone.db fingerprint-merge VP-spurious VP-canonical --dry-run
undertone --db ./undertone.db fingerprint-merge VP-spurious VP-canonical --yes
```

`--expected-speaker-count` collapses over-detected speakers toward the expected count. `--fingerprint-similarity-threshold` (or `UNDERTONE_SPEAKER_MERGE_THRESHOLD`) controls how aggressively new speakers match an existing voiceprint. `fingerprint-merge` folds a spurious voiceprint into a canonical one and restamps the affected speaker rows; it refuses to merge voiceprints from different embedding models.

## Export and import

```bash
undertone --db ./undertone.db fingerprint-export --output ./voiceprints.json
undertone --db ./undertone.db fingerprint-import ./voiceprints.json --dry-run
undertone --db ./undertone.db fingerprint-import ./voiceprints.json --yes
```

Export and import preserve the embedding model tag, dimension, and timestamps, so a labeled library moves between DBs or machines without losing provenance. Import validates the payload `schema_version` and rejects duplicate fingerprint ids within the file. See `references/upgrades.md` for cross-DB portability and model-namespace details.

## Embedding-model namespace

Fingerprints are namespaced by the effective embedding model. For `fluidaudio-pyannote` the model is the resolved pyannote model; otherwise it is `UNDERTONE_EMBEDDING_MODEL` / `--embedding-model`. Matching only ever compares embeddings within the same model namespace, so a model change never silently matches new speakers against vectors from a different embedding space. Legacy fingerprints with no model tag are dormant for normal ingest until explicitly adopted, and `doctor`/`models`/`--progress json` surface dormant and incompatible counts.

```bash
undertone --db ./undertone.db fingerprint-adopt-model --dry-run
undertone --db ./undertone.db fingerprint-adopt-model --yes
```

`fingerprint-adopt-model` asserts provenance for old vectors; it does not convert vectors between model spaces. Only adopt when the stored vectors were actually produced by the active model. If the embedder genuinely changed, rerun audio to rebuild embeddings instead of adopting. See `references/upgrades.md`.

## Safety

Fingerprint import, merge, and model adoption are identity-changing operations. Always run `--dry-run` first. Writes require `--yes` and create a timestamped `.bak` SQLite copy beside the active DB before changing rows. A dry-run never mutates the real DB; it plans against a migrated temporary copy and discards it.
