# Undertone Speaker Fingerprints

Voice fingerprints give a speaker a durable, cross-recording identity. Undertone mints them from speaker embeddings, matches new speakers against the stored library by cosine similarity, and lets you name them once so they self-name on later recordings.

## Inspect

```bash
undertone --db ./undertone.db fingerprints
undertone --db ./undertone.db fingerprints --format json
undertone --db ./undertone.db fingerprints --unnamed --excerpts
undertone --db ./undertone.db fingerprints --status all
```

`fingerprints` lists active voiceprints by default with display name, sample count, status, and embedding model. `--status discarded` or `--status all` shows retired prints. `--unnamed` filters to the ones still needing a label; `--excerpts` shows sample transcript lines so you can recognize the voice. Add `--json` (or `--format json`) for machine-readable output, including `status` and `discard_reason`.

## Label and relabel

```bash
undertone --db ./undertone.db fingerprint-label VP-abc123 "Speaker Name"
undertone --db ./undertone.db relabel meeting-1
undertone --db ./undertone.db relabel --all
```

`fingerprint-label` sets the display name on a voiceprint. `relabel` re-stamps saved transcript speaker names from the current fingerprint DB without re-running ASR, diarization, enrichment, or fingerprint matching, so it is the cheap "label then fix the back catalog" path. Pass a single transcript id or `--all`; passing both an id and `--all` is rejected.

`reenrich` is different: it rebuilds enrichment from the saved raw transcript without retranscribing audio. Use `reenrich` after changing thresholds or feature toggles. It re-runs fingerprint matching against today's fingerprint centroids, so fingerprint match diagnostics reflect the current library, not necessarily the exact state at original ingest time.

## Corrective actions

Use discard for a bad, mixed, or over-merged voiceprint. Discard is reversible and keeps the historical row for audit; future ingest ignores discarded prints and mints a fresh fingerprint when that person appears again.

```bash
undertone --db ./undertone.db fingerprint-discard VP-bad --reason "mixed speaker" --dry-run
undertone --db ./undertone.db fingerprint-discard VP-bad --reason "mixed speaker" --yes
undertone --db ./undertone.db fingerprint-restore VP-bad --dry-run
undertone --db ./undertone.db fingerprint-restore VP-bad --yes
undertone --db ./undertone.db fingerprint-destroy VP-bad --dry-run
undertone --db ./undertone.db fingerprint-destroy VP-bad --yes
```

`fingerprint-discard` sets `status=discarded` and stores `discard_reason`; `fingerprint-restore` makes the print active again. `fingerprint-destroy` permanently deletes the fingerprint row and cascades its `fingerprint_sources` rows. Saved transcript speaker rows keep their historical fingerprint id, so use destroy only when a voiceprint should leave the library entirely.

## Duration gates

Fingerprinting is duration-gated on every engine, not just pyannote. A speaker below the enroll threshold of total talk time does not mint a durable cross-recording fingerprint, and a sample below the update threshold is not folded into a stored centroid. This is deliberate: short, noisy speakers are the main source of garbage identities. A brief speaker still appears in the transcript with per-recording diarization; it just does not get a stable durable identity.

## Known Counts, Spurious Splits, and Merge

Use `--expected-speaker-count` only when you know how many speakers are actually present in the recording. It is a known-count hint, not a de-duplication or merge knob. Sponsor reads, ad voices, brief guests, and short co-speakers can be legitimate distinct speakers.

If diarization splits one person into two voiceprints, tune fingerprint matching at ingest or merge after the fact:

```bash
undertone --db ./undertone.db run-wav ./ep.wav --fingerprint-similarity-threshold 0.78
undertone --db ./undertone.db fingerprint-merge VP-spurious VP-canonical --dry-run
undertone --db ./undertone.db fingerprint-merge VP-spurious VP-canonical --yes
```

`--fingerprint-similarity-threshold` (or `UNDERTONE_FINGERPRINT_SIMILARITY_THRESHOLD`) controls how aggressively new speakers match an existing voiceprint across recordings. `UNDERTONE_SPEAKER_MERGE_THRESHOLD` is separate: it controls same-recording speaker collapse before fingerprint assignment. `fingerprint-merge` folds a spurious active voiceprint into a canonical active one and restamps the affected speaker rows; it refuses to merge discarded prints or voiceprints from different embedding models.

## Match Diagnostics

Saved transcript speakers include a nullable `match` object. `match.kind` is the authoritative signal (`strong`, `margin`, `new`, `no_enroll`, `name_match`, `preassigned`, or `no_embedding`). Similarity and margin values are diagnostic scalars from the fingerprint matcher, not portable probabilities; thresholds depend on the active embedding model and configuration. If no comparison happened, `similarity` is `null`; if there was no real runner-up, `second_similarity` and `margin` are `null`. Legacy transcripts show `match=null` until they are reenriched or ingested again.

## Export and import

```bash
undertone --db ./undertone.db fingerprint-export --output ./voiceprints.json
undertone --db ./undertone.db fingerprint-import ./voiceprints.json --dry-run
undertone --db ./undertone.db fingerprint-import ./voiceprints.json --yes
```

Export and import preserve the embedding model tag, dimension, timestamps, `status`, and `discard_reason`, so a labeled library moves between DBs or machines without losing provenance or accidentally reactivating discarded prints. Import validates the payload `schema_version` and rejects duplicate fingerprint ids within the file. See `references/upgrades.md` for cross-DB portability and model-namespace details.

## Embedding-model namespace

Fingerprints are namespaced by the effective embedding model. For `fluidaudio-pyannote` the model is the resolved pyannote model; otherwise it is `UNDERTONE_EMBEDDING_MODEL` / `--embedding-model`. Matching only ever compares embeddings within the same model namespace, so a model change never silently matches new speakers against vectors from a different embedding space. Legacy fingerprints with no model tag are dormant for normal ingest until explicitly adopted, and `doctor`/`models`/`--progress json` surface dormant and incompatible counts.

```bash
undertone --db ./undertone.db fingerprint-adopt-model --dry-run
undertone --db ./undertone.db fingerprint-adopt-model --yes
```

`fingerprint-adopt-model` asserts provenance for old vectors; it does not convert vectors between model spaces. Only adopt when the stored vectors were actually produced by the active model. If the embedder genuinely changed, rerun audio to rebuild embeddings instead of adopting. See `references/upgrades.md`.

## Safety

Fingerprint import, merge, model adoption, discard, restore, and destroy are identity-changing operations. Always run `--dry-run` first. Writes require `--yes` and create a timestamped `.bak` SQLite copy beside the active DB before mutating rows. No-op writes do not create backups. A dry-run never mutates the real DB; it plans against a migrated temporary copy and discards it.
