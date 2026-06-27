# Undertone Upgrades and Migration

General guidance for moving an Undertone install or database forward: version upgrades, automatic schema migrations, the fingerprint embedding-model namespace, and moving a fingerprint library between DBs or machines. This is the general migration surface; setup-specific backfills are not covered here.

## Schema migrations

Schema changes are additive and applied automatically when a `TranscriptStore` opens. Column adds are idempotent, so opening an older database upgrades its schema in place without data loss. A dry-run of a fingerprint command never mutates the real DB: it plans against a migrated temporary copy and discards it, so previewing on a pre-upgrade database leaves the original schema untouched.

## Fingerprint model namespace on upgrade

Fingerprints are namespaced by the effective embedding model. When you upgrade into a version that tracks the model and your existing voiceprints predate it, those rows carry no model tag and become **dormant**: they are not matched against new speakers under the active model. This is intentional. Comparing embeddings across model spaces silently produces wrong matches, so the safe default is dormancy, not a blind assumption that old vectors belong to the current model.

Dormant and incompatible counts are surfaced loudly:

```bash
undertone --db ./undertone.db doctor
undertone --db ./undertone.db models
# under --progress json, ingest emits a warning event when the library has dormant rows
```

`doctor` fails its fingerprint check (with the fix command) while the library has dormant or incompatible rows.

## Activating a legacy library

If the dormant voiceprints really were produced by the model you run now, assert that provenance:

```bash
undertone --db ./undertone.db fingerprint-adopt-model --dry-run
undertone --db ./undertone.db fingerprint-adopt-model --yes
```

`fingerprint-adopt-model` is a provenance assertion, not a vector conversion. It stamps the active model onto untagged rows; it does not move vectors between embedding spaces. If the embedder actually changed (a different model, not just an upgrade), do not adopt. Rerun the source audio so embeddings are rebuilt under the new model, then relabel.

## Moving a library between DBs or machines

```bash
undertone --db ./source.db fingerprint-export --output ./voiceprints.json
undertone --db ./target.db fingerprint-import ./voiceprints.json --dry-run
undertone --db ./target.db fingerprint-import ./voiceprints.json --yes
```

Export and import preserve the embedding model tag, dimension, and timestamps, so provenance survives the move and the imported voiceprints land in the correct model namespace. Import validates `schema_version` and rejects duplicate fingerprint ids in the file. Merging across DBs still refuses to combine voiceprints from different embedding models.

## Version pinning

Pin a known-good version when a downstream integration reads Undertone output, and review changes to diarization, fingerprint identity, or the CLI surface before moving up. Output field names for speakers (`speaker_id`, `fingerprint_id`, `display_name`) are stable; new fields are added without renaming existing ones, and `schema_version` marks the contract.
