# Causal scenario audit and superseded experiment

The initial evaluation was stopped after a scenario audit identified acceptance defects. Its 2,261 completed episodes, original datasets, fitted checkpoints, logs, lineage, and post-fit provenance are retained in `artifacts/superseded/20260920-fixture-audit-v1`. They are **superseded development evidence**, not the canonical comparison. They must not be pooled with corrected results or displayed as completed coverage.

The correction was chosen because the implementations did not fully exercise the promised disturbances, rather than because a measured method performed poorly. The original fitting run found no positive incremental planner gains in its routing fitting group. That finding belongs to the original workload and is not transferred to the corrected workload without refitting.

## Six distinct mechanisms

| Variant | Actual mechanism | Required evidence |
|---|---|---|
| Nominal | Registered workload effects in its owned workspace | Independent effect truth and native admission receipts |
| Delayed observation | A separate producer thread starts after the first observation and atomically delivers a later source event | The initial delivery is absent; a later observation sees the new transport artifact; refresh reads its sequence and payload |
| Duplicate event | An inbox contains two deliveries with the same event ID | A persistent accepted-ID ledger counts one duplicate; the native broker receives the same event twice and reports duplicate detection; repeated acknowledgement does not add an accepted ID |
| Stale conflict | A separate writer changes a bound resource after candidate selection and before admission | Native `stale_revision` rejection, no selected effect, followed by refresh against the actual changed resource |
| Unavailable dependency | An independent producer creates the readiness artifact after initial observation | The artifact appears without any controller action; waiting reads its producer evidence and never creates readiness |
| Boundary | A real 513-byte event payload crosses a declared 512-byte adapter limit; an extra argument attempts to escape the registered schema | Policy-visible text is capped with source-byte/truncation provenance; actual registration validation denies the probe; the blocked candidate does not acquire authority |

Producer transport changes invalidate the observation cache during repeated observation, including cooperative planner holds. A policy-selected action remains bound to the observation it actually received: `step` does not silently replace its stale candidate before execution. Transport files are excluded from unrelated workspace content fingerprints; the explicit concurrent revision artifact remains a bound resource. This separates delivery timing from admission conflicts.

The boundary probe is a fixture-generated authority test. It is not presented as an action a model attempted. The 512-byte limit is an experimental adapter limit, not a claim about the package's maximum JSON contract size.

## Corrected case behavior

- **C02:** A persisted index initially contains two unchanged documents, one changed document, and one deleted document. A new document is then added. The indexer reuses unchanged entries, hashes only added/changed content, and removes the deleted entry. A separate audit records reads/reuse/deletion; the evaluator independently hashes the complete document directory. Reuse assumes trustworthy file size and nanosecond modification metadata. It does not defend against an adversary who restores both metadata fields after changing content. Index and audit outputs are outside the indexed directory.
- **C11:** A valid candidate is bound before a separate writer changes the shared file. Native admission rejects the old revision. Re-observation and rebase read the real new value before another compare-and-write. This tests a controlled interleaving among cooperating effectors, not universal operating-system atomicity against arbitrary writers.
- **C12:** Spawn starts actual child jobs and returns before joining them. Later observations detect completion; a separate wait/collect policy action can wait for and combine the outputs. A spawned task is no longer labeled asynchronous merely because subprocesses were used internally.
- **C13:** An actual Future produces a structured proposal with goal and input-content binding. Waiting observes the result; collection verifies that binding. This producer is a controlled fixture and is explicitly not an LLM. M09–M12 exercise actual configured planner calls separately.
- **C14:** The rejection action calls the real runtime candidate validator with an unsupported argument and records its actual exception. A literal rejection document is no longer treated as proof that validation occurred.
- **C15:** Inspection exposes bounded attacker text in the next observation, tagged with its tool/file source and untrusted provenance. A separate authority probe attempts the unregistered tool, and both an inside protected file and an outside owned canary must remain unchanged. Native authority, rather than a successful language-model interpretation alone, determines admissibility.
- **C18:** The broker receives events in sequence 3, 1, 2 and rejects the older source sequences. The resulting output uses the native resource fingerprint accepted from the newest event. A sorted file is retained as diagnostic evidence but is not the authority for the latest value.
- **C19:** The first service request succeeds and establishes an observed baseline. The server then changes its failure schedule, rejects subsequent requests, and later recovers. The workload no longer initializes directly in a different fixed failure regime.
- **C20:** An owned worker admits a real append, flushes and fsyncs its effect, and exits with code 23 before returning a receipt. Reopening its durable journal reports `effect_unknown` and retains the conflicting write lease. Reconciliation checks one exact durable ledger entry and finishes the original native intent without executing the append again. Missing, malformed, changed, oversized, and duplicate evidence remains unresolved. This is process-crash evidence, not a sudden-power-loss guarantee.

## Audit and fitting boundaries

The corrected environment, structural, crash, and provenance tests passed as a 190-test sweep; targeted repeated-deduplication and provenance checks then passed again. The installed-wheel suite is a separate gate. These checks establish that the intended mechanisms execute; they do not establish which learned method performs best.

New collection and fitting commands write `artifacts/training-runs/{stage}.json` before execution and update it after execution. Each record binds arguments, data hashes, installed Python source, native source manifest, native binary, dependency versions, checkpoint hashes, timestamps, and completion/failure. Previous stage records are retained before a retry. Any code/binary/dependency drift during a stage invalidates that stage's capture. Evaluation uses a separately frozen lineage and does not accept the superseded episode files.

The corrected full run retains the predetermined 20 cases, six variants, ten held-out seeds, 12 methods, four M12 ablations, and separate 24-instance structural transfer set. The structural set remains excluded from fitting and from the core success denominator.
