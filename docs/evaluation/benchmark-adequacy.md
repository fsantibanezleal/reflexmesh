# Frozen benchmark adequacy audit

The corrected 19,488-episode evaluation remains the historical result. This audit asks a narrower, post-hoc question: can its 14,400-episode core matrix distinguish a successful learned controller from simpler working methods? It does not refit a model, edit a trace or replace the corrected result.

`scripts/audit_benchmark_adequacy.py` reads the canonical `artifacts/evaluation/index.json`, verifies the SHA-256 of each indexed core episode, checks its method/case/variant/seed identity, and requires all 12 × 20 × 6 × 10 cells. Its [machine-readable output](benchmark-adequacy.json) binds the source index digest. Reproduce it with:

```sh
python scripts/audit_benchmark_adequacy.py --index artifacts/evaluation/index.json --output docs/evaluation/benchmark-adequacy.json
```

Seven methods solve all 1,200 of their core episodes: M01, M02, M03, M04, M05, M10 and M11. In particular, M01 is a guarded state machine. M12 solves 1,160/1,200. All 40 failures occur in C07 cancellation: ten in each of delayed observation, duplicate event, stale conflict and unavailable dependency. The state-machine ceiling means this matrix cannot establish a *task-success* gain for a learned allocation policy. Other measured properties, such as native admission, recovery, latency and traceability, have separate evidence and are not invalidated by that ceiling.

Only 71 of 1,440 method/case/variant cells change their binary success outcome across ten seeds. M12 has no such cell. Inputs can vary even when the success bit does not; the audit counts outcome variability, not identical byte-for-byte episodes. Seeds within a fixed task template are not independent draws from the population of software work. The degenerate seed-bootstrap interval reported for M12 therefore does not imply certainty about new tasks.

The fitted allocation gates had seven positive incremental-utility labels among 228 fit states and selected threshold 1.01. Neither fitted gate chooses to hand off through that threshold. M12's other planner triggers must be analyzed separately. All four fixed-gate ablations matched M12 core success; this establishes no measured component contribution under that frozen configuration.

The C07 implementation now clears stale-state, duplicate-event and unavailable-dependency preconditions before requesting cancellation, and tests exercise all six C07 variants. This is a **post-hoc repair** after inspecting held-out failures. The frozen 1,160/1,200 score is not rewritten or replaced by a repaired-code score. Any rerun belongs to a separately named evaluation with new lineage and an explicit selection-history caveat.

The [semantic-transfer protocol](semantic-transfer.md) adds a separate designed stress lane in which candidate IDs are opaque, compatible tools produce different real files, and goal language must be grounded in operation descriptions. It complements rather than shrinks the 20-family core, structural-transfer and ablation records. It is not, by itself, proof of a learned routing advantage or general intelligence.
