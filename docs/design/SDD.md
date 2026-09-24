# Reflexmesh software design
Date: 2026-09-23. Release: 0.1.0.

## Review and migration record
This document consolidates the research-backed architecture and implementation authorized on September 20. ADR-0075 was adopted on September 22, after that implementation and experiment had started. This is an explicit migration, not a retroactive claim that this document was reviewed before development. The user's September 23 instruction authorizes completing the existing scope. New scope requires its own design.

## Problem and non-goals
Select and coordinate registered software effects with low decision latency, durable authority checks and bounded asynchronous deliberation. The proposed contribution is the integration of typed authority, effect-aware recurrent control and state-bound computation allocation. Novelty and utility are hypotheses tested against twelve baselines, not established by implementation.
Non-goals: biological equivalence, AGI, hostile-code sandboxing, hard real time, arbitrary workspace access, autonomous credential discovery, and guarantees outside registered cooperating effectors.

## Contracts
Observations, candidates, capabilities, intents and receipts are typed finite JSON values. Model scores cannot grant authority. Every effect passes current capability, schema, revision, cancellation and lease checks in the Rust reducer. SQLite persists intent before external execution; unknown effects retain leases until independently verified. Replay never re-executes effects.
Ingestion is a resettable software environment with observed action/outcome and separate teacher labels. Nonfinite values, overlapping seed/group splits, invalid candidates and changed lineage are rejected. Failed policy outputs remain failed observations rather than disappearing from the denominator.
Evaluation artifacts contain exact case/method/variant/seed identity, source/native/checkpoint hashes, installed binary identity, events, decisions, receipts and independently measured outcomes.

## Lanes and measured basis
Rust admission and cached scoring run locally. Warm seven-candidate scoring is measured separately from Python feature construction, fsync and effects. The Python runtime and registered services execute actual tools. A local or HTTPS planner runs asynchronously and may return too late; source/state bindings determine whether a proposal remains admissible.
Offline fitting and the 19,488-episode experiment require installed scientific dependencies and real software effects. They do not run in CI or in a browser. Portable wheels are built in release jobs. No hosted arbitrary-effect service is supplied.

## Method ladder
| Method | Acceptance criterion |
| --- | --- |
| M01 | Maintained state rules execute independent postconditions across the complete core matrix. |
| M02 | Explicit behavior-tree traversal executes the same candidate/authority contract. |
| M03 | Fitted logistic imitation, disjoint calibration and Rust export match within 1e-5. |
| M04 | Fitted boosted-tree imitation, calibration and Rust export match within 1e-5. |
| M05 | Trained GRU carries episode-local hidden state; ONNX export parity is measured. |
| M06 | Logged executed rewards and propensities fit the contextual linear bandit. |
| M07 | Masked PPO records achieved online transitions and held-out execution outcomes. |
| M08 | Learned observed transitions, reward and termination support bounded lookahead and validation error. |
| M09 | A real model receives the current bounded candidate set; its selected action is independently admitted. |
| M10 | Fixed asynchronous delegation records request, acceptance, rejection and unused completion. |
| M11 | Paired executed incremental utility fits a calibrated gate with separate threshold selection. |
| M12 | Recurrent/effect preference, surprise response and learned allocation execute with four component ablations. |
Each method has 1,200 core and 24 structural episodes. The full comparison and hashes are checked by the experiment audit; acceptance means implemented and measured, not superior success.

## Case taxonomy and coverage
C01-C04 cover files, indexing, validation and build recovery. C05-C08 cover HTTP retry, ambiguous writes, process cancellation and restart. C09-C12 cover bounded storage, DAGs, revision conflict and agent jobs. C13-C16 cover planner deadlines, invalid arguments, untrusted text and missing evidence. C17-C20 cover new capabilities, event ordering, reliability drift and actual crash recovery.
Twenty families x six variants x ten held-out seeds x twelve methods = 14,400 core episodes. Four unseen structural compositions x two variants x three seeds x twelve methods = 288 transfer episodes. Four M12 ablations each repeat all 1,200 core cells. Template overlap across train/test is explicit; structural transfer is reported separately.

## Oracle
Truth is verified by actual files, hashes, HTTP service state, child-process termination and external crash/reopen observations. Success requires independent postconditions. Teacher agreement is an imitation metric, never a safety oracle. Native receipt rejection measures declared-contract enforcement, not absence of all possible harm.
Paired comparisons match exact identities and bootstrap over environment seed groups. Timing budgets are not equalized. Ten core groups and three structural seeds limit generalization. A native scorer microbenchmark is not end-to-end latency.

## Deploy driver
This repository is an installable library, so the target is PyPI plus versioned GitHub model assets. The approximately 11 MB checkpoint set stays out of the wheel. A base installation must import and execute rules/native authority without training frameworks. Windows AMD64, Linux manylinux x86_64 and macOS ARM64 CPython 3.11-3.13 wheels are the measured release targets. Other targets require source builds and are not claimed verified.

## Risks and kill criteria
Do not publish if receipt/restart tests fail, exported scores exceed tolerance, a matrix is incomplete, a trace fails its digest, or fitting/evaluation lineage changes. Do not claim a learned advantage when ablations or baselines fail to support it. A PyPI authentication or billing failure remains an explicit release blocker; it is never recorded as a passed gate.
Unresolved process descendants and ambiguous remote writes require trusted external evidence; no heuristic reconciliation clears their leases.

## Requirements
```text
R-001 WHEN an action is requested, THE broker SHALL recheck current authority before a real effect.
Gate: tests/test_native_broker.py
R-002 IF a process dies with an incomplete intent, THE runtime SHALL preserve uncertainty across restart.
Gate: tests/test_environments_crash.py
R-003 WHILE a journal is owned, THE runtime SHALL reject a concurrent owner.
Gate: tests/test_runtime_ownership.py
R-004 WHEN a workflow is submitted, THE service SHALL validate the entire recipe before execution.
Gate: tests/test_workflows.py::test_whole_recipe_is_validated_before_any_effect
R-005 WHEN mutable goal policies are created, THE controller SHALL enforce independent instances.
Gate: tests/test_controller.py::test_factory_cannot_accidentally_share_episode_state
R-006 WHEN fitting splits overlap, THE learning pipeline SHALL reject the fit.
Gate: tests/test_learning_integrity.py::test_group_and_seed_overlap_rejected_even_different_episode_names
R-007 WHEN a fitting stage changes its source, THE provenance capture SHALL reject completion.
Gate: tests/test_pipeline_training_capture.py::test_training_capture_rejects_source_change_during_fitting
R-008 WHEN an evaluation is resumed with changed lineage, THE evaluator SHALL reject the resumption.
Gate: tests/test_pipeline_artifacts.py::test_cancel_before_effect_and_resume_lineage_rejection
R-009 WHEN corrected evidence is released, THE audit SHALL verify all 19,488 identities and trace hashes.
Gate: scripts/audit_science.py
R-010 WHERE native policy export is used, THE parity check SHALL enforce maximum probability error of 1e-5.
Gate: python/reflexmesh/benchmarks/native_parity.py
R-011 WHEN a model bundle is fetched, THE installer SHALL verify its archive and file hashes.
Gate: tests/test_model_artifacts.py
R-012 WHEN CI executes, THE workflows SHALL avoid the training stack and respect the trunk-only budget.
Gate: scripts/check_ci_budget.py
```

Feature requirements, design and convergence are in [features](features/). The evidence report is [corrected evaluation](../evaluation/corrected-results.md).
