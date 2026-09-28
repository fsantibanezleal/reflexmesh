# Semantic transfer stress protocol

This lane tests an identifiable gap in the frozen core: a fixed controller can solve every original task, so the core cannot reveal when a slow reasoner helps. It retains the 12 published methods and their frozen checkpoints. The separate suite has eight D01–D08 compositional families, nominal and boundary variants, and eight held-out seeds (82000–82007), yielding 128 episodes per method and 1,536 for a complete 12-method comparison. The numbers never enter the original core or structural denominators.

Every episode starts from a seeded JSON list with negative values, zeros, repeated values and distinct positives. It requires three dependent transformations. At each stage, three registered operations have the same input/output type and numeric `known` feature, but produce distinct values on the oracle path. Tool identifiers are SHA-derived opaque tokens, candidate order and input bytes vary by seed, and two families change the required transformation across seeds. The goal uses natural-language paraphrases rather than the operation labels in candidate descriptions. All methods receive the same visible goal, candidates, outcome history and native file-effect interface. The hidden correct IDs and expected values stay in the evaluation environment and are used only for demonstration-action diagnostics and independent truth.

The executor writes `stage-1.json`, `stage-2.json` and `final.json` through registered, journaled real-file effects. Independent verification reads and compares all three files with the expected transformation chain. A wrong admissible action writes a wrong file; a subsequent correct action can repair it within the 12-step limit. An accepted native receipt does not imply task success. Tests run all 128 configurations, deliberately perform a wrong first action, repair it, check all file predicates and inspect the four actual receipts.

Run a complete lane only with the declared model and fitted checkpoint assets available:

```sh
python -m reflexmesh.pipeline semantic-transfer --artifacts artifacts --seeds 8 --output artifacts/semantic-transfer --model qwen3.5:4b
```

The output records per-episode actions, receipts, latency, planner work, source/native/model digests and a separate index. A smoke run, including one seed for four methods, is exploratory and must not be silently promoted to the complete result. Evaluate task success, invalid output, action agreement, planner calls and wall time separately. A method that does well because a hand-written rule can parse this suite is a useful ceiling diagnostic, not evidence of learned allocation. These designed transformations do not stand for arbitrary software tasks, and a model failure can reflect grounding or output-format errors as well as poor routing. Comparing the fitted gate against an alternative gate requires paired training and held-out evaluation with enough positive incremental-utility cases; the frozen gate does not supply that evidence.

## Completed run and verdict (2026-09-28)

The full run is saved in `artifacts/semantic-transfer-complete/` and was checked with `scripts/audit_semantic_transfer.py` against its index and report. All 1,536 episodes are present (128 per method, eight families, two variants and eight seeds). The index SHA-256 is `05a57e6e4709c0c2e2226ba804e8c8f2fc233580e912e777d578cbc2ea2a44d8`. The recorded model is `qwen3.5:4b`; model and source digests are in the index. This is a deliberately harder transfer lane, not an addition to the frozen core's 19,488-episode denominators.

| Method | Verified successes / 128 | Invalid policy outputs | Planner delegations | Mean wall time / episode |
|---|---:|---:|---:|---:|
| M04 XGBoost | 2 | 0 | 0 | 0.329 s |
| M07 GRU | 0 | 128 | 0 | 0.072 s |
| M09 planner | 38 | 40 | 925 | 18.882 s |
| M12 controller | 18 | 17 | 375 | 7.706 s |

No method is perfect. M12 beats M09 on only 2 paired cases, loses on 22 and ties on 104. The new suite does remove the trivial perfect-state-machine ceiling, but it does **not** establish a benefit from M12's learned hand-off: M09 succeeds more often, while M12 is faster by using the planner less. In particular, planner invalid outputs are a separate failure mode; their count cannot be read as routing errors alone. Thirty-five method/case/variant cells vary across seeds, so this lane introduces outcome variation, but the designed families still limit external validity.

An exploratory *offline* initial-arm routing study (`artifacts/semantic-routing-study/analysis.json`) fit a visible-text and fast-score logistic model on D01-D04, calibrated on D05-D06, and held out D07-D08. It used 19 positive fit cases and eight positive calibration cases. On the 32 held-out episodes the learned gate delegated **zero** times and verified zero successes, equal to always-fast M04; always-planner M09 succeeded 10 times, and a hash-half selector succeeded 7. A post-hoc oracle upper bound selects the 10 successful planner episodes and succeeds 10 times. That oracle uses outcomes and is not a deployable controller. The study chooses among already-recorded episode arms and excludes routing/inference latency; it is neither an online M12 run nor evidence of general metacognitive transfer. A useful next experiment needs a prespecified train/calibration/test split with more positive incremental-utility states and a controller actually run on the held-out episodes.

The routing audit runs with `scripts/audit_semantic_routing.py`: it checks every paired first observation and candidate set, the fitted-model digest and all held-out comparator summaries against the executed traces.
