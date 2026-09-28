# Benchmark adequacy and semantic-transfer requirements

```text
R-001 WHEN the frozen core is interpreted, THE diagnostic SHALL hash-check all 14,400 indexed traces and report method ceilings and seed-variable cells without changing results.
Gate: scripts/audit_benchmark_adequacy.py
R-002 WHEN a new semantic result is exported, THE auditor SHALL reject missing or altered episodes, lineage and aggregates.
Gate: scripts/audit_semantic_transfer.py
R-003 WHEN a semantic choice is executed, THE environment SHALL require actual stage files with independently computed values, not just an accepted receipt.
Gate: tests/test_semantic_transfer.py::test_semantic_choices_have_independent_file_truth
R-004 WHEN C07 asks for cancellation, THE metacontroller SHALL resolve observation prerequisites before the urgent effect.
Gate: tests/test_learning_metacontrol.py::test_urgent_cancellation_respects_observation_prerequisites
R-005 WHEN file workflow postconditions are supplied, THE validator SHALL reject a verifier bound to another target or write digest.
Gate: tests/test_workflows.py::test_file_write_cannot_claim_an_unrelated_existing_postcondition
```
