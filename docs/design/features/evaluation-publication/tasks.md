# evaluation-publication convergence

1. Bind input and authority contracts (R-001, R-002, R-003).
2. Execute the named contract gates before release.
3. Check persisted artifacts and user-visible behavior, then record external release gates separately.

19,488 episodes passed the independent identity/hash audit on September 23. Native parity passed 484 vectors per exported model at 1e-5. PyPI publication succeeded through Trusted Publishing in workflow 35944473929: nine portable wheels and one source distribution. A fresh isolated, uncached PyPI installation executed and verified a real file workflow; all 18 downloaded model files passed their release hashes. See docs/evaluation/release-validation.json.

| Requirement | Gate | Current verdict |
| --- | --- | --- |
| R-001 | scripts/audit_science.py | Implemented and locally verified; see evidence above. |
| R-002 | tests/test_pipeline_training_capture.py | Implemented and locally verified; see evidence above. |
| R-003 | tests/test_model_artifacts.py | Implemented and locally verified; see evidence above. |
