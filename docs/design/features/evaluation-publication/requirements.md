# evaluation-publication requirements

Date: 2026-09-23. Migration of the authorized existing implementation.

```text
R-001 WHEN corrected evidence is audited, THE audit SHALL check full matrix identities and every trace digest.
Gate: scripts/audit_science.py
R-002 WHEN training provenance is captured, THE stage SHALL preserve failed attempts and exact inputs.
Gate: tests/test_pipeline_training_capture.py
R-003 WHEN model assets are installed, THE loader SHALL reject mismatched digests.
Gate: tests/test_model_artifacts.py
```
