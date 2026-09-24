# Release convergence, September 23, 2026

The product SDD requirements were checked against the actual installed-wheel tests and released artifacts. Implementation acceptance does not imply scientific superiority.

| Requirement | Gate and measured result |
| --- | --- |
| R-001 | Native broker contract tests passed, included in 297 installed-wheel tests. |
| R-002 | Actual child crash/reopen tests passed; no effect replay. |
| R-003 | Cross-process journal ownership and release-on-death tests passed. |
| R-004 | Whole-recipe validation before effects passed, including invalid paths and registrations. |
| R-005 | Shared mutable policy-factory rejection passed. |
| R-006 | Overlapping group/seed rejection passed. |
| R-007 | Fitting-time source-change rejection and failed-attempt retention passed. |
| R-008 | Changed-lineage resumption rejection passed. |
| R-009 | Full 19,488-episode identity, byte digest and cross-experiment lineage audit passed. |
| R-010 | M03/M04 export parity passed 484 vectors per model at tolerance 1e-5. |
| R-011 | Model-integrity tests passed; public release download verified all 18 actual model files. |
| R-012 | Budget guard passed; develop and main CI succeeded without training dependencies. |

Release workflow [35944473929](https://github.com/fsantibanezleal/reflexmesh/actions/runs/35944473929) published nine platform wheels and a source distribution. A clean public-index installation executed a real file write and checked its independent hash through native authority. Exact public distribution digests are in [release-validation.json](../evaluation/release-validation.json).

The source/evaluation freeze and subsequently rebuilt publication wheels have distinct binary identities. The release includes the exact frozen Windows evaluation installation, original fitting data, all owned traces, checkpoint manifests and model artifacts. No untested CPU architecture is claimed supported by a wheel.

Scientific verdict: the integrated controller is implemented; learned delegation benefit and M12 core superiority were not demonstrated. All four fixed-gate ablations matched M12 core success. These are recorded results, not unfinished experiments or concealed failures.
