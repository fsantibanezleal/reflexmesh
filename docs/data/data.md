# Data

The [owned data card](owned-data-card.md) defines actual workload families, effect predicates, collection policies, split construction and known limitations. The [causal audit](../evaluation/causal-scenario-audit.md) records why the first fixture generation and its fitted artifacts were superseded before canonical evaluation.

Collection runs the registered environments and records observations, selected candidates and independently observed outcomes. Training, validation, calibration and test seeds are disjoint; routing experiments use additional groups. Labels derived from the demonstration policy are imitation targets, while verified task success comes from the environment's separate filesystem, process, HTTP or journal predicate.

Bring-your-own-data integration begins by defining the event provenance, resource contracts and verifiable outcome before fitting a scorer. Do not copy a task-success label into a policy-visible feature. Preserve missing values, delayed delivery and ambiguous effects in the record. Map application observations into the versioned Candidate/Observation contract, fit on owned training groups, select and calibrate on separate groups, and evaluate against simpler policies on unchanged test groups. The shipped checkpoints are research-domain models; applying them to a new feature distribution requires validation and may require refitting.
