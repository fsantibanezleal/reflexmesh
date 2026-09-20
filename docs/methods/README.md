# Method cards

These cards describe the executable implementations, not benchmark conclusions. Model hashes, training counts, calibration outcomes, device measurements and final evaluation denominators belong to the frozen release artifacts. Regenerated checkpoints invalidate older performance claims.

| Method | Card | Distinguishing mechanism |
|---|---|---|
| M01 | [Guarded finite-state policy](01-guarded-fsm.md) | Ordered observed-state rules |
| M02 | [Reactive behavior tree](02-behavior-tree.md) | Independent selector/sequence interpreter |
| M03 | [Calibrated logistic scorer](03-logistic-scorer.md) | Linear candidate preference and separate calibration |
| M04 | [Calibrated boosted trees](04-boosted-trees.md) | Nonlinear candidate preference and cached native trees |
| M05 | [Recurrent behavior policy](05-recurrent-policy.md) | Per-goal GRU state and demonstrated action likelihood |
| M06 | [Shared LinUCB](06-contextual-bandit.md) | Executed-action reward updates and uncertainty bonus |
| M07 | [Masked PPO](07-masked-ppo.md) | Nonrecurrent actor/critic and admissibility mask |
| M08 | [Learned transition lookahead](08-transition-lookahead.md) | Measured transition regression and bounded search |
| M09 | [Structured local planner](09-structured-planner.md) | Real model request for each control decision |
| M10 | [Asynchronous FSM and planner](10-asynchronous-planner.md) | Fast progression with a background bound proposal |
| M11 | [Learned deferral](11-learned-deferral.md) | Paired executed utility and calibrated recruitment |
| M12 | [Persistent metacontroller](12-persistent-metacontroller.md) | Recurrent/effect blend, residuals, recruitment and holds |

All methods receive a causal observation and registered candidates. Authority comes from the executor, including capabilities, revisions, leases and cancellation. No method's confidence, utility, mask or natural-language explanation authorizes an effect. [Scientific overview](scientific-methods.md) describes the common feature schema and experimental utility. The separate [workspace workflow controller](../workspace-workflows.md) is a verified dependency FSM; it is not an arbitrary-task deployment of these learned policies.

The terms fast and deliberative identify computational roles. They do not classify every LLM computation as psychological System 2, every small policy as biological System 1, or this integration as AGI. Papers motivate mechanisms; package source establishes what was actually implemented.
