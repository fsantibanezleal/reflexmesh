# M10: asynchronous finite-state policy plus planner

M10 begins one real background planner request while continuing to choose M01 actions. A single-worker thread pool stores the pending proposal and a binding of the observation from which it was requested. On a later call, a completed proposal can replace the rule decision only if that binding still matches and the response validates against the current observation. Otherwise the stale proposal is recorded as discarded and the finite-state policy continues.

The binding includes goal, revision, observed state, capabilities, budget, deadline and the sorted candidate registry including arguments, effects and resource versions. Writing canonical serialization as $C$ and SHA-256 as $H$,

$$
b_0=H(C(o_0)),\qquad
\operatorname{apply}(p_t)\iff\operatorname{done}(p_t)\land b_0=H(C(o_t))\land V(p_t,o_t).
$$

The hash detects a change in the modeled proposal context; it does not prove that every relevant external change is observable. Native revision/capability checks remain mandatory. The global binding is intentionally conservative: unrelated changes may invalidate a useful proposal. Read-set-scoped invalidation is an alternative research design, not an unreported feature of this baseline.

No combined neural checkpoint is fitted. M10 uses the M01 rule source plus an explicitly configured real planner and its model provenance. The pending job persists across decision calls, while `reset` waits for prior work before clearing episode state. `finish_episode` accounts for the completed proposal even when the fast controller already finished the task; `close` shuts down the executor and waits for outstanding work. The current implementation launches a single proposal per episode, not an unrestricted continuous planning service.

Diagnostics distinguish request, pending state, application and stale discard. End-to-end wall time includes drained background work, whereas per-call controller time can be short because it overlaps the request. Reporting only hot-path latency would hide consumed model work. The meaningful quantities include planner calls, completed unused proposals, token/time cost, application fraction, task success and time to first verified effect.

Short owned workflows may finish before any planner result arrives. That is a measured behavior, not a reason to omit the planner cost or fabricate useful delegation. Compare with M01, direct M09, and a cooperative-hold policy such as M12. Vary actual response delay and changing observations to determine whether overlap helps. Removing policy-level invalidation can be an ablation, but must never disable the executor's authority checks. This is an adapted asynchronous controller baseline, not a reproduction of an external agent's entire architecture or scores.

Implementation: [planning.py](../../python/reflexmesh/policies/planning.py), [runner.py](../../python/reflexmesh/evaluation/runner.py). Research context: [Kool et al., 2016](https://doi.org/10.1371/journal.pcbi.1005090) motivates charging for deliberation; it does not establish that asynchronous LLM work improves this software suite.
