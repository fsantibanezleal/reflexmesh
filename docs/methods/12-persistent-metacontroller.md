# M12: persistent outcome-aware metacontroller

M12 integrates recurrent state, learned effect utility, causal prediction error and bounded asynchronous planner recruitment. Its fast branch uses M05 candidate probabilities plus 0.15 times M08 lookahead utility. The sum is explicitly a utility score, not a calibrated probability. Crucially, paired gate fitting executes this exact blended action; using the unmodified GRU action would train the gate against the wrong counterfactual.

For recurrent preference $p_t(a)$, lookahead value $Q_t(a)$, expected next-state feature vector $\widehat s_t$, next observed vector $s_{t+1}$, and feature count $d$,

$$
a_t^{fast}=\arg\max_a[p_t(a)+0.15Q_t(a)],\qquad
e_{t+1}=d^{-1}\lVert s_{t+1}-\widehat s_t\rVert_1,\qquad
\bar e_{t+1}=0.8\bar e_t+0.2e_{t+1}.
$$

The residual is unavailable until the next observation. It can signal delayed evidence, an inaccurate model, noise or changed dynamics; it does not identify a unique cause or establish safety. Hidden state and expected effects reset between independent episodes. An accepted planner action updates the expected effect for the action actually selected.

Recruitment is requested by positive fitted incremental utility, current insufficient-evidence flag, or residual above 0.25. The latter two triggers are declared policy choices, separate from learned value of computation. A zero-positive learned gate can therefore coexist with manual evidence-triggered planner calls. Default limits are at most three calls, 10,000 ms proposal TTL constrained by current deadline, and up to 150 ms waiting per policy call. While a proposal is pending, affected work is held; these holds consume elapsed time and decision calls but are not software transitions or action steps.

A changed observation binding invalidates pending work. Arrival still undergoes decision validation and fresh executor admission. Urgent registered cancellation can act during a hold. Expired proposals are discarded; background jobs are drained and their work recorded even when unused. This mechanism improves neither biological fidelity nor task success by definition: responsiveness and useful planner application require measurement.

Dependencies are `M05.pt`, `M08.pt`, `M12.gate.joblib`, source configuration and a configured real planner. The gate uses the disjoint executed-branch protocol described in [M11](11-learned-deferral.md). Diagnostics include recurrent confidence, effect predictions, residual/EWMA, expected gain, delegation trigger, holds, proposal application/discard and ablation flags. The release manifest supplies actual hashes and calibration evidence.

Required comparisons remove memory, effects, deferral and policy-level invalidation separately, retaining the common broker in every run. A no-invalidation ablation cannot disable capability or revision enforcement. Match cost and step budgets; report decision calls separately because holds inflate them without advancing software. Compare against M01, M09 and M10, and keep structural transfer distinct from familiar-template seeds. Negative results challenge the integration hypothesis rather than being relabeled as emergence.

Implementation: [control.py](../../python/reflexmesh/policies/control.py), [routing.py](../../python/reflexmesh/learning/routing.py), [runner.py](../../python/reflexmesh/evaluation/runner.py). Scientific motivation: [Daw et al., 2005](https://doi.org/10.1038/nn1560), [Lee et al., 2014](https://doi.org/10.1016/j.neuron.2013.11.028), [Kool et al., 2016](https://doi.org/10.1371/journal.pcbi.1005090). The implemented combination is an engineering hypothesis, not a neural model established by those studies.
