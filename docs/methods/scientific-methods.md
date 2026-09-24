# Executable controller methods

The comparison concerns bounded software control, not a claim of general intelligence. Every method receives the same observed state and registered candidates. Every selected effect passes through the same native broker and actual disposable workspace. Candidate preference never grants authority.

| ID | Actual engine | Fit and prediction |
|---|---|---|
| M01 | Explicit guarded finite-state policy | Ordered causal state guards, exact registered operation selection. No fitted parameters. |
| M02 | Reactive behavior-tree interpreter | Selector, sequence, condition and action nodes with success/failure/running statuses. Independently declared task subtrees; not a call to M01. |
| M03 | scikit-learn LogisticRegression | L2, C=5, liblinear, seed17, max600 iterations. Binary candidate demonstration labels, followed by disjoint Platt calibration. |
| M04 | XGBoost XGBClassifier | 160 trees, depth5, learning-rate0.08, histogram training, seed17, two CPU threads. Same labels and separate Platt calibration. |
| M05 | PyTorch recurrent behavior cloning | 27-dimensional observed state to GRU32; 934-dimensional candidate to Tanh48; concatenate recurrent state and candidate projection; Tanh48 and scalar logit. Masked candidate softmax. |
| M06 | Shared action-feature LinUCB | Fixed seeded Rademacher projection934 to64, ridge identity initialization, alpha0.2, exact Sherman-Morrison inverse updates on executed actions and their observed rewards. Frozen at test. |
| M07 | sb3-contrib MaskablePPO | Nonrecurrent MLP actor/critic64,32; padded seven-candidate input; native-enforced real software environment;256 rollout steps, minibatch64, five epochs, learning-rate0.0005, gamma0.95, seed29. |
| M08 | Learned transition plus bounded search | PyTorch934-64-48-29 ReLU network predicts27 next-state features, observed reward and termination. Horizon2, maximum24 second-level expansions. |
| M09 | Structured local language-model planner | Actual Ollama model request for every decision, dynamic candidate schema, fixed temperature0 and seed42. No mock model fallback. |
| M10 | Asynchronous FSM plus planner | A real background proposal overlaps M01 actions. Global observation/resource binding invalidates stale proposals. Completed unused work still incurs recorded cost. |
| M11 | Calibrated learned deferral | M03 action versus actual planner action, executed in independently reset copies. Logistic positive-gain classifier, separate Platt calibration, Ridge gain magnitude and disjoint threshold selection. |
| M12 | Persistent outcome-aware metacontroller | GRU plus learned-effect utility, observed prediction residual and EWMA, action-aligned learned benefit gate, bounded asynchronous planner recruitment, affected-work hold, arrival validation, TTL and urgent cancellation. |

## Observation and features

`Observation` contains event/goal identifiers, revision, causal state, candidates, capabilities, budget, deadline and past actions. Feature extraction ignores family, seed, event/goal IDs, filenames, evaluator labels and future outcomes. The934 inputs comprise27 state features,11 context/candidate features,32 operation indicators and864 state-operation interactions. The latter allow a linear scorer to condition action preference on state rather than adding a state-only term that cancels across all candidates. Goal-description token overlap is public semantic input, not a teacher label. Candidate order is shuffled between episodes and steps.

The GRU keeps state per goal and resets between independent episodes. It is trained with causal sequence unrolling, Adam0.003, gradient norm cap1, batch32 and25 epochs. A scalar temperature is selected from41 logarithmically spaced values on the separate calibration sequences. `M05.metadata.json` records actual export parity. The ONNX interface uses float32 `state[1,27]`, `candidates[1,K,934]`, `hidden[1,1,32]`, producing `logits[1,K]` and `next_hidden[1,1,32]`. Candidate count is dynamic; batch size is fixed at one.

## Reward, dynamics and routing targets

The environment reward is1 for independently verified terminal success, -1 for a contract violation, -0.1 for an observed failed action and -0.01 for other incomplete transitions. These are declared experimental utilities, not real monetary costs. A nonterminal `success=None` is unresolved task success. The transition model predicts the separately observed `terminal` flag; unresolved success is never relabeled failure.

M08 estimates `r(x,a) + 0.8 (1 - terminal(x,a)) max_b r(f(x,a),b)` with a bounded hypothetical search. Candidate schemas remain fixed within that short search; it does not synthesize new tools or prove the predicted next state is reachable. Scores are utilities and are not labeled probabilities.

For one routing decision, each reset branch executes the chosen action, then the same bounded teacher continuation. Let `U = verified_success + first_action_reward - 0.005 * executed_steps`. The paired target is `Delta = U_slow - U_fast - 0.005 * measured_planner_seconds`. This controlled target isolates a single routing choice under shared continuation. It is not the total causal value of replacing the full policy. The gate fits both `P(Delta>0 | observed_gate_features)` and `E[Delta | observed_gate_features]`; delegation requires calibrated probability above the selected threshold and positive predicted gain. If training or calibration contains one class, the empirical constant model is retained and reported. Positive examples are never invented.

Base models use seeds10000-10004; base validation20000 and calibration30000. Routing fit50000, probability calibration60000 and threshold selection70000 are independent reset groups. M12 routing targets the exact recurrent/effect blended action, not the unmodified GRU action. Threshold selection reliability is descriptive; the held-out evaluation provides the final independent comparison.

M12 also has a declared insufficient-evidence/surprise trigger. It is a conservative policy choice, separate from the fitted value-of-computation estimate. The default surprise threshold is0.25 mean absolute normalized state residual. It makes no calibrated risk guarantee. The effect residual is only available after the next observation; EWMA is0.8 previous plus0.2 new residual.

## Planner arrival and effects

M12 holds affected work while a proposal is pending, polling for at most150ms per policy call. A hold is elapsed computation time, not a software transition, so it does not consume environment action steps. A10-second default TTL, current deadline and maximum three planner calls bound recruitment. Urgent cancellation may act during a hold and thereby invalidate the proposal. Every arrived proposal is checked against goal, observation revision, candidate resources and the native executor's fresh resource versions. The no-invalidation ablation removes the policy check only; it cannot remove broker authority.

M10 is an adapted asynchronous FSM/planner baseline, not a reproduction of DPT-Agent's complete agent, benchmark, training or reported results. Its short owned tasks may finish before the proposal arrives; this outcome is measured, not hidden. M12's cooperative hold is tested to apply repeated delayed proposals while preserving action-step budgets.

## Learning and novelty limits

This implementation uses offline controlled learning and frozen evaluation. It does not update model weights online in the user's workspace. LinUCB receives only rewards of actually executed actions; the exploratory data records the behavior propensity. The current benchmark reports direct execution, not an unsupported off-policy causal estimate. Deterministic evaluation does not provide overlap for arbitrary unchosen actions.

GRU, linear/tree scorers, PPO, learned dynamics, calibration, asynchronous planning and resource arbitration are established components. The system integration and its falsifiable hypotheses are the research contribution under test. Improvements must survive matched-budget baselines, the independent structural-transfer suite and component ablations. A perfect hand-written controller on the owned templates may leave no useful planner benefit; negative findings remain valid results.

Mamba, RWKV, liquid networks, spiking networks and neuromorphic hardware are reviewed research alternatives, not implemented methods mislabeled as comparisons. The delivered native paths and Python inference paths must be distinguished in the export/parity reports.

Loaded M03/M04 evaluation uses cached Rust scorers with packed little-endian float32 input. Export parity and warmed scoring measurements are in `artifacts/native-cached-parity.json`; those microbenchmarks exclude shared feature extraction and effect execution. Native scorers load JSON exports without importing the training frameworks. M01/M02/M06 and planner-only M09/M10 also support the base package dependency set. Recurrent, PPO, dynamics and fitted routing paths require training extras; their inference uses two PyTorch CPU threads in the canonical factory.
