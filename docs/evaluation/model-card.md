# ReflexMesh controller model card

**Scope:** bounded selection and coordination of explicitly registered software actions. These controllers are not general-purpose pretrained foundation models, operating-system sandboxes, or demonstrations of AGI. Every method uses the same observed candidate contract and native execution authority.

**Current experiment status:** corrected collection, fitting and all 19,488 evaluation episodes completed with captured stage provenance and frozen installed-wheel lineage. The independent September 23 audit passed. Read the [complete corrected results and negative findings](corrected-results.md): M12 achieved 1,160/1,200 core and 21/24 structural successes; its four core ablations matched its success and both learned gates selected never-defer. The original tranche remains superseded following the [causal scenario audit](causal-scenario-audit.md).

## Intended use and boundaries

Use the package for inspectable local control experiments and for selecting among capabilities an application has explicitly registered. A candidate's preference score cannot grant capabilities, change its argument schema, or override a resource revision. Model decisions pass through fresh native admission. The runtime controls cooperating registered effectors; executing arbitrary hostile Python or shell code requires an external isolation boundary.

The owned workloads create disposable files, local HTTP services and child processes. Their fault schedules are designed experiments. They do not establish failure probabilities, latency guarantees, or autonomous competence in an arbitrary user's workspace. Structural transfer tests renamed capabilities and four new transformation compositions, separately from the 20 familiar workload templates.

## Data, grouping and labels

| Purpose | Environment seeds | Scheduled episodes | Use |
|---|---|---:|---|
| Base fit | 10000–10004 | 600 | Candidate imitation, recurrent imitation, logged executed rewards and transition targets |
| Base validation | 20000 | 120 | Independent transition-error reporting |
| Base calibration | 30000 | 120 | Candidate probability/temperature calibration |
| Router fit | 50000 | 120, up to two sampled states each | Actual reset fast/slow action comparisons |
| Router probability calibration | 60000 | 120, up to two states each | Disjoint calibration of positive incremental utility |
| Router threshold selection | 70000 | 120, up to two states each | Choose a threshold including never-defer |
| Core held-out evaluation | 40000–40009 | 1,200 per method | Direct measured execution, excluded from fitting |
| Structural transfer | 81000–81002 | 24 per method | Four unseen compositions with opaque registered names; excluded from fitting |

All 20 base templates and six variants occur in the base splits. Distinct seeds provide instance separation, not task-family holdout. Manifests record episode, group and seed identities; a group spans related families/variants for one environment seed. The code checks cross-split overlap before fitting. Candidate features omit family/seed identifiers, event and goal IDs, filenames, teacher labels and future outcomes. Public goal-description overlap and registered operation/state interactions are permitted inputs.

Training exploration executes a sampled action with epsilon 0.15 and records the behavior propensity. Candidate demonstration labels are separate from actual executed actions and outcomes. A preferred teacher action is not a certified safe action or necessarily a unique successful action. Transition targets are 27 observed next-state values, observed reward and observed episode termination. Unresolved task success stays `None`; it is not changed into failure to form a target.

PPO uses fresh actual software environments during learning, sampling the 120 training specifications at seed 10000. Its 16,384 requested transitions are separate from the offline 600-episode dataset. The checkpoint and report contain actual achieved transition counts.

## Parameters and artifacts

| Method | Fitted object and configuration | Published artifact |
|---|---|---|
| M01 | Explicit maintained guards; no fitting | Python rule source |
| M02 | Independently declared reactive behavior tree; no fitting | Python node/interpreter source |
| M03 | Logistic candidate classifier, C=5, liblinear, max600 iterations, seed17; separate Platt map | `M03.joblib`, `M03.linear.json`, `classical-training.json` |
| M04 | XGBoost160 trees, depth5, learning rate0.08, histogram training, two CPU threads, seed17; separate Platt map | `M04.joblib`, `M04.xgboost.json`, `M04.trees.json` |
| M05 | GRU32 over27 state features; candidate934→Tanh48; joint80→Tanh48→1; Adam0.003,25 epochs, sequence batch32, norm cap1, seed17 | `M05.pt`, `M05.onnx`, metadata/recurrent training report |
| M06 | LinUCB projected934→64 with seeded Rademacher matrix, alpha0.2, identity ridge initialization, seed17; logged executed-action reward updates | `M06.npz`, classical training report |
| M07 | Nonrecurrent MaskablePPO actor/critic64,32; rollout256, batch64, five epochs/update, learning rate0.0005, gamma0.95, seed29 | `M07.zip`, complete stored PPO parameters, optimizer and system information, `ppo-training.json` |
| M08 | Transition934→ReLU64→ReLU48→29; Adam0.003,40 epochs, batch128, seed31; horizon2, max24 second-level expansions | `M08.pt`, `transition-training.json` |
| M09 | External pretrained local `qwen3.5:4b`; no local weight fitting | Configured Ollama model digest and actual request diagnostics |
| M10 | Explicit asynchronous M01/planner composition; no local model fitting | Source plus configured model digest/request records |
| M11 | M03 fast branch, positive-gain classifier, Ridge gain regression, separate probability calibration and threshold selection | `M11.gate.joblib`, paired JSONL, routing training report |
| M12 | M05 recurrence, M08 effects, action-aligned M12 benefit gate, residual memory, explicit surprise/evidence request rule and bounded asynchronous arrival control | `M12.gate.joblib` plus component artifacts and routing training report |

M03/M04 inference loads cached Rust scorers from JSON exports without importing their training frameworks. Native/Python export parity is measured separately. M05 has a measured ONNX export; canonical recurrent inference uses PyTorch on CPU. The ONNX export is not evidence that all recurrence runs inside Rust. M06 uses NumPy; M07/M08/M12 use the declared Python scientific engines. Heavy-policy factory inference sets two PyTorch CPU threads. Device and library versions are captured in each actual run, rather than inferred from GPU availability on the host.

## Calibration and delegation meaning

M03/M04 probabilities target binary candidate demonstration preference. M05's masked softmax targets the teacher's selected candidate; a temperature is chosen on the separate calibration split from `exp(linspace(-2,2,41))`. Calibration reliability on these designed instances is not a distribution-free safety or error guarantee. Utilities from M06/M08/M12 are not relabeled as probabilities.

For routing, a reset branch executes the chosen action and then a bounded shared teacher continuation. Its declared utility is verified success plus first-action reward minus 0.005 per executed step. The slow branch also pays 0.005 per measured planner second. This is a one-decision experimental target, not a monetary objective or the causal effect of replacing an entire controller. Both branch starts must match the original complete candidate-feature matrix and causal history. Mismatched reset scheduling is identified **before** the selected action's outcome, retained with observed cost/outcomes, and excluded from fitting. Hidden operating-system/process state is not claimed to be cloned.

The positive-gain classifier uses an empirical constant when the fitting labels contain one class. That is reported, rather than inventing examples or forcing a logistic model. Ridge estimates gain magnitude. Calibration and threshold selection use separate groups. The selector includes threshold1.01, representing never-defer. M12 additionally requests deliberation for declared insufficient evidence or sufficiently large observed effect surprise; this remains an explicit design rule even when its learned gate never delegates.

## Provenance and reproducibility

The canonical machine-readable record is `artifacts/training-provenance.json`, accompanied by `artifacts/training-runs/collect.json`, `train.json`, `ppo.json`, and `gates.json`. The stage records begin before stage imports/execution and end after completion or failure. They capture arguments, timestamps, exact installed source/native manifest/binary, dependencies, input/output data and checkpoint hashes. A changed source, native binary or dependency set invalidates the stage. Retried stage records are retained under digest-qualified names.

Training configuration written into the completed stage/checkpoint report is captured evidence. A value inspected from a stored estimator/checkpoint afterward is identified as such. Configuration reconstructed from source is not retroactively described as captured at fitting time. In particular, the original archived fitting run lacked source snapshots at launch and preceded C06 receipt handling and later scenario corrections; that limitation remains in its archived provenance. The corrected experiment addresses this prospectively.

Evaluation records have their own exact frozen lineage. Resumption rejects changed relevant source, native source/binary, dependencies, model checkpoints, specifications or configured planner identity. Source-control commits are useful release identities but do not replace the executed wheel and artifact hashes. Actual local model requests retain provider usage only when available; failed or malformed requests are not assigned invented token counts.

## Interpretation and limitations

Report complete coverage, task success, policy/effect/wall latency separately, contract violations, unresolved effects, planner requests and accepted/stale/unused completions. The 20 templates can favor maintained rules; a strong M01 baseline is an important comparison, not an inconvenient result. Small held-out seed groups and designed fixtures limit statistical generalization. Structural transfer and four fixed-gate M12 ablations are separate evidence, with their own denominators and lineage.

The system integrates established learning/control mechanisms. Any contribution claim must survive the measured baselines and ablations. Neither component presence, a zero observed violation count, a microsecond scorer benchmark, nor a positive result on this suite establishes AGI, biological equivalence, a universal safety guarantee, or useful learned delegation in production.
