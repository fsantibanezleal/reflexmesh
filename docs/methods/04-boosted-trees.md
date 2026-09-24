# M04: calibrated gradient-boosted candidate trees

M04 applies a fitted XGBoost classifier to the same causal state–candidate features and binary demonstration-preference labels as M03. Its distinguishing capacity is nonlinear partitioning and interaction discovery through an ensemble of trees. The model does not receive future task success, hidden evaluator state or a special privileged tool set. Runtime admissibility is enforced independently after the policy proposes an action.

Writing $f_k(x)$ for the already weighted leaf contribution of tree $k$, $b_0$ for the base margin and $\sigma$ for the logistic link,

$$
m(x)=b_0+\sum_{k=1}^{K}f_k(x),\qquad
p_{cal}(x)=\sigma(\alpha m(x)+\beta).
$$

The fitted ensemble has $K=160$ trees, maximum depth 5, learning rate 0.08, histogram training, `binary:logistic` objective, two CPU threads and seed 17. The exported leaf values already incorporate boosting contributions; the native scorer must not multiply by the learning rate again. Calibration uses a distinct partition and a logistic fit with `C=10`, `max_iter=300`, on clipped raw log odds.

Artifacts are `M04.joblib`, `M04.xgboost.json` and `M04.trees.json`. The last format stores feature order, base margin, explicit child identifiers, split thresholds, missing-value branches, leaf contributions and calibration coefficients. The cached Rust scorer traverses those trees using packed float32 feature rows. The Python reference uses XGBoost. Feature-order validation and prediction parity are release requirements, especially around split boundaries and missing-value behavior; a matching file extension is not evidence of semantic equivalence.

Emitted diagnostics include `probability_target`, `safety_bound: false` and the inference backend. Candidate scores are separately calibrated binary preferences rather than a normalized multiclass distribution. The evaluator reports actual task effects, not merely correct classification of the demonstrated choice. Training and warmed scoring times describe different workloads and must have separate units and boundaries.

Tree depth, dataset coverage and explicit operation indicators can make template memorization effective. Test held-out episode seeds separately from unseen compositions and opaque tool names. Ablate interaction features and calibration; compare against M03 on the same split and matched controller budget. A small classification gain may increase unsafe proposals or fail to improve task success after executor rejection. Calibration fitted on one population cannot guarantee reliability under arbitrary new software, malicious tool text or an altered action registry.

Implementation: [classical.py](../../python/reflexmesh/policies/classical.py), [train.py](../../python/reflexmesh/learning/train.py). Primary research: [Chen and Guestrin, 2016](https://arxiv.org/abs/1603.02754), [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html). These establish the algorithm families, not results for the owned workflow suite.
