# M03: calibrated logistic candidate scorer

M03 learns a binary preference score for each state–candidate pair. Its target is agreement with a demonstrated preferred action, not the probability that an action is safe or that an entire workflow will succeed. The feature vector includes observed state, candidate properties, public operation indicators and state–operation interactions. Family identifiers, filenames, future outcomes and evaluator labels are excluded. Interactions matter: a state-only additive term would cancel when ranking candidates from the same observation.

With candidate features $x=\phi(o,a)$, fitted coefficients $w,b$, calibration coefficients $\alpha,\beta$, and logistic function $\sigma(z)=(1+e^{-z})^{-1}$, prediction is

$$
p_{raw}=\sigma(w^\top x+b),\qquad
p_{cal}=\sigma(\alpha(w^\top x+b)+\beta),\qquad
a^*=\arg\max_{a\in A(o)}p_{cal}(o,a).
$$

Each $p_{cal}$ is a binary candidate-preference estimate. Scores across a candidate set need not sum to one. The reported confidence is the selected candidate's score. Ties use the first maximum in admissible order; changing candidate order therefore remains a useful robustness check.

Training uses scikit-learn `LogisticRegression`, L2 regularization, `C=5`, `solver='liblinear'`, `max_iter=600`, and seed 17. A separate logistic calibration fit uses `C=10`, at most 300 iterations and only the calibration partition. Whole episode groups must be separated before fitting. `classical-training.json` records observed candidate counts, duration and calibration measurements; no number is implied by the method definition.

Checkpoints include `M03.joblib` for the Python reference path and `M03.linear.json` for the cached Rust scorer. The JSON carries feature order, coefficients, intercept and calibration. Native loading verifies feature schema; prediction receives packed little-endian float32 feature rows. The canonical factory requests the native backend. Export/parity results must accompany any equivalence claim; warmed scorer microbenchmarks omit shared feature extraction and effect dispatch.

Observe Brier score, log loss, reliability bins, action confusion, complete task success and policy latency separately. Calibration can improve probability interpretation without changing ranking. Distribution shift, missing state and teacher bias can defeat an apparently reliable scorer. Ablate state–operation interactions, calibration and candidate descriptions while retaining the same data partitions. Compare with exact rules and boosted trees, rather than treating a learned linear score as inherently more general.

Implementation: [classical.py](../../python/reflexmesh/policies/classical.py), [train.py](../../python/reflexmesh/learning/train.py), [features.py](../../python/reflexmesh/features.py). Scientific reference: [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html) motivates separate probability calibration and evaluation; its reported neural-network results are not performance claims for this controller.
