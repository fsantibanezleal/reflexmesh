# M11: calibrated learned deferral

M11 selects between the actual M03 candidate and a real planner using a fitted incremental-utility gate. Gate labels come from separately executed reset branches, not planner self-ratings or copied candidate correctness labels. For each sampled state, a shared recorded prefix reconstructs the environment; each branch executes its own first action and then the same bounded M01 continuation. This design isolates one routing choice under that continuation rather than the value of replacing an entire policy.

For verified terminal success $S$, first-action reward $r_0$, total executed steps $N$, and measured planner seconds $T_p$, the declared experimental utility and paired target are

$$
U=S+r_0-0.005N,\qquad
\Delta=U_{slow}-U_{fast}-0.005T_p.
$$

These coefficients specify an experimental preference, not financial costs. A gate feature vector $z$ contains fast confidence, top-score margin, budget, normalized deadline, uncertainty, staleness, preceding-action failure and normalized history length. A positive-benefit classifier estimates $P(\Delta>0\mid z)$; a separate Ridge regressor estimates $E[\Delta\mid z]$. Dispatch to the planner requires probability above the selected threshold, positive predicted gain and remaining budget greater than 0.1.

Training uses independent reset groups for fit (seed 50000), probability calibration (60000) and threshold selection (70000). The classifier is logistic regression (`C=1`, liblinear, at most 500 iterations) when both classes exist; one-class data retain an empirical constant classifier. The gain model is Ridge with `alpha=1`. Probability calibration uses a separate logistic/constant model. Threshold selection checks 21 values from zero to one plus 1.01, sums actual selected paired gains, and favors the higher threshold on ties. A never-defer solution is legitimate; positive benefit examples must not be invented.

`M11.gate.joblib` stores gate and threshold; `M03.linear.json` or its reference checkpoint supplies the fast policy. `routing-training.json` records target, group seeds, counts and calibration evidence. Prediction emits benefit probability, expected incremental utility, threshold and delegation status. If recruitment selects an unavailable planner, a model-required error is retained rather than silently replacing it.

Evaluate paired net utility, referral coverage, calibration, full-task outcomes and total cost. A zero-positive training partition can make constant deferral behavior scientifically correct while leaving unseen planner-beneficial tasks unlearned. Test cost sensitivity, group transfer and a matched random gate; do not tune on held-out outcomes. [RouteLLM](https://arxiv.org/abs/2406.18665) is relevant routing research, but its preference-derived targets differ from this executed one-decision utility.

Implementation: [routing.py](../../python/reflexmesh/learning/routing.py), [control.py](../../python/reflexmesh/policies/control.py). Arbitration context: [Lee, Shimojo and O'Doherty, 2014](https://doi.org/10.1016/j.neuron.2013.11.028); calibration context: [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html). These references do not make the fitted gate a safety certificate.
