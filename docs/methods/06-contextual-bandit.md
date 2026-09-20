# M06: shared action-feature LinUCB

M06 is a shared linear contextual bandit over candidate-conditioned features. It is not the disjoint per-action LinUCB parameterization. A fixed seeded random-sign projection reduces the common feature vector to 64 dimensions, scaled by $1/\sqrt{64}$. All actions share one regularized covariance inverse and one reward accumulator. The default exploration coefficient is 0.2; training constructs the projection with seed 17.

Let $x_{t,a}$ be a projected candidate vector, $A$ the regularized feature covariance, $b$ accumulated observed rewards, and $\alpha$ the exploration coefficient. Selection uses

$$
\widehat\theta=A^{-1}b,\qquad
a_t=\arg\max_{a\in A_t}\left[x_{t,a}^{\top}\widehat\theta+
\alpha\sqrt{x_{t,a}^{\top}A^{-1}x_{t,a}}\right].
$$

Here $A_t$ denotes admissible actions and should not be confused with the covariance matrix $A$. On an actually executed vector $x$ with observed reward $r$, the implementation applies exact Sherman–Morrison updates:

$$
A^{-1}\leftarrow A^{-1}-\frac{A^{-1}xx^\top A^{-1}}{1+x^\top A^{-1}x},\qquad b\leftarrow b+rx.
$$

Initialization is identity covariance and zero reward accumulator. Offline fitting updates only the executed action from exploratory logs and its real observed one-step reward. It does not supply counterfactual rewards for unchosen actions. Evaluation loads and freezes these statistics; `reset` does not retrain or zero the checkpoint. `M06.npz` preserves inverse covariance, accumulator, projection, exploration coefficient and update count. Prediction uses NumPy and the common native executor; no neural runtime is required.

UCB scores are mean-plus-bonus values, not calibrated probabilities. Diagnostics explicitly set `ucb_not_probability` and report update count. Deterministic first-maximum selection has propensity one for its selected action, which does not supply overlap for off-policy evaluation of arbitrary alternatives. Exploration-policy propensities in training logs have a different meaning.

The objective is myopic reward under a linear model. A repair that costs a step but enables later success can be misranked without suitable context or a task-specific reward signal. Compare immediate reward and complete verified task success; evaluate delayed feedback, feature aliasing and new operation compositions. Ablate the bonus with `alpha=0`, vary projection size, and check numerical symmetry/positive covariance behavior after many updates. Those interventions test this implementation rather than proving contextual-bandit regret assumptions in a nonstationary software workflow.

Implementation: [classical.py](../../python/reflexmesh/policies/classical.py), [train.py](../../python/reflexmesh/learning/train.py). Primary algorithm context: [Li et al., 2010](https://arxiv.org/abs/1003.0146). Its recommendation experiments are not evidence of this controller's software-task performance.
