# M07: nonrecurrent masked PPO

M07 is a trained policy-gradient baseline using SB3-Contrib MaskablePPO. It is explicitly nonrecurrent. The Gym adapter orders candidates by identifier, represents up to seven admissible candidates in a padded feature matrix, and flattens that matrix for a multilayer perceptron. The action mask marks actual admissible slots. The selected index maps back to a registered candidate; native admission independently checks the proposed effect.

For binary availability mask $m_a$, policy logits $z_a$, probability ratio $r_t$, estimated advantage $\widehat A_t$, and PPO clipping width $\epsilon$, the relevant objectives are

$$
\pi_m(a\mid o)=\frac{m_a e^{z_a}}{\sum_bm_be^{z_b}},\qquad
L^{clip}=\mathbb E_t\min\left(r_t\widehat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t\right).
$$

The ratio compares new and behavior-policy probabilities under the same action-mask convention. An empty mask produces a stop decision. A validity mask cannot predict every external side effect or grant a capability that was not configured.

The training function uses separate actor and critic networks with hidden sizes 64 and 32, 256 rollout steps, minibatches of 64, five optimization epochs per rollout, learning rate 0.0005, discount 0.95 and seed 29. Its default request is 16,384 environment steps on CPU; the final report records actual steps and timing. Library defaults not overridden by the code, including the clipping implementation, are identified through the installed dependency provenance. `M07.zip` is the model checkpoint; `ppo-training.json` records the environment and training configuration. Inference is deterministic and emits the actual mask with `recurrent: false`.

The reward is experimental utility: independently verified terminal success, contract violation, observed failure and incomplete progress have distinct values. Training uses real disposable software through the Gym adapter, not an invented differentiable simulator. Termination and truncation remain distinct. Neither high return nor action agreement substitutes for verified task success and violation reporting.

Candidate padding introduces a fixed-capacity limitation; a larger tool set requires a changed representation and new training/evaluation. Candidate ordering can also become a shortcut. Test alternative ordering, longer dependency horizons, perturbation families and multiple independent training seeds. A memory ablation is inapplicable to this nonrecurrent network; compare against M05 or add a separately identified recurrent PPO method instead. Small seed counts and correlated task families constrain uncertainty claims.

Implementation: [rl.py](../../python/reflexmesh/policies/rl.py), [gym.py](../../python/reflexmesh/environments/gym.py). Primary sources: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347), [Huang and Ontanon, 2020](https://arxiv.org/abs/2006.14171), and [MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).
