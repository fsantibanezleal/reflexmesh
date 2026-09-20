# M08: learned transition model with bounded lookahead

M08 fits an action-conditioned transition regressor and uses its predictions in a short lookahead. The network maps the common candidate vector through 64 and 48 ReLU units to 29 outputs: 27 normalized next-state features, observed one-step reward and observed episode termination. The last output targets termination, not success probability. An unresolved nonterminal outcome is not relabeled as task failure.

For fitted transition $\widehat f$, reward $\widehat r$, terminal output $\widehat d$, current observation $o$, candidate $a$, and the available candidate set $A(o)$, the implemented two-level utility is

$$
\widehat Q_2(o,a)=\widehat r(o,a)+0.8[1-\operatorname{clip}(\widehat d(o,a),0,1)]
\max_{b\in A(o)}\widehat r(\widehat f(o,a),b).
$$

The network predictions replace state values in a hypothetical observation. Candidate schemas remain fixed during the short search, so the expression is not a complete simulator of newly created actions, permissions or resource revisions. Default horizon is two and the second-level expansion cap is 24. Setting horizon one removes the additional lookahead. Larger numeric horizon values do not implement arbitrary-depth tree search in the shipped two-level code.

Training uses only executed transitions. It minimizes mean squared error across state, reward and terminal targets, with equal per-output weighting. Defaults are 40 epochs, Adam learning rate 0.003, minibatches of 128 and seed 31. Validation reports overall, per-state-feature, reward and terminal MSE. Those diagnostics are distinct from rollout success and do not establish calibration. `M08.pt` contains the transition weights; the canonical factory uses CPU PyTorch with bounded thread configuration.

Decisions report `effect_predicted`, `predicted_reward`, horizon, actual expansion count and `scores_are_utility: true`. Prediction residuals must pair a forecast with the subsequent observed state, not with the forecast's input. A final observation absent from the record remains absent. Regression outputs may leave a binary feature's natural range; clipping the terminal contribution is a search convention, not a probability calibration result.

Compare horizon one versus two at matched inference budgets, remove the prediction head, and contrast against exact recovery and model-free policies with equivalent history. An oracle transition model is useful only as an explicitly labeled diagnostic. Model error can compound and make a deeper predicted plan worse; familiar operation indicators may mask failures on novel compositions. The forward-model analogy to cerebellar adaptation is a design inspiration, not biological validation of this network.

Implementation: [control.py](../../python/reflexmesh/policies/control.py), [train.py](../../python/reflexmesh/learning/train.py). Scientific context: [Daw, Niv and Dayan, 2005](https://doi.org/10.1038/nn1560), [Tseng et al., 2007](https://doi.org/10.1152/jn.00266.2007). Their bounded empirical claims concern learning/arbitration and human reaching, respectively; this package supplies separate software evidence.
