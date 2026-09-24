# M05: GRU behavior policy

M05 maintains a 32-dimensional hidden state separately for each goal. A GRU consumes the current normalized observed-state vector; candidate features enter a separate 48-unit tanh projection. Concatenating candidate encoding with the recurrent state gives a 48-unit tanh scoring layer and a scalar logit. The implementation reads causal state predicates that reflect previous outcomes; it does not append an unrestricted transcript, retrieve arbitrary long-term memory or observe hidden evaluator truth.

For normalized state $s_t$, previous hidden state $h_{t-1}$, candidate vector $x_{t,a}$, scoring network $q$, candidate projection parameters $W_c,b_c$, and calibrated temperature $T>0$,

$$
h_t=\operatorname{GRU}(s_t,h_{t-1}),\quad
\ell_{t,a}=q([\tanh(W_cx_{t,a}+b_c);h_t]),\quad
p_{t,a}=\frac{e^{\ell_{t,a}/T}}{\sum_{b\in A_t}e^{\ell_{t,b}/T}}.
$$

Behavior cloning minimizes negative log probability of the demonstrated action. The default fit uses causal sequence unrolling, 25 epochs, Adam learning rate 0.003, batches of 32 episodes, gradient-norm cap 1 and seed 17. Temperature is selected by calibration-sequence negative log likelihood over 41 values from `exp(-2)` to `exp(2)`. Episode boundaries reset hidden state; online inference updates state but does not change learned weights.

`M05.pt` contains weights, hidden size and temperature. `M05.metadata.json` fixes feature/state order and export evidence. The ONNX opset-17 interface is `state[1,27]`, `candidates[1,K,934]`, `hidden[1,1,32]` to `logits[1,K]` and `next_hidden[1,1,32]` for the current feature schema. Candidate count is dynamic; batch size is one. Export checks counts 1, 3, 6 and 7 against PyTorch with maximum absolute tolerance `1e-4`. Actual parity belongs to the metadata, and does not establish browser execution parity.

Diagnostics expose temperature and hidden-state norm. Norm measures magnitude, not intelligence, conceptual content or confidence. Candidate softmax targets demonstrated choice. Observe calibration, task success, hidden reset behavior, long/short histories and delayed evidence. Resetting memory every decision is a useful ablation; shuffling future observations into a sequence is leakage. Teacher-distribution imitation may fail after the policy's own mistakes, a problem identified by [Ross, Gordon and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html). This implementation does not claim to implement DAgger.

Implementation: [recurrent.py](../../python/reflexmesh/policies/recurrent.py), [train.py](../../python/reflexmesh/learning/train.py). Architecture reference: [Cho et al., 2014](https://arxiv.org/abs/1406.1078). Biological recurrent dynamics motivate hypotheses, but neither the GRU nor these software tasks establish neural fidelity.
