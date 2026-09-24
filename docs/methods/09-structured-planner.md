# M09: direct structured local language-model planner

M09 calls an actual configured model for every decision. The default local adapter uses Ollama and sends the causal observation, public candidates, capabilities and history. Its dynamic JSON schema restricts output to a registered candidate identifier or null, a mode (`act`, `wait`, `stop`) and a bounded reason. It does not allow the model to invent tool arguments. The parser checks exact keys, types, candidate membership and consistency between mode and identifier before returning a policy proposal.

Let $S(o)$ be the valid response schema for observation $o$, $L(o)$ the model's response and $V$ the independent response validator. The operative boundary is

$$
\widehat a=V(L(o),S(o)),\qquad
\operatorname{dispatch}(\widehat a)\text{ requires fresh executor admission}.
$$

Schema-valid output can still be semantically wrong. The equation expresses a validation pipeline, not a proof that a constrained model selects the correct action. File contents and tool outputs remain untrusted data even when they appear inside the planner context.

The local adapter defaults are model identifier `qwen3.5:4b`, 4,096 context tokens, 192 output tokens, temperature zero, seed 42, disabled thinking flag, 120-second request timeout and a 15-minute keep-alive request. The actual configured model and provider metadata are authoritative; these defaults neither install a model nor establish a reproducible hardware result. There is no implicit provider fallback. Requests above 1 MiB and responses above 2 MiB are rejected; unencrypted transport is restricted to loopback and redirects are not followed.

The package does not train the planner checkpoint. A local model identifier, digest/provenance and runtime configuration must accompany evaluation. Adapter output records provider-supplied token counts and durations, model identity and measured wall time. These are usage evidence, not calibrated confidence. M09 adds measured planner duration and a delegation marker to each decision; no invented probability is displayed.

Compare full-episode success, structured-output failures, unsupported choices, tool-effect correctness, prompt sensitivity, latency tails and total token use. Fixed seed and temperature do not guarantee bitwise reproducibility across runtimes or hardware. A slower model can still use shallow pattern completion; an LLM is not categorically psychological System 2. Candidate-only output narrows the task and must be disclosed when comparing with unrestricted coding agents. Deliberate malicious tool content must reach the planner in a real evaluation before injection-resistance claims are justified.

Implementation: [planners.py](../../python/reflexmesh/planners.py), [planning.py](../../python/reflexmesh/policies/planning.py). Research context: [ToolSandbox, Lu et al., 2024](https://arxiv.org/abs/2408.04682) motivates stateful tool-use evaluation; this owned suite does not reproduce its benchmark scores. [Bago and De Neys, 2017](https://doi.org/10.1016/j.cognition.2016.10.014) cautions against equating logical competence with a uniformly slow cognitive process.
