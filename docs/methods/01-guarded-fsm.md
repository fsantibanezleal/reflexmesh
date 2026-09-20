# M01: guarded finite-state policy

M01 is an explicit deterministic controller over observed software state. `GuardedFSMPolicy` calls an ordered operation-selection function, then selects a registered admissible candidate whose public operation matches. It has no fitted parameters, latent recurrent state or language-model fallback. `reset` is intentionally empty because its state resides in the current observation and controlled environment.

Let $o_t$ be the current observation, $A(o_t)$ its admissible candidate set, $g_j$ ordered Boolean guards and $u_j$ the operation attached to guard $j$. The implemented decision can be written

$$
j^*=\min\{j:g_j(o_t)=1\},\qquad
a_t=\operatorname{first}\{a\in A(o_t):\operatorname{operation}(a)=u_{j^*}\}.
$$

Here `first` follows the supplied candidate order; absent a matching supported operation, the controller returns mode `stop` with `no_supported_rule`. This equation describes an ordered rule system, not an optimal policy or a complete formal verification result. The guards first address stale evidence, duplicate delivery, missing dependency and cancellation. Task branches then cover validation, build recovery, ambiguous HTTP writes, retry, workflow prerequisites, resource conflicts, delegation and other explicitly registered cases. Unsupported tool families remain unsupported.

There is no training split, calibration fit or checkpoint. The operative configuration is the source revision plus observation/candidate schema. Inference runs in Python with the base dependency set; actual effects still pass through the Rust-backed admission runtime. Diagnostics record `operation` and `learned: false`; reason `explicit_state_guard` identifies selection. A rule priority is neither an empirical confidence nor a safety probability.

The important observables are verified task success, unsupported stops, action count, reaction latency, stale/repeated-effect prevention and completed native receipts. Teacher agreement can be circular if this same controller supplied the demonstration labels. Report independently checked effects alongside imitation accuracy. A perfect score on hand-designed templates may reflect explicit task knowledge and a narrow task distribution.

Useful tests remove or reorder one guard, alter observation delay, hide an important predicate, or introduce a genuinely new operation/composition. Cancellation tests must include an effect already in progress; a fast stop decision does not undo a completed write. Compare M01 with M02 to separate operational coverage from interpreter structure, and with learned methods under identical candidates and executor checks. Do not weaken its rules to manufacture a learned-model advantage.

Implementation: [rules.py](../../python/reflexmesh/policies/rules.py), [contracts.py](../../python/reflexmesh/contracts.py), [runtime.py](../../python/reflexmesh/runtime.py). Context: [Kool, Cushman and Gershman, 2016](https://doi.org/10.1371/journal.pcbi.1005090) motivates testing when extra computation pays for itself; it does not validate these software guards or identify them with a specific neural circuit.
