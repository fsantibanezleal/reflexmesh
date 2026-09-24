# M02: reactive behavior-tree policy

M02 uses an independent tree interpreter with condition, action, sequence and selector nodes. It does not call M01's recommendation function. Conditions inspect the current observation; an action searches current admissible candidates for its registered operation. Every tick emits a node trace with actual `success`, `failure` or `running` statuses. The root is reactive: relevant conditions are reconsidered on every decision.

For child status sequence $s_1,\ldots,s_n$, the control-flow semantics are

$$
\operatorname{Sequence}(s)=\text{first }s_i\ne\mathrm{success},\qquad
\operatorname{Selector}(s)=\text{first }s_i\ne\mathrm{failure}.
$$

If no such child exists, a sequence succeeds and a selector fails. A condition returns success or failure. An available action returns running with a candidate identifier; unavailable actions fail. `Running` here means that this tick selected an action to dispatch. The tree itself does not own the asynchronous effect lifetime: the common runtime and next environment observation supply that information. This distinction prevents reading tree status as verified completion.

The declaration contains guarded subtrees for freshness, duplicates, dependency resolution and cancellation before task-specific subtrees. Build repair, HTTP status reconciliation and extraction/transformation/reporting have explicit conditional branches. Other subtrees select retry, compare-and-swap recovery, validation and registered operation families. These are authored task semantics rather than a learned universal planner.

No checkpoint or probability calibration exists. Tree structure, guard ordering and source hash are the effective model configuration. Python inference requires base dependencies. The output has reason `reactive_behavior_tree`, `tree_status` and `node_trace`; the latter allows a reviewer to distinguish failure of a condition, an absent candidate and a dispatched action. Independent native receipts and task predicates remain the completion evidence.

A meaningful comparison against M01 preserves equivalent available operations and authority. Similar results are unsurprising when both cover the same task vocabulary. Perturbations should test changes during a running effect, a prerequisite invalidated between ticks, absent actions and cancellation. Removing reactivity or a specific subtree tests a stated mechanism; replacing the tree with an invocation of M01 would invalidate the comparison. Structural transfer with new compositions is especially relevant because a fixed subtree vocabulary may fail even when individual tools are familiar.

Implementation: [rules.py](../../python/reflexmesh/policies/rules.py), [controller.py](../../python/reflexmesh/controller.py). Primary implementation reference: [BehaviorTree.CPP introduction](https://www.behaviortree.dev/docs/intro/) explains the wider behavior-tree design family. Reflexmesh implements its own small interpreter and does not claim complete BehaviorTree.CPP feature parity, real-time guarantees or a neuroscience mapping.
