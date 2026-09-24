# Corrected evaluation: measured result and contribution verdict

All 19,488 corrected episodes completed on September 21 and passed an independent identity/hash audit on September 23: 14,400 core, 288 structural transfer and 4,800 component ablations. The frozen wheel's native binary SHA-256 is `f1ed93084ccb5081e247bb14952d17a7ab3c162a8f940c993bd96aa84be1880c`. The [machine-readable report](corrected-results.json) binds the exact index and lineage digests, captured fitting stages, comparisons and upstream run inventory.

The software integration is implemented. This experiment does **not** establish a new learning algorithm, AGI, biological equivalence, or a general performance advantage for M12. Several maintained and fitted baselines solve every core episode. M12 solves 1,160 of 1,200, and its four fixed-gate ablations solve exactly the same number. These negative findings are retained; held-out data were not used for retuning.

## Complete comparison

| Method | Core success / 1200 | Policy p50 / p95 ms | Mean wall ms | Structural success / 24 |
| --- | ---: | ---: | ---: | ---: |
| M01 | 1200 | 0.0219 / 0.0394 | 160.56 | 0 |
| M02 | 1200 | 0.0689 / 0.1177 | 183.68 | 0 |
| M03 | 1200 | 2.5695 / 4.2114 | 189.87 | 0 |
| M04 | 1200 | 2.6327 / 4.0172 | 192.90 | 24 |
| M05 | 1200 | 3.1890 / 4.4434 | 183.24 | 0 |
| M06 | 253 | 2.5673 / 4.0432 | 262.28 | 12 |
| M07 | 1170 | 2.3607 / 2.9987 | 156.36 | 0 |
| M08 | 664 | 9.3875 / 14.0328 | 325.01 | 0 |
| M09 | 595 | 1175.2670 / 1718.1800 | 13081.51 | 22 |
| M10 | 1200 | 0.0641 / 0.1588 | 1314.88 | 0 |
| M11 | 1200 | 1.8156 / 3.1422 | 147.05 | 0 |
| M12 | 1160 | 18.1303 / 158.9701 | 332.77 | 21 |

Policy latency is measured per decision and excludes actual execution. Wall latency is measured per episode and includes execution and planner effects. Rates and times are not combined into a synthetic score. M09 used 12,341 core planner starts and had five invalid-policy-output episodes; M10 started one background request per episode. M12 started 140 core delegations. Failed and invalid episodes remain in every denominator. No declared-contract violation was recorded in these runs; that is a scoped observation, not a safety guarantee.

## Core failure analysis and paired uncertainty

All 40 M12 core failures are in C07, long-job cancellation. Compare their exact trace decisions and receipts before attributing a mechanism; the aggregate does not establish why any one component failed. M12 minus M01 core success is -0.033333. Its ten seed-group means are identical, so the paired seed-bootstrap 95% interval is the same point. That degenerate interval reflects the repeated fixture structure, not certainty about production performance. The machine-readable report gives every baseline comparison and preserves independent-episode descriptive intervals in the canonical index.

## Allocation and ablations

Both M11 and M12 fit on 228 paired states with seven positive incremental-utility labels; 228 disjoint calibration states also contain seven positives, and threshold selection uses another 228 states. No pair was excluded for mismatched starting features/history. Both selected threshold **1.01**, meaning never defer through the learned gate. M12's separately declared surprise/evidence trigger can still request a planner; those requests are not evidence that the learned gate is useful.

Each of no-memory, no-effects, no-deferral and no-invalidation completed 1,200 exact matched core cells. Full-minus-ablation success is zero for every comparison. Mean wall overhead of the full system is approximately 2.59, 39.28, 131.89 and 5.00 ms respectively. These sequential run timings and unequal compute budgets do not justify a causal latency ranking. Fixed-gate ablations test this fitted configuration; they do not estimate the best refitted system without each component.

## Structural transfer

Four unseen compositions, two variants and three seeds give 24 episodes per method. M04 solves 24, M09 solves 22 and M12 solves 21. M12 has two invalid-output episodes on this set; M09 also has two. This is useful evidence that the full controller can handle some renamed and recomposed registrations, but the boosted-tree baseline is stronger here. The tiny designed sample cannot establish broad generalization. Core and structural scores are never pooled.

## Model and export diagnostics

The transition model's held-out observed-target MSE is 0.00710938, with reward MSE 0.01807934 and terminal MSE 0.01468785. Termination is not success probability. Per-state errors are available in the JSON report. Candidate reliability and confusion in the workbench measure demonstration-action agreement, not action safety or eventual task success.

Fresh corrected native parity checked 484 vectors per M03/M04, including adjacent float32 values around all 76 exported split thresholds. Maximum probability errors are 3.14e-14 and 3.79e-7, below 1e-5. Warm seven-candidate Rust scoring p50 is 0.0090/0.0185 ms and p95 is 0.0118/0.0231 ms; Python references are 0.1091/0.2917 ms p50. This excludes common feature construction, fsync and effect execution. There is no end-to-end microsecond or hard-real-time claim.

## External upstream evidence

The pinned ToolSandbox subset completed eight cases with mean upstream similarity 0.954604. The pinned tau2 retail subset completed four cases with mean upstream reward 0.75. The selected corrected runs are explicitly identified, and all earlier failed/interrupted runs remain in the inventory. These are a local System Two agent plus native admission, using official upstream evaluators; they are not full leaderboard runs or learned-policy transfer. Reward may disagree with action checks, so the report retains each task's reward basis and failed-check count.

Native acting admission across 47 upstream calls has p50 0.1124 ms and p95 0.22064 ms. It excludes model inference, fingerprints, durable logging and effects. The upstream task elapsed metric includes evaluator time. ToolSandbox and tau2 have separate licenses and protocols; their scores are not comparable to owned core success.

## Reproduction and release boundary

Run `python scripts/audit_science.py` against the complete experiment tree, then `python scripts/export_science_report.py`. Model assets are deterministic, contain 18 allowlisted files and carry individual and archive SHA-256 digests. Fitting stage provenance, canonical indexes and lineage are published with the experiment records; superseded fixture results remain excluded. An installed wheel, a validated experiment, a PyPI publication and a live website are distinct gates.
