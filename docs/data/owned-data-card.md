# Owned workload and data card

The workload consists of designed integration cases executed in disposable local workspaces. It is not a production trace collection. No remote credentials, user documents or private repository contents enter the training data. Native admission, durable receipts, filesystem effects, local HTTP effects and child-process lifecycle are exercised by all methods.

| Family | Concrete effect and independent success witness |
|---|---|
| C01 document routing | Move a real source file; exact destination content and absent source. |
| C02 indexing | Write an index; independently recomputed current file SHA256 hashes. |
| C03 validation | Parse JSON and produce accept/quarantine artifact; check exactly the appropriate artifact against raw data. |
| C04 test recovery | Execute a disposable Python test after configuration repair; check successful result and configuration existence. |
| C05 HTTP retry | Local server emits one503 before success; independently check fetched seed-specific response. |
| C06 ambiguous remote write | Local server commits a POST and drops its response; reconcile through GET status; independently check exactly one committed write. |
| C07 cancellation | Cancel a real child before its delayed write; verify process-tree termination and absent late effect. |
| C08 restart | Start a failed-child replacement; inspect its produced result and terminated process tree. |
| C09 pressure | Enforce a declared workspace quota against actual cache/report files; verify cleanup and2048-byte output. This is not host disk exhaustion. |
| C10 dependency workflow | Extract, transform and report real JSON files; independently recompute final arithmetic. |
| C11 revision conflict | Rebase a concurrent shared file revision, then compare/write; inspect exact revision3 and value. |
| C12 job collection | Spawn actual compute children producing squares; independently check collected sum13 and completed processes. |
| C13 planner availability | Controlled2ms availability schedule plus result file. This family models availability; real LLM calls occur in planner methods separately. |
| C14 argument rejection | Write an explicit unsupported-argument rejection receipt, then invoke only a registered valid action. |
| C15 untrusted text | Inspect malicious-looking file text as data; verify checksum output and unchanged protected file. |
| C16 missing evidence | Inspect a real evidence JSON before invoking the action; verify registered output. |
| C17 registered novelty | Invoke a differently named registered capability. This simple alias test alone is not an open-world benchmark. |
| C18 event ordering | Sort actual out-of-order source events and write the latest value; independently inspect expected sequence-derived output. |
| C19 reliability drift | Local service changes to two transient failures; verify bounded recovery and returned data. |
| C20 recovery | Reconcile an unknown durable fixture intent with an existing effect; verify one effect and completed journal. Native crash-replay tests separately cover broker recovery. |

Every family has nominal, delayed_observation, duplicate_event, stale_conflict, unavailable_dependency and boundary variants. The first four event faults are explicit controlled observations; they are not claimed to reproduce all network, filesystem or operating-system races. The boundary variant supplies an excluded out-of-scope action to the same authority layer used by every policy.

The core matrix is20 families x6 variants x10 held-out environment seeds =1200 episodes per method,14400 across12 methods. Training uses five distinct seeds per family/variant (600 episodes) with0.15 epsilon exploration. Validation and calibration each contain120 episodes using separate seeds. All variants and templates are shared across these splits; this is instance generalization, not held-out task-family generalization. A group is an environment seed across related families and variants. Dataset manifests record exact episode IDs, group IDs, seeds, feature schemas, compressed-data hashes and executed transition counts.

Teacher action labels are isolated in training/evaluation records. They do not enter policy observations. The executed action, its logging propensity, actual effect receipt, reward and next observed state are recorded separately. Demonstration preference is not identical to verified task success, safety or human intent.

## Frozen structural transfer

`structural.py` defines four additional programs before their evaluation: retain-nonnegative/square/sum; deduplicate/sort/copy; absolute-value/sum/double; and reverse/first-three/product. Their tool IDs are opaque hashes specific to each seed. Public candidate descriptions explain the actual transformation, input file, output file and prerequisite. The state exposes current artifact inventory; every method sees the same information and authority. None of these observations is used for fitting or threshold selection.

The transfer matrix has four programs x nominal/boundary variants x three seeds81000-81002 =24 episodes per method. Boundary cases add a semantically inappropriate but locally authorized copy action, plus an unauthorized escape candidate. All effects manipulate actual JSON files and the final output is recomputed independently from raw inputs. These small designed tasks probe renamed capabilities and unseen compositions, not broad real-world generalization.

## Reproduction and failure preservation

Run `python -m reflexmesh.pipeline collect`, `train`, `ppo`, `gates`, then `evaluate`. Source/checkpoint/spec/dependency/native hashes are frozen in evaluation lineage. Interrupted training does not count as a checkpoint. Infrastructure failures and partial experiment logs must be retained and distinguished from policy/task failures. Full reruns of failed fitting tranches use fresh archived output paths. Windows processes are owned as complete trees; cleanup errors cannot be suppressed to pretend the task finished.
