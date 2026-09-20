# Runtime and controller contracts

`Runtime` is the native authority boundary and registered effect dispatcher. `Controller` coordinates persistent goals above it. Learned policies return decisions; they cannot create capabilities or tool registrations. Exact resources and recurrent policy features are separate state representations.

## Register, observe, decide, execute

```python
from pathlib import Path
from reflexmesh import Observation, Decision
from reflexmesh.runtime import Runtime
from reflexmesh.effectors import WorkspaceFiles

root = Path("workspace")
root.mkdir(exist_ok=True)
with Runtime(root, ("file.read", "file.write")) as runtime:
    WorkspaceFiles(root).register(runtime)
    candidate = runtime.candidate("file.write", {"path": "result.txt", "content": "verified"}, action_id="write")
    observation = Observation("event-1", "goal-1", 0, "requested", {}, (candidate,),
                              ("file.read", "file.write"))
    receipt = runtime.execute(observation, Decision("explicit", "write"))
    assert receipt["status"] == "completed_verified"
```

The application registers immutable `ToolSpec` instances. Each validates exact arguments, declares concrete read/write resources, computes resource fingerprints and returns independently checked effect evidence. Candidate metadata cannot override those registrations. Before dispatch, the executor validates arguments again, observes current fingerprints, submits the intent and asks the native broker to begin. A SQLite FULL commit precedes the effect. A failure in that commit prevents dispatch.

The default file adapter normalizes relative paths, rejects traversal, Windows device paths, reserved journal directories, symbolic links, reparse points and existing hardlinks. Alias paths share resource identities. Writes use a same-directory temporary file, fsync, atomic replacement and an immediate content hash check. Cooperating operations share native leases. A hostile external process can race path checks; isolation from such processes requires a separate OS sandbox. The library does not claim that Python validation creates that sandbox.

## Uncertain outcomes and recovery

`completed_verified`, `failed_verified`, `cancelled_verified` and `effect_unknown` have different meanings. A tool raising an exception after dispatch becomes unknown. An HTTP write response alone does not establish the remote application's postcondition, so the generic HTTP adapter keeps the outcome unknown until a trusted adapter supplies independently checked evidence. Unknown intents retain conflicting leases.

On restart, the runtime verifies the native journal hash chain and SQLite payload checksum. Accepted intents are invalidated; running intents become unknown. Replay does not execute tools. The application can register the same immutable action contracts and reconcile unknown receipts with external evidence. A receipt is evidence of its declared postcondition, not an automatic reward or generalized success guarantee.

Every 256 journal transitions, the runtime durably archives the complete preceding journal, acknowledges its tip to the native broker, then saves the new checkpoint. Archive retention is an operator decision. Configured broker capacities remain explicit; exhausting capacity blocks new admission rather than silently discarding unresolved work or idempotency history.

## Goals and asynchronous planning

`Controller.submit(Goal(...))` accepts an observation factory, an independent goal verifier, prior-goal dependencies, a step bound and a deadline. Dependencies form an acyclic graph because references must name previously submitted goals. The verifier decides success; a policy `stop` without verified completion is failure. Unknown effects block the goal for reconciliation.

Normal events and urgent cancellation/revocation events use separate bounded queues. Policy inference, effect execution and slow-model calls run outside the coordinator's event loop. Actual worker cancellation is cooperative. Windows processes are created suspended, admitted to a Job with no breakaway and kill-on-close, then resumed; inherited descendants share its lifetime. POSIX workers use a separate process group. Neither mechanism rolls back earlier effects or isolates a program from privileged external launch services. The event API never upgrades requested cancellation to verified rollback.

Windows Store-backed Python virtualenv redirectors can launch through an activation service outside the Job. That case was reproduced with a descendant writing after cancellation. The adapter therefore rejects those redirectors. `processes.worker_python()` explicitly selects the current direct base interpreter for stdlib-only workers, and tests verify that its nested children cannot produce a delayed file after Job termination. Workloads requiring virtualenv-only dependencies should use a regular Python installation, not a Store-backed redirector. This is a concrete platform boundary, not a silent fallback.

When a policy requests delegation, the coordinator submits a bounded slow request. Arrival must match goal identity and observation revision, then the executor still rechecks resources and authority. Request cancellation does not imply a provider stopped billing or computing. The adapter records whether the future was cancelled before execution. A stale model response is discarded.

Factories should use stable candidate IDs and causal revisions. Re-reading an unchanged file must not invent a new resource revision. Learned policy state is retained by goal ID; authority is always read from the native broker.

## Local service

`reflexmesh serve --checkpoints artifacts/checkpoints --artifacts artifacts/evaluation --model qwen3.5:4b` binds only `127.0.0.1:8151`. A per-session bearer token is printed locally; it is never put in URLs. Read-only configuration is public on loopback, but run, trace, replay and cancellation endpoints require the token and enforce same-origin requests. Host validation rejects DNS-rebinding aliases. No remote origin receives CORS authority.

The service runs explicitly registered disposable integration workloads. General host access uses the Python SDK or an explicit CLI workflow with registered tools. Opening a static workbench does not grant filesystem access. The `replay` endpoint returns imported data labeled unverified and executes nothing. Traces can contain selected tool inputs or outputs; callers choose which application data to retain and export.
