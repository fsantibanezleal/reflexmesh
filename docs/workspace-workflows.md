# User workspace workflows

`reflexmesh workflow recipe.json --workspace /absolute/owned/directory` executes a dependency graph against real files and explicitly registered process templates. Each step has a unique ID, a registered tool, fixed arguments and a SHA256 file postcondition. Success means that predicate holds. An existing verified output makes the step idempotently complete without repeating its effect. All recipes are validated before the first effect; missing dependencies, cycles and unknown tools fail as a whole.

```json
{"schema_version":1,"steps":[{"id":"write-note","tool":"file.write","arguments":{"path":"note.txt","content":"hello"},"verify":{"path":"note.txt","sha256":"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}}]}
```

The `file.read` arguments are `{"path":"note.txt"}`. `file.write` accepts `path` and UTF-8 `content`, bounded to 1MiB. Paths are relative to the configured root. Traversal, journal access, symlink/reparse components and hardlinks reject. Subdirectories must already exist. These cooperating tool boundaries do not isolate adversarial code from the operating system.

Add `depends_on:["write-note"]` to a later step to require the predecessor's verified success. The controller orders the DAG, runs independent goals through bounded workers and records every native receipt. A failed, cancelled or unresolved predecessor blocks dependents. Deadline cancellation does not claim rollback. Interrupted processes can retain unknown effects that require reconciliation through the Runtime API.

For trusted process jobs, a CLI recipe may include `processes:[{"name":"build","argv":["/absolute/python","build.py"],"timeout_seconds":30}]`. The step selects `process.build` with empty arguments and an independent output-file digest. The argv is immutable after registration; there is no model-generated shell command. A process must be trusted because a child program can itself access resources beyond its working directory. Windows Job Objects own the child tree; POSIX groups cover cooperating descendants. Store app execution aliases are not valid owned process executables; use a direct interpreter image.

## Local workbench API

Start `reflexmesh serve --checkpoints models/checkpoints --static frontend/dist --workspace /absolute/owned/directory`. An optional `--process-config trusted-processes.json` supplies the same registration list locally. The browser cannot choose another root, add a process registration, change argv or expand capabilities. The same-origin bearer session protects workspace metadata and execution. General workspace workflows use the verified dependency FSM; the twelve learned/research methods are separately evaluated policies, and are not claimed to generalize automatically to an arbitrary user recipe.

The authenticated protocol is:

1. `GET /api/workspace/config` returns the configured root and registered tool names, or `enabled:false`.
2. `POST /api/workspace/preview` accepts the recipe directly. It returns the normalized steps, capability names, current file fingerprints and a single-use `preview_id` valid for 120 seconds. It does not execute tools.
3. `POST /api/workspace/run` accepts `{"preview_id":"..."}`. The worker rechecks those fingerprints before dispatch. Changed files require a new preview. Exactly one workflow can occupy the workspace worker; benchmark runs use a separate disposable workspace.
4. `GET /api/workspace/events?run_id=...&after=0` streams sequenced decisions, goal changes and final `workflow` evidence. The final evidence retains per-goal decision history, effect receipts, verified predicates, recipe hash and native journal sequence.
5. `POST /api/workspace/cancel/{run_id}` requests interruption. Poll to a terminal state; cancellation does not assert the absence of earlier effects.

The preview is a human-readable preflight, not a distributed transaction. Cooperating runtime actions receive fresh resource bindings and native admission checks. An external hostile writer can race filesystem checks; use an OS-isolated workspace for that threat model. The local service does not upload workspace content to the hosted companion. Exported user traces may contain file content and process output, so publication requires deliberate sanitization.
