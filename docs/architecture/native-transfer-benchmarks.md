# External benchmark execution

The adapters execute actual upstream tools and original upstream evaluators,
using a local open model and the compiled Rust admission broker. They do not
substitute a handwritten scoring simulator or require a paid model provider.
The measured policy is **local System Two plus native admission**. These runs
validate external integration and actual effects; they do not demonstrate that
the learned System One policy transfers to these external task distributions.

## Pinned inputs and environment isolation

| Source | Immutable revision | Upstream terms | Executed environment |
| --- | --- | --- | --- |
| [Apple ToolSandbox](https://github.com/apple-aiml-research/ToolSandbox) | `c8571d7854316d2e1c5f288e59fe1e34e53f6dd1` | Custom Apple ToolSandbox License; not relicensed by this package | Python 3.11, upstream pinned NumPy 1.26.4/Pydantic 2.7.4 |
| [tau2-bench v0.1.3](https://github.com/sierra-research/tau2-bench/tree/5ba9e3e56db57c5e4114bf7f901291f09b2c5619) | `5ba9e3e56db57c5e4114bf7f901291f09b2c5619` | MIT repository; preserve source/data notices | Python 3.12, upstream declared dependencies |

`python -m reflexmesh.benchmarks.sources <toolsandbox|tau2>` downloads the
immutable official archive, checks extraction paths/expanded bounds, and records
the archive SHA-256 beside the source. Source code and datasets stay under the
ignored `.external/` directory. The recorded hash is reproducibility evidence;
it is not an independent signature. Do not replace pins with current `main`:
the tau repository has since changed its benchmark surface.

`native-transfer-source-audit.json` records a post-run comparison of all 77
original ToolSandbox files and 200 original tau2 files against the downloaded
archives. All original source/data files matched. The adapter now repeats this
streamed integrity check before acting; added build caches do not replace any
original file. This checks reproducibility against the recorded archive, not
the security of upstream code or its publisher identity.

ToolSandbox's declared `ccy==1.3.1` dependency rejected Python 3.12 in the
observed install. Its declared NumPy/Pydantic pins also differ from reflexmesh's
normal runtime requirements. The explicit compatibility harness therefore uses
Python 3.11, installs the full original dependencies, then installs reflexmesh
with `--no-deps` in that isolated environment. This intentionally does not
satisfy the package's general dependency constraints; only the selected adapter
and native broker were exercised there. It is not a supported general reflexmesh
installation. Do not loosen upstream pins or overwrite the training environment.

Example PowerShell setup, from the repository root (Rust/MSVC must already be
installed for native builds):

```powershell
python -m reflexmesh.benchmarks.sources toolsandbox
python -m reflexmesh.benchmarks.sources tau2
py -3.11 -m venv .external/toolsandbox-py311
.external/toolsandbox-py311/Scripts/python -m pip install -e .external/sources/toolsandbox-c8571d7854316d2e1c5f288e59fe1e34e53f6dd1
.external/toolsandbox-py311/Scripts/python -m pip install --no-deps -e .
py -3.12 -m venv .external/tau2-venv
.external/tau2-venv/Scripts/python -m pip install -e .external/sources/tau2-5ba9e3e56db57c5e4114bf7f901291f09b2c5619
.external/tau2-venv/Scripts/python -m pip install -e .
```

Use a fresh source checkout per editable interpreter build if the build backend
changes shared extension filenames. The executed installs produced separate
`cp311` and `cp312` files. Full environment freezes are in each run's provenance.

Ollama must already serve the selected model on loopback. The adapter records
its actual model digest from `/api/tags`; the mutable name alone is not a model
pin. Runs used `qwen3.5:4b`, temperature 0, seed 0, thinking disabled, context
16,384, maximum output 1,024. The same local model acts as user simulator where
upstream requires one. Run model jobs serially on constrained local hardware.

```powershell
.external/toolsandbox-py311/Scripts/python -m reflexmesh.benchmarks.toolsandbox --source .external/sources/toolsandbox-c8571d7854316d2e1c5f288e59fe1e34e53f6dd1 --output artifacts/transfer/new-toolsandbox-run
.external/tau2-venv/Scripts/python -m reflexmesh.benchmarks.tau2 --source .external/sources/tau2-5ba9e3e56db57c5e4114bf7f901291f09b2c5619 --output artifacts/transfer/new-tau2-run --domain retail --task-ids 0 1 2 3
```

Both adapters reject existing output directories to preserve prior evidence.
Raw model responses, token counts, model-side durations, trajectories, native
journals, evaluator results, errors, and dependency provenance are retained.
No paid requests are made by the overridden model transport; hardware, energy,
and elapsed-time costs are unmeasured monetary costs, not zero-cost claims.

## Acting boundary and evaluation

ToolSandbox preserves `play_and_evaluate`, original stateful tool functions,
and milestone/minefield scoring. The guarded execution environment accepts only
the upstream-generated single-call representation, checks tool availability and
role visibility, and dispatches the original function. Credentialed RapidAPI
tools are excluded. The eight executed scenarios are cellular/wifi off and
state reads, two contact searches, contact creation, and contact deletion.

tau2 preserves ordinary `LLMAgent`, `UserSimulator`, `Orchestrator`, task loaders
and deterministic `ALL` evaluation. Gold-trace and solo agents are not used.
Task evaluation criteria, expected actions, and evaluator database hashes never
enter acting model prompts. Tasks requiring an external language-model judge
are rejected. The actual tool implementations mutate the fixture databases.

Each allowed tool execution passes native capability, resource revision,
idempotency, and lease checks. A conservative world-level write lease is used
for each benchmark environment. Its hash is kept in authority/evidence, not
policy input. This choice limits concurrency and is not equivalent to each
upstream benchmark's default tool scheduling protocol.

tau2 evaluation constructs additional environments and executes replay/golden
actions. The adapter now records those separately from acting calls. Historical
runs are reconstructed by matching unique intent IDs to environment journals
and verifying acting counts against the saved conversation. The summary refuses
ambiguous counts. Episode elapsed time includes evaluation; native admission
latency excludes fingerprinting, observations, journal fsync and external effects.

## Executed evidence and limits

`native-transfer-results.json` is derived from the preserved raw artifacts:
eight ToolSandbox cases evaluated with mean official similarity
**0.9546040025**, and four tau2 retail tasks with official reward **1.0** each.
ToolSandbox similarity is a continuous upstream metric, not a binary task
success rate. The tau2 tasks are related variants in one retail context. In
tasks 2 and 3 the upstream action-check details include failures (a referenced
product is absent), while the specified reward basis remains DB plus
COMMUNICATE. The raw evaluator output and summary retain that distinction.

The first tau2 run was interrupted after tasks 0 and 1; a separate run completed
2 and 3. A preliminary concurrent ToolSandbox run had one HTTP 500; a complete
isolated rerun followed. Neither failure was relabeled a successful attempt.
The inventory includes failed probes and incomplete manifests. Run selection is
explicit, and duplicate selected case IDs are rejected rather than selecting
favorable retries. There are 47 actual acting admissions in the selected set.

```powershell
python -m reflexmesh.benchmarks.summarize --selected-runs toolsandbox-isolated-seed0 tau2-retail-seed0 tau2-retail-remaining-seed0 --output docs/architecture/native-transfer-results.json
```

These bounded integration subsets are not leaderboard-comparable, a full
external benchmark suite, evidence of AGI, or evidence of learned transfer.
The local owned fixture matrix separately tests files, subprocesses, services,
concurrency, cancellation, stale state and adversarial output. Broad external
generalization needs held-out task families, multiple seeds, a frozen selection
protocol and learned-policy adapters that use only observable external state.

Raw upstream-derived artifacts should be distributed separately with their
source attribution and applicable licenses; they are not part of the Python
wheel. The summary records input and evidence hashes without bundling upstream
source or training data.
