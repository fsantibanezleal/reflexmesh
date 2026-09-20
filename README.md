# reflexmesh

An event-driven software controller with a Rust authority core, Python tool adapters, learned policies and asynchronous language-model coordination. The library is the engine behind Neuraxis.

The controller maintains exact capabilities and resource versions separately from learned recurrent state. A policy proposes a bounded action or requests deliberation. The executor revalidates authority and state before dispatch, records effects durably, and preserves unknown outcomes for reconciliation. Model confidence never grants permission.

## Install and execute

Python 3.11+ is required. Source builds require Rust and platform C/C++ build tools. Training and the local workbench server are optional dependencies.

```sh
python -m pip install '.[train,server,dev]'
reflexmesh doctor
mkdir workspace
reflexmesh workflow examples/verified-workflow.json --workspace workspace
```

The example performs a real file write and checks its SHA-256 postcondition through the native broker. It does not call an LLM. Use the Runtime SDK to register files, fixed process templates, explicit HTTP endpoints and application-specific tools. Use Controller to submit dependent goals, subscribe to events and coordinate a slow planner.

```python
from pathlib import Path
from reflexmesh.effectors import WorkspaceFiles
from reflexmesh.runtime import Runtime

root = Path('workspace')
root.mkdir(exist_ok=True)
with Runtime(root, ('file.read', 'file.write')) as runtime:
    WorkspaceFiles(root).register(runtime)
    candidate = runtime.candidate('file.write', {'path': 'answer.txt', 'content': '42'})
    receipt = runtime.execute_candidate(candidate)
    print(receipt['status'])
```

## Scientific pipeline

The owned workload suite contains 20 independent software-effect families, six fault variants and disjoint train/validation/calibration/test environments. It executes disposable real files, local HTTP services and child processes with independent postcondition checks. Twelve policy methods cover FSM, behavior tree, calibrated logistic/XGBoost scoring, GRU behavior cloning, LinUCB, masked PPO, learned-transition lookahead, direct LLM, asynchronous FSM+LLM, learned deferral and recurrent metacontrol.

```sh
python -m reflexmesh.pipeline collect --artifacts artifacts
python -m reflexmesh.pipeline train --artifacts artifacts
python -m reflexmesh.pipeline ppo --artifacts artifacts
python -m reflexmesh.pipeline gates --artifacts artifacts --model qwen3.5:4b
python -m reflexmesh.pipeline evaluate --artifacts artifacts --seeds 10
```

Model-dependent methods require a real configured local Ollama model. No unavailable planner is silently replaced with a rule policy. Raw traces, fitted checkpoints, lineage digests and aggregate statistics remain separate. External benchmarks have independent code/model/data license and environment provenance. Scores on designed workloads do not establish production or AGI performance.

## Workbench service

```sh
reflexmesh serve --checkpoints artifacts/checkpoints --artifacts artifacts/evaluation --model qwen3.5:4b
```

The service binds loopback and prints a session token. Every executable endpoint requires that token and the same origin. The service executes registered disposable workloads; public static replays have no host-tool authority.

## Evidence and limits

Fast/slow routing, recurrent policies, predictive control and learned deferral are prior art. The research hypothesis concerns outcome-grounded deliberation allocation under asynchronous state change, resource-versioned proposals and persistent effect residuals. It requires matched held-out comparisons and component ablations. Implementation alone does not establish novelty or superiority.

This is a process-level coordination library, not an operating-system sandbox or a hard real-time system. A cancelled request does not imply rollback or cancellation of a remote provider. Hostile native executables require independent isolation. See [runtime contracts](docs/runtime.md), [native architecture](docs/architecture/native-broker.md), [contributing](CONTRIBUTING.md) and [security](SECURITY.md).

Authored source is Apache-2.0. External models, data and benchmarks retain their own licenses.
