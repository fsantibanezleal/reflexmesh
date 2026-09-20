"""Run real pinned tau2 tasks with official orchestration and deterministic evaluators.

Local Ollama replaces only the upstream LLM transport. The acting agent never
receives task evaluation criteria, expected actions, or evaluator database state.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .common import LocalOllama, NativeGuard, digest, provenance, save_json
from .sources import SOURCES


def run(
    source: Path,
    output: Path,
    domain: str,
    task_ids: list[str],
    model: str,
    seed: int,
    max_steps: int = 40,
) -> dict:
    os.environ["TAU2_DATA_DIR"] = str(source.resolve() / "data")
    # The source package loads this variable at import time.
    import tau2.agent.llm_agent as agent_module
    import tau2.user.user_simulator as user_module
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.run import run_task
    from tau2.utils.llm_utils import to_litellm_messages

    if output.exists():
        raise FileExistsError("choose a new output directory to preserve prior run evidence")
    output.mkdir(parents=True)
    marker = json.loads((source / "reflexmesh-source.json").read_text(encoding="utf-8"))
    if marker["revision"] != SOURCES["tau2"].revision:
        raise ValueError("unsupported tau2 revision")
    local = LocalOllama(model=model, seed=seed)
    metadata = provenance(source, local)
    metadata["evaluation"] = "upstream ALL deterministic DB/action/communication; NL judge excluded"
    metadata["max_steps"] = max_steps
    save_json(output / "provenance.json", metadata)

    def generate(model, messages, tools=None, **kwargs):
        response = local.chat(
            to_litellm_messages(messages), [tool.openai_schema for tool in tools] if tools else None
        )
        message = response["choices"][0]["message"]
        calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=json.loads(call["function"]["arguments"]),
            )
            for call in message.get("tool_calls") or []
        ]
        return AssistantMessage(
            role="assistant",
            content=message["content"],
            tool_calls=calls or None,
            cost=None,
            usage=response["usage"],
            raw_data={"transport": "local_ollama", "model": local.model},
        )

    agent_module.generate = generate
    user_module.generate = generate
    task_map = {task.id: task for task in registry.get_tasks_loader(domain)()}
    missing = set(task_ids) - set(task_map)
    if missing:
        raise ValueError(f"unknown upstream task IDs: {sorted(missing)}")
    constructor = registry.get_env_constructor(domain)
    active_guards: list[NativeGuard] = []

    def guarded_constructor(*args, **kwargs):
        environment = constructor(*args, **kwargs)
        index = len(active_guards)
        guard = NativeGuard(
            output / "native" / f"environment-{index}" / "journal.json",
            lambda: digest([environment.get_db_hash(), environment.get_user_db_hash()]),
        )
        active_guards.append(guard)
        original = environment.get_response
        agent_tools = {t.name: t for t in environment.get_tools()}
        user_tools = (
            {t.name: t for t in environment.get_user_tools()} if environment.user_tools else {}
        )

        def get_response(message):
            permitted = user_tools if message.requestor == "user" else agent_tools
            tool = permitted.get(message.name)
            if tool is None:
                return ToolMessage(
                    id=message.id,
                    role="tool",
                    requestor=message.requestor,
                    content="Admission denied: unregistered tool",
                    error=True,
                )
            # Preserve upstream argument semantics; native contract binds the validated object.
            try:
                tool.params.model_validate(message.arguments)
            except Exception as exc:  # noqa: BLE001 - third-party schema failures become explicit tool errors
                return ToolMessage(
                    id=message.id,
                    role="tool",
                    requestor=message.requestor,
                    content=f"Admission denied: {type(exc).__name__}",
                    error=True,
                )
            return guard.call(
                f"{message.requestor}.{message.name}",
                message.arguments,
                lambda: original(message),
                failed=lambda result: result.error,
            )

        environment.get_response = get_response
        return environment

    transfer_domain = "reflexmesh_" + domain
    registry.register_domain(guarded_constructor, transfer_domain)
    results = []
    for task_id in task_ids:
        task = task_map[task_id]
        criteria = task.evaluation_criteria
        if criteria is None or any(
            str(basis.value) == "NL_ASSERTION" or str(basis.value).lower() == "nl_assertion"
            for basis in criteria.reward_basis
        ):
            raise ValueError(
                "eligible tasks must have deterministic criteria and no remote LLM judge"
            )
        started = time.perf_counter_ns()
        call_start = len(local.calls)
        guard_start = len(active_guards)
        entry = {"task_id": task_id, "task_sha256": digest(task.model_dump(mode="json"))}
        try:
            simulation = run_task(
                domain=transfer_domain,
                task=task,
                agent="llm_agent",
                user="user_simulator",
                llm_agent=model,
                llm_user=model,
                llm_args_agent={},
                llm_args_user={},
                max_steps=max_steps,
                max_errors=5,
                evaluation_type=EvaluationType.ALL,
                seed=seed,
            )
            save_json(
                output / "trajectories" / f"{task_id}.json", simulation.model_dump(mode="json")
            )
            entry.update(
                status="evaluated",
                upstream_reward=simulation.reward_info.model_dump(mode="json"),
                termination_reason=str(simulation.termination_reason.value),
            )
        except Exception as exc:  # noqa: BLE001 - record every upstream failure without inventing a score
            entry.update(status="error", error_type=type(exc).__name__, error=str(exc)[:2000])
        entry["elapsed_ms"] = (time.perf_counter_ns() - started) / 1e6
        entry["model_calls"] = local.calls[call_start:]
        # At the pinned revision, run_task constructs one acting environment;
        # evaluate_simulation then constructs separate replay/golden worlds.
        # This adapter selects ordinary LLMAgent, never the solo branch which
        # would construct an extra acting environment.
        task_guards = active_guards[guard_start:]
        entry["native_admissions"] = task_guards[0].timings if task_guards else []
        entry["evaluator_native_admissions"] = [
            timing for guard in task_guards[1:] for timing in guard.timings
        ]
        entry["native_environment_roles"] = [
            {
                "journal": str(guard.path.relative_to(output)),
                "role": "acting" if index == 0 else "evaluation_replay",
            }
            for index, guard in enumerate(task_guards)
        ]
        for guard in active_guards[guard_start:]:
            guard.persist()
        results.append(entry)
        save_json(
            output / "results.json",
            {
                "benchmark": "tau2",
                "domain": domain,
                "provenance": metadata,
                "task_manifest": task_ids,
                "results": results,
            },
        )
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "status": entry["status"],
                    "reward": entry.get("upstream_reward"),
                    "error": entry.get("error"),
                }
            ),
            flush=True,
        )
    return {"benchmark": "tau2", "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=("retail", "airline", "telecom"), default="retail")
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()
    result = run(
        args.source, args.output, args.domain, args.task_ids, args.model, args.seed, args.max_steps
    )
    raise SystemExit(1 if any(r["status"] == "error" for r in result["results"]) else 0)
