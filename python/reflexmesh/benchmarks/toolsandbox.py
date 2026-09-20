"""Real ToolSandbox scenarios/evaluator with a local model and native tool admission.

Run in the pinned upstream Python 3.11 environment; its dependency requirements
are intentionally isolated from the main training environment.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .common import LocalOllama, NativeGuard, digest, provenance, save_json
from .sources import SOURCES

DEFAULT_SCENARIOS = (
    "cellular_off",
    "wifi_off",
    "get_cellular",
    "get_wifi",
    "search_phone_number_with_name",
    "search_name_with_relationship",
    "add_contact_with_name_and_phone_number",
    "remove_contact_with_id",
)


def run(source: Path, output: Path, scenarios: list[str], model: str, seed: int) -> dict:
    import attrs
    from openai import NotGiven
    from openai.types.chat import ChatCompletion
    from tool_sandbox.common.execution_context import (
        DatabaseNamespace,
        RoleType,
        get_current_context,
    )
    from tool_sandbox.common.message_conversion import python_code_to_openai_tool_call
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.roles.execution_environment import (
        ExecutionEnvironment,
        get_messages_to_process,
    )
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
    from tool_sandbox.roles.openai_api_user import OpenAIAPIUser
    from tool_sandbox.scenarios import named_scenarios

    if output.exists():
        raise FileExistsError("choose a new output directory to preserve prior run evidence")
    output.mkdir(parents=True)
    marker = json.loads((source / "reflexmesh-source.json").read_text(encoding="utf-8"))
    if marker["revision"] != SOURCES["toolsandbox"].revision:
        raise ValueError("unsupported ToolSandbox revision")
    random.seed(seed)
    local = LocalOllama(model=model, seed=seed)
    metadata = provenance(source, local)
    save_json(output / "provenance.json", metadata)

    class LocalInference:
        def __init__(self):
            self.model_name = model

        def model_inference(self, openai_messages, openai_tools):
            tools = None if isinstance(openai_tools, NotGiven) else list(openai_tools)
            return ChatCompletion.model_validate(local.chat(openai_messages, tools))

    class LocalAgent(LocalInference, OpenAIAPIAgent):
        pass

    class LocalUser(LocalInference, OpenAIAPIUser):
        pass

    def fingerprint():
        context = get_current_context()
        return digest(
            {
                str(namespace): context.get_database(namespace).to_dicts()
                for namespace in DatabaseNamespace
                if namespace != DatabaseNamespace.SANDBOX
            }
        )

    available = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
    results = []
    for name in scenarios:
        if name not in available:
            raise ValueError(f"unknown upstream scenario: {name}")
        scenario = available[name]
        allowed = scenario.starting_context.get_available_tools(scrambling_allowed=False)
        if any("rapid_api" in tool.__module__ for tool in allowed.values()):
            raise ValueError(f"scenario {name} includes credentialed/external search tools")
        guard = NativeGuard(output / "native" / name / "journal.json", fingerprint)

        class GuardedEnvironment(ExecutionEnvironment):
            def __init__(self, broker_guard):
                self.guard = broker_guard

            def respond(self, ending_index=None):
                messages = self.get_messages(ending_index)
                pending = get_messages_to_process(messages, self.role_type)
                if pending and all(m.sender == RoleType.SYSTEM for m in pending):
                    return super().respond(ending_index)  # trusted upstream scenario initialization
                if len(pending) != 1:
                    raise ValueError("adapter requires one explicit tool call at a time")
                call = python_code_to_openai_tool_call(pending[0].content, None)
                context_tools = get_current_context().get_available_tools(scrambling_allowed=False)
                if call.function.name not in context_tools:
                    raise ValueError("tool unavailable in current upstream scenario")
                tool = context_tools[call.function.name]
                if pending[0].sender not in getattr(tool, "visible_to", (RoleType.AGENT,)):
                    raise ValueError("upstream role cannot access requested tool")
                arguments = json.loads(call.function.arguments)
                parent = super().respond
                return self.guard.call(
                    call.function.name,
                    arguments,
                    lambda: parent(ending_index),
                    failed=lambda _: bool(self.get_messages()[-1].tool_call_exception),
                )

        started = time.perf_counter_ns()
        call_start = len(local.calls)
        entry = {"scenario": name, "categories": list(map(str, scenario.categories))}
        try:
            result = scenario.play_and_evaluate(
                roles={
                    RoleType.AGENT: LocalAgent(),
                    RoleType.USER: LocalUser(),
                    RoleType.EXECUTION_ENVIRONMENT: GuardedEnvironment(guard),
                },
                output_directory=output,
                scenario_name=name,
            )
            entry.update(
                status="evaluated", upstream_evaluation=attrs.asdict(result.evaluation_result)
            )
        except Exception as exc:  # noqa: BLE001 - record every upstream failure without inventing a score
            entry.update(status="error", error_type=type(exc).__name__, error=str(exc)[:2000])
        entry["elapsed_ms"] = (time.perf_counter_ns() - started) / 1e6
        entry["model_calls"] = local.calls[call_start:]
        entry["native_admissions"] = guard.timings
        guard.persist()
        results.append(entry)
        save_json(
            output / "results.json",
            {
                "benchmark": "ToolSandbox",
                "provenance": metadata,
                "scenario_manifest": scenarios,
                "results": results,
            },
        )
        print(
            json.dumps(
                {
                    "scenario": name,
                    "status": entry["status"],
                    "evaluation": entry.get("upstream_evaluation"),
                    "error": entry.get("error"),
                }
            ),
            flush=True,
        )
    return {"benchmark": "ToolSandbox", "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(args.source, args.output, args.scenarios, args.model, args.seed)
    raise SystemExit(1 if any(r["status"] == "error" for r in result["results"]) else 0)
