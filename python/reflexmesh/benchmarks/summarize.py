"""Derive an auditable transfer summary without changing raw upstream evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .common import save_json


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def quantile(q):
        position = (len(ordered) - 1) * q
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"n": len(values), **{f"p{q}": quantile(q / 100) for q in (50, 95, 99)}}


def _tau_acting_admissions(directory: Path, entry: dict) -> tuple[list, list, dict]:
    if "evaluator_native_admissions" in entry:
        return (
            entry["native_admissions"],
            entry["evaluator_native_admissions"],
            {
                "method": "explicit_environment_roles",
                "roles": entry["native_environment_roles"],
            },
        )
    # Historical v1 logs predate explicit phase labeling. The pinned ordinary
    # LLMAgent run_task creates its acting environment first, and subsequent
    # worlds execute evaluator replay. Map unique intent IDs back to journals.
    by_intent = {}
    journals = {}
    for path in (directory / "native").glob("environment-*/journal.json"):
        index = int(path.parent.name.split("-")[-1])
        journal = json.loads(path.read_text(encoding="utf-8"))
        journals[index] = {"path": str(path.relative_to(directory)), "sha256": file_hash(path)}
        for event in journal["entries"]:
            command = event["command"]
            if command["operation"] == "submit":
                intent_id = command["intent"]["intent_id"]
                if intent_id in by_intent:
                    raise ValueError("ambiguous intent ID across external worlds")
                by_intent[intent_id] = index
    timings = entry.get("native_admissions", [])
    indices = {by_intent[timing["intent_id"]] for timing in timings}
    if not indices:
        return [], [], {"method": "no_recorded_native_admissions"}
    acting_index = min(indices)
    acting = [t for t in timings if by_intent[t["intent_id"]] == acting_index]
    evaluation = [t for t in timings if by_intent[t["intent_id"]] != acting_index]
    trajectory_path = directory / "trajectories" / f"{entry['task_id']}.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    actual_calls = sum(len(message.get("tool_calls") or []) for message in trajectory["messages"])
    if actual_calls != len(acting):
        raise ValueError(
            "historical phase reconstruction differs from actual trajectory tool calls"
        )
    return (
        acting,
        evaluation,
        {
            "method": "journal_intent_matching_and_first_environment_verified_against_trajectory",
            "acting_environment_index": acting_index,
            "trajectory_tool_calls": actual_calls,
            "trajectory_sha256": file_hash(trajectory_path),
            "journals": [journals[index] for index in sorted(indices)],
        },
    )


def summarize(base: Path, selected: list[str], output: Path) -> dict:
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate selected run")
    inventory, chosen, identities = [], [], set()
    all_admission_times, all_effect_times = [], []
    for path in sorted(base.glob("*/results.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        manifest = raw.get("task_manifest", raw.get("scenario_manifest", []))
        entries = raw["results"]
        is_selected = path.parent.name in selected
        inventory.append(
            {
                "run": path.parent.name,
                "selected": is_selected,
                "benchmark": raw["benchmark"],
                "results_sha256": file_hash(path),
                "declared": len(manifest),
                "recorded": len(entries),
                "evaluated": sum(e["status"] == "evaluated" for e in entries),
                "errors": [
                    {
                        "case": e.get("task_id", e.get("scenario")),
                        "error_type": e.get("error_type"),
                        "error": e.get("error"),
                    }
                    for e in entries
                    if e["status"] != "evaluated"
                ],
                "manifest_complete": len(manifest) == len(entries),
                "provenance_sha256": file_hash(path.parent / "provenance.json"),
                "source": raw["provenance"]["source"],
                "model": raw["provenance"]["model"],
                "python": raw["provenance"]["python"],
                "platform": raw["provenance"]["platform"],
                "seed": raw["provenance"]["seed"],
                "model_options": raw["provenance"]["model_options"],
            }
        )
        if not is_selected:
            continue
        for entry in entries:
            if entry["status"] != "evaluated":
                raise ValueError("selected run contains a non-evaluated case; report it explicitly")
            case = entry.get("task_id", entry.get("scenario"))
            identity = (raw["benchmark"], case)
            if identity in identities:
                raise ValueError("selected runs repeat a case; do not select favorable retries")
            identities.add(identity)
            if raw["benchmark"] == "tau2":
                acting, evaluation, reconstruction = _tau_acting_admissions(path.parent, entry)
                reward = entry["upstream_reward"]
                metric = {
                    "name": "upstream_reward",
                    "value": reward["reward"],
                    "basis": reward["reward_basis"],
                    "failed_action_checks": sum(
                        not v["action_match"] for v in reward.get("action_checks") or []
                    ),
                }
            else:
                acting, evaluation = entry["native_admissions"], []
                reconstruction = {"method": "single_acting_world_no_tool_replay_evaluator"}
                metric = {
                    "name": "upstream_similarity",
                    "value": entry["upstream_evaluation"]["similarity"],
                    "minefield_similarity": entry["upstream_evaluation"]["minefield_similarity"],
                }
            admission = [t["native_submit_begin_ms"] for t in acting]
            effects = [t["effect_and_finish_ms"] for t in acting]
            all_admission_times.extend(admission)
            all_effect_times.extend(effects)
            chosen.append(
                {
                    "run": path.parent.name,
                    "benchmark": raw["benchmark"],
                    "case": case,
                    "metric": metric,
                    "elapsed_ms_including_evaluation": entry["elapsed_ms"],
                    "acting_native_calls": len(acting),
                    "evaluator_replay_native_calls": len(evaluation),
                    "acting_admission_ms": _percentiles(admission),
                    "model_calls_agent_and_user": len(entry["model_calls"]),
                    "policy": raw["provenance"]["policy"],
                    "phase_evidence": reconstruction,
                }
            )
    if set(selected) != {item["run"] for item in inventory if item["selected"]}:
        raise ValueError("selected run missing")
    report = {
        "schema_version": 1,
        "selected_runs": selected,
        "run_inventory": inventory,
        "cases": chosen,
        "aggregates": {
            benchmark: {
                "evaluated_cases": len(values),
                "mean_upstream_metric": statistics.mean(values),
            }
            for benchmark in sorted({entry["benchmark"] for entry in chosen})
            if (
                values := [
                    entry["metric"]["value"] for entry in chosen if entry["benchmark"] == benchmark
                ]
            )
        },
        "acting_admission_ms": _percentiles(all_admission_times),
        "acting_effect_and_finish_ms": _percentiles(all_effect_times),
        "limitations": [
            "Local System Two acting agent with native admission; no learned-policy transfer claim",
            "Eligible upstream subsets only; inspect selected case identities and categories",
            "Official upstream evaluators used; local model/transport/subset differ from leaderboard protocols",
            "Historical failed or interrupted attempts remain in the inventory with explicit selection flags",
            "Upstream reward can disagree with action checks; each tau2 case retains the reward basis and failed-check count",
            "Admission excludes fingerprints, observations, fsync and effects; task elapsed includes evaluator time",
            "Small samples and mixed cold/warm state do not establish tail-latency or generalization guarantees",
        ],
    }
    save_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("artifacts/transfer"))
    parser.add_argument("--selected-runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/transfer/summary.json"))
    args = parser.parse_args()
    result = summarize(args.base, args.selected_runs, args.output)
    print(
        json.dumps(
            {
                "aggregates": result["aggregates"],
                "acting_admission_ms": result["acting_admission_ms"],
            },
            indent=2,
        )
    )
