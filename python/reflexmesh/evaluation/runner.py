"""Actual execution traces and explicitly scoped descriptive statistics."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..contracts import Decision, ModelRequiredError
from ..environments import CASES, VARIANTS, SoftwareEnvironment
from ..policies import METHODS


def run_episode(
    policy,
    spec,
    parameters=None,
    on_step=None,
    cancel_event=None,
    environment_factory=SoftwareEnvironment,
):
    started = time.perf_counter()
    policy.reset()
    events, decisions = [], []
    policy_errors = []
    controller_ms, execution_ms, violations, delegations = 0.0, 0.0, 0, 0
    status = "failed"
    run_id = f"{policy.policy_id}-{spec.family}-{spec.variant}-{spec.seed}"
    with environment_factory(spec) as environment:
        while not environment.terminal:
            if cancel_event is not None and cancel_event.is_set():
                status = "cancelled"
                break
            observation = environment.observe()
            begin = time.perf_counter()
            invalid_output = False
            try:
                decision = policy.predict(observation)
            except (ValueError, KeyError) as error:
                invalid_output = True
                record = {
                    "phase": "prediction",
                    "type": type(error).__name__,
                    "message": str(error)[:2000],
                }
                policy_errors.append(record)
                # Usage is copied only if the real provider supplied it.
                usage = getattr(getattr(policy, "planner", None), "last_usage", None)
                decision = Decision(
                    policy.policy_id,
                    None,
                    "stop",
                    reason="invalid_policy_output",
                    diagnostics={"policy_error": record, "provider_usage": usage},
                )
            duration = (time.perf_counter() - begin) * 1000
            controller_ms += duration
            target = environment.expert_action()
            _, outcome = environment.step(decision)
            execution_ms += outcome.duration_ms
            violations += int(outcome.contract_violation)
            delegations += int(decision.diagnostics.get("delegated", False))
            confidence_target = decision.diagnostics.get("probability_target")
            probabilities = confidence_target is not None and not decision.diagnostics.get(
                "scores_are_utility"
            )
            event = {
                "sequence": len(decisions) + 1,
                "timestamp_ms": (time.perf_counter() - started) * 1000,
                "kind": observation.event_kind,
                "state": observation.state,
                "revision": observation.revision,
            }
            events.append(event)
            snapshot = environment.runtime.snapshot()["state"]
            resource_records = [
                {
                    "id": key,
                    "kind": "workspace",
                    "version": record["revision"],
                    "status": "observed",
                    "value": record.get("fingerprint"),
                    "owner": "native-broker",
                }
                for key, record in snapshot["resources"].items()
            ]
            choices = [
                {
                    "action": c.action_id,
                    "score": decision.scores.get(c.action_id),
                    "probability": decision.scores.get(c.action_id) if probabilities else None,
                    "allowed": c in observation.admissible,
                    "arguments": c.arguments,
                    "blocked_reason": c.blocked_reason,
                }
                for c in observation.candidates
            ]
            decisions.append(
                {
                    "sequence": len(decisions) + 1,
                    "timestamp_ms": event["timestamp_ms"],
                    "event_kind": observation.event_kind,
                    "action": decision.candidate_id,
                    "mode": decision.mode,
                    "status": outcome.status,
                    "candidates": choices,
                    "reason": decision.reason,
                    "uncertainty": {
                        "confidence": decision.confidence,
                        "target": confidence_target,
                        "safety_bound": False,
                    },
                    "threshold": decision.diagnostics.get("threshold"),
                    "effect_predicted": decision.diagnostics.get("effect_predicted"),
                    "effect_observed": outcome.effects,
                    "residual": decision.diagnostics.get("prediction_residual"),
                    "budget_remaining_ms": observation.deadline_remaining_ms,
                    "delegation_id": f"{run_id}:{environment.steps}"
                    if decision.diagnostics.get("delegated")
                    else None,
                    "resource_version": observation.revision,
                    "controller_ms": duration,
                    "execution_ms": outcome.duration_ms,
                    "diagnostics": decision.diagnostics,
                    "resources": resource_records,
                    "demonstration_action": target,
                    "demonstration_agreement": decision.candidate_id == target,
                }
            )
            if on_step is not None:
                on_step(event, decisions[-1])
            if invalid_output:
                break
        success = environment.verify()
        status = "succeeded" if success else (status if status == "cancelled" else "failed")
        native_snapshot = environment.runtime.snapshot()
        native_receipts = len(native_snapshot["state"]["intents"])
        actual_steps = environment.steps
    # Pending asynchronous work remains part of the episode cost, including stale
    # proposals. This synchronization is not included in hot-path decision time.
    try:
        planner_work = policy.finish_episode() if hasattr(policy, "finish_episode") else {}
    except (ValueError, KeyError) as error:
        policy_errors.append(
            {
                "phase": "background_completion",
                "type": type(error).__name__,
                "message": str(error)[:2000],
            }
        )
        planner_work = {
            "invalid_background_proposal": True,
            "provider_usage": getattr(getattr(policy, "planner", None), "last_usage", None),
        }
    try:
        policy.reset()
    except (ValueError, KeyError):
        # Completion above recorded the bad proposal; reset clears state in a
        # finally clause so the next independent episode remains evaluable.
        pass
    wall_ms = (time.perf_counter() - started) * 1000
    return {
        "schema_version": 1,
        "run_id": run_id,
        "case_id": spec.family,
        "method_id": policy.policy_id,
        "variant": spec.variant,
        "seed": spec.seed,
        "lane": "owned-structural-transfer"
        if spec.split == "structural_transfer"
        else "owned-real-software",
        "parameters": parameters or {},
        "status": status,
        "events": events,
        "decisions": decisions,
        "planner_work": planner_work,
        "policy_errors": policy_errors,
        "resources": [
            {
                "id": key,
                "kind": "workspace",
                "version": record["revision"],
                "status": "observed",
                "value": record.get("fingerprint"),
                "owner": "native-broker",
                "native_receipts": native_receipts,
            }
            for key, record in native_snapshot["state"]["resources"].items()
        ],
        "metrics": {
            "task_success": int(success),
            "steps": actual_steps,
            "invalid_policy_output": int(bool(policy_errors)),
            "decision_calls": len(decisions),
            "deliberation_wait_ms": sum(
                d["controller_ms"]
                for d in decisions
                if d["mode"] == "wait" and d["diagnostics"].get("deliberation_hold")
            ),
            "controller_ms": controller_ms,
            "execution_ms": execution_ms,
            "wall_ms": wall_ms,
            "violations": violations,
            "delegations": delegations,
            "native_receipts": native_receipts,
        },
        "provenance": {
            "episode_id": spec.episode_id,
            "group_id": spec.group_id,
            "split": spec.split,
            "truth": "independent filesystem/process/HTTP effect predicates",
            "fixture_limitations": (
                {
                    "C09": "configured quota model, actual files",
                    "C13": "scheduled planner availability fixture",
                }.get(spec.family)
            ),
            "planner_model": getattr(getattr(policy, "planner", None), "model_id", None),
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }


def aggregate(episodes):
    result = []
    for method in sorted({e["method_id"] for e in episodes}):
        selected = [e for e in episodes if e["method_id"] == method]
        metrics = {
            key: float(np.mean([e["metrics"][key] for e in selected]))
            for key in (
                "task_success",
                "controller_ms",
                "execution_ms",
                "wall_ms",
                "violations",
                "delegations",
                "steps",
            )
        }
        latency = [d["controller_ms"] for e in selected for d in e["decisions"]]
        successes = sum(e["metrics"]["task_success"] for e in selected)
        n = len(selected)
        z = 1.959963984540054
        p = successes / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        metrics.update(
            method_id=method,
            episodes=n,
            controller_p50_ms=float(np.quantile(latency, 0.5)),
            controller_p95_ms=float(np.quantile(latency, 0.95)),
            controller_p99_ms=float(np.quantile(latency, 0.99)),
            success_interval_descriptive=[float(center - radius), float(center + radius)],
            interval_assumption="Wilson independent-episode descriptive interval; families are designed fixtures",
        )
        groups = sorted({e["provenance"]["group_id"] for e in selected})
        cluster_means = np.asarray(
            [
                np.mean(
                    [
                        e["metrics"]["task_success"]
                        for e in selected
                        if e["provenance"]["group_id"] == group
                    ]
                )
                for group in groups
            ]
        )
        if len(groups) > 1:
            rng = np.random.default_rng(20260920)
            draws = cluster_means[rng.integers(0, len(groups), size=(2000, len(groups)))].mean(
                axis=1
            )
            metrics["success_interval_group_bootstrap"] = list(
                map(float, np.quantile(draws, [0.025, 0.975]))
            )
        metrics["bootstrap_groups"] = len(groups)
        metrics["bootstrap_scope"] = (
            "environment-seed clusters; templates shared, small designed cluster population"
        )
        metrics["by_variant"] = [
            {
                "variant": variant,
                "episodes": len(group),
                "task_success": float(np.mean([e["metrics"]["task_success"] for e in group])),
            }
            for variant in VARIANTS
            if (group := [e for e in selected if e["variant"] == variant])
        ]
        bins = []
        calibrated = [
            d
            for e in selected
            for d in e["decisions"]
            if d["uncertainty"].get("target") and d["uncertainty"]["confidence"] is not None
        ]
        for lower in np.arange(0, 1, 0.1):
            group = [
                d
                for d in calibrated
                if lower <= d["uncertainty"]["confidence"]
                and (d["uncertainty"]["confidence"] < lower + 0.1 or lower >= 0.9)
            ]
            if group:
                bins.append(
                    {
                        "lower": float(lower),
                        "upper": float(lower + 0.1),
                        "count": len(group),
                        "confidence": float(
                            np.mean([d["uncertainty"]["confidence"] for d in group])
                        ),
                        "accuracy": float(np.mean([d["demonstration_agreement"] for d in group])),
                    }
                )
        metrics["calibration"] = {
            "target": "demonstration_action_agreement, not safety",
            "bins": bins,
        }
        confusion = {}
        for episode in selected:
            for d in episode["decisions"]:
                key = f"{d['demonstration_action']}|{d['action']}"
                confusion[key] = confusion.get(key, 0) + 1
        metrics["confusion"] = {"target": "demonstration_action_agreement", "counts": confusion}
        result.append(metrics)
    return result


def evaluate(
    policies, specs, directory, progress=None, lineage=None, environment_factory=SoftwareEnvironment
):
    directory = Path(directory)
    (directory / "episodes").mkdir(parents=True, exist_ok=True)
    lineage = lineage or {}
    signature = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    lineage_path = directory / "lineage.json"
    if lineage_path.exists() and json.loads(lineage_path.read_text(encoding="utf8")) != lineage:
        raise ValueError("refusing stale evaluation lineage")
    lineage_path.write_text(json.dumps(lineage, indent=2, sort_keys=True), encoding="utf8")
    episodes = []
    manifest = []
    for policy in policies:
        for number, spec in enumerate(specs):
            run_id = f"{policy.policy_id}-{spec.family}-{spec.variant}-{spec.seed}"
            path = directory / "episodes" / (run_id + ".json")
            if path.exists():
                episode = json.loads(path.read_text(encoding="utf8"))
                if episode.get("provenance", {}).get("lineage_signature") != signature:
                    raise ValueError("refusing stale episode: model/code/config lineage changed")
            else:
                try:
                    episode = run_episode(policy, spec, environment_factory=environment_factory)
                except ModelRequiredError as error:
                    failure = {
                        "run_id": run_id,
                        "method_id": policy.policy_id,
                        "error": str(error),
                        "status": "infrastructure_error",
                        "lineage_signature": signature,
                        "generated_at": datetime.now(UTC).isoformat(),
                    }
                    with (directory / "infrastructure-failures.jsonl").open(
                        "a", encoding="utf8"
                    ) as stream:
                        stream.write(json.dumps(failure) + "\n")
                    raise
                episode["provenance"].update(
                    lineage_signature=signature, lineage_document="../lineage.json"
                )
                path.write_text(json.dumps(episode, separators=(",", ":")), encoding="utf8")
            episodes.append(episode)
            manifest.append(
                {
                    k: episode[k]
                    for k in ("run_id", "case_id", "method_id", "variant", "seed", "status")
                }
                | {
                    "path": "episodes/" + path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            if progress and (number + 1) % 20 == 0:
                progress({"method": policy.policy_id, "completed": number + 1, "total": len(specs)})
    index = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "scope": "owned integration cases, not external benchmark scores",
            "split": specs[0].split,
            "lineage": lineage,
            "lineage_signature": signature,
            "complete_method_ids": [p.policy_id for p in policies],
        },
        "methods": [
            {"id": mid, "name": METHODS[mid], "lane": "owned-real-software"} for mid in METHODS
        ],
        "cases": (
            [
                {"id": cid, "name": entry[0], "description": entry[2], "variants": list(VARIANTS)}
                for cid, entry in CASES.items()
            ]
            if specs[0].split != "structural_transfer"
            else [
                {
                    "id": spec.family,
                    "name": spec.family,
                    "description": spec.goal,
                    "variants": ["nominal", "boundary"],
                }
                for spec in specs
                if spec.variant == "nominal" and spec.seed == 81000
            ]
        ),
        "episodes": manifest,
        "aggregate_results": aggregate(episodes),
    }
    (directory / "index.json").write_text(json.dumps(index, indent=2), encoding="utf8")
    return index
