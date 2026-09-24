"""Independent post-run coverage/hash audit. Does not change measured episodes."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
VARIANTS = [
    "nominal",
    "delayed_observation",
    "duplicate_event",
    "stale_conflict",
    "unavailable_dependency",
    "boundary",
]
METHODS = [f"M{i:02d}" for i in range(1, 13)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, families, variants, seeds, methods):
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf8"))
    lineage = json.loads((directory / "lineage.json").read_text(encoding="utf8"))
    signature = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    assert index["provenance"]["lineage"] == lineage
    assert index["provenance"]["lineage_signature"] == signature
    for field in ("source_sha256", "native_source_sha256", "checkpoint_sha256"):
        assert lineage[field] and all(len(value) == 64 for value in lineage[field].values())
    assert len(lineage["native_binary"]["sha256"]) == 64
    expected = {(m, f, v, s) for m in methods for f in families for v in variants for s in seeds}
    seen, episodes = set(), []
    for item in index["episodes"]:
        path = (directory / item["path"]).resolve()
        assert path.is_relative_to(directory.resolve())
        assert digest(path) == item["sha256"]
        episode = json.loads(path.read_text(encoding="utf8"))
        key = tuple(episode[k] for k in ("method_id", "case_id", "variant", "seed"))
        assert key not in seen and key in expected, key
        seen.add(key)
        assert episode["provenance"]["lineage_signature"] == signature
        assert episode["status"] in ("succeeded", "failed", "cancelled")
        assert episode["metrics"]["task_success"] == int(episode["status"] == "succeeded")
        assert episode["metrics"]["decision_calls"] == len(episode["decisions"])
        episodes.append(episode)
    assert seen == expected, f"coverage mismatch: missing {len(expected - seen)}"
    assert len(list((directory / "episodes").glob("*.json"))) == len(expected)
    summary = []
    for method in methods:
        rows = [e for e in episodes if e["method_id"] == method]
        decisions = [d for e in rows for d in e["decisions"]]
        latency = np.asarray([d["controller_ms"] for d in decisions])
        summary.append(
            {
                "method_id": method,
                "episodes": len(rows),
                "successes": sum(e["metrics"]["task_success"] for e in rows),
                "success_rate": float(np.mean([e["metrics"]["task_success"] for e in rows])),
                "total_wall_seconds": sum(e["metrics"]["wall_ms"] for e in rows) / 1000,
                "mean_wall_ms": float(np.mean([e["metrics"]["wall_ms"] for e in rows])),
                "decision_latency_ms": dict(
                    zip(["p50", "p95", "p99"], map(float, np.quantile(latency, [0.5, 0.95, 0.99])))
                ),
                "contract_violations": sum(e["metrics"]["violations"] for e in rows),
                "invalid_policy_output_episodes": sum(
                    e["metrics"]["invalid_policy_output"] for e in rows
                ),
                "recorded_delegation_starts": sum(e["metrics"]["delegations"] for e in rows),
                "applied_planner_decisions": sum(
                    bool(d["diagnostics"].get("planner_applied")) for d in decisions
                ),
                "delegation_triggers": dict(
                    Counter(
                        d["diagnostics"].get("delegation_trigger")
                        for d in decisions
                        if d["diagnostics"].get("delegated")
                    )
                ),
                "discarded_proposals": sum(
                    len(e.get("planner_work", {}).get("discarded", [])) for e in rows
                ),
                "failed_cells": [
                    {k: e[k] for k in ("case_id", "variant", "seed", "run_id")}
                    for e in rows
                    if not e["metrics"]["task_success"]
                ],
            }
        )
    return {
        "directory": str(directory.relative_to(ROOT)).replace("\\", "/"),
        "episodes": len(episodes),
        "index_sha256": digest(index_path),
        "lineage_signature": signature,
        "methods": summary,
    }, episodes


def paired_difference(left, right):
    def keyed(rows):
        return {(e["case_id"], e["variant"], e["seed"]): e for e in rows}

    a, b = keyed(left), keyed(right)
    assert a.keys() == b.keys()
    groups = defaultdict(list)
    cost = []
    for key in a:
        groups[key[-1]].append(
            a[key]["metrics"]["task_success"] - b[key]["metrics"]["task_success"]
        )
        cost.append(a[key]["metrics"]["wall_ms"] - b[key]["metrics"]["wall_ms"])
    means = np.asarray([np.mean(values) for _, values in sorted(groups.items())])
    rng = np.random.default_rng(20260920)
    draws = means[rng.integers(0, len(means), size=(10000, len(means)))].mean(axis=1)
    return {
        "paired_cells": len(a),
        "seed_groups": len(means),
        "success_difference_left_minus_right": float(means.mean()),
        "paired_seed_bootstrap95": list(map(float, np.quantile(draws, [0.025, 0.975]))),
        "mean_wall_difference_ms_left_minus_right": float(np.mean(cost)),
        "scope": "designed shared templates; small seed-group population; compute budgets are not equalized",
    }


report = {
    "schema_version": 1,
    "generated_at": datetime.now(UTC).isoformat(),
    "status": "validated",
    "training": {},
}
for stage in ("collect", "train", "ppo", "gates"):
    path = ARTIFACTS / "training-runs" / (stage + ".json")
    record = json.loads(path.read_text())
    assert record["status"] == "complete" and record["source_unchanged"]
    report["training"][stage] = {
        "sha256": digest(path),
        "elapsed_seconds": record["elapsed_seconds"],
        "started_at": record["started_at"],
        "finished_at": record["finished_at"],
    }
report["core"], core = audit(
    ARTIFACTS / "evaluation",
    [f"C{i:02d}" for i in range(1, 21)],
    VARIANTS,
    range(40000, 40010),
    METHODS,
)
reference_lineage = json.loads((ARTIFACTS / "evaluation" / "lineage.json").read_text())
for experiment in [
    ARTIFACTS / "structural-transfer",
    *(
        ARTIFACTS / "ablations" / name
        for name in ("no_memory", "no_effects", "no_deferral", "no_invalidation")
    ),
]:
    lineage = json.loads((experiment / "lineage.json").read_text())
    for field in (
        "source_sha256",
        "native_source_sha256",
        "native_binary",
        "checkpoint_sha256",
        "packages",
        "planner",
    ):
        assert lineage[field] == reference_lineage[field], (
            f"cross-experiment drift: {experiment.name}/{field}"
        )
report["transfer"], transfer = audit(
    ARTIFACTS / "structural-transfer",
    [f"S{i:02d}" for i in range(1, 5)],
    ["nominal", "boundary"],
    range(81000, 81003),
    METHODS,
)
full = [e for e in core if e["method_id"] == "M12"]
report["core_paired_M12_minus_method"] = {
    m: paired_difference(full, [e for e in core if e["method_id"] == m])
    for m in METHODS
    if m != "M12"
}
report["ablations"] = {}
for name in ("no_memory", "no_effects", "no_deferral", "no_invalidation"):
    summary, rows = audit(
        ARTIFACTS / "ablations" / name,
        [f"C{i:02d}" for i in range(1, 21)],
        VARIANTS,
        range(40000, 40010),
        ["M12"],
    )
    report["ablations"][name] = {
        **summary,
        "paired_full_minus_ablation": paired_difference(full, rows),
    }
report["completed_episodes"] = (
    report["core"]["episodes"]
    + report["transfer"]["episodes"]
    + sum(v["episodes"] for v in report["ablations"].values())
)
assert report["completed_episodes"] == 19488
(ARTIFACTS / "corrected-science-audit.json").write_text(
    json.dumps(report, indent=2, allow_nan=False), encoding="utf8"
)
print(
    json.dumps(
        {
            "status": "validated",
            "episodes": report["completed_episodes"],
            "core": report["core"]["episodes"],
            "transfer": report["transfer"]["episodes"],
            "ablations": 4800,
        }
    )
)
