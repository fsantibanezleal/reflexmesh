"""Audit the exploratory routing study against paired traces and its fitted model.

This validates the recorded offline arm selection and fitted-model bytes, not
generalisation of the model or an online M12 policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts"
SPLIT = {"fit": ["D01", "D02", "D03", "D04"],
         "calibration": ["D05", "D06"], "test": ["D07", "D08"]}


def _episode(root: Path, entry: dict) -> dict:
    relative = Path(entry["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("routing arm trace escapes its evidence root")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("routing arm trace digest mismatch")
    return json.loads(raw)


def _summary(rows: list[tuple[dict, dict]], selected: list[bool]) -> dict:
    chosen = [slow if use_slow else fast
              for (fast, slow), use_slow in zip(rows, selected, strict=True)]
    return {"episodes": len(rows), "planner_episodes": sum(selected),
            "verified_successes": sum(int(row["task_success"]) for row in chosen),
            "invalid_policy_outputs": sum(int(row["invalid_policy_output"]) for row in chosen),
            "mean_wall_ms": sum(float(row["wall_ms"]) for row in chosen) / len(chosen)}


def validate(root: Path = ROOT) -> dict:
    root = root.resolve()
    semantic = root / "semantic-transfer-complete"
    index_bytes = (semantic / "index.json").read_bytes()
    index = json.loads(index_bytes)
    report = json.loads((root / "semantic-routing-study/analysis.json").read_text(encoding="utf-8"))
    if (report.get("schema_version") != 1
            or report.get("status") != "exploratory_offline_paired_selection_not_an_online_M12_result"
            or report.get("source", {}).get("index_sha256") != hashlib.sha256(index_bytes).hexdigest()
            or report.get("split") != SPLIT
            or not re.fullmatch(r"[0-9a-f]{64}", report.get("model_sha256", ""))):
        raise ValueError("routing study lineage, split or scope differs")
    model = root / "semantic-routing-study/initial-router.joblib"
    if hashlib.sha256(model.read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("fitted routing model digest differs")

    references = {(row["method_id"], row["case_id"], row["variant"], row["seed"]): row
                  for row in index["episodes"] if row["method_id"] in {"M04", "M09"}}
    if len(references) != 256:
        raise ValueError("128 paired routing environments are required")
    pairs = {}
    for family in [*SPLIT["fit"], *SPLIT["calibration"], *SPLIT["test"]]:
        for variant in ("boundary", "nominal"):
            for seed in range(82000, 82008):
                key = family, variant, seed
                fast = _episode(semantic, references[("M04", *key)])
                slow = _episode(semantic, references[("M09", *key)])
                if fast["events"][0]["state"] != slow["events"][0]["state"]:
                    raise ValueError(f"routing arms do not share an initial state: {key}")
                if [(c["action"], c["arguments"], c["allowed"])
                    for c in fast["decisions"][0]["candidates"]] != [
                    (c["action"], c["arguments"], c["allowed"])
                    for c in slow["decisions"][0]["candidates"]
                ]:
                    raise ValueError(f"routing arms do not share initial candidates: {key}")
                pairs[key] = (fast["metrics"], slow["metrics"])

    positives = {name: sum(slow["task_success"] > fast["task_success"]
                           for key, (fast, slow) in pairs.items() if key[0] in families)
                 for name, families in SPLIT.items()}
    if any(report[f"{name}_positives"] != positives[name] for name in SPLIT):
        raise ValueError("incremental-utility counts differ from executed arms")

    predictions = report.get("held_out_predictions", [])
    expected = {key for key in pairs if key[0] in SPLIT["test"]}
    if len(predictions) != 32 or {(p["family"], p["variant"], p["seed"])
                                  for p in predictions} != expected:
        raise ValueError("held-out prediction grid is incomplete")
    ordered = sorted(predictions, key=lambda row: (row["family"], row["variant"], row["seed"]))
    test_rows = []
    learned = []
    half = []
    oracle = []
    for prediction in ordered:
        key = prediction["family"], prediction["variant"], prediction["seed"]
        fast, slow = pairs[key]
        if (prediction["fast_success"] != fast["task_success"]
                or prediction["slow_success"] != slow["task_success"]
                or not 0 <= prediction["estimated_incremental_success_probability"] <= 1
                or prediction["delegated"] is not
                (prediction["estimated_incremental_success_probability"] > report["selected_threshold"])):
            raise ValueError(f"held-out prediction differs from its executed arms: {key}")
        test_rows.append((fast, slow))
        learned.append(prediction["delegated"])
        half.append(int(hashlib.sha256(f"{key[0]}:{key[1]}:{key[2]}".encode()).hexdigest(), 16) % 2 == 0)
        oracle.append(slow["task_success"] > fast["task_success"])
    strategies = {"learned_gate": learned, "always_fast_M04": [False] * 32,
                  "always_planner_M09": [True] * 32, "hash_half": half,
                  "post_hoc_oracle_upper_bound": oracle}
    if set(report["test"]) != set(strategies):
        raise ValueError("routing comparators differ")
    for name, selected in strategies.items():
        actual = _summary(test_rows, selected)
        recorded = report["test"][name]
        if any(not math.isclose(actual[key], recorded[key], rel_tol=1e-12, abs_tol=1e-6)
               for key in actual):
            raise ValueError(f"routing comparator differs from executed arms: {name}")
    return {"paired_environments": len(pairs), "held_out": len(ordered),
            "held_out_incremental_positives": positives["test"],
            "learned_delegations": sum(learned)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(validate(args.root)))
