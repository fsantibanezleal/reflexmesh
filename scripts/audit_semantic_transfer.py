"""Hash-audit the complete, separate 12-method semantic-transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

METHODS = {f"M{i:02d}" for i in range(1, 13)}
FAMILIES = {f"D{i:02d}" for i in range(1, 9)}
VARIANTS = {"nominal", "boundary"}
SEEDS = set(range(82000, 82008))


def audit(index_path: Path) -> dict:
    root = index_path.parent.resolve()
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    provenance = index.get("provenance", {})
    lineage = json.loads((root / "lineage.json").read_text(encoding="utf8"))
    signature = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    if (
        index.get("schema_version") != 1
        or provenance.get("split") != "semantic_transfer"
        or set(provenance.get("complete_method_ids", [])) != METHODS
        or provenance.get("lineage") != lineage
        or provenance.get("lineage_signature") != signature
    ):
        raise ValueError("semantic-transfer index lineage or method set is incomplete")
    expected = {(m, c, v, s) for m in METHODS for c in FAMILIES for v in VARIANTS for s in SEEDS}
    observed = {}
    rows: dict[str, list[dict]] = defaultdict(list)
    for entry in index.get("episodes", []):
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("episode path escapes evidence root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("episode digest mismatch: " + entry["run_id"])
        episode = json.loads(raw)
        identity = tuple(episode[key] for key in ("method_id", "case_id", "variant", "seed"))
        if identity in observed or identity not in expected:
            raise ValueError("duplicate or unexpected semantic-transfer cell")
        if any(episode[key] != entry[key] for key in ("run_id", "case_id", "method_id", "variant", "seed", "status")):
            raise ValueError("episode identity differs from index")
        if (
            episode.get("lane") != "owned-semantic-transfer"
            or episode.get("provenance", {}).get("split") != "semantic_transfer"
            or episode["provenance"].get("lineage_signature") != signature
            or episode["metrics"].get("task_success") not in (0, 1)
            or not episode.get("decisions")
            or type(episode["metrics"].get("native_receipts")) is not int
        ):
            raise ValueError("episode lacks actual semantic execution evidence")
        observed[identity] = int(episode["metrics"]["task_success"])
        rows[identity[0]].append(episode)
    if set(observed) != expected:
        raise ValueError(f"semantic-transfer matrix incomplete: {len(observed)}/{len(expected)}")
    aggregates = {row["method_id"]: row for row in index.get("aggregate_results", [])}
    if set(aggregates) != METHODS:
        raise ValueError("aggregate method set is incomplete")
    methods = []
    for method in sorted(METHODS):
        episodes = rows[method]
        successes = sum(e["metrics"]["task_success"] for e in episodes)
        if aggregates[method]["episodes"] != len(episodes) or abs(
            aggregates[method]["task_success"] - successes / len(episodes)
        ) > 1e-12:
            raise ValueError("aggregate success differs from episode truth")
        methods.append(
            {
                "method_id": method,
                "episodes": len(episodes),
                "successes": successes,
                "invalid_policy_outputs": sum(e["metrics"]["invalid_policy_output"] for e in episodes),
                "planner_delegations": sum(e["metrics"]["delegations"] for e in episodes),
                "native_receipts": sum(e["metrics"]["native_receipts"] for e in episodes),
                "mean_wall_ms": sum(e["metrics"]["wall_ms"] for e in episodes) / len(episodes),
                "mean_steps": sum(e["metrics"]["steps"] for e in episodes) / len(episodes),
            }
        )
    by_family = [
        {
            "family": family,
            "methods": {
                method: sum(observed[(method, family, variant, seed)] for variant in VARIANTS for seed in SEEDS)
                for method in sorted(METHODS)
            },
            "episodes_per_method": len(VARIANTS) * len(SEEDS),
        }
        for family in sorted(FAMILIES)
    ]
    paired = {}
    for comparator in ("M01", "M04", "M09"):
        counts = Counter()
        for family in FAMILIES:
            for variant in VARIANTS:
                for seed in SEEDS:
                    left = observed[("M12", family, variant, seed)]
                    right = observed[(comparator, family, variant, seed)]
                    counts["m12_wins" if left > right else "m12_losses" if left < right else "ties"] += 1
        paired[comparator] = dict(counts)
    return {
        "schema_version": 1,
        "scope": "complete separate owned semantic-transfer lane; fixed pre-existing checkpoints",
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "lineage_signature": signature,
        "episodes_checked": len(observed),
        "methods": methods,
        "by_family": by_family,
        "m12_paired": paired,
        "seed_variable_cells": sum(
            len({observed[(method, family, variant, seed)] for seed in SEEDS}) > 1
            for method in METHODS for family in FAMILIES for variant in VARIANTS
        ),
        "perfect_methods": [row["method_id"] for row in methods if row["successes"] == 128],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("artifacts/semantic-transfer-complete/index.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect", type=Path, help="compare with a committed report")
    args = parser.parse_args()
    result = audit(args.index)
    if args.expect and result != json.loads(args.expect.read_text(encoding="utf8")):
        raise ValueError("semantic-transfer report differs from audited traces")
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
