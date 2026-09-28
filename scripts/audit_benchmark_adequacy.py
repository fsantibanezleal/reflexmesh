"""Audit how much the frozen core matrix can discriminate controller methods.

This is a post-hoc diagnostic, not a new result or a replacement for the
source/lineage audit. It verifies every indexed trace digest before counting
outcomes and keeps template and seed clustering explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def diagnose(index_path: Path) -> dict:
    raw = index_path.read_bytes()
    index = json.loads(raw)
    if index.get("schema_version") != 1:
        raise ValueError("expected canonical schema version 1")
    root = index_path.parent
    outcomes: dict[tuple[str, str, str], dict[int, int]] = defaultdict(dict)
    by_method: dict[str, list[int]] = defaultdict(list)
    for entry in index["episodes"]:
        path = root / entry["path"]
        if path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("indexed episode escapes the evidence root")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"episode digest mismatch: {entry['path']}")
        episode = json.loads(payload)
        identity = (entry["method_id"], entry["case_id"], entry["variant"], entry["seed"])
        if identity != (
            episode["method_id"], episode["case_id"], episode["variant"], episode["seed"]
        ):
            raise ValueError(f"episode identity mismatch: {entry['path']}")
        result = episode["metrics"]["task_success"]
        if result not in (0, 1):
            raise ValueError("task success must be binary")
        cell = outcomes[identity[:3]]
        if identity[3] in cell:
            raise ValueError("duplicate method/case/variant/seed")
        cell[identity[3]] = result
        by_method[identity[0]].append(result)
    if len(index["episodes"]) != 14400 or len(outcomes) != 12 * 20 * 6:
        raise ValueError("canonical core matrix is incomplete")
    if any(len(seeds) != 10 for seeds in outcomes.values()):
        raise ValueError("a core cell is missing a held-out seed")
    method_rows = []
    for method in sorted(by_method):
        cells = {key: values for key, values in outcomes.items() if key[0] == method}
        variable = sum(len(set(values.values())) > 1 for values in cells.values())
        successes = sum(by_method[method])
        method_rows.append(
            {
                "method_id": method,
                "successes": successes,
                "episodes": len(by_method[method]),
                "seed_variable_case_variant_cells": variable,
                "seed_invariant_case_variant_cells": len(cells) - variable,
            }
        )
    m12_failures = [
        {"case_id": case, "variant": variant, "failures": 10 - sum(seeds.values())}
        for (method, case, variant), seeds in sorted(outcomes.items())
        if method == "M12" and sum(seeds.values()) != 10
    ]
    return {
        "schema_version": 1,
        "scope": "frozen canonical core; post-hoc discrimination diagnostic",
        "index_sha256": hashlib.sha256(raw).hexdigest(),
        "episodes_checked": len(index["episodes"]),
        "case_variant_method_cells": len(outcomes),
        "seed_variable_cells": sum(len(set(v.values())) > 1 for v in outcomes.values()),
        "perfect_methods": [row["method_id"] for row in method_rows if row["successes"] == 1200],
        "methods": method_rows,
        "m12_failures": m12_failures,
        "interpretation": (
            "Template-shared seed repetitions are not independent tasks. A perfect exact baseline "
            "prevents this matrix from demonstrating a success advantage for learned deferral. "
            "Keep this historical result separate from later stress evaluations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("artifacts/evaluation/index.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.index)
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
