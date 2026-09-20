"""Check fitted Python models against the actual compiled Rust scorers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from .common import save_json


def check(directory: Path, output: Path, extension: Path | None = None) -> dict:
    from reflexmesh.policies.classical import CandidateScorer

    if extension:
        loader = importlib.machinery.ExtensionFileLoader("_native", str(extension.resolve()))
        spec = importlib.util.spec_from_loader("_native", loader)
        native = importlib.util.module_from_spec(spec)
        loader.exec_module(native)
    else:
        from reflexmesh import _native as native
    tree_export = json.loads((directory / "M04.trees.json").read_text())
    linear_export = json.loads((directory / "M03.linear.json").read_text())
    dimension = len(tree_export["feature_names"])
    if tree_export["feature_names"] != linear_export["feature_names"]:
        raise ValueError("model feature schema mismatch")
    rng = np.random.default_rng(44128)
    rows = [*rng.uniform(-1, 1, size=(256, dimension)).astype(np.float32)]
    # Probe every unique exported split at adjacent representable float32 values.
    splits = {
        (node["feature_index"], node["threshold"])
        for tree in tree_export["trees"]
        for node in tree["nodes"]
        if "threshold" in node
    }
    for index, threshold in sorted(splits):
        threshold = np.float32(threshold)
        for value in (
            np.nextafter(threshold, np.float32(-np.inf)),
            threshold,
            np.nextafter(threshold, np.float32(np.inf)),
        ):
            row = np.zeros(dimension, dtype=np.float32)
            row[index] = value
            rows.append(row)
    rows = np.stack(rows)
    results = {}
    for policy_id, filename, scorer in (
        ("M03", "M03.linear.json", native.score_linear),
        ("M04", "M04.trees.json", native.score_trees),
    ):
        policy = CandidateScorer.load(directory, policy_id)
        if policy_id == "M04":
            policy.model.set_params(n_jobs=1)
        expected = policy.probabilities(rows)
        native_values = []
        durations = []
        encoded_model = (directory / filename).read_text(encoding="utf-8")
        for start in range(0, len(rows), 32):
            payload = json.dumps(rows[start : start + 32].tolist(), allow_nan=False)
            started = time.perf_counter_ns()
            native_values.extend(json.loads(scorer(encoded_model, payload)))
            durations.append((time.perf_counter_ns() - started) / 1e6)
        differences = np.abs(np.asarray(native_values) - expected)
        cached_class = native.LinearScorer if policy_id == "M03" else native.TreeScorer
        cold_started = time.perf_counter_ns()
        cached = cached_class(encoded_model)
        cold_ms = (time.perf_counter_ns() - cold_started) / 1e6
        if cached.feature_names != tree_export["feature_names"]:
            raise ValueError("cached native feature schema mismatch")
        cached_values = np.asarray(cached.score_f32(rows.astype("<f4").tobytes(), len(rows)))
        cached_error = float(np.max(np.abs(cached_values - expected)))
        batch = np.ascontiguousarray(rows[:7], dtype="<f4")
        warmed = {"native_cached": [], "python_reference": []}
        # Alternate order to reduce a systematic warmup/scheduling advantage.
        for repetition in range(520):
            order = ("native_cached", "python_reference")
            for backend in order if repetition % 2 == 0 else reversed(order):
                started = time.perf_counter_ns()
                if backend == "native_cached":
                    cached.score_f32(batch.tobytes(), len(batch))
                else:
                    policy.probabilities(batch)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if repetition >= 20:
                    warmed[backend].append(elapsed)
        results[policy_id] = {
            "feature_dimension": dimension,
            "vectors": len(rows),
            "split_thresholds": len(splits),
            "max_abs_probability_error": float(differences.max()),
            "mean_abs_probability_error": float(differences.mean()),
            "tolerance": 1e-5,
            "passed": bool(max(differences.max(), cached_error) <= 1e-5),
            "cached_max_abs_probability_error": cached_error,
            "cached_constructor_ms": cold_ms,
            "warmed_candidate_count": 7,
            "warmed_repetitions": 500,
            "warmed_latency_ms": {
                backend: {str(q): float(np.percentile(values, q)) for q in (50, 95, 99)}
                for backend, values in warmed.items()
            },
            "warmed_timing_scope": "Native: packed feature serialization + cached Rust scoring + return list; Python: fitted model predict + Platt; both exclude shared feature construction and effect execution",
            "batch_size": 32,
            "call_latency_ms": {str(q): float(np.percentile(durations, q)) for q in (50, 95, 99)},
            "model_sha256": hashlib.sha256(encoded_model.encode()).hexdigest(),
            "timing_scope": "JSON model parse + feature parse + Rust batch scoring + return decode; excludes Python feature construction",
        }
    extension_path = Path(native.__file__)
    report = {
        "schema_version": 1,
        "native_engine": native.__engine__,
        "extension": extension_path.name,
        "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        "seed": 44128,
        "results": results,
        "note": "Numerical export parity, not outcome generalization or safety calibration evidence",
    }
    save_json(output, report)
    if not all(result["passed"] for result in results.values()):
        raise AssertionError(f"native model parity failed: {results}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/native-parity.json"))
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.output, args.extension), indent=2))
