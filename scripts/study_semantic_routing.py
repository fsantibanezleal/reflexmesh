"""Exploratory, family-held-out initial routing from paired executed episodes.

Both arms are complete, independently executed M04 and M09 episodes. The fitted
gate chooses one arm before its first action; this is not per-step M12 deferral
or a new online success claim. Fit/calibration/test families are disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SPLITS = {
    "fit": {"D01", "D02", "D03", "D04"},
    "calibration": {"D05", "D06"},
    "test": {"D07", "D08"},
}


def _features(episode: dict) -> tuple[str, list[float]]:
    first = episode["decisions"][0]
    state = episode["events"][0]["state"]
    descriptions = sorted(
        str(candidate.get("arguments", {}).get("description", ""))
        for candidate in first["candidates"]
    )
    scores = sorted(
        (float(c["score"]) for c in first["candidates"] if c.get("score") is not None),
        reverse=True,
    )
    if not scores or not isinstance(state.get("goal"), str):
        raise ValueError("routing study requires a scored first observation and visible goal")
    probabilities = np.asarray(scores, dtype=float)
    probabilities = probabilities / max(1e-12, probabilities.sum())
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    numeric = [scores[0], scores[0] - scores[1] if len(scores) > 1 else scores[0], entropy]
    return state["goal"] + "\n" + "\n".join(descriptions), numeric


def load_pairs(root: Path) -> tuple[dict, list[dict]]:
    index_bytes = (root / "index.json").read_bytes()
    index = json.loads(index_bytes)
    if index.get("provenance", {}).get("split") != "semantic_transfer":
        raise ValueError("expected separately indexed semantic transfer evidence")
    grouped: dict[tuple[str, str, int], dict[str, dict]] = {}
    for entry in index["episodes"]:
        if entry["method_id"] not in {"M04", "M09"}:
            continue
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("trace escapes evidence root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("trace digest mismatch")
        episode = json.loads(raw)
        key = (episode["case_id"], episode["variant"], episode["seed"])
        grouped.setdefault(key, {})[episode["method_id"]] = episode
    if len(grouped) != 128 or any(set(pair) != {"M04", "M09"} for pair in grouped.values()):
        raise ValueError("all 128 complete M04/M09 pairs are required")
    rows = []
    for key, pair in sorted(grouped.items()):
        fast, slow = pair["M04"], pair["M09"]
        fast_candidates = [
            (c["action"], c["arguments"], c["allowed"]) for c in fast["decisions"][0]["candidates"]
        ]
        slow_candidates = [
            (c["action"], c["arguments"], c["allowed"]) for c in slow["decisions"][0]["candidates"]
        ]
        if (
            fast["events"][0]["state"] != slow["events"][0]["state"]
            or fast_candidates != slow_candidates
        ):
            raise ValueError("paired arms did not start from the same visible state and choices")
        wording, numeric = _features(fast)
        rows.append(
            {
                "family": key[0],
                "variant": key[1],
                "seed": key[2],
                "text": wording,
                "numeric": numeric,
                "fast": fast["metrics"],
                "slow": slow["metrics"],
            }
        )
    return {"index_sha256": hashlib.sha256(index_bytes).hexdigest()}, rows


def _outcomes(rows: list[dict], selected: list[bool]) -> dict:
    chosen = [
        row["slow"] if use_slow else row["fast"]
        for row, use_slow in zip(rows, selected, strict=True)
    ]
    return {
        "episodes": len(rows),
        "planner_episodes": sum(selected),
        "verified_successes": sum(int(m["task_success"]) for m in chosen),
        "invalid_policy_outputs": sum(int(m["invalid_policy_output"]) for m in chosen),
        "mean_wall_ms": sum(float(m["wall_ms"]) for m in chosen) / len(chosen),
    }


def study(root: Path, output: Path) -> dict:
    provenance, rows = load_pairs(root.resolve())
    split = {
        name: [r for r in rows if r["family"] in families] for name, families in SPLITS.items()
    }
    if [len(split[name]) for name in ("fit", "calibration", "test")] != [64, 32, 32]:
        raise ValueError("family-held-out routing partition is incomplete")
    fit, calibration, test = (split[name] for name in ("fit", "calibration", "test"))
    vocabulary = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    vocabulary.fit([row["text"] for row in fit])
    scaler = StandardScaler().fit([row["numeric"] for row in fit])

    def transform(rows: list[dict]):
        text = vocabulary.transform([row["text"] for row in rows])
        numeric = csr_matrix(scaler.transform([row["numeric"] for row in rows]))
        return hstack([text, numeric], format="csr")

    labels = np.asarray(
        [int(row["slow"]["task_success"] > row["fast"]["task_success"]) for row in fit]
    )
    if len(set(labels)) < 2:
        model = DummyClassifier(strategy="prior").fit(transform(fit), labels)
        kind = "constant_prior"
    else:
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=500).fit(
            transform(fit), labels
        )
        kind = "logistic_text_and_fast_score"

    def probability(rows: list[dict]) -> np.ndarray:
        scores = model.predict_proba(transform(rows))
        return (
            scores[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(rows))
        )

    cal_probability = probability(calibration)
    test_probability = probability(test)
    call_budget = 16
    candidates = np.r_[-0.01, np.unique(cal_probability), 1.01]
    rankings = []
    for threshold in candidates:
        selected = [bool(p > threshold) for p in cal_probability]
        if sum(selected) <= call_budget:
            outcome = _outcomes(calibration, selected)
            rankings.append(
                (
                    outcome["verified_successes"],
                    -outcome["planner_episodes"],
                    -outcome["mean_wall_ms"],
                    float(threshold),
                )
            )
    threshold = max(rankings)[-1]
    chosen = [bool(p > threshold) for p in test_probability]
    fixed_half = [
        int(hashlib.sha256(f"{r['family']}:{r['variant']}:{r['seed']}".encode()).hexdigest(), 16)
        % 2
        == 0
        for r in test
    ]
    oracle = [row["slow"]["task_success"] > row["fast"]["task_success"] for row in test]
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "initial-router.joblib"
    joblib.dump(
        {"vocabulary": vocabulary, "scaler": scaler, "model": model, "threshold": threshold},
        model_path,
    )
    result = {
        "schema_version": 1,
        "status": "exploratory_offline_paired_selection_not_an_online_M12_result",
        "source": provenance,
        "scope": "initial arm selection only; both arms actually executed before this analysis",
        "split": {key: sorted(value) for key, value in SPLITS.items()},
        "algorithm": kind,
        "features": "visible initial goal and candidate descriptions plus M04 first-decision scores; no family ID, seed or future outcome",
        "fit_positives": int(labels.sum()),
        "calibration_positives": sum(
            r["slow"]["task_success"] > r["fast"]["task_success"] for r in calibration
        ),
        "test_positives": sum(oracle),
        "calibration_planner_budget": call_budget,
        "selected_threshold": threshold,
        "test": {
            "learned_gate": _outcomes(test, chosen),
            "always_fast_M04": _outcomes(test, [False] * len(test)),
            "always_planner_M09": _outcomes(test, [True] * len(test)),
            "hash_half": _outcomes(test, fixed_half),
            "post_hoc_oracle_upper_bound": _outcomes(test, oracle),
        },
        "held_out_predictions": [
            {
                "family": row["family"],
                "variant": row["variant"],
                "seed": row["seed"],
                "estimated_incremental_success_probability": float(probability),
                "delegated": delegated,
                "fast_success": int(row["fast"]["task_success"]),
                "slow_success": int(row["slow"]["task_success"]),
            }
            for row, probability, delegated in zip(test, test_probability, chosen, strict=True)
        ],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "limits": "Post-hoc protocol choice after exploratory smoke; only two held-out task families and recorded arm outcomes. Inference and routing latency are not added to wall time. No causal claim about per-step M12 or general software tasks.",
    }
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, default=Path("artifacts/semantic-transfer-complete")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/semantic-routing"))
    args = parser.parse_args()
    print(json.dumps(study(args.evidence, args.output)["test"], indent=2))


if __name__ == "__main__":
    main()
