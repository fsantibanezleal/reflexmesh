"""Outcome-paired local planner experiments; no inferred counterfactual rewards."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from ..contracts import Decision
from ..environments import CaseSpec, SoftwareEnvironment, case_matrix
from ..policies.classical import CandidateScorer
from ..policies.control import LookaheadPolicy, MetaFastPolicy, deferral_features
from ..policies.recurrent import RecurrentPolicy
from ..policies.rules import GuardedFSMPolicy
from .train import calibration_report


class CalibratedBenefit:
    """Separate gain regression and probability calibration on disjoint groups."""

    classes_ = np.asarray([0, 1])

    def __init__(self, classifier, regressor, calibrator):
        self.classifier, self.regressor, self.calibrator = classifier, regressor, calibrator

    def raw(self, x):
        values = self.classifier.predict_proba(x)
        return (
            values[:, list(self.classifier.classes_).index(1)]
            if 1 in self.classifier.classes_
            else np.zeros(len(x))
        )

    def predict_proba(self, x):
        p = np.clip(self.raw(x), 1e-7, 1 - 1e-7)
        raw = np.log(p / (1 - p)).reshape(-1, 1)
        values = self.calibrator.predict_proba(raw)
        calibrated = (
            values[:, list(self.calibrator.classes_).index(1)]
            if 1 in self.calibrator.classes_
            else np.zeros(len(x))
        )
        return np.c_[1 - calibrated, calibrated]

    def predict_gain(self, x):
        return self.regressor.predict(x)


def _branch(spec, prefix, action):
    """Execute the specified branch then bounded shared continuation in a fresh environment."""
    teacher = GuardedFSMPolicy()
    with SoftwareEnvironment(spec) as env:
        for item in prefix:
            if env.terminal:
                raise ValueError("prefix unexpectedly terminal")
            env.step(item)
        _, outcome = env.step(action if action is not None else Decision("paired", None, "stop"))
        immediate = outcome.reward
        while not env.terminal:
            env.step(teacher.predict(env.observe()))
        success = env.verify()
        return {
            "success": success,
            "immediate_reward": immediate,
            "steps": env.steps,
            "utility": float(success) + immediate - 0.005 * env.steps,
            "native_receipts": len(env.runtime.snapshot()["state"]["intents"]),
        }


def paired_experiments(specs, fast_policies, planner, path, progress=None):
    """At most two observed states per episode; all three actions execute separately.

    Utility = verified task success + first-step reward - .005 executed steps;
    the slow branch additionally pays .005 per measured planner second.
    Shared continuation isolates the value of this one routing decision. It does
    not estimate whole-policy long-horizon value or production financial cost.
    """
    path = Path(path)
    records = []
    path.parent.mkdir(parents=True, exist_ok=True)
    teacher = GuardedFSMPolicy()
    with path.open("w", encoding="utf8") as stream:
        for number, spec in enumerate(specs):
            prefix = []
            for policy in fast_policies.values():
                policy.reset()
            with SoftwareEnvironment(spec) as env:
                for step in range(2):
                    if env.terminal:
                        break
                    observation = env.observe()
                    fast = {
                        key: policy.predict(observation) for key, policy in fast_policies.items()
                    }
                    began = time.perf_counter()
                    slow = planner.plan(observation)
                    elapsed = time.perf_counter() - began
                    slow_branch = _branch(spec, prefix, slow.candidate_id)
                    branches = {
                        key: _branch(spec, prefix, decision.candidate_id)
                        for key, decision in fast.items()
                    }
                    record = {
                        "spec": asdict(spec),
                        "episode_id": spec.episode_id,
                        "group_id": spec.group_id,
                        "prefix": list(prefix),
                        "observation": observation.to_dict(),
                        "slow": slow.to_dict(),
                        "planner_seconds": elapsed,
                        "planner_model": planner.model_id,
                        "slow_branch": slow_branch,
                        "fast": {
                            key: {
                                "decision": decision.to_dict(),
                                "features": deferral_features(observation, decision).tolist(),
                                "branch": branches[key],
                                "incremental_utility": slow_branch["utility"]
                                - 0.005 * elapsed
                                - branches[key]["utility"],
                            }
                            for key, decision in fast.items()
                        },
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    stream.flush()
                    records.append(record)
                    action = teacher.predict(observation).candidate_id
                    env.step(action)
                    prefix.append(action)
            if progress and (number + 1) % 10 == 0:
                progress({"paired_episodes": number + 1, "total": len(specs)})
    return records


def train_gates(checkpoints, data, planner, progress=None):
    checkpoints, data = Path(checkpoints), Path(data)
    fast = {
        "M11": CandidateScorer.load(checkpoints, "M03"),
        "M12": MetaFastPolicy(RecurrentPolicy.load(checkpoints), LookaheadPolicy.load(checkpoints)),
    }
    # Independent environment seeds, beyond every base model fit/calibration and
    # every held-out test seed. Group IDs are recorded for audit and resampling.
    fit_specs = [
        CaseSpec(s.family, s.variant, 50000, "validation") for s in case_matrix("validation", 1)
    ]
    cal_specs = [
        CaseSpec(s.family, s.variant, 60000, "calibration") for s in case_matrix("calibration", 1)
    ]
    select_specs = [
        CaseSpec(s.family, s.variant, 70000, "validation") for s in case_matrix("validation", 1)
    ]
    fit = paired_experiments(fit_specs, fast, planner, data / "router-fit.jsonl", progress)
    calibration = paired_experiments(
        cal_specs, fast, planner, data / "router-calibration.jsonl", progress
    )
    selection = paired_experiments(
        select_specs, fast, planner, data / "router-selection.jsonl", progress
    )
    results = {}
    for method in fast:
        x = np.asarray([r["fast"][method]["features"] for r in fit])
        y = np.asarray([r["fast"][method]["incremental_utility"] > 0 for r in fit], dtype=int)
        cx = np.asarray([r["fast"][method]["features"] for r in calibration])
        cy = np.asarray(
            [r["fast"][method]["incremental_utility"] > 0 for r in calibration], dtype=int
        )
        if len(set(y)) < 2:
            classifier = DummyClassifier(strategy="prior").fit(x, y)
            fitted_kind = "empirical_constant_gate_no_positive_or_negative_examples"
        else:
            classifier = LogisticRegression(C=1.0, max_iter=500, solver="liblinear").fit(x, y)
            fitted_kind = "logistic_incremental_benefit"
        # Select a threshold using actual paired calibration utility, including
        # measured planner cost. Includes never-defer; no positives are invented.
        regressor = Ridge(alpha=1.0).fit(x, [r["fast"][method]["incremental_utility"] for r in fit])
        uncalibrated = classifier.predict_proba(cx)
        raw = (
            uncalibrated[:, list(classifier.classes_).index(1)]
            if 1 in classifier.classes_
            else np.zeros(len(cx))
        )
        raw = np.clip(raw, 1e-7, 1 - 1e-7)
        raw = np.log(raw / (1 - raw)).reshape(-1, 1)
        calibrator = (
            LogisticRegression(C=1.0, solver="liblinear")
            if len(set(cy)) > 1
            else DummyClassifier(strategy="prior")
        ).fit(raw, cy)
        model = CalibratedBenefit(classifier, regressor, calibrator)
        sx = np.asarray([r["fast"][method]["features"] for r in selection])
        sy = np.asarray(
            [r["fast"][method]["incremental_utility"] > 0 for r in selection], dtype=int
        )
        probability = model.predict_proba(sx)[:, 1]
        gain = np.asarray([r["fast"][method]["incremental_utility"] for r in selection])
        positive_prediction = model.predict_gain(sx) > 0
        thresholds = np.r_[np.linspace(0, 1, 21), 1.01]
        values = [
            float(np.sum(gain[(probability > threshold) & positive_prediction]))
            for threshold in thresholds
        ]
        # Prefer larger thresholds under exact ties.
        threshold = float(
            max(t for t, v in zip(thresholds, values, strict=True) if v == max(values))
        )
        joblib.dump(
            {"model": model, "threshold": threshold}, checkpoints / (method + ".gate.joblib")
        )
        results[method] = {
            "kind": fitted_kind,
            "fit_states": len(x),
            "positive_fit_states": int(y.sum()),
            "calibration_states": len(cx),
            "positive_calibration_states": int(cy.sum()),
            "threshold": threshold,
            "calibration_net_incremental_utility": max(values),
            "calibration": calibration_report(probability, sy)
            | {"target": "positive_incremental_utility"},
            "planner_model": planner.model_id,
            "target": "positive executed one-decision incremental utility including measured planner cost",
            "scope": "resettable owned environments; shared teacher continuation; not whole-policy causal effect",
            "fit_seed": 50000,
            "calibration_seed": 60000,
            "threshold_selection_seed": 70000,
            "probability_calibration": "Platt fit on60000; empirical constant if one class; reliability on70000; no safety bound",
            "expected_gain_model": "Ridge on actual paired incremental utility including measured local planner cost",
            "fast_action": "actual M03 action"
            if method == "M11"
            else "actual recurrent/effect blended action",
        }
    (checkpoints / "routing-training.json").write_text(
        json.dumps(results, indent=2), encoding="utf8"
    )
    return results
