from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..contracts import Decision, ModelRequiredError, Observation
from ..features import FEATURE_NAMES, feature_matrix


def sigmoid(value):
    value = np.clip(value, -40, 40)
    return 1 / (1 + np.exp(-value))


def ranked_decision(
    policy_id: str,
    observation: Observation,
    values: np.ndarray,
    reason: str,
    diagnostics: dict | None = None,
) -> Decision:
    candidates = observation.admissible
    if not candidates:
        return Decision(policy_id, None, "stop", reason="no_admissible_candidates")
    index = int(np.argmax(values))
    return Decision(
        policy_id,
        candidates[index].action_id,
        confidence=float(values[index]),
        scores={c.action_id: float(v) for c, v in zip(candidates, values, strict=True)},
        reason=reason,
        diagnostics=diagnostics or {},
    )


class CandidateScorer:
    def __init__(self, policy_id: str, model, calibrator=None):
        if policy_id not in {"M03", "M04"}:
            raise ValueError("candidate scorer must be M03 or M04")
        self.policy_id, self.model, self.calibrator = policy_id, model, calibrator

    def reset(self, goal_id: str | None = None) -> None:
        pass

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ModelRequiredError("fit or load the candidate scorer first")
        if not len(x):
            return np.empty(0)
        if self.policy_id == "M03":
            raw = self.model.decision_function(x)
        else:
            probability = self.model.predict_proba(x)[:, 1].clip(1e-7, 1 - 1e-7)
            raw = np.log(probability / (1 - probability))
        return (
            self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
            if self.calibrator is not None
            else sigmoid(raw)
        )

    def predict(self, observation: Observation) -> Decision:
        matrix = feature_matrix(observation)
        native = getattr(self, "native_scorer", None)
        values = (
            np.asarray(
                native.score_f32(np.asarray(matrix, dtype="<f4", order="C").tobytes(), len(matrix))
            )
            if native is not None
            else self.probabilities(matrix)
        )
        return ranked_decision(
            self.policy_id,
            observation,
            values,
            "calibrated_demonstration_preference",
            {
                "probability_target": "verified_demonstration_action",
                "safety_bound": False,
                "inference_backend": "native_cached" if native is not None else "python_reference",
            },
        )

    def save(self, directory: str | Path) -> None:
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "calibrator": self.calibrator},
            directory / f"{self.policy_id}.joblib",
        )
        if self.policy_id == "M03":
            exported = {
                "schema_version": 1,
                "feature_names": list(FEATURE_NAMES),
                "coef": self.model.coef_[0].tolist(),
                "intercept": float(self.model.intercept_[0]),
                "calibration": {
                    "kind": "platt",
                    "coef": float(self.calibrator.coef_[0, 0]),
                    "intercept": float(self.calibrator.intercept_[0]),
                },
            }
            (directory / "M03.linear.json").write_text(json.dumps(exported), encoding="utf8")
        else:
            self.model.save_model(directory / "M04.xgboost.json")
            booster = self.model.get_booster()
            config = json.loads(booster.save_config())
            base = config["learner"]["learner_model_param"]["base_score"]
            base = float(json.loads(base)[0] if base.startswith("[") else base)
            trees = []
            for serialized in booster.get_dump(dump_format="json"):
                nodes = []

                def visit(node, nodes=nodes):
                    if "leaf" in node:
                        nodes.append({"node_id": node["nodeid"], "leaf": node["leaf"]})
                    else:
                        nodes.append(
                            {
                                "node_id": node["nodeid"],
                                "feature_index": int(node["split"].removeprefix("f")),
                                "threshold": node["split_condition"],
                                "left": node["yes"],
                                "right": node["no"],
                                "missing": node["missing"],
                            }
                        )
                        for child in node["children"]:
                            visit(child)

                visit(json.loads(serialized))
                trees.append({"nodes": sorted(nodes, key=lambda node: node["node_id"])})
            exported = {
                "schema_version": 1,
                "feature_names": list(FEATURE_NAMES),
                "base_margin": float(np.log(base / (1 - base))),
                "trees": trees,
                "calibration": {
                    "kind": "platt",
                    "coef": float(self.calibrator.coef_[0, 0]),
                    "intercept": float(self.calibrator.intercept_[0]),
                },
            }
            (directory / "M04.trees.json").write_text(json.dumps(exported), encoding="utf8")

    @classmethod
    def load(cls, directory: str | Path, policy_id: str, backend="python"):
        import joblib

        loaded = joblib.load(Path(directory) / f"{policy_id}.joblib")
        result = cls(policy_id, loaded["model"], loaded["calibrator"])
        if backend == "native":
            from .. import _native

            constructor = getattr(
                _native, "LinearScorer" if policy_id == "M03" else "TreeScorer", None
            )
            if constructor is None:
                raise ModelRequiredError(
                    "installed native extension lacks cached trained scorer; rebuild package"
                )
            suffix = "linear" if policy_id == "M03" else "trees"
            result.native_scorer = constructor(
                (Path(directory) / f"{policy_id}.{suffix}.json").read_text(encoding="utf8")
            )
            if tuple(result.native_scorer.feature_names) != FEATURE_NAMES:
                raise ValueError("native model feature schema mismatch")
        elif backend != "python":
            raise ValueError("unknown candidate scoring backend")
        return result


class LinUCBPolicy:
    """Shared linear action-feature model with exact Sherman-Morrison updates."""

    policy_id = "M06"

    def __init__(self, dimension: int = 64, alpha: float = 0.2, seed: int = 0):
        self.alpha = alpha
        self.a_inv = np.eye(dimension, dtype=np.float64)
        self.b = np.zeros(dimension, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.projection = self.rng.choice(
            [-1.0, 1.0], size=(len(FEATURE_NAMES), dimension)
        ) / np.sqrt(dimension)
        self.updates = 0

    def reset(self, goal_id: str | None = None) -> None:
        pass

    def update(self, features: np.ndarray, reward: float) -> None:
        x = np.asarray(features, dtype=np.float64)
        if len(x) == len(FEATURE_NAMES):
            x = x @ self.projection
        transformed = self.a_inv @ x
        self.a_inv -= np.outer(transformed, transformed) / (1 + x @ transformed)
        self.b += reward * x
        self.updates += 1

    def predict(self, observation: Observation) -> Decision:
        x = feature_matrix(observation).astype(np.float64) @ self.projection
        if not len(x):
            return Decision(self.policy_id, None, "stop", reason="no_admissible_candidates")
        theta = self.a_inv @ self.b
        mean = x @ theta
        bonus = self.alpha * np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", x, self.a_inv, x)))
        ucb = mean + bonus
        # Deterministic tie-breaking is explicit. Propensity is 1 for the selected
        # action; this policy alone does not provide overlap for unchosen actions.
        index = int(np.argmax(ucb))
        return Decision(
            self.policy_id,
            observation.admissible[index].action_id,
            scores={
                c.action_id: float(v) for c, v in zip(observation.admissible, ucb, strict=True)
            },
            reason="linucb_myopic",
            diagnostics={"propensity": 1.0, "ucb_not_probability": True, "updates": self.updates},
        )

    def save(self, directory: str | Path) -> None:
        np.savez_compressed(
            Path(directory) / "M06.npz",
            a_inv=self.a_inv,
            b=self.b,
            alpha=self.alpha,
            updates=self.updates,
            projection=self.projection,
        )

    @classmethod
    def load(cls, directory: str | Path):
        data = np.load(Path(directory) / "M06.npz")
        result = cls(len(data["b"]), float(data["alpha"]))
        result.a_inv, result.b, result.updates = data["a_inv"], data["b"], int(data["updates"])
        result.projection = data["projection"]
        return result
