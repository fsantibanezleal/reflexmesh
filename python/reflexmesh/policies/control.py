from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..contracts import Decision, ModelRequiredError, Observation, validate_decision
from ..features import FEATURE_NAMES, STATE_KEYS, feature_matrix, state_features
from .planning import proposal_binding


class TransitionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, len(STATE_KEYS) + 2),
        )

    def forward(self, values):
        return self.net(values)


class LookaheadPolicy:
    policy_id = "M08"

    def __init__(self, model: TransitionNetwork, horizon: int = 2, expansions: int = 24):
        self.model = model.eval().cpu()
        self.horizon, self.expansions = horizon, expansions

    def reset(self, goal_id: str | None = None) -> None:
        pass

    def consequences(self, observation: Observation) -> np.ndarray:
        with torch.inference_mode():
            return self.model(torch.from_numpy(feature_matrix(observation))).numpy()

    def predict(self, observation: Observation) -> Decision:
        if not observation.admissible:
            return Decision(self.policy_id, None, "stop", reason="no_admissible_candidates")
        first = self.consequences(observation)
        values = first[:, -2].copy()
        expanded = 0
        if self.horizon > 1:
            for i, predicted in enumerate(first):
                if expanded + len(observation.admissible) > self.expansions:
                    break
                new_state = dict(observation.state)
                new_state.update(
                    {k: float(v) for k, v in zip(STATE_KEYS, predicted[:-2], strict=True)}
                )
                hypothetical = replace(observation, state=new_state)
                second = self.consequences(hypothetical)
                values[i] += (
                    0.8 * (1 - float(np.clip(predicted[-1], 0, 1))) * float(np.max(second[:, -2]))
                )
                expanded += len(second)
        # Scores are expected utility, explicitly not calibrated probabilities.
        index = int(np.argmax(values))
        return Decision(
            self.policy_id,
            observation.admissible[index].action_id,
            scores={
                c.action_id: float(v) for c, v in zip(observation.admissible, values, strict=True)
            },
            reason="learned_transition_bounded_search",
            diagnostics={
                "horizon": self.horizon,
                "expansions": expanded,
                "effect_predicted": {
                    k: float(v) for k, v in zip(STATE_KEYS, first[index, :-2], strict=True)
                },
                "predicted_reward": float(first[index, -2]),
                "scores_are_utility": True,
            },
        )

    def save(self, directory: str | Path) -> None:
        torch.save(self.model.state_dict(), Path(directory) / "M08.pt")

    @classmethod
    def load(cls, directory: str | Path):
        model = TransitionNetwork()
        model.load_state_dict(
            torch.load(Path(directory) / "M08.pt", map_location="cpu", weights_only=True)
        )
        return cls(model)


def deferral_features(observation: Observation, decision: Decision) -> np.ndarray:
    values = sorted(decision.scores.values(), reverse=True)
    return np.asarray(
        [
            decision.confidence or 0,
            (values[0] - values[1]) if len(values) > 1 else 0,
            observation.budget_remaining,
            observation.deadline_remaining_ms / 10000,
            float(observation.state.get("uncertain", 0)),
            float(observation.state.get("stale", 0)),
            float(observation.state.get("last_action_failed", 0)),
            len(observation.history) / 20,
        ],
        dtype=np.float64,
    )


class DeferralPolicy:
    policy_id = "M11"

    def __init__(self, fast, benefit_model, planner, threshold: float = 0.5):
        if benefit_model is None:
            raise ModelRequiredError("deferral requires fitted paired-outcome benefit model")
        self.fast, self.benefit_model, self.planner, self.threshold = (
            fast,
            benefit_model,
            planner,
            threshold,
        )

    def reset(self, goal_id: str | None = None) -> None:
        self.fast.reset(goal_id)

    def predict(self, observation: Observation) -> Decision:
        fast = self.fast.predict(observation)
        gate_x = deferral_features(observation, fast).reshape(1, -1)
        probability = self.benefit_model.predict_proba(gate_x)[0]
        classes = list(self.benefit_model.classes_)
        benefit = float(probability[classes.index(1)]) if 1 in classes else 0.0
        gain = (
            float(self.benefit_model.predict_gain(gate_x)[0])
            if hasattr(self.benefit_model, "predict_gain")
            else benefit - self.threshold
        )
        diagnostics = {
            **fast.diagnostics,
            "benefit_probability": benefit,
            "delegated": False,
            "threshold": self.threshold,
            "paired_outcome_target": True,
            "expected_incremental_utility": gain,
        }
        if benefit > self.threshold and gain > 0 and observation.budget_remaining > 0.1:
            if self.planner is None:
                raise ModelRequiredError("learned deferral selected unavailable planner")
            slow = self.planner.plan(observation)
            validate_decision(observation, slow)
            return replace(
                slow,
                policy_id=self.policy_id,
                diagnostics={**slow.diagnostics, **diagnostics, "delegated": True},
            )
        return replace(
            fast,
            policy_id=self.policy_id,
            diagnostics=diagnostics,
            reason="learned_incremental_benefit_gate",
        )


class MetaFastPolicy:
    """The exact recurrent/effect action evaluated when fitting the M12 gate."""

    policy_id = "M12-fast"

    def __init__(
        self, recurrent, effect_model, effect_weight=0.15, use_memory=True, use_effects=True
    ):
        self.recurrent, self.effect_model = recurrent, effect_model
        self.effect_weight, self.use_memory, self.use_effects = (
            effect_weight,
            use_memory,
            use_effects,
        )
        self.expected = None
        self.residual_ema = 0.0

    def reset(self, goal_id=None):
        self.recurrent.reset(goal_id)
        self.expected = None
        self.residual_ema = 0.0

    def predict(self, observation):
        if not self.use_memory:
            self.recurrent.reset(observation.goal_id)
        residual = (
            None
            if self.expected is None
            else float(np.mean(np.abs(state_features(observation) - self.expected)))
        )
        if residual is not None:
            self.residual_ema = 0.8 * self.residual_ema + 0.2 * residual
        fast = self.recurrent.predict(observation)
        if not observation.admissible:
            return fast
        diagnostics = {
            **fast.diagnostics,
            "prediction_residual": residual,
            "residual_ema": self.residual_ema,
        }
        if not self.use_effects:
            return replace(fast, diagnostics=diagnostics)
        predicted = self.effect_model.predict(observation)
        scores = {
            c.action_id: fast.scores.get(c.action_id, 0)
            + self.effect_weight * predicted.scores.get(c.action_id, 0)
            for c in observation.admissible
        }
        action = max(scores, key=scores.get)
        effects = self.effect_model.consequences(observation)
        selected = next(i for i, c in enumerate(observation.admissible) if c.action_id == action)
        self.expected = effects[selected, :-2]
        diagnostics.update(
            effect_predicted=dict(zip(STATE_KEYS, map(float, self.expected), strict=True)),
            effect_weight=self.effect_weight,
            scores_are_utility=True,
            recurrent_confidence=fast.confidence,
        )
        diagnostics.pop("probability_target", None)
        return replace(
            fast,
            policy_id=self.policy_id,
            candidate_id=action,
            confidence=None,
            scores=scores,
            reason="recurrent_effect_utility_blend",
            diagnostics=diagnostics,
        )


class Metacontroller(DeferralPolicy):
    policy_id = "M12"

    def __init__(
        self,
        recurrent,
        effect_model,
        benefit_model,
        planner,
        threshold=0.5,
        effect_weight=0.15,
        use_memory=True,
        use_effects=True,
        use_deferral=True,
        use_invalidation=True,
        surprise_threshold=0.25,
        planner_ttl_ms=10000,
        max_planner_calls=3,
    ):
        super().__init__(
            MetaFastPolicy(recurrent, effect_model, effect_weight, use_memory, use_effects),
            benefit_model,
            planner,
            threshold,
        )
        self.use_deferral, self.use_invalidation = use_deferral, use_invalidation
        self.surprise_threshold, self.planner_ttl_ms = surprise_threshold, planner_ttl_ms
        self.max_planner_calls = max_planner_calls
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflexmesh-M12")
        self.pending = None
        self.jobs = []
        self.discards = []
        self.pending_fast = None
        self.pending_diagnostics = None

    def reset(self, goal_id=None):
        try:
            for job in self.jobs:
                job.result()
        finally:
            super().reset(goal_id)
            self.pending = None
            self.jobs = []
            self.discards = []
            self.pending_fast = None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def finish_episode(self):
        return {
            "planner_calls": len(self.jobs),
            "completed_proposals": [job.result().to_dict() for job in self.jobs],
            "discarded": self.discards,
            "background_work_accounted": True,
        }

    @staticmethod
    def _binding(observation):
        return proposal_binding(observation)

    def _poll(self, observation, diagnostics):
        if self.use_invalidation and self.origin != self._binding(observation):
            self.discards.append({"reason": "binding_changed", "origin": self.origin})
            self.pending = None
            return None
        remaining = self.expires - time.monotonic()
        if remaining <= 0:
            self.discards.append({"reason": "ttl_expired", "origin": self.origin})
            self.pending = None
            return replace(
                self.pending_fast,
                policy_id=self.policy_id,
                diagnostics={**diagnostics, "planner_timeout": True},
            )
        try:
            proposal = self.pending.result(timeout=min(0.15, remaining))
        except FutureTimeout:
            return Decision(
                self.policy_id,
                None,
                "wait",
                reason="affected_resource_deliberation_hold",
                diagnostics={
                    **diagnostics,
                    "deliberation_hold": True,
                    "planner_pending": True,
                    "hold_remaining_ms": remaining * 1000,
                },
            )
        self.pending = None
        validate_decision(observation, proposal)
        self.fast.expected = None
        if self.fast.use_effects and proposal.candidate_id is not None:
            predictions = self.fast.effect_model.consequences(observation)
            selected = next(
                i
                for i, c in enumerate(observation.admissible)
                if c.action_id == proposal.candidate_id
            )
            self.fast.expected = predictions[selected, :-2]
            diagnostics = {
                **diagnostics,
                "effect_predicted": dict(
                    zip(STATE_KEYS, map(float, self.fast.expected), strict=True)
                ),
            }
        return replace(
            proposal,
            policy_id=self.policy_id,
            diagnostics={
                **proposal.diagnostics,
                **diagnostics,
                "planner_applied": True,
                "planner_pending": False,
            },
        )

    def predict(self, observation):
        # Urgent cancellation is handled while an affected-work proposal waits.
        # The next poll invalidates the old plan; authority is always rechecked.
        urgent = next(
            (c for c in observation.admissible if c.arguments.get("operation") == "cancel"), None
        )
        if urgent and observation.state.get("cancel_requested"):
            return Decision(
                self.policy_id, urgent.action_id, reason="urgent_cancellation_during_deliberation"
            )
        if self.pending is not None:
            decision = self._poll(observation, {**self.pending_diagnostics, "delegated": False})
            if decision is not None:
                return decision
        fast = self.fast.predict(observation)
        if not observation.admissible:
            return replace(fast, policy_id=self.policy_id)
        x = deferral_features(observation, fast).reshape(1, -1)
        probability = self.benefit_model.predict_proba(x)[0]
        classes = list(self.benefit_model.classes_)
        benefit = float(probability[classes.index(1)]) if 1 in classes else 0.0
        gain = (
            float(self.benefit_model.predict_gain(x)[0])
            if hasattr(self.benefit_model, "predict_gain")
            else benefit - self.threshold
        )
        residual = fast.diagnostics.get("prediction_residual")
        surprising = (
            self.fast.use_effects and residual is not None and residual > self.surprise_threshold
        )
        # Explicit epistemic insufficiency is a separate conservative request,
        # not claimed to be the fitted value-of-computation optimum.
        insufficient = bool(observation.state.get("uncertain"))
        request = (benefit > self.threshold and gain > 0) or surprising or insufficient
        diagnostics = {
            **fast.diagnostics,
            "benefit_probability": benefit,
            "expected_incremental_utility": gain,
            "threshold": self.threshold,
            "delegated": False,
            "planner_calls": len(self.jobs),
            "ablations": {
                "memory": self.fast.use_memory,
                "effects": self.fast.use_effects,
                "deferral": self.use_deferral,
                "invalidation": self.use_invalidation,
            },
        }
        if (
            self.use_deferral
            and request
            and len(self.jobs) < self.max_planner_calls
            and observation.budget_remaining > 0.1
        ):
            if self.planner is None:
                raise ModelRequiredError("metacontroller requested unavailable planner")
            self.origin = self._binding(observation)
            self.expires = (
                time.monotonic()
                + min(self.planner_ttl_ms, observation.deadline_remaining_ms) / 1000
            )
            self.pending = self.executor.submit(self.planner.plan, observation)
            self.jobs.append(self.pending)
            self.pending_fast = fast
            diagnostics.update(
                delegated=True,
                delegation_trigger="insufficient_evidence"
                if insufficient
                else "surprise"
                if surprising
                else "positive_expected_utility",
                planner_model=self.planner.model_id,
            )
            self.pending_diagnostics = diagnostics
            return self._poll(observation, diagnostics)
        return replace(
            fast,
            policy_id=self.policy_id,
            reason="persistent_outcome_aware_metacontrol",
            diagnostics=diagnostics,
        )
