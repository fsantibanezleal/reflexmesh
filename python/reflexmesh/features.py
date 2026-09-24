"""Fixed causal feature schema shared by fitted policies and native exports."""

from __future__ import annotations

import math
import re

import numpy as np

from .contracts import Candidate, Observation

STATE_KEYS = (
    "stale",
    "duplicate",
    "dependency_missing",
    "uncertain",
    "cancel_requested",
    "running",
    "failed",
    "write_unknown",
    "confirmed",
    "pressure",
    "job_ready",
    "needs_validation",
    "valid",
    "source_exists",
    "destination_exists",
    "index_stale",
    "config_missing",
    "test_passed",
    "input_ready",
    "transformed",
    "result_ready",
    "conflict",
    "recovered",
    "drift",
    "attempts",
    "progress",
    "last_action_failed",
)
OPERATIONS = (
    "refresh",
    "ack_event",
    "wait_dependency",
    "move",
    "index",
    "validate",
    "quarantine",
    "accept",
    "run_test",
    "repair_config",
    "http_get",
    "http_post",
    "http_status",
    "cancel",
    "restart",
    "cleanup",
    "write",
    "extract",
    "transform",
    "report",
    "rebase",
    "cas_write",
    "collect",
    "spawn",
    "wait",
    "reject",
    "inspect",
    "invoke",
    "reorder",
    "backoff",
    "reconcile",
    "finish",
)
BASE_FEATURE_NAMES = (
    tuple("state." + k for k in STATE_KEYS)
    + (
        "budget",
        "deadline",
        "history_length",
        "candidate.goal_overlap",
        "candidate.cost",
        "candidate.duration",
        "candidate.write",
        "candidate.destructive",
        "candidate.known",
        "candidate.prerequisites",
        "candidate.completed",
    )
    + tuple("operation." + op for op in OPERATIONS)
)
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    f"interaction.{op}.{key}" for op in OPERATIONS for key in STATE_KEYS
)
FEATURE_SCHEMA_VERSION = 1


def _bounded(value: object, divisor: float = 1.0) -> float:
    try:
        value = float(value) / divisor
    except (TypeError, ValueError):
        return 0.0
    return max(-10.0, min(10.0, value)) if math.isfinite(value) else 0.0


def candidate_features(observation: Observation, candidate: Candidate) -> np.ndarray:
    """No IDs, family labels, filenames, future outcomes, or oracle values enter X."""
    state = observation.state
    goal_tokens = set(re.findall(r"[a-z]+", str(state.get("goal", "")).lower()))
    description = str(candidate.arguments.get("description", candidate.tool))
    action_tokens = set(re.findall(r"[a-z]+", description.lower()))
    overlap = len(goal_tokens & action_tokens) / max(1, len(action_tokens))
    operation = str(candidate.arguments.get("operation", candidate.tool))
    values = [_bounded(state.get(k, 0), 10 if k == "attempts" else 1) for k in STATE_KEYS]
    values += [
        _bounded(observation.budget_remaining),
        _bounded(observation.deadline_remaining_ms, 10_000),
        _bounded(len(observation.history), 20),
        overlap,
        _bounded(candidate.features.get("cost", 0.01)),
        _bounded(candidate.features.get("duration_ms", 1), 1000),
        float("write" in candidate.effects),
        float("delete" in candidate.effects),
        _bounded(candidate.features.get("known", 1)),
        _bounded(candidate.features.get("prerequisites", 1)),
        _bounded(candidate.features.get("completed", 0)),
    ]
    values += [float(operation == op) for op in OPERATIONS]
    values += [
        float(operation == op) * _bounded(state.get(k, 0), 10 if k == "attempts" else 1)
        for op in OPERATIONS
        for k in STATE_KEYS
    ]
    return np.asarray(values, dtype=np.float32)


def feature_matrix(observation: Observation, admissible_only: bool = True) -> np.ndarray:
    candidates = observation.admissible if admissible_only else observation.candidates
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack([candidate_features(observation, c) for c in candidates])


def state_features(observation: Observation) -> np.ndarray:
    """Candidate-independent causal state for the recurrent encoder."""
    return np.asarray(
        [_bounded(observation.state.get(k, 0), 10 if k == "attempts" else 1) for k in STATE_KEYS],
        dtype=np.float32,
    )
