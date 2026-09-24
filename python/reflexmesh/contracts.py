"""Portable, JSON serializable policy contracts.

Policy observations contain only information available before a decision. Task
oracles and counterfactual labels belong to training records, never observations.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

Mode = Literal["act", "observe", "wait", "delegate", "stop"]


def _json_record(value, name):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    try:
        serialized = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    if len(serialized.encode("utf8")) > 1_048_576:
        raise ValueError(f"{name} exceeds one MiB")


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} below minimum")


def _numeric_record(value, name):
    _json_record(value, name)
    for item in value.values():
        _number(item, name)


def _strings(values, name):
    if not isinstance(values, (tuple, list)) or any(not isinstance(item, str) for item in values):
        raise ValueError(f"{name} must be a string sequence")


@dataclass(frozen=True)
class Candidate:
    action_id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    effects: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    resource_versions: dict[str, int] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    allowed: bool = True
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action_id, str)
            or not isinstance(self.tool, str)
            or not self.action_id
            or not self.tool
        ):
            raise ValueError("candidate action_id and tool must be nonempty")
        _json_record(self.arguments, "arguments")
        _numeric_record(self.features, "features")
        _json_record(self.resource_versions, "resource_versions")
        if any(type(value) is not int or value < 0 for value in self.resource_versions.values()):
            raise ValueError("resource revisions must be nonnegative integers")
        _strings(self.effects, "effects")
        _strings(self.required_capabilities, "capabilities")
        if type(self.allowed) is not bool:
            raise ValueError("allowed must be boolean")


@dataclass(frozen=True)
class Observation:
    event_id: str
    goal_id: str
    revision: int
    event_kind: str
    state: dict[str, Any]
    candidates: tuple[Candidate, ...]
    capabilities: tuple[str, ...] = ()
    budget_remaining: float = 1.0
    deadline_remaining_ms: float = 10_000.0
    history: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, (tuple, list)) or any(
            not isinstance(c, Candidate) for c in self.candidates
        ):
            raise ValueError("candidates must be Candidate records")
        if len(self.candidates) > 1024:
            raise ValueError("too many candidates")
        ids = [candidate.action_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate identifiers must be unique")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be nonnegative")
        if any(
            not isinstance(value, str) or not value
            for value in (self.event_id, self.goal_id, self.event_kind)
        ):
            raise ValueError("observation identifiers must be nonempty strings")
        _json_record(self.state, "state")
        _numeric_record(self.features, "features")
        _strings(self.capabilities, "capabilities")
        _strings(self.history, "history")
        _number(self.budget_remaining, "budget", 0)
        _number(self.deadline_remaining_ms, "deadline", 0)

    @property
    def admissible(self) -> tuple[Candidate, ...]:
        capabilities = set(self.capabilities)
        return tuple(
            c
            for c in self.candidates
            if c.allowed and set(c.required_capabilities).issubset(capabilities)
        )

    def candidate(self, action_id: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.action_id == action_id:
                return candidate
        raise KeyError(action_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Observation:
        fields = dict(value)
        fields["candidates"] = tuple(
            Candidate(
                **{
                    **c,
                    "effects": tuple(c.get("effects", ())),
                    "required_capabilities": tuple(c.get("required_capabilities", ())),
                }
            )
            for c in fields.get("candidates", ())
        )
        fields["capabilities"] = tuple(fields.get("capabilities", ()))
        fields["history"] = tuple(fields.get("history", ()))
        return cls(**fields)


@dataclass(frozen=True)
class Decision:
    policy_id: str
    candidate_id: str | None
    mode: Mode = "act"
    confidence: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id required")
        if self.candidate_id is not None and not isinstance(self.candidate_id, str):
            raise ValueError("candidate_id must be string or null")
        if self.mode not in {"act", "observe", "wait", "delegate", "stop"}:
            raise ValueError("invalid decision mode")
        if self.mode == "act" and self.candidate_id is None:
            raise ValueError("act decisions need a candidate")
        if self.confidence is not None:
            _number(self.confidence, "confidence")
            if not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be within [0, 1]")
        _numeric_record(self.scores, "scores")
        _json_record(self.diagnostics, "diagnostics")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    goal_id: str
    action_id: str | None
    success: bool | None
    terminal: bool
    reward: float
    duration_ms: float
    status: str
    effects: dict[str, Any] = field(default_factory=dict)
    contract_violation: bool = False
    observations: tuple[str, ...] = ()

    def __post_init__(self):
        if self.success is not None and type(self.success) is not bool:
            raise ValueError("success must be boolean or unresolved null")
        if type(self.terminal) is not bool or type(self.contract_violation) is not bool:
            raise ValueError("outcome flags must be boolean")
        _number(self.reward, "reward")
        _number(self.duration_ms, "duration", 0)
        _json_record(self.effects, "effects")
        _strings(self.observations, "observations")
        if not isinstance(self.goal_id, str) or not isinstance(self.status, str):
            raise TypeError("outcome identifiers must be strings")


class Policy(Protocol):
    policy_id: str

    def predict(self, observation: Observation) -> Decision: ...

    def reset(self, goal_id: str | None = None) -> None: ...


class Planner(Protocol):
    """Synchronous planner adapter; async servers can run this in a worker."""

    model_id: str

    def plan(self, observation: Observation) -> Decision: ...


def validate_decision(observation: Observation, decision: Decision) -> None:
    """Admission only; effectors must revalidate resource versions at dispatch."""
    if decision.candidate_id is not None:
        allowed = {c.action_id for c in observation.admissible}
        if decision.candidate_id not in allowed:
            raise ValueError("decision names an unavailable or unauthorized candidate")


class ModelRequiredError(RuntimeError):
    """A required trained policy or real language model has not been supplied."""
