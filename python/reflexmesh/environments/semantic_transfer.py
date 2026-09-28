"""Held-out semantic choice tasks with real file effects and independent truth.

The frozen core and structural-transfer matrices are never rewritten by this
suite. Each stage offers several type-compatible operations with identical
numeric tool features. The visible goal and operation descriptions carry the
distinguishing information; opaque IDs, hidden correct actions and verifier
outputs do not. Seeds change inputs, candidate identities/order and distractors.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace

from .software import SoftwareEnvironment
from .structural import transform

LIST_TO_LIST = (
    "retain_nonnegative",
    "square_each",
    "deduplicate",
    "sort_values",
    "absolute_each",
    "reverse_values",
    "first_three",
    "copy_list",
)
LIST_TO_SCALAR = ("sum_values", "product_values", "minimum_value", "maximum_value", "count_values")
SCALAR_TO_SCALAR = ("double_scalar", "increment_scalar", "negate_scalar", "square_scalar")

PROGRAMS = {
    "D01": ("retain_nonnegative", "square_each", "sum_values"),
    "D02": ("deduplicate", "sort_values", "copy_list"),
    "D03": ("absolute_each", "sum_values", "double_scalar"),
    "D04": ("reverse_values", "first_three", "product_values"),
    "D05": ("square_each", "sort_values", "sum_values"),
    "D06": ("retain_nonnegative", "reverse_values", "product_values"),
    "D07": ("deduplicate", "first_three", "sum_values"),
    "D08": ("absolute_each", "sort_values", "first_three"),
}

# Goals and capability descriptions deliberately use different wording. A
# candidate must be grounded in the requested transformation and current file
# stage; exact string equality with a hidden program name is insufficient.
GOAL_PHRASES = {
    "retain_nonnegative": ("discard entries below zero", "keep zero and positive entries"),
    "square_each": ("multiply every element by itself", "replace each entry by its square"),
    "deduplicate": (
        "keep only the first occurrence of each value",
        "remove repeated values while preserving their order",
    ),
    "sort_values": ("arrange entries from smallest to largest", "order the list ascending"),
    "absolute_each": ("replace entries by their magnitudes", "ignore every entry's sign"),
    "reverse_values": ("put the list in reverse order", "move the last entry to the front"),
    "first_three": ("keep the leading three entries", "select positions one through three"),
    "copy_list": ("preserve the list unchanged", "copy its entries without alteration"),
    "sum_values": ("add every entry together", "compute the total of the list"),
    "product_values": ("multiply all entries together", "compute their joint product"),
    "double_scalar": ("multiply the resulting number by two", "double the scalar result"),
}


def program_for(family: str, seed: int) -> tuple[str, str, str]:
    if family not in PROGRAMS:
        raise ValueError("unknown semantic transfer family")
    program = list(PROGRAMS[family])
    # Two families change the required transformation itself across seeds,
    # rather than merely changing the numbers or candidate positions.
    if family == "D07" and seed % 2:
        program[-1] = "product_values"
    if family == "D08" and seed % 2:
        program[1] = "reverse_values"
    return tuple(program)


def apply_transform(operation: str, value):
    if operation == "minimum_value":
        return min(value)
    if operation == "maximum_value":
        return max(value)
    if operation == "count_values":
        return len(value)
    if operation == "increment_scalar":
        return value + 1
    if operation == "negate_scalar":
        return -value
    if operation == "square_scalar":
        return value * value
    return transform(operation, value)


@dataclass(frozen=True)
class SemanticSpec:
    family: str
    variant: str
    seed: int
    split: str = "semantic_transfer"
    max_steps: int = 12

    def __post_init__(self) -> None:
        if self.family not in PROGRAMS or self.variant not in {"nominal", "boundary"}:
            raise ValueError("unknown semantic transfer family or variant")
        if type(self.seed) is not int or not 0 <= self.seed <= 999999:
            raise ValueError("invalid semantic transfer seed")

    @property
    def goal(self) -> str:
        stages = program_for(self.family, self.seed)
        phrases = [GOAL_PHRASES[operation][self.seed % 2] for operation in stages]
        return (
            "Read input.json. First "
            + phrases[0]
            + " into stage-1.json; then "
            + phrases[1]
            + " into stage-2.json; finally "
            + phrases[2]
            + " into final.json. Check the actual files after each effect."
        )

    @property
    def episode_id(self) -> str:
        return f"{self.split}:{self.family}:{self.variant}:{self.seed}"

    @property
    def group_id(self) -> str:
        return f"{self.split}:environment:{self.seed}"


def semantic_matrix(seeds: int = 8) -> list[SemanticSpec]:
    if not 1 <= seeds <= 20:
        raise ValueError("semantic transfer uses 1..20 preregistered seeds")
    return [
        SemanticSpec(family, variant, 82000 + seed)
        for family in PROGRAMS
        for variant in ("nominal", "boundary")
        for seed in range(seeds)
    ]


def _domain(operation: str) -> tuple[str, ...]:
    if operation in LIST_TO_LIST:
        return LIST_TO_LIST
    if operation in LIST_TO_SCALAR:
        return LIST_TO_SCALAR
    if operation in SCALAR_TO_SCALAR:
        return SCALAR_TO_SCALAR
    raise ValueError("operation has no compatible semantic domain")


class SemanticEnvironment(SoftwareEnvironment):
    """Execute opaque registered tools; keep the answer outside observations."""

    def _setup(self) -> None:
        rng = random.Random(self.spec.seed)
        self.inputs = [-9, -4, -1, 0, 2, 2, 5, 8]
        rng.shuffle(self.inputs)
        # The amount and signs vary while every input retains negatives,
        # duplicate positives and distinct values for nontrivial distractors.
        self.inputs.append(9 + self.spec.seed % 4)
        self._write("input.json", json.dumps(self.inputs))
        self.program = program_for(self.spec.family, self.spec.seed)
        self.expected: dict[str, object] = {}
        self.actions: dict[str, tuple[str, str, str, int]] = {}
        self.correct_ids: list[str] = []
        value = list(self.inputs)
        for stage, correct in enumerate(self.program):
            source = "input.json" if stage == 0 else f"stage-{stage}.json"
            destination = "final.json" if stage == 2 else f"stage-{stage + 1}.json"
            desired = apply_transform(correct, value)
            self.expected[destination] = desired
            alternatives = [
                operation
                for operation in _domain(correct)
                if operation != correct and apply_transform(operation, value) != desired
            ]
            rng.shuffle(alternatives)
            selected = [correct]
            outputs = [desired]
            for operation in alternatives:
                outcome = apply_transform(operation, value)
                if outcome not in outputs:
                    selected.append(operation)
                    outputs.append(outcome)
                if len(selected) == 3:
                    break
            if len(selected) != 3:
                raise RuntimeError("semantic fixture lacks distinct plausible choices")
            rng.shuffle(selected)
            for operation in selected:
                opaque = "cap_" + hashlib.sha256(
                    f"{self.spec.seed}:{self.spec.family}:{stage}:{operation}".encode()
                ).hexdigest()[:12]
                self.actions[opaque] = (operation, source, destination, stage)
                if operation == correct:
                    self.correct_ids.append(opaque)
            value = desired
        self.state.update(uncertain=1, artifact_inventory=["input.json"], input_ready=1)

    def _operations(self) -> list[str]:
        return list(self.actions)

    def observe(self):
        observation = super().observe()
        candidates = []
        for candidate in observation.candidates:
            opaque = candidate.arguments.get("operation")
            if opaque in self.actions:
                operation, source, destination, _ = self.actions[opaque]
                candidate = replace(
                    candidate,
                    arguments={
                        **candidate.arguments,
                        "description": (
                            f"{operation.replace('_', ' ')}: read JSON {source}, "
                            f"apply that operation, write JSON {destination}. Requires {source}."
                        ),
                    },
                    features={
                        **candidate.features,
                        "known": 0.0,
                        "prerequisites": float((self.root / source).exists()),
                        "completed": float((self.root / destination).exists()),
                    },
                )
            candidates.append(candidate)
        return replace(observation, candidates=tuple(candidates))

    def _execute(self, opaque: str) -> dict:
        operation, source, destination, stage = self.actions[opaque]
        value = json.loads((self.root / source).read_text(encoding="utf8"))
        result = apply_transform(operation, value)
        self._write(destination, json.dumps(result))
        self.state["artifact_inventory"] = sorted(p.name for p in self.root.iterdir() if p.is_file())
        self.state["progress"] = sum((self.root / p).exists() for p in self.expected) / 3
        return {
            "read": source,
            "written": destination,
            "observed_value": result,
            "stage": stage + 1,
        }

    def expert_action(self) -> str:
        # Evaluation-only witness; neither these IDs nor expected values enter
        # the policy observation or fitted feature vectors.
        for stage, destination in enumerate(("stage-1.json", "stage-2.json", "final.json")):
            path = self.root / destination
            if not path.exists() or json.loads(path.read_text(encoding="utf8")) != self.expected[destination]:
                return "a-" + self.correct_ids[stage]
        return "unresolved"

    def verify(self) -> bool:
        try:
            return all(
                json.loads((self.root / path).read_text(encoding="utf8")) == expected
                for path, expected in self.expected.items()
            )
        except (FileNotFoundError, ValueError, TypeError):
            return False
