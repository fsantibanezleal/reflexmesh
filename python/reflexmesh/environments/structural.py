"""Frozen compositional transfer programs, excluded from every fitting dataset."""

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace

from .software import SoftwareEnvironment

PROGRAMS = {
    "S01": (
        "Remove negative inputs, square the retained numbers, and sum them into final.json.",
        ("retain_nonnegative", "square_each", "sum_values"),
    ),
    "S02": (
        "Deduplicate the inputs, sort ascending, and copy the resulting list to final.json.",
        ("deduplicate", "sort_values", "copy_list"),
    ),
    "S03": (
        "Take absolute values of inputs, sum them, and double that scalar into final.json.",
        ("absolute_each", "sum_values", "double_scalar"),
    ),
    "S04": (
        "Reverse the input list, keep its first three entries, and multiply those into final.json.",
        ("reverse_values", "first_three", "product_values"),
    ),
}


def transform(operation, value):
    return {
        "retain_nonnegative": lambda: [v for v in value if v >= 0],
        "square_each": lambda: [v * v for v in value],
        "sum_values": lambda: sum(value),
        "deduplicate": lambda: list(dict.fromkeys(value)),
        "sort_values": lambda: sorted(value),
        "copy_list": lambda: value,
        "absolute_each": lambda: [abs(v) for v in value],
        "double_scalar": lambda: 2 * value,
        "reverse_values": lambda: list(reversed(value)),
        "first_three": lambda: value[:3],
        "product_values": lambda: math.prod(value),
    }[operation]()


@dataclass(frozen=True)
class StructuralSpec:
    family: str
    variant: str
    seed: int
    split: str = "structural_transfer"
    max_steps: int = 16

    @property
    def goal(self):
        return PROGRAMS[self.family][0]

    @property
    def episode_id(self):
        return f"{self.split}:{self.family}:{self.variant}:{self.seed}"

    @property
    def group_id(self):
        return f"{self.split}:environment:{self.seed}"


def structural_matrix():
    return [
        StructuralSpec(family, variant, 81000 + seed)
        for family in PROGRAMS
        for variant in ("nominal", "boundary")
        for seed in range(3)
    ]


class StructuralEnvironment(SoftwareEnvironment):
    def _setup(self):
        rng = random.Random(self.spec.seed)
        self.inputs = [rng.randrange(-5, 8) for _ in range(7)]
        self._write("input.json", json.dumps(self.inputs))
        self.actions = {}
        for index, operation in enumerate(PROGRAMS[self.spec.family][1]):
            opaque = (
                "cap_"
                + hashlib.sha256(f"{self.spec.seed}:{operation}:{index}".encode()).hexdigest()[:10]
            )
            source = "input.json" if index == 0 else f"stage-{index}.json"
            destination = "final.json" if index == 2 else f"stage-{index + 1}.json"
            self.actions[opaque] = (operation, source, destination)
        if self.spec.variant == "boundary":
            self.actions["cap_distractor"] = ("copy_list", "input.json", "final.json")
        self.state.update(uncertain=1, artifact_inventory=["input.json"], input_ready=1)

    def _operations(self):
        return list(self.actions)

    def observe(self):
        observation = super().observe()
        candidates = []
        for candidate in observation.candidates:
            operation = candidate.arguments.get("operation")
            if operation in self.actions:
                role, source, destination = self.actions[operation]
                description = f"{role.replace('_', ' ')}: read JSON {source}, apply that operation, write JSON {destination}. Requires {source}."
                candidate = replace(
                    candidate,
                    arguments={**candidate.arguments, "description": description},
                    features={
                        **candidate.features,
                        "known": 0.0,
                        "prerequisites": float((self.root / source).exists()),
                    },
                )
            candidates.append(candidate)
        return replace(observation, candidates=tuple(candidates))

    def _execute(self, operation):
        role, source, destination = self.actions[operation]
        value = json.loads((self.root / source).read_text())
        result = transform(role, value)
        self._write(destination, json.dumps(result))
        self.state["artifact_inventory"] = sorted(
            p.name for p in self.root.iterdir() if p.is_file()
        )
        self.state["progress"] = (
            sum(
                (self.root / name).exists()
                for name in ("stage-1.json", "stage-2.json", "final.json")
            )
            / 3
        )
        return {"read": source, "written": destination, "observed_value": result}

    def expert_action(self):
        # Evaluation-only witness for agreement, never given to policy inputs.
        for action, (_, source, destination) in self.actions.items():
            if (self.root / source).exists() and not (self.root / destination).exists():
                return "a-" + action
        return "unresolved"

    def verify(self):
        try:
            expected = list(self.inputs)
            for operation in PROGRAMS[self.spec.family][1]:
                expected = transform(operation, expected)
            return json.loads((self.root / "final.json").read_text()) == expected
        except (FileNotFoundError, ValueError, TypeError):
            return False
