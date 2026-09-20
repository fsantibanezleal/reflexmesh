"""Gymnasium adapter for actual disposable software episodes."""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np

from ..features import FEATURE_NAMES, feature_matrix
from .software import SoftwareEnvironment, case_matrix

MAX_CANDIDATES = 7


def ordered(observation):
    return replace(
        observation, candidates=tuple(sorted(observation.candidates, key=lambda c: c.action_id))
    )


def gym_features(observation):
    observation = ordered(observation)
    values = feature_matrix(observation)
    output = np.zeros((MAX_CANDIDATES, len(FEATURE_NAMES)), dtype="float32")
    output[: len(values)] = values
    return output.ravel()


class SoftwareGym(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012 - Gymnasium class metadata contract

    def __init__(self, specs=None):
        super().__init__()
        self.specs = specs or case_matrix("train", 1)
        self.action_space = gym.spaces.Discrete(MAX_CANDIDATES)
        self.observation_space = gym.spaces.Box(
            -10, 10, (MAX_CANDIDATES * len(FEATURE_NAMES),), dtype=np.float32
        )
        self.environment = None
        self.current = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.environment:
            self.environment.close()
        spec = self.specs[int(self.np_random.integers(len(self.specs)))]
        self.environment = SoftwareEnvironment(spec)
        self.current = ordered(self.environment.observe())
        return gym_features(self.current), {"episode_id": spec.episode_id}

    def action_masks(self):
        mask = np.zeros(MAX_CANDIDATES, dtype=bool)
        if self.current:
            mask[: len(self.current.admissible)] = True
        return mask

    def step(self, action):
        if int(action) >= len(self.current.admissible):
            raise ValueError("masked action was unavailable")
        observation, outcome = self.environment.step(self.current.admissible[int(action)].action_id)
        self.current = ordered(observation)
        terminated = bool(outcome.success) or outcome.contract_violation
        truncated = self.environment.terminal and not terminated
        return (
            gym_features(self.current),
            outcome.reward,
            terminated,
            truncated,
            {
                "is_success": bool(outcome.success),
                "status": outcome.status,
                "contract_violation": outcome.contract_violation,
            },
        )

    def close(self):
        if self.environment:
            self.environment.close()
            self.environment = None
