from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..contracts import Observation
from ..environments import CaseSpec, SoftwareEnvironment
from ..features import FEATURE_NAMES, STATE_KEYS, feature_matrix, state_features
from ..policies.rules import GuardedFSMPolicy


def assert_disjoint(*manifests: dict) -> None:
    seen_episodes, seen_groups, seen_seeds = set(), set(), set()
    for manifest in manifests:
        episodes = set(manifest["episode_ids"])
        groups = set(manifest["group_ids"])
        seeds = set(manifest["environment_seeds"])
        if seen_episodes & episodes or seen_groups & groups or seen_seeds & seeds:
            raise ValueError("data split overlap detected")
        seen_episodes |= episodes
        seen_groups |= groups
        seen_seeds |= seeds


def collect(specs: list[CaseSpec], directory: str | Path, exploration: float = 0.0) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not specs or len({s.split for s in specs}) != 1:
        raise ValueError("collection requires one nonempty declared split")
    split = specs[0].split
    path = directory / f"{split}.episodes.jsonl.gz"
    teacher = GuardedFSMPolicy()
    count, successes, transitions = 0, 0, 0
    with gzip.open(path, "wt", encoding="utf8") as stream:
        for spec in specs:
            rng = random.Random(spec.seed)
            steps = []
            with SoftwareEnvironment(spec) as environment:
                while not environment.terminal:
                    observation = environment.observe()
                    target = teacher.predict(observation)
                    action = target.candidate_id
                    propensity = 1 - exploration + exploration / len(observation.admissible)
                    if rng.random() < exploration:
                        action = rng.choice(observation.admissible).action_id
                        if action != target.candidate_id:
                            propensity = exploration / len(observation.admissible)
                    next_observation, outcome = environment.step(action)
                    steps.append(
                        {
                            "observation": observation.to_dict(),
                            "teacher_action": target.candidate_id,
                            "executed_action": action,
                            "propensity": propensity,
                            "outcome": asdict(outcome),
                            "next_state": state_features(next_observation).tolist(),
                        }
                    )
                success = environment.verify()
            record = {
                "spec": asdict(spec),
                "episode_id": spec.episode_id,
                "group_id": spec.group_id,
                "success": success,
                "steps": steps,
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
            successes += int(success)
            transitions += len(steps)
    manifest = {
        "schema_version": 1,
        "split": split,
        "episodes": count,
        "transitions": transitions,
        "verified_successes": successes,
        "exploration": exploration,
        "episode_ids": [s.episode_id for s in specs],
        "group_ids": sorted({s.group_id for s in specs}),
        "environment_seeds": sorted({s.seed for s in specs}),
        "feature_names": list(FEATURE_NAMES),
        "state_keys": list(STATE_KEYS),
        "generalization_scope": "held-out environment instances; owned templates shared across splits",
        "label_target": "demonstration preference; execution outcomes separately recorded",
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (directory / f"{split}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf8"
    )
    return manifest


def load_episodes(directory: str | Path, split: str) -> list[dict]:
    directory = Path(directory)
    manifest = json.loads((directory / f"{split}.manifest.json").read_text(encoding="utf8"))
    path = directory / manifest["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("dataset hash mismatch")
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def candidate_dataset(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for episode in episodes:
        for step in episode["steps"]:
            observation = Observation.from_dict(step["observation"])
            x.extend(feature_matrix(observation))
            y.extend(int(c.action_id == step["teacher_action"]) for c in observation.admissible)
    return np.asarray(x, dtype="float32"), np.asarray(y, dtype="int64")
