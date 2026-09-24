from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..contracts import Observation
from ..features import FEATURE_NAMES, STATE_KEYS, candidate_features, feature_matrix, state_features
from ..policies.classical import CandidateScorer, LinUCBPolicy
from ..policies.control import LookaheadPolicy, TransitionNetwork
from ..policies.recurrent import RecurrentNetwork, RecurrentPolicy
from .dataset import candidate_dataset


def calibration_report(probability: np.ndarray, truth: np.ndarray) -> dict:
    probability = np.clip(np.asarray(probability), 1e-7, 1 - 1e-7)
    truth = np.asarray(truth)
    bins = []
    for low in np.arange(0, 1, 0.1):
        selected = (probability >= low) & (probability < low + 0.1)
        if selected.any():
            bins.append(
                {
                    "lower": float(low),
                    "upper": float(low + 0.1),
                    "count": int(selected.sum()),
                    "confidence": float(probability[selected].mean()),
                    "accuracy": float(truth[selected].mean()),
                }
            )
    return {
        "n": len(truth),
        "target": "demonstration_preference",
        "bins": bins,
        "brier": float(np.mean((probability - truth) ** 2)),
        "log_loss": float(
            -np.mean(truth * np.log(probability) + (1 - truth) * np.log(1 - probability))
        ),
    }


def train_classical(train: list[dict], calibration: list[dict], directory: str | Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x, y = candidate_dataset(train)
    cx, cy = candidate_dataset(calibration)
    results = {}
    for method, model in [
        ("M03", LogisticRegression(max_iter=600, C=5, solver="liblinear", random_state=17)),
        (
            "M04",
            XGBClassifier(
                n_estimators=160,
                max_depth=5,
                learning_rate=0.08,
                n_jobs=2,
                random_state=17,
                objective="binary:logistic",
                tree_method="hist",
            ),
        ),
    ]:
        start = time.perf_counter()
        model.fit(x, y)
        scorer = CandidateScorer(method, model)
        raw = np.log(
            np.clip(scorer.probabilities(cx), 1e-7, 1 - 1e-7)
            / (1 - np.clip(scorer.probabilities(cx), 1e-7, 1 - 1e-7))
        )
        calibrator = LogisticRegression(C=10, max_iter=300).fit(raw.reshape(-1, 1), cy)
        scorer.calibrator = calibrator
        scorer.save(directory)
        results[method] = {
            "training_seconds": time.perf_counter() - start,
            "training_candidates": len(y),
            "calibration": calibration_report(scorer.probabilities(cx), cy),
            "fit_configuration": {
                key: value
                for key, value in model.get_params().items()
                if value is None
                or isinstance(value, (str, int, bool))
                or isinstance(value, float)
                and np.isfinite(value)
            },
            "calibrator_configuration": calibrator.get_params(),
            "device": "cpu",
        }
    # LinUCB receives only the actual action and observed one-step reward from each
    # exploratory log, never full-information labels for unexecuted actions.
    linucb = LinUCBPolicy(dimension=64, seed=17)
    for episode in train:
        for step in episode["steps"]:
            obs = Observation.from_dict(step["observation"])
            linucb.update(
                candidate_features(obs, obs.candidate(step["executed_action"])),
                step["outcome"]["reward"],
            )
    linucb.save(directory)
    results["M06"] = {
        "updates": linucb.updates,
        "reward_target": "observed_one_step_reward",
        "sequential_interpretation": "myopic",
        "fit_configuration": {
            "dimension": 64,
            "seed": 17,
            "alpha": linucb.alpha,
            "initial_ridge": 1.0,
        },
    }
    (directory / "classical-training.json").write_text(
        json.dumps(results, indent=2), encoding="utf8"
    )
    return results


def _sequence_arrays(episodes: list[dict]):
    result = []
    for episode in episodes:
        sequence = []
        for step in episode["steps"]:
            observation = Observation.from_dict(step["observation"])
            ids = [c.action_id for c in observation.admissible]
            sequence.append(
                (
                    state_features(observation),
                    feature_matrix(observation),
                    ids.index(step["teacher_action"]),
                )
            )
        result.append(sequence)
    return result


def train_recurrent(
    train: list[dict],
    calibration: list[dict],
    directory: str | Path,
    epochs: int = 25,
    seed: int = 17,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    directory = Path(directory)
    model = RecurrentNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    sequences = _sequence_arrays(train)
    rng = np.random.default_rng(seed)
    losses = []
    start = time.perf_counter()
    for epoch in range(epochs):
        total, count = 0.0, 0
        indices = rng.permutation(len(sequences))
        for offset in range(0, len(indices), 32):
            batch = [sequences[i] for i in indices[offset : offset + 32]]
            hidden = torch.zeros(1, len(batch), model.hidden_size, device=device)
            loss = torch.tensor(0.0, device=device)
            steps_count = 0
            for t in range(max(map(len, batch))):
                active = [i for i, sequence in enumerate(batch) if t < len(sequence)]
                max_candidates = max(len(batch[i][t][1]) for i in active)
                states = np.zeros((len(batch), len(STATE_KEYS)), dtype="float32")
                candidates = np.zeros(
                    (len(batch), max_candidates, len(FEATURE_NAMES)), dtype="float32"
                )
                masks = np.zeros((len(batch), max_candidates), dtype=bool)
                targets = np.zeros(len(batch), dtype="int64")
                for i in active:
                    state, features, target = batch[i][t]
                    states[i], candidates[i, : len(features)], targets[i] = state, features, target
                    masks[i, : len(features)] = True
                logits, hidden = model(
                    torch.from_numpy(states).to(device),
                    torch.from_numpy(candidates).to(device),
                    hidden,
                )
                logits = logits.masked_fill(~torch.from_numpy(masks).to(device), -1e9)
                loss = loss + nn.functional.cross_entropy(
                    logits[active], torch.from_numpy(targets[active]).to(device)
                )
                steps_count += 1
            optimizer.zero_grad()
            (loss / steps_count).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            count += steps_count
        losses.append(total / max(1, count))
    policy = RecurrentPolicy(model.cpu())
    # Tune one temperature on disjoint calibration sequences; no test labels used.
    calibration_logits, calibration_targets = [], []
    with torch.no_grad():
        for sequence in _sequence_arrays(calibration):
            hidden = torch.zeros(1, 1, model.hidden_size)
            for state, candidates, target in sequence:
                logits, hidden = model(
                    torch.from_numpy(state).unsqueeze(0),
                    torch.from_numpy(candidates).unsqueeze(0),
                    hidden,
                )
                calibration_logits.append(logits[0])
                calibration_targets.append(target)
    temperatures = np.exp(np.linspace(-2, 2, 41))
    losses_t = [
        np.mean(
            [
                float(
                    nn.functional.cross_entropy((logit / temp).unsqueeze(0), torch.tensor([target]))
                )
                for logit, target in zip(calibration_logits, calibration_targets, strict=True)
            ]
        )
        for temp in temperatures
    ]
    policy.temperature = float(temperatures[int(np.argmin(losses_t))])
    metadata = policy.save(directory)
    result = {
        "epochs": epochs,
        "seed": seed,
        "device": device,
        "training_seconds": time.perf_counter() - start,
        "losses": losses,
        "temperature": policy.temperature,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "sequence_batch_size": 32,
        "gradient_norm_cap": 1.0,
        "temperature_grid": "exp(linspace(-2,2,41))",
        **metadata,
    }
    (directory / "recurrent-training.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    return result


def train_transition(
    train: list[dict],
    validation: list[dict],
    directory: str | Path,
    epochs: int = 40,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(31)
    torch.set_num_threads(2)

    def tensors(episodes):
        x, y = [], []
        for episode in episodes:
            for step in episode["steps"]:
                obs = Observation.from_dict(step["observation"])
                x.append(candidate_features(obs, obs.candidate(step["executed_action"])))
                # This head predicts observed episode termination, not success.
                # Nonterminal success=None is not relabeled as task failure.
                y.append(
                    step["next_state"]
                    + [step["outcome"]["reward"], float(step["outcome"]["terminal"])]
                )
        return torch.tensor(np.asarray(x), device=device), torch.tensor(
            np.asarray(y, dtype="float32"), device=device
        )

    x, y = tensors(train)
    vx, vy = tensors(validation)
    model = TransitionNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    for epoch in range(epochs):
        ordering = torch.randperm(len(x), device=device)
        for indices in ordering.split(128):
            predicted = model(x[indices])
            # Reward and terminal heads receive the same measured target weight.
            loss = nn.functional.mse_loss(predicted, y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        errors = ((model(vx) - vy) ** 2).mean(0).cpu().numpy()
    LookaheadPolicy(model.cpu()).save(directory)
    result = {
        "epochs": epochs,
        "losses": losses,
        "device": device,
        "seed": 31,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "batch_size": 128,
        "validation_mse": float(errors.mean()),
        "validation_state_mse": dict(zip(STATE_KEYS, map(float, errors[:-2]), strict=True)),
        "validation_reward_mse": float(errors[-2]),
        "validation_terminal_mse": float(errors[-1]),
        "last_head_target": "observed episode termination; not success probability",
    }
    (Path(directory) / "transition-training.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    return result
