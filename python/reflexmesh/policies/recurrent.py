from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..contracts import Decision, Observation
from ..features import FEATURE_NAMES, STATE_KEYS, feature_matrix, state_features


class RecurrentNetwork(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(len(STATE_KEYS), hidden_size, batch_first=True)
        self.candidate = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 48), nn.Tanh())
        self.scorer = nn.Sequential(nn.Linear(48 + hidden_size, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, state, candidates, hidden):
        memory, next_hidden = self.gru(state.unsqueeze(1), hidden)
        context = memory[:, -1, :].unsqueeze(1).expand(-1, candidates.size(1), -1)
        encoded = self.candidate(candidates)
        logits = self.scorer(torch.cat((encoded, context), dim=-1)).squeeze(-1)
        return logits, next_hidden


class RecurrentPolicy:
    policy_id = "M05"

    def __init__(self, model: RecurrentNetwork, temperature: float = 1.0):
        self.model = model.eval().cpu()
        self.temperature = temperature
        self.hidden: dict[str, torch.Tensor] = {}

    def reset(self, goal_id: str | None = None) -> None:
        if goal_id is None:
            self.hidden.clear()
        else:
            self.hidden.pop(goal_id, None)

    def predict(self, observation: Observation) -> Decision:
        candidates = observation.admissible
        if not candidates:
            return Decision(self.policy_id, None, "stop", reason="no_admissible_candidates")
        hidden = self.hidden.get(observation.goal_id, torch.zeros(1, 1, self.model.hidden_size))
        with torch.inference_mode():
            logits, hidden = self.model(
                torch.from_numpy(state_features(observation)).unsqueeze(0),
                torch.from_numpy(feature_matrix(observation)).unsqueeze(0),
                hidden,
            )
            values = torch.softmax(logits[0] / self.temperature, dim=0).numpy()
        self.hidden[observation.goal_id] = hidden.detach()
        index = int(np.argmax(values))
        return Decision(
            self.policy_id,
            candidates[index].action_id,
            confidence=float(values[index]),
            scores={c.action_id: float(v) for c, v in zip(candidates, values, strict=True)},
            reason="recurrent_behavior_cloning",
            diagnostics={
                "temperature": self.temperature,
                "hidden_norm": float(hidden.norm()),
                "probability_target": "demonstration_action",
            },
        )

    def save(self, directory: str | Path, export_onnx: bool = True) -> dict:
        directory = Path(directory)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_size": self.model.hidden_size,
                "temperature": self.temperature,
            },
            directory / "M05.pt",
        )
        metadata = {
            "hidden_size": self.model.hidden_size,
            "temperature": self.temperature,
            "feature_names": list(FEATURE_NAMES),
            "state_keys": list(STATE_KEYS),
        }
        if export_onnx:
            inputs = (
                torch.zeros(1, len(STATE_KEYS)),
                torch.zeros(1, 6, len(FEATURE_NAMES)),
                torch.zeros(1, 1, self.model.hidden_size),
            )
            torch.onnx.export(
                self.model,
                inputs,
                directory / "M05.onnx",
                opset_version=17,
                dynamo=False,
                input_names=["state", "candidates", "hidden"],
                output_names=["logits", "next_hidden"],
                dynamic_axes={
                    "candidates": {1: "candidate_count"},
                    "logits": {1: "candidate_count"},
                },
            )
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(directory / "M05.onnx"), providers=["CPUExecutionProvider"]
            )
            rng = np.random.default_rng(812)
            maximum = 0.0
            for count in (1, 3, 6, 7):
                arrays = {
                    "state": rng.normal(size=(1, len(STATE_KEYS))).astype("float32"),
                    "candidates": rng.normal(size=(1, count, len(FEATURE_NAMES))).astype("float32"),
                    "hidden": rng.normal(size=(1, 1, self.model.hidden_size)).astype("float32"),
                }
                with torch.no_grad():
                    expected = self.model(
                        *(torch.from_numpy(arrays[k]) for k in ("state", "candidates", "hidden"))
                    )
                actual = session.run(None, arrays)
                for a, b in zip(actual, expected, strict=True):
                    maximum = max(maximum, float(np.max(np.abs(a - b.numpy()))))
            metadata["onnx_max_abs_error"] = maximum
            if maximum > 1e-4:
                raise RuntimeError("GRU ONNX export failed numerical parity")
        (directory / "M05.metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf8"
        )
        return metadata

    @classmethod
    def load(cls, directory: str | Path):
        checkpoint = torch.load(Path(directory) / "M05.pt", map_location="cpu", weights_only=True)
        model = RecurrentNetwork(checkpoint["hidden_size"])
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, checkpoint["temperature"])
