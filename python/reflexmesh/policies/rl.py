from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import Decision
from ..environments.gym import MAX_CANDIDATES, SoftwareGym, gym_features, ordered


class MaskedPPOPolicy:
    policy_id = "M07"

    def __init__(self, model):
        self.model = model

    def reset(self, goal_id=None):
        pass

    def predict(self, observation):
        observation = ordered(observation)
        mask = np.zeros(MAX_CANDIDATES, dtype=bool)
        mask[: len(observation.admissible)] = True
        if not mask.any():
            return Decision(self.policy_id, None, "stop", reason="no_admissible_candidates")
        action, _ = self.model.predict(
            gym_features(observation), action_masks=mask, deterministic=True
        )
        return Decision(
            self.policy_id,
            observation.admissible[int(action)].action_id,
            reason="trained_nonrecurrent_maskable_ppo",
            diagnostics={"action_mask": mask.tolist(), "recurrent": False},
        )

    @classmethod
    def load(cls, directory):
        from sb3_contrib import MaskablePPO

        return cls(MaskablePPO.load(Path(directory) / "M07.zip", device="cpu"))


def train_ppo(directory, timesteps=16384, seed=29):
    import json
    import time

    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(2)
    environment = SoftwareGym()
    start = time.perf_counter()
    try:
        model = MaskablePPO(
            "MlpPolicy",
            environment,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
            learning_rate=0.0005,
            gamma=0.95,
            seed=seed,
            device="cpu",
            verbose=0,
            policy_kwargs={"net_arch": {"pi": [64, 32], "vf": [64, 32]}},
        )

        class Progress(BaseCallback):
            def _on_step(self):
                if self.num_timesteps % 1024 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "ppo",
                                "timesteps": self.num_timesteps,
                                "requested": timesteps,
                                "elapsed_seconds": time.perf_counter() - start,
                            }
                        ),
                        flush=True,
                    )
                return True

        model.learn(total_timesteps=timesteps, callback=Progress())
        model.save(Path(directory) / "M07.zip")
    finally:
        environment.close()
    result = {
        "actual_timesteps": model.num_timesteps,
        "requested_timesteps": timesteps,
        "seed": seed,
        "device": "cpu",
        "training_seconds": time.perf_counter() - start,
        "environment": "real disposable software with native admission",
        "recurrent": False,
        "fit_configuration": {
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "learning_rate": model.learning_rate,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef,
            "max_grad_norm": model.max_grad_norm,
            "clip_range": model.clip_range(1.0),
            "normalize_advantage": model.normalize_advantage,
            "policy_kwargs": model.policy_kwargs,
            "optimizer": model.policy.optimizer.__class__.__name__,
        },
        "training_spec_ids": [spec.episode_id for spec in environment.specs],
    }
    (Path(directory) / "ppo-training.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    return result
