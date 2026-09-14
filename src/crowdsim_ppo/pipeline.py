"""Configuration helpers shared by training and evaluation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig
from .ppo_policy import RobotPPOConfig, RobotPPOTrainer


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_pipeline(
    values: dict[str, Any], device: torch.device, num_envs: int | None = None
) -> tuple[PointGoalBatchEnv, RobotPPOTrainer]:
    env_values = dict(values.get("environment", {}))
    if num_envs is not None:
        env_values["num_envs"] = num_envs
    env_config = PointGoalEnvConfig(**env_values)
    env = PointGoalBatchEnv(env_config, device)

    network = values.get("network", {})
    algorithm = values.get("algorithm", {})
    ppo_config = RobotPPOConfig(
        obs_dim=env.obs_dim,
        action_dim=2,
        hidden_dims=tuple(network.get("hidden_dims", [128, 128])),
        vector_hidden_dims=tuple(network.get("vector_hidden_dims", [128, 128])),
        actor_hidden_dims=tuple(network.get("actor_hidden_dims", [128])),
        critic_hidden_dims=tuple(network.get("critic_hidden_dims", [128])),
        vector_obs_dim=env.obs_dim,
        map_enabled=False,
        depth_enabled=False,
        num_neighbors=env_config.num_neighbors,
        neighbor_dim=env.neighbor_dim,
        lr=float(algorithm.get("lr", 3.0e-4)),
        gamma=float(algorithm.get("gamma", 0.99)),
        gae_lambda=float(algorithm.get("gae_lambda", 0.95)),
        clip_ratio=float(algorithm.get("clip_ratio", 0.2)),
        value_coef=float(algorithm.get("value_coef", 0.5)),
        entropy_coef=float(algorithm.get("entropy_coef", 0.005)),
        max_grad_norm=float(algorithm.get("max_grad_norm", 1.0)),
        ppo_epochs=int(algorithm.get("ppo_epochs", 4)),
        minibatch_size=int(algorithm.get("minibatch_size", 256)),
    )
    return env, RobotPPOTrainer(ppo_config, device)
