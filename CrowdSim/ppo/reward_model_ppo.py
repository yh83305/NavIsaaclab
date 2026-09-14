# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

"""Online reward-model inference and task reward composition for PPO.

This module deliberately has no Isaac Lab imports, which keeps the temporal
alignment and reward arithmetic independently testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from CrowdSim.pref.offline_reward_model import OfflineRewardModel


@dataclass(frozen=True)
class RewardCompositionConfig:
    """Weights for the learned view-clearance and navigation task rewards."""

    learned_weight: float = 1.0
    progress_weight: float = 2.0
    goal_bonus: float = 10.0
    collision_penalty: float = -10.0
    timeout_penalty: float = -3.0
    stuck_penalty: float = -3.0
    time_penalty: float = -0.005
    environment_reward_weight: float = 0.0
    learned_reward_clip: float = 5.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compose_reward(
    learned_reward: torch.Tensor,
    environment_reward: torch.Tensor,
    info: dict[str, Any],
    config: RewardCompositionConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose a non-degenerate navigation reward and return each component.

    The preference teacher only describes pedestrian clearance.  Progress and
    terminal task terms are therefore explicit so that standing still or
    turning away cannot maximize the objective.
    """

    reference = environment_reward

    def _float_info(key: str) -> torch.Tensor:
        value = info.get(key)
        if value is None:
            return torch.zeros_like(reference)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=reference.device)
        return value.to(device=reference.device, dtype=reference.dtype)

    clip = abs(float(config.learned_reward_clip))
    learned = learned_reward.to(reference).clamp(-clip, clip)
    components = {
        "reward_learned": float(config.learned_weight) * learned,
        "reward_progress_task": float(config.progress_weight) * _float_info("progress"),
        "reward_goal_task": float(config.goal_bonus) * _float_info("reached"),
        "reward_collision_task": float(config.collision_penalty) * _float_info("collision"),
        "reward_timeout_task": float(config.timeout_penalty) * _float_info("timeout"),
        "reward_stuck_task": float(config.stuck_penalty) * _float_info("stuck"),
        "reward_time_task": torch.full_like(reference, float(config.time_penalty)),
        "reward_environment": float(config.environment_reward_weight) * reference,
    }
    total = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return total, components


def reward_model_timing(
    checkpoint_payload: dict[str, Any],
    *,
    runtime_hz: float,
    source_hz_override: float | None = None,
    stride_override: int | None = None,
) -> tuple[int, int, float]:
    """Return ``(sequence_steps, runtime_stride, model_hz)`` from metadata.

    Refuse non-integral timing conversions instead of silently feeding a model
    a temporal distribution different from the one it was trained on.
    """

    metadata = checkpoint_payload.get("dataset_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    sequence_steps = int(metadata.get("segment_steps", 0))
    if sequence_steps <= 0:
        raise ValueError("Reward checkpoint lacks a positive dataset segment_steps")
    frame_stride = int(metadata.get("frame_stride", 1))
    if frame_stride <= 0:
        raise ValueError("Reward checkpoint has an invalid frame_stride")

    source_hz = source_hz_override
    if source_hz is None:
        metadata_source_hz = metadata.get("source_hz")
        if metadata_source_hz is not None:
            source_hz = float(metadata_source_hz)
    if source_hz is None:
        values: list[float] = []
        for source in metadata.get("sources", []):
            if isinstance(source, dict) and isinstance(source.get("meta"), dict):
                value = source["meta"].get("source_hz")
                if value is not None:
                    values.append(float(value))
        if values:
            if max(values) - min(values) > 1.0e-6:
                raise ValueError(f"Reward dataset mixes source rates: {values}")
            source_hz = values[0]
    if source_hz is None:
        raise ValueError(
            "Reward checkpoint lacks source_hz metadata; pass --reward-source-hz"
        )
    if source_hz <= 0 or runtime_hz <= 0:
        raise ValueError("source_hz and runtime_hz must be positive")

    model_hz = float(source_hz) / frame_stride
    if stride_override is not None:
        runtime_stride = int(stride_override)
        if runtime_stride <= 0:
            raise ValueError("reward runtime stride must be positive")
    else:
        exact_stride = float(runtime_hz) / model_hz
        runtime_stride = int(round(exact_stride))
        if runtime_stride <= 0 or abs(exact_stride - runtime_stride) > 1.0e-4:
            raise ValueError(
                f"Cannot align runtime {runtime_hz:g} Hz with reward model "
                f"{model_hz:g} Hz using an integer stride"
            )
    return sequence_steps, runtime_stride, model_hz


class OnlineRewardHistory:
    """GPU-vectorized, asynchronously resettable reward-model history."""

    def __init__(
        self,
        model: OfflineRewardModel,
        *,
        num_envs: int,
        sequence_steps: int,
        depth_size: int,
        min_history_steps: int = 1,
        device: str | torch.device,
    ) -> None:
        if num_envs <= 0 or sequence_steps <= 0:
            raise ValueError("num_envs and sequence_steps must be positive")
        if not 1 <= min_history_steps <= sequence_steps:
            raise ValueError("min_history_steps must be within the history length")
        self.model = model.to(device).eval()
        model_max_len = int(getattr(self.model, "max_len", sequence_steps))
        if sequence_steps > model_max_len:
            raise ValueError(
                f"history length {sequence_steps} exceeds model max_len={model_max_len}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.num_envs = int(num_envs)
        self.sequence_steps = int(sequence_steps)
        self.depth_size = int(depth_size)
        self.min_history_steps = int(min_history_steps)
        self.device = torch.device(device)
        self.obs = torch.zeros(
            self.num_envs, self.sequence_steps, self.model.obs_dim, device=self.device
        )
        self.action = torch.zeros(
            self.num_envs, self.sequence_steps, self.model.action_dim, device=self.device
        )
        self.depth = (
            torch.zeros(
                self.num_envs, self.sequence_steps, self.depth_size, self.depth_size,
                device=self.device,
            )
            if self.depth_size > 0 else None
        )
        self.count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, done: torch.Tensor | None = None) -> None:
        mask = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if done is None else done.to(device=self.device, dtype=torch.bool)
        )
        if mask.shape != (self.num_envs,):
            raise ValueError(f"reset mask must have shape ({self.num_envs},)")
        self.obs[mask] = 0
        self.action[mask] = 0
        if self.depth is not None:
            self.depth[mask] = 0
        self.count[mask] = 0

    def _prepare_depth(self, depth: torch.Tensor | None) -> torch.Tensor | None:
        if self.depth is None:
            return None
        if depth is None:
            raise ValueError("Reward model was trained with depth but runtime depth is None")
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3 or depth.shape[0] != self.num_envs:
            raise ValueError("depth must be [N,H,W] or [N,1,H,W]")
        depth = depth.to(device=self.device, dtype=torch.float32)
        if depth.shape[-2:] != (self.depth_size, self.depth_size):
            depth = F.interpolate(
                depth[:, None], size=(self.depth_size, self.depth_size), mode="area"
            )[:, 0]
        return depth

    @torch.no_grad()
    def append_and_score(
        self,
        obs: torch.Tensor,
        executed_action: torch.Tensor,
        depth: torch.Tensor | None,
    ) -> torch.Tensor:
        """Append a model-rate sample and return its latest per-step reward."""

        obs = obs.to(device=self.device, dtype=torch.float32)
        executed_action = executed_action.to(device=self.device, dtype=torch.float32)
        if obs.shape != (self.num_envs, self.model.obs_dim):
            raise ValueError(
                f"obs must be ({self.num_envs},{self.model.obs_dim}), got {tuple(obs.shape)}"
            )
        if executed_action.shape != (self.num_envs, self.model.action_dim):
            raise ValueError(
                "executed_action must be "
                f"({self.num_envs},{self.model.action_dim}), got {tuple(executed_action.shape)}"
            )
        depth_sample = self._prepare_depth(depth)
        new_env = self.count == 0

        # Initial padding repeats the first real sample, matching offline fixed
        # length clips without injecting out-of-distribution all-zero frames.
        if new_env.any():
            self.obs[new_env] = obs[new_env, None, :]
            self.action[new_env] = executed_action[new_env, None, :]
            if self.depth is not None and depth_sample is not None:
                self.depth[new_env] = depth_sample[new_env, None, :, :]
        old_env = ~new_env
        if old_env.any():
            self.obs[old_env] = torch.roll(self.obs[old_env], shifts=-1, dims=1)
            self.action[old_env] = torch.roll(self.action[old_env], shifts=-1, dims=1)
            self.obs[old_env, -1] = obs[old_env]
            self.action[old_env, -1] = executed_action[old_env]
            if self.depth is not None and depth_sample is not None:
                self.depth[old_env] = torch.roll(self.depth[old_env], shifts=-1, dims=1)
                self.depth[old_env, -1] = depth_sample[old_env]
        self.count.clamp_max_(self.sequence_steps - 1).add_(1)
        self.count.clamp_max_(self.sequence_steps)

        reward = self.model.per_step(self.obs, self.action, self.depth)[:, -1]
        ready = self.count >= self.min_history_steps
        return torch.where(ready, reward, torch.zeros_like(reward))
