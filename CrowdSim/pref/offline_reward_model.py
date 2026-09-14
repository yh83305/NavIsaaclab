# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward model used by the offline view-clearance preference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from CrowdSim.pref.reward_model_utils import RewardNet_Traj_Sys_SelfAttn


class OfflineRewardModel(nn.Module):
    """Map an observation/action/depth clip to per-step and clip rewards."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        max_len: int = 200,
        depth_embed_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_len = int(max_len)
        self.depth_embed_dim = int(depth_embed_dim)
        self.backbone = RewardNet_Traj_Sys_SelfAttn(
            vec_dim=self.obs_dim,
            map_size=0,
            action_dim=self.action_dim,
            max_len=self.max_len,
            depth_embed_dim=self.depth_embed_dim,
        )
        # Bradley-Terry temperature learned jointly with the reward function.
        self.logit_scale_log = nn.Parameter(torch.zeros(()))

    def per_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        depth: torch.Tensor | None,
    ) -> torch.Tensor:
        if obs.ndim != 3 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs must be [B,T,{self.obs_dim}], got {tuple(obs.shape)}")
        if action.shape[:2] != obs.shape[:2] or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must be [B,T,{self.action_dim}], got {tuple(action.shape)}"
            )
        if obs.shape[1] > self.max_len:
            raise ValueError(f"clip length {obs.shape[1]} exceeds max_len={self.max_len}")
        system = torch.cat((obs.float(), action.float()), dim=-1)
        return self.backbone(
            system,
            depth_feats=None if depth is None else depth.float(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        depth: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.per_step(obs, action, depth).mean(dim=1)

    def preference_logits(self, reward_a, reward_b) -> torch.Tensor:
        scale = self.logit_scale_log.exp().clamp(max=100.0)
        return scale * (reward_a - reward_b)

    def model_config(self) -> dict[str, int]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "max_len": self.max_len,
            "depth_embed_dim": self.depth_embed_dim,
        }


def load_offline_reward_model(
    checkpoint: str | Path | dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> tuple[OfflineRewardModel, dict[str, Any]]:
    payload = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, (str, Path)) else checkpoint
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Offline reward checkpoint must contain model weights")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Offline reward checkpoint lacks model_config")
    model = OfflineRewardModel(**config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload
