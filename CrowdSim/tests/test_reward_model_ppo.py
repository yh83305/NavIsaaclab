# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from CrowdSim.ppo.reward_model_ppo import (
    OnlineRewardHistory,
    RewardCompositionConfig,
    compose_reward,
    reward_model_timing,
)


class _FakeRewardModel(nn.Module):
    obs_dim = 3
    action_dim = 2

    def per_step(self, obs, action, depth):
        if depth is not None:
            assert depth.shape[-2:] == (4, 4)
        return obs[..., 0] + action[..., 0]


def test_reward_model_timing_uses_offline_sampling_metadata():
    payload = {"dataset_metadata": {
        "segment_steps": 16,
        "frame_stride": 3,
        "sources": [{"meta": {"source_hz": 30.0}}],
    }}
    steps, stride, model_hz = reward_model_timing(payload, runtime_hz=30.0)
    assert (steps, stride, model_hz) == (16, 3, 10.0)


def test_online_history_pads_first_sample_resizes_depth_and_resets_per_env():
    history = OnlineRewardHistory(
        _FakeRewardModel(), num_envs=2, sequence_steps=3,
        depth_size=4, min_history_steps=2, device="cpu",
    )
    obs = torch.tensor([[1.0, 0, 0], [2.0, 0, 0]])
    action = torch.tensor([[0.5, 0], [0.25, 0]])
    first = history.append_and_score(obs, action, torch.ones(2, 8, 8))
    assert torch.equal(first, torch.zeros(2))
    assert torch.equal(history.obs[0, :, 0], torch.ones(3))

    second = history.append_and_score(obs + 1, action, torch.ones(2, 8, 8))
    torch.testing.assert_close(second, torch.tensor([2.5, 3.25]))
    history.reset(torch.tensor([False, True]))
    history.append_and_score(obs + 2, action, torch.ones(2, 8, 8))
    assert history.count.tolist() == [3, 1]
    assert torch.equal(history.obs[1, :, 0], torch.full((3,), 4.0))


def test_reward_composition_keeps_clearance_and_navigation_objectives():
    learned = torch.tensor([2.0, -9.0])
    environment = torch.tensor([4.0, 4.0])
    info = {
        "progress": torch.tensor([0.5, -0.25]),
        "reached": torch.tensor([True, False]),
        "collision": torch.tensor([False, True]),
        "timeout": torch.tensor([False, False]),
        "stuck": torch.tensor([False, False]),
    }
    config = RewardCompositionConfig(
        learned_weight=1.5, progress_weight=2.0, goal_bonus=10.0,
        collision_penalty=-8.0, time_penalty=-0.1,
        environment_reward_weight=0.25, learned_reward_clip=3.0,
    )
    total, components = compose_reward(learned, environment, info, config)
    torch.testing.assert_close(total, torch.tensor([14.9, -12.1]))
    torch.testing.assert_close(components["reward_learned"], torch.tensor([3.0, -4.5]))
