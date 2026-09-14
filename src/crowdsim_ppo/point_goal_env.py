"""A small vectorized PointGoal environment used by the public PPO example.

This is deliberately simulator-free: dynamics, moving neighbors and rewards are
implemented with PyTorch tensors so the complete training pipeline can run on
CPU or CUDA without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class PointGoalEnvConfig:
    num_envs: int = 64
    num_neighbors: int = 4
    world_size: float = 10.0
    dt: float = 0.1
    max_episode_steps: int = 250
    min_start_goal_distance: float = 2.0
    max_start_goal_distance: float = 8.0
    rl_initial_heading_min_offset_degrees: float = 0.0
    rl_initial_heading_max_offset_degrees: float = 180.0
    goal_radius: float = 0.35
    collision_radius: float = 0.55
    max_linear_velocity: float = 1.2
    max_angular_velocity: float = 1.5
    neighbor_speed: float = 0.45


class PointGoalBatchEnv:
    """Batched 2-D unicycle navigation with bouncing moving neighbors."""

    obs_dim = 5
    neighbor_dim = 5

    def __init__(self, config: PointGoalEnvConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        n, k = config.num_envs, config.num_neighbors
        self.position = torch.zeros(n, 2, device=device)
        self.heading = torch.zeros(n, device=device)
        self.goal = torch.zeros(n, 2, device=device)
        self.neighbor_position = torch.zeros(n, k, 2, device=device)
        self.neighbor_velocity = torch.zeros(n, k, 2, device=device)
        self.last_action = torch.zeros(n, 2, device=device)
        self.episode_step = torch.zeros(n, dtype=torch.long, device=device)
        self.episode_return = torch.zeros(n, device=device)
        self.reset()

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    def reset(self, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        count = int(ids.numel())
        if count == 0:
            return self.observe()

        margin = 0.75
        span = self.config.world_size - 2.0 * margin
        self.position[ids] = margin + torch.rand(count, 2, device=self.device) * span

        goal_angle = torch.rand(count, device=self.device) * (2.0 * math.pi)
        goal_distance = self.config.min_start_goal_distance + torch.rand(
            count, device=self.device
        ) * (self.config.max_start_goal_distance - self.config.min_start_goal_distance)
        offset = torch.stack((goal_angle.cos(), goal_angle.sin()), dim=-1) * goal_distance[:, None]
        self.goal[ids] = (self.position[ids] + offset).clamp(margin, self.config.world_size - margin)

        min_offset = math.radians(self.config.rl_initial_heading_min_offset_degrees)
        max_offset = math.radians(self.config.rl_initial_heading_max_offset_degrees)
        signed_offset = min_offset + torch.rand(count, device=self.device) * (max_offset - min_offset)
        signed_offset *= torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        self.heading[ids] = self._wrap_angle(goal_angle + signed_offset)

        k = self.config.num_neighbors
        self.neighbor_position[ids] = margin + torch.rand(count, k, 2, device=self.device) * span
        velocity_angle = torch.rand(count, k, device=self.device) * (2.0 * math.pi)
        speed = self.config.neighbor_speed * (0.4 + 0.6 * torch.rand(count, k, device=self.device))
        self.neighbor_velocity[ids] = torch.stack(
            (velocity_angle.cos(), velocity_angle.sin()), dim=-1
        ) * speed[..., None]
        self.last_action[ids] = 0.0
        self.episode_step[ids] = 0
        self.episode_return[ids] = 0.0
        return self.observe()

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_delta = self.goal - self.position
        goal_distance = goal_delta.norm(dim=-1)
        goal_angle = torch.atan2(goal_delta[:, 1], goal_delta[:, 0])
        goal_bearing = self._wrap_angle(goal_angle - self.heading)
        obs = torch.stack(
            (
                (goal_distance / max(self.config.max_start_goal_distance, 1.0)).clamp(0.0, 1.5),
                goal_bearing.cos(),
                goal_bearing.sin(),
                self.last_action[:, 0],
                self.last_action[:, 1],
            ),
            dim=-1,
        )

        relative = self.neighbor_position - self.position[:, None, :]
        distance = relative.norm(dim=-1).clamp_min(1.0e-6)
        bearing = self._wrap_angle(
            torch.atan2(relative[..., 1], relative[..., 0]) - self.heading[:, None]
        )
        c, s = self.heading.cos()[:, None], self.heading.sin()[:, None]
        vx, vy = self.neighbor_velocity[..., 0], self.neighbor_velocity[..., 1]
        robot_vx = c * vx + s * vy
        robot_vy = -s * vx + c * vy
        neighbors = torch.stack(
            (
                (distance / self.config.world_size).clamp(0.0, 1.5),
                bearing.cos(), bearing.sin(), robot_vx, robot_vy,
            ),
            dim=-1,
        )
        neighbor_mask = torch.ones(
            self.num_envs, self.config.num_neighbors, dtype=torch.bool, device=self.device
        )
        return obs, neighbors, neighbor_mask

    def step(
        self, action: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        action = torch.stack(
            (action[:, 0].clamp(0.0, 1.0), action[:, 1].clamp(-1.0, 1.0)),
            dim=-1,
        )
        previous_distance = (self.goal - self.position).norm(dim=-1)
        linear = action[:, 0] * self.config.max_linear_velocity
        angular = action[:, 1] * self.config.max_angular_velocity
        self.heading = self._wrap_angle(self.heading + angular * self.config.dt)
        direction = torch.stack((self.heading.cos(), self.heading.sin()), dim=-1)
        self.position = (
            self.position + direction * linear[:, None] * self.config.dt
        ).clamp(0.0, self.config.world_size)
        self.last_action = action

        self.neighbor_position += self.neighbor_velocity * self.config.dt
        outside = (self.neighbor_position < 0.0) | (
            self.neighbor_position > self.config.world_size
        )
        self.neighbor_velocity = torch.where(outside, -self.neighbor_velocity, self.neighbor_velocity)
        self.neighbor_position.clamp_(0.0, self.config.world_size)

        self.episode_step += 1
        distance = (self.goal - self.position).norm(dim=-1)
        min_neighbor_distance = (
            self.neighbor_position - self.position[:, None, :]
        ).norm(dim=-1).min(dim=-1).values
        reached = distance <= self.config.goal_radius
        collision = min_neighbor_distance <= self.config.collision_radius
        timeout = self.episode_step >= self.config.max_episode_steps
        done = reached | collision | timeout

        progress = previous_distance - distance
        reward = 2.5 * progress - 0.01 - 0.01 * action[:, 1].square()
        reward = reward + 3.0 * reached.float() - 2.0 * collision.float()
        self.episode_return += reward
        terminal_return = self.episode_return.clone()
        terminal_length = self.episode_step.clone()
        info = {
            "reached": reached,
            "collision": collision,
            "timeout": timeout & ~reached & ~collision,
            "episode_return": terminal_return,
            "episode_length": terminal_length,
            "goal_distance": distance,
        }
        self.reset(done)
        return self.observe(), reward, done, info

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(angle.sin(), angle.cos())
