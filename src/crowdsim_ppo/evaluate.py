"""Evaluate a trained policy in the public PointGoal environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .pipeline import build_pipeline, load_config, resolve_device
from .ppo_policy import bounded_robot_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/point_goal.json")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    values = load_config(args.config)
    device = resolve_device(args.device)
    env, trainer = build_pipeline(values, device, num_envs=args.num_envs)
    trainer.load(args.checkpoint)
    trainer.model.eval()
    obs, neighbors, neighbor_mask = env.observe()
    completed = successes = collisions = 0
    total_return = total_length = 0.0
    while completed < args.episodes:
        mean, _, _ = trainer.model(
            obs, neighbors=neighbors, neighbor_mask=neighbor_mask
        )
        action = bounded_robot_action(mean)
        next_state, _, done, info = env.step(action)
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            remaining = args.episodes - completed
            ids = ids[:remaining]
            completed += int(ids.numel())
            successes += int(info["reached"][ids].sum().item())
            collisions += int(info["collision"][ids].sum().item())
            total_return += float(info["episode_return"][ids].sum().item())
            total_length += float(info["episode_length"][ids].sum().item())
        obs, neighbors, neighbor_mask = next_state
    return {
        "episodes": float(completed),
        "success_rate": successes / completed,
        "collision_rate": collisions / completed,
        "mean_return": total_return / completed,
        "mean_episode_length": total_length / completed,
    }


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2))


if __name__ == "__main__":
    main()
