"""Collect PPO rollouts with explicit route guidance.

The PPO policy and raw-buffer schema remain unchanged. Each mode supplies a
safe lateral A* route whose sparse waypoint replaces PPO's control goal. The
original final goal is still used for success and dataset labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CrowdSim.flow.collect_ppo_buffer import _atomic_torch_save, collect  # noqa: E402
from CrowdSim.utils.config_loader import cfg_path, load_config  # noqa: E402

MODE_VALUES = (-2, -1, 0, 1, 2)
MODE_NAMES = ("far_right", "right", "center", "left", "far_left")


class RouteGuide:
    def __init__(self, config: dict, robots: int, repeats_per_mode: int,
                 *, random_modes: bool = False):
        self.robots = robots
        self.repeats_per_mode = repeats_per_mode
        self.random_modes = random_modes
        self.saved_counts = np.zeros(robots, dtype=np.int64)
        seed = int(config.get("navigation", {}).get("path", {}).get("seed", 42))
        self.rng = np.random.default_rng(seed + 83_917)
        initial_modes = np.resize(np.asarray(MODE_VALUES, dtype=np.int64), robots)
        self.rng.shuffle(initial_modes)
        self.current_modes = initial_modes
        self.offset = float(config.get("route_guidance", {}).get("lateral_offset_m", 1.5))
        self.static_clearance = float(
            config.get("route_guidance", {}).get("static_clearance_m", 0.8)
        )
        self.waypoint_spacing = float(
            config.get("route_guidance", {}).get("waypoint_spacing_m", 2.0)
        )
        self.waypoint_tolerance = float(
            config.get("route_guidance", {}).get("waypoint_tolerance_m", 1.0)
        )
        self._paths: list[list[np.ndarray]] = []
        self._indices = np.zeros(robots, dtype=np.int64)

    @property
    def mode(self) -> np.ndarray:
        if self.random_modes:
            return self.current_modes
        slots = np.minimum(
            self.saved_counts // self.repeats_per_mode, len(MODE_VALUES) - 1
        )
        return np.asarray(MODE_VALUES, dtype=np.int64)[slots]

    def _sparse(self, path):
        spacing = self.waypoint_spacing
        keep, acc = [0], 0.0
        for j in range(1, len(path)):
            acc += float(np.linalg.norm(path[j] - path[j - 1]))
            if acc >= spacing or j == len(path) - 1:
                keep.append(j); acc = 0.0
        return np.asarray(path, dtype=np.float32)[keep]

    def _safe_via(self, task, via: np.ndarray) -> bool:
        py, px = (int(value) for value in task.world_to_pixel(via))
        radius = max(1, int(np.ceil(self.static_clearance / task.config.map_resolution)))
        y0, y1 = max(0, py - radius), min(task.height, py + radius + 1)
        x0, x1 = max(0, px - radius), min(task.width, px + radius + 1)
        if y0 != py - radius or y1 != py + radius + 1:
            return False
        if x0 != px - radius or x1 != px + radius + 1:
            return False
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - py) ** 2 + (xx - px) ** 2 <= radius ** 2
        return not bool(np.asarray(task.obstacle_map[y0:y1, x0:x1])[disk].any())

    def initialize(self, nav_manager):
        self._paths = [None] * self.robots
        self._indices[:] = 0
        for i in range(self.robots):
            self._paths[i] = self._build_robot_paths(nav_manager, i)

    def _build_robot_paths(self, nav_manager, index: int) -> list[np.ndarray]:
        task = getattr(nav_manager, "task", None)
        agent_id = nav_manager.config.num_humanoids + index
        start = np.asarray(nav_manager.starts_xy[agent_id])
        goal = np.asarray(nav_manager.goals_xy[agent_id])
        delta = goal - start
        norm = max(float(np.linalg.norm(delta)), 1e-6)
        left = np.array([-delta[1], delta[0]], dtype=np.float32) / norm
        robot_paths = []
        for sign in MODE_VALUES:
            best = np.asarray(nav_manager.paths_xy[agent_id], dtype=np.float32)
            if task is not None:
                for frac in (.35, .5, .65):
                    for scale in (1.0, .75, 1.25, .5, 1.5):
                        via = start + frac * delta + sign * self.offset * scale * left
                        if not self._safe_via(task, via):
                            continue
                        a = task.planner.get_astar_path(
                            task.world_to_pixel(start), task.world_to_pixel(via)
                        )
                        b = task.planner.get_astar_path(
                            task.world_to_pixel(via), task.world_to_pixel(goal)
                        )
                        if a is not None and b is not None:
                            pa = np.asarray([task.pixel_to_world(x) for x in a], np.float32)
                            pb = np.asarray([task.pixel_to_world(x) for x in b], np.float32)
                            best = np.vstack((pa[:-1], pb))
                            break
                    if len(best) > 2 and not np.array_equal(best, nav_manager.paths_xy[agent_id]):
                        break
            robot_paths.append(self._sparse(best))
        return robot_paths

    def route_mode(self, index: int) -> int:
        return int(self.mode[index])

    @staticmethod
    def route_robot_id(index: int) -> int:
        return int(index)

    def on_episode_saved(self, index: int) -> None:
        self.saved_counts[index] += 1

    def reset(self, robot_ids, nav_manager) -> None:
        robot_ids = np.asarray(robot_ids, dtype=np.int64)
        self._indices[robot_ids] = 0
        if self.random_modes:
            for index in robot_ids:
                previous = int(self.current_modes[index])
                choices = [mode for mode in MODE_VALUES if mode != previous]
                self.current_modes[index] = int(self.rng.choice(choices))
        for index in robot_ids:
            self._paths[int(index)] = self._build_robot_paths(nav_manager, int(index))

    def collection_complete(self) -> bool:
        if self.random_modes:
            return False
        return bool((self.saved_counts >= len(MODE_VALUES) * self.repeats_per_mode).all())

    def _mode_slot(self, index: int) -> int:
        return MODE_VALUES.index(self.route_mode(index))

    def route_sparse_path(self, index: int) -> np.ndarray:
        return self._paths[index][self._mode_slot(index)]

    def route_sparse_target(self, index: int) -> np.ndarray:
        path = self.route_sparse_path(index)
        return path[min(int(self._indices[index]), len(path) - 1)]

    def _advance_target(self, index: int, position: np.ndarray) -> None:
        path = self.route_sparse_path(index)
        while (self._indices[index] < len(path) - 1
               and np.linalg.norm(path[self._indices[index]] - position)
               < self.waypoint_tolerance):
            self._indices[index] += 1

    def transform_observations(self, obs, positions, yaws, nav_manager):
        transformed = obs.clone()
        offset = int(nav_manager.config.num_humanoids)
        max_goal_distance = max(
            float(nav_manager.config.rl_goal_observation_max_distance), 1.0e-6
        )
        for index in range(min(len(transformed), self.robots)):
            position = np.asarray(positions[offset + index, :2], dtype=np.float32)
            self._advance_target(index, position)
            delta = self.route_sparse_target(index) - position
            distance = float(np.linalg.norm(delta))
            world_angle = float(np.arctan2(delta[1], delta[0]))
            relative_angle = float(np.arctan2(
                np.sin(world_angle - float(yaws[index])),
                np.cos(world_angle - float(yaws[index])),
            ))
            transformed[index, 0] = distance / max_goal_distance
            transformed[index, 1] = np.sin(relative_angle)
            transformed[index, 2] = np.cos(relative_angle)
        return transformed

    def __call__(self, action, positions, yaws, goals, nav_manager):
        return action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="CrowdSim/config/collect/ppo.yaml")
    p.add_argument("--ppo-ckpt", required=True)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--num-robots", type=int, default=1)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--random-modes", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    config = load_config(cfg_path(args.env_config, PROJECT_ROOT))
    config.setdefault("navigation", {})["num_robots"] = args.num_robots
    route_cfg = config.setdefault("route_guidance", {})
    endpoint_clearance = float(route_cfg.get("static_clearance_m", 0.8))
    path_cfg = config["navigation"].setdefault("path", {})
    path_cfg["planning_clearance"] = max(
        float(path_cfg.get("planning_clearance", 0.0)), endpoint_clearance
    )
    c = config.setdefault("collection", {})
    if args.output_dir is not None:
        c["output_dir"] = args.output_dir
    robots = int(c.get("num_robots", config.get("navigation", {}).get("num_robots", 1)))
    mode_count = len(MODE_VALUES)
    if not args.random_modes and args.episodes % (mode_count * robots) != 0:
        raise ValueError("--episodes must be divisible by 5 * --num-robots")
    repeats_per_mode = 1 if args.random_modes else args.episodes // (mode_count * robots)
    guide = RouteGuide(config, robots, repeats_per_mode,
                       random_modes=args.random_modes)
    collect_steps = int(args.steps if args.steps is not None else c.get("collect_steps", 10000))
    target_episodes = 0 if args.random_modes else args.episodes
    output_dir = Path(c.get("output_dir", "output/collect_buffer/route_guided"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_every = int(c.get("checkpoint_every", 0))
    checkpoint_dir = output_dir / "checkpoints" if checkpoint_every > 0 else None
    gif_cfg = config.get("gif", {})
    gif_out = gif_cfg.get("out")
    if gif_out is not None:
        configured_gif = Path(gif_out)
        gif_out = configured_gif.parent / run_timestamp / configured_gif.name
    wandb_run = None
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", True):
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "crowdsim-flow-data"),
                name=f"collect_route_guided_{run_timestamp}",
                dir=str(output_dir / "wandb_collect"),
                config={
                    "ppo_ckpt": args.ppo_ckpt,
                    "collect_steps": collect_steps,
                    "num_robots": robots,
                    "random_modes": args.random_modes,
                    "waypoint_spacing_m": guide.waypoint_spacing,
                    "waypoint_tolerance_m": guide.waypoint_tolerance,
                    "static_clearance_m": guide.static_clearance,
                },
            )
        except Exception as error:
            print(
                "[RouteGuided][WARN] W&B initialization failed; "
                f"continuing without W&B: {error}"
            )
            wandb_run = None
    started = time.time()
    result = collect(config, args.ppo_ckpt, num_envs=max(robots, int(config["navigation"].get("num_humanoids", 1))),
            headless=bool(c.get("headless", True)), collect_steps=collect_steps,
            min_episode_len=int(c.get("min_episode_len", 32)), only_success=bool(c.get("only_success", False)),
            device=str(c.get("device", "cuda")), target_episodes=target_episodes,
            wandb_run=wandb_run,
            checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
            gif_out=str(gif_out) if gif_out is not None else None,
            gif_fps=float(gif_cfg.get("fps", 7.5)),
            gif_every_episodes=int(gif_cfg.get("every_episodes", 10)),
            repeat_robot_routes=False, action_transform=guide)
    print(f"[RouteGuided] Elapsed: {time.time() - started:.0f}s")
    output_path = output_dir / f"raw_route_guided_{run_timestamp}.pt"
    result["meta"]["route_guidance"] = {
        "modes": list(MODE_NAMES),
        "mode_values": list(MODE_VALUES),
        "lateral_offset_m": guide.offset,
        "static_clearance_m": guide.static_clearance,
        "episodes_per_robot_mode": repeats_per_mode,
        "assignment": "random_per_episode" if args.random_modes else "balanced",
        "waypoint_spacing_m": guide.waypoint_spacing,
        "waypoint_tolerance_m": guide.waypoint_tolerance,
        "final_goal_preserved": True,
    }
    _atomic_torch_save(result, output_path)
    print(f"[RouteGuided] Saved -> {output_path}")
    if checkpoint_dir is not None:
        for checkpoint_path in checkpoint_dir.glob("ckpt_*.pt"):
            checkpoint_path.unlink()
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {-2: "navy", -1: "tab:blue", 0: "tab:gray", 1: "tab:orange", 2: "darkred"}
    for robot_id, robot_paths in enumerate(guide._paths):
        for mode, path in zip(MODE_VALUES, robot_paths):
            ax.plot(path[:, 0], path[:, 1], ":", color=colors[mode], alpha=0.65)
            ax.scatter(path[:, 0], path[:, 1], s=18, color=colors[mode], marker="x",
                       label=f"robot{robot_id} A* mode{mode}")
    for idx, episode in enumerate(result["episodes"]):
        world = episode.get("world", {})
        xy = np.asarray(world.get("robot_xy", []), dtype=np.float32)
        if xy.ndim == 3: xy = xy[:, 0]
        mode = int(torch.as_tensor(episode["route_mode"])[0])
        robot_id = int(torch.as_tensor(episode["route_robot_id"]))
        if len(xy): ax.plot(xy[:, 0], xy[:, 1], color=colors[mode], alpha=0.8,
                            label=f"robot{robot_id} mode{mode} ep{idx:02d}")
    ax.set_aspect("equal"); ax.set_title("Route-guided PPO trajectories"); ax.legend(fontsize=7)
    plot_path = output_path.with_suffix(".png"); fig.savefig(plot_path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[RouteGuided] Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()
