#!/usr/bin/env python3
"""Grid-based policy evaluation and visualization for CrowdSim.

Loads a checkpoint, runs evaluation from a grid of start poses, and generates
three static plots: visit frequency heatmap, speed streamplot, and trajectory
overlay.

Usage::

    python CrowdSim/tools/eval_policy.py \\
        --ckpt output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt \\
        --num-envs 1 --headless \\
        --num-episodes 200 --search-step 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")

_PROJECT = Path(__file__).resolve().parents[2]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


def _save_evaluation_gif(nav_manager, output_dir: Path) -> Path | None:
    """Render the recent simulator trajectory log as an evaluation GIF."""
    try:
        from CrowdSim.tools.render_navigation_fast import render_nav_gif_bytes

        cfg = nav_manager.config
        payload = render_nav_gif_bytes(
            obstacle_map=nav_manager.obstacle_map,
            resolution=float(cfg.map_resolution),
            origin_xy=tuple(cfg.map_origin_xy),
            num_humanoids=int(cfg.num_humanoids),
            num_robots=int(cfg.num_robots),
            paths_xy=[np.asarray(path) for path in nav_manager.paths_xy],
            goals_xy=np.asarray(nav_manager.goals_xy, dtype=np.float32),
            trajectory_log_path=getattr(nav_manager, "trajectory_log_path", None),
            n_frames=min(int(cfg.rl_max_episode_steps) * 2, 600),
            fps=float(cfg.update_hz),
            scale=1.0,
        )
        if payload is None:
            return None
        path = output_dir / "evaluation.gif"
        path.write_bytes(payload)
        return path
    except Exception as error:
        print(f"Evaluation GIF skipped: {error}")
        return None


# ═══════════════════════════════════════════════════════════════════
# grid helpers
# ═══════════════════════════════════════════════════════════════════

def _build_grid(task, search_step: float = 0.5, min_dist: float = 1.0):
    """Generate candidate start poses on a grid within the free space."""
    origin_x, origin_y = task.config.map_origin_xy
    res = task.config.map_resolution

    free = task.planner_free_mask
    h, w = free.shape

    candidates = []
    for py in range(0, h, int(search_step / res)):
        for px in range(0, w, int(search_step / res)):
            if not free[py, px]:
                continue
            wx = origin_x + px * res
            wy = origin_y + (h - 1 - py) * res
            # basic spacing from map border
            if wx - origin_x < min_dist or wy - origin_y < min_dist:
                continue
            candidates.append((wx, wy))
    return candidates


def _sample_goal(map_meta, start_xy, min_d=7.0, max_d=10.0):
    """Sample a goal around *start_xy* within [min_d, max_d]."""
    import random
    rng = random.Random(hash(tuple(start_xy)) % (2**31))
    for _ in range(200):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(min_d, max_d)
        gx = start_xy[0] + dist * np.cos(angle)
        gy = start_xy[1] + dist * np.sin(angle)
        # Basic bounds check — caller should verify with map
        return (gx, gy)
    return (start_xy[0] + max_d, start_xy[1])


# ═══════════════════════════════════════════════════════════════════
# visualization
# ═══════════════════════════════════════════════════════════════════

def save_plots(
    vectors: np.ndarray,        # (grid_size, 2) mean velocity vectors
    visits: np.ndarray,         # (grid_size,) visit counts
    avg_speeds: np.ndarray,     # (grid_size,) mean speeds
    map_meta,
    grid_meta: dict,            # {x_range, y_range, cols, rows, cell_size}
    robot_trajs: list,
    actor_trajs: list | None,
    actor_positions: list | None,
    goal_pose,
    output_dir: str,
    name_prefix: str,
) -> None:
    x_range = grid_meta["x_range"]
    y_range = grid_meta["y_range"]
    cols = grid_meta["cols"]
    rows = grid_meta["rows"]
    xx, yy = np.meshgrid(x_range, y_range)
    u = vectors[:, 0].reshape(rows, cols)
    v = vectors[:, 1].reshape(rows, cols)
    vis = visits.reshape(rows, cols)
    spd = avg_speeds.reshape(rows, cols)

    def _draw_obstacles(ax):
        """Overlay occupancy map obstacles."""
        origin_x, origin_y = map_meta.config.map_origin_xy
        res = map_meta.config.map_resolution
        occ = map_meta.obstacle_map
        h, w = occ.shape
        for py in range(0, h, 2):
            for px in range(0, w, 2):
                if occ[py, px]:
                    wx = origin_x + px * res
                    wy = origin_y + (h - 1 - py) * res
                    ax.add_patch(plt.Rectangle(
                        (wx - res / 2, wy - res / 2), res, res,
                        color="gray", alpha=0.3, zorder=3,
                    ))

    def _draw_actors(ax):
        if actor_positions:
            for pos in actor_positions:
                ax.plot(pos[0], pos[1], "o", color="firebrick",
                        markersize=8, zorder=5, markeredgecolor="white")
        if goal_pose:
            ax.plot(goal_pose[0], goal_pose[1], "*", color="dodgerblue",
                    markersize=15, zorder=6, markeredgecolor="white")

    # ── frequency plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 12))
    _draw_obstacles(ax)
    visits_clip = np.clip(vis, 1, None)
    cntr = ax.contourf(xx, yy, visits_clip,
                       levels=np.logspace(0, np.log10(visits_clip.max() + 1), 15),
                       cmap="OrRd", norm=LogNorm(), alpha=0.6, zorder=1)
    u_m, v_m = np.ma.array(u, mask=(vis == 0)), np.ma.array(v, mask=(vis == 0))
    ax.streamplot(xx, yy, u_m, v_m, density=1.5, color="black",
                  linewidth=0.7, arrowsize=1.0, zorder=2)
    _draw_actors(ax)
    plt.colorbar(cntr, ax=ax, label="Visit Counts", format=FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_aspect("equal")
    ax.set_title(f"{name_prefix} — Visit Frequency & Streamplot")
    fig.savefig(os.path.join(output_dir, f"{name_prefix}_freq.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── speed plot ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 12))
    _draw_obstacles(ax)
    spd_m = np.ma.array(spd, mask=(vis == 0))
    cntr = ax.contourf(xx, yy, spd_m, levels=15, cmap="Blues",
                       alpha=0.75, vmin=0.0, vmax=1.0, zorder=1)
    ax.streamplot(xx, yy, u_m, v_m, density=1.5, color="black",
                  linewidth=0.7, arrowsize=1.0, zorder=2)
    _draw_actors(ax)
    plt.colorbar(cntr, ax=ax, label="Mean Linear Speed (m/s)")
    ax.set_aspect("equal")
    ax.set_title(f"{name_prefix} — Speed Heatmap & Streamplot")
    fig.savefig(os.path.join(output_dir, f"{name_prefix}_speed.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── trajectory plot ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 12))
    _draw_obstacles(ax)

    def _draw_traj(traj, cmap_name="Blues", lw=2, z=5):
        if traj is None or len(traj) < 2:
            return
        traj = np.array(traj)
        speed = traj[:, 2] if traj.shape[1] >= 3 else np.ones(len(traj))
        points = traj[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap(cmap_name),
                            norm=plt.Normalize(vmin=0, vmax=1), linewidths=lw, zorder=z)
        lc.set_array(np.clip(speed[:-1], 0, 1))
        ax.add_collection(lc)
        ax.plot(traj[0, 0], traj[0, 1], "o", color="limegreen", markersize=5, zorder=z + 1)
        ax.plot(traj[-1, 0], traj[-1, 1], "x", color="red", markersize=6, zorder=z + 1)

    for t in robot_trajs:
        _draw_traj(t, cmap_name="Blues", lw=1, z=6)
    if actor_trajs:
        for t in actor_trajs:
            _draw_traj(t, cmap_name="Reds", lw=1, z=5)
    _draw_actors(ax)
    ax.set_aspect("equal")
    ax.set_title(f"{name_prefix} — Trajectories")
    fig.savefig(os.path.join(output_dir, f"{name_prefix}_traj.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# evaluation loop
# ═══════════════════════════════════════════════════════════════════

def run_evaluation(args):
    from CrowdSim.crowd_sim import cfg_path, load_config
    from CrowdSim.world.builder import build_env

    config = load_config(cfg_path(args.env_config))
    # Point to the checkpoint being eval'd
    config.setdefault("car", {})["policy_checkpoint"] = args.ckpt
    ckpt_name = Path(args.ckpt).stem
    output_dir = Path(args.ckpt).parent / f"eval_{ckpt_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single-robot eval: force num_robots=1 (this tool only drives env 0) but
    # preserve the crowd size (num_humanoids) so the benchmark reflects
    # multi-humanoid navigation.  num_envs = max(num_humanoids, 1) so the
    # active robot (env 0) sees all humanoids via the shared positions array.
    nav_cfg = config.setdefault("navigation", {})
    nav_cfg["num_robots"] = 1
    recording = nav_cfg.setdefault("recording", {})
    recording["enabled"] = True
    recording["output_dir"] = str(output_dir / "navigation")
    n_h = int(nav_cfg.get("num_humanoids", 1))
    args.num_envs = max(n_h, 1)

    result = build_env(config, num_envs=args.num_envs, headless=args.headless)
    if result is None:
        return
    env, agent, nav_manager, runtime = result

    # ── Grid setup ─────────────────────────────────────────
    map_meta = nav_manager.task
    candidates = _build_grid(map_meta, args.search_step, args.min_dist)
    n_episodes = min(args.num_episodes, len(candidates))
    if n_episodes == 0:
        raise RuntimeError("No valid start poses found")
    rng = np.random.default_rng(args.seed)
    chosen = [candidates[i] for i in rng.choice(len(candidates), n_episodes, replace=False)]

    print(f"[Eval] {n_episodes} starts from {len(candidates)} candidates")

    # ── Grid cells ─────────────────────────────────────────
    origin_x, origin_y = map_meta.config.map_origin_xy
    res = map_meta.config.map_resolution
    h, w = map_meta.height, map_meta.width
    grid_cell = args.grid_cell
    cols = int(w * res / grid_cell)
    rows = int(h * res / grid_cell)
    min_x, min_y = origin_x, origin_y
    grid_size = cols * rows
    x_range = np.linspace(min_x + grid_cell / 2, min_x + cols * grid_cell - grid_cell / 2, cols)
    y_range = np.linspace(min_y + grid_cell / 2, min_y + rows * grid_cell - grid_cell / 2, rows)
    grid_meta = {"x_range": x_range, "y_range": y_range, "cols": cols, "rows": rows, "cell_size": grid_cell}

    vectors = np.zeros((grid_size, 2))
    speed_sum = np.zeros(grid_size)
    visit_counts = np.zeros(grid_size)

    def _pos_to_cell(xy):
        ix = int((xy[0] - min_x) // grid_cell)
        iy = int((xy[1] - min_y) // grid_cell)
        ix = max(0, min(ix, cols - 1))
        iy = max(0, min(iy, rows - 1))
        return iy * cols + ix

    # ── Load model ──────────────────────────────────────────
    from CrowdSim.ppo.ppo_policy import (
        RobotPPOConfig, RobotPPOTrainer, bounded_robot_action, robot_network_kwargs,
    )
    from CrowdSim.crowd_sim import get_network_config

    network_cfg = get_network_config(config)
    device = runtime.fabric.device

    ppo = RobotPPOTrainer(
        RobotPPOConfig(
            obs_dim=nav_manager.robot_rl_obs_dim,
            vector_obs_dim=nav_manager.robot_rl_vector_obs_dim,
            **robot_network_kwargs(
                network_cfg,
                depth_enabled=nav_manager.config.rl_depth_enabled,
                depth_size=nav_manager.config.rl_depth_size,
                map_enabled=nav_manager.config.rl_map_size > 0,
                map_size=nav_manager.config.rl_map_size,
                num_neighbors=nav_manager.config.rl_num_neighbors,
            ),
        ), device,
    )
    ppo.load(Path(args.ckpt).expanduser().resolve())

    robot_trajs: list = []
    stats = {"success": 0, "collision": 0, "timeout": 0, "stuck": 0}

    skipped = 0
    for ep in range(n_episodes):
        sx, sy = chosen[ep]
        car_idx = nav_manager.config.num_humanoids

        # Try to find a valid goal+path (7-10m from start, in free space)
        start_xy = np.array([sx, sy], dtype=np.float32)
        try:
            start_px, goal_px, path_xy = nav_manager.task.sample_goal_and_plan_path(start_xy)
        except RuntimeError:
            skipped += 1
            continue
        actual_start = nav_manager.task.pixel_to_world(start_px)
        actual_goal = nav_manager.task.pixel_to_world(goal_px)
        nav_manager.starts_xy[car_idx] = actual_start
        nav_manager.goals_xy[car_idx] = actual_goal
        # Sync pixel-level state
        nav_manager.starts_px[car_idx] = start_px
        nav_manager.goals_px[car_idx] = goal_px

        # ── Teleport ──────────────────────────────────────
        pose = nav_manager.robot.data.default_root_state[0:1, :7].clone()
        pose[0, 0:2] = torch.as_tensor([sx, sy], dtype=torch.float32, device=device)
        nav_manager.robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=device))
        nav_manager.robot.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=device), env_ids=torch.tensor([0], device=device),
        )
        # Clear episode state
        nav_manager._robot_episode_steps[0] = 0
        nav_manager.paths_xy[car_idx] = path_xy
        nav_manager.waypoint_ids[car_idx] = 1
        nav_manager.reached[car_idx] = False
        nav_manager._robot_prev_progress_dist[0] = np.float32(
            np.linalg.norm(actual_goal - actual_start)
        )
        nav_manager._robot_progress_targets[0] = start_xy

        traj = []
        done_flag = False
        step = 0
        max_steps = nav_manager.config.rl_max_episode_steps

        obs, _ = env.reset(None)
        while not done_flag and step < max_steps:
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                model_outs = agent.model(obs_td)
                humanoid_action = model_outs.get("mean_action", model_outs["action"])
                robot_obs, robot_neighbors, neighbor_mask, robot_depth, robot_map = nav_manager.get_robot_rl_observations()
                mean, _, _ = ppo.model(robot_obs, robot_depth, robot_map, robot_neighbors, neighbor_mask)
                robot_action = bounded_robot_action(mean)
            nav_manager.set_robot_rl_actions(robot_action)
            _, _, dones, _, _ = env.step(humanoid_action)
            if nav_manager.config.rl_depth_enabled and args.headless:
                env.simulator._sim.render()
            _, _, robot_done, info, _, _, _, _ = nav_manager.get_robot_rl_feedback()

            robot_xy = nav_manager.drive.positions_xy()[0]
            robot_speed = float(abs(robot_action[0, 0].item()))
            traj.append([float(robot_xy[0]), float(robot_xy[1]), robot_speed])

            cell = _pos_to_cell(robot_xy)
            visit_counts[cell] += 1
            speed_sum[cell] += robot_speed
            vectors[cell] += np.array([
                robot_speed * np.cos(float(nav_manager.drive.yaws()[0])),
                robot_speed * np.sin(float(nav_manager.drive.yaws()[0])),
            ])

            if robot_done[0]:
                robot_done_np = robot_done.cpu().numpy()
                if robot_done_np[0]:
                    if info.get("reached", torch.zeros(1))[0]:
                        stats["success"] += 1
                    elif info.get("collision", torch.zeros(1))[0]:
                        stats["collision"] += 1
                    elif info.get("stuck", torch.zeros(1))[0]:
                        stats["stuck"] += 1
                    else:
                        stats["timeout"] += 1
                    done_flag = True

            nav_manager.reset_robot_rl_episodes(robot_done)
            h_done = dones.nonzero(as_tuple=False).squeeze(-1)
            if h_done.numel() > 0:
                done_indices = h_done
            step += 1

        robot_trajs.append(traj)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep + 1}/{n_episodes}  success={stats['success']} "
                  f"collision={stats['collision']} timeout={stats['timeout']}")

    # ── Normalize ───────────────────────────────────────────
    nz = visit_counts > 0
    vectors[nz] /= visit_counts[nz][:, None]
    avg_speeds = np.zeros(grid_size)
    avg_speeds[nz] = speed_sum[nz] / visit_counts[nz]

    # ── Output ──────────────────────────────────────────────
    # Save raw data
    data = {
        "vectors": vectors, "visits": visit_counts, "avg_speeds": avg_speeds,
        "robot_trajs": robot_trajs, "stats": stats,
        "grid_meta": grid_meta,
    }
    data_path = output_dir / f"{ckpt_name}_eval.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    gif_path = _save_evaluation_gif(nav_manager, output_dir)
    total_eval = n_episodes - skipped
    evaluation_metrics = {
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
        "requested_episodes": n_episodes,
        "evaluated_episodes": total_eval,
        "skipped_episodes": skipped,
        **{key: int(value) for key, value in stats.items()},
        "success_rate": stats["success"] / max(total_eval, 1),
        "collision_rate": stats["collision"] / max(total_eval, 1),
        "timeout_rate": stats["timeout"] / max(total_eval, 1),
        "stuck_rate": stats["stuck"] / max(total_eval, 1),
        "seed": args.seed,
        "gif": None if gif_path is None else str(gif_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(evaluation_metrics, indent=2), encoding="utf-8"
    )

    # Save plots
    save_plots(vectors, visit_counts, avg_speeds, map_meta, grid_meta,
               robot_trajs, None, None, None, str(output_dir), ckpt_name)

    total = n_episodes - skipped
    print(f"\n{'═' * 40}")
    if skipped:
        print(f"  Skipped:   {skipped} (no valid goal found)")
    print(f"  Success:  {stats['success']}/{total} ({stats['success'] / max(total,1) * 100:.1f}%)")
    print(f"  Collision:{stats['collision']}/{total} ({stats['collision'] / max(total,1) * 100:.1f}%)")
    print(f"  Timeout:  {stats['timeout']}/{total} ({stats['timeout'] / max(total,1) * 100:.1f}%)")
    print(f"  Stuck:    {stats['stuck']}/{total} ({stats['stuck'] / max(total,1) * 100:.1f}%)")
    print(f"{'═' * 40}")
    print(f"Results → {output_dir}")

    # Cleanup
    import omni.kit.app
    omni.kit.app.get_app().post_quit()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Grid-based policy evaluation")
    p.add_argument("--ckpt", required=True, help="Path to PPO checkpoint")
    p.add_argument("--env-config", default="CrowdSim/config/env.yaml")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--search-step", type=float, default=1.0,
                   help="Grid spacing for start pose sampling (m)")
    p.add_argument("--min-dist", type=float, default=1.5,
                   help="Min distance from obstacles / map border")
    p.add_argument("--grid-cell", type=float, default=0.5,
                   help="Grid cell size for statistics (m)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-data", type=str, default=None,
                   help="Skip simulation, regenerate plots from .pkl")
    args = p.parse_args()

    if args.load_data:
        # Regenerate plots from cached data
        with open(args.load_data, "rb") as f:
            data = pickle.load(f)
        from CrowdSim.utils.map_metadata import load_occupancy_map_metadata
        from CrowdSim.crowd_sim import cfg_path
        env_config = __import__("CrowdSim.crowd_sim", fromlist=["load_config"]).load_config(
            cfg_path(args.env_config)
        )
        map_meta = load_occupancy_map_metadata(
            __import__("CrowdSim.scene_setup", fromlist=["resolve_repo_path"]).resolve_repo_path(
                env_config["scene"]["scene_map"]
            )
        )
        ckpt_name = Path(args.load_data).stem.replace("_eval", "")
        output_dir = Path(args.load_data).parent
        save_plots(
            data["vectors"], data["visits"], data["avg_speeds"],
            map_meta, data["grid_meta"],
            data.get("robot_trajs", []), None, None, None,
            str(output_dir), ckpt_name,
        )
        print(f"Plots regenerated → {output_dir}")
        return

    run_evaluation(args)


if __name__ == "__main__":
    main()
