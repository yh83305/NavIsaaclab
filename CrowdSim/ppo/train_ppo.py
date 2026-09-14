"""Train a simple PPO policy for CrowdSim Jetbot navigation."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import torch  # noqa: E402

torch.set_float32_matmul_precision("high")

from CrowdSim.ppo.ppo_policy import (  # noqa: E402
    RobotPPOConfig,
    RobotPPOTrainer,
    RobotRolloutBuffer,
    robot_network_kwargs,
)
from CrowdSim.utils.config_loader import cfg_path as _cfg_path, load_config  # noqa: E402


def cfg_path(path_like: str) -> Path:
    """Module-local shim: resolve path relative to this file's PROJECT_ROOT."""
    return _cfg_path(path_like, PROJECT_ROOT)


# ── Logging helpers ────────────────────────────────────────────────────────

def _flatten_config(d: dict, prefix: str = "", sep: str = "/") -> dict:
    """Recursively flatten a nested dict into a single-level dict with path keys."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def _get_git_info() -> dict:
    """Return git branch, short commit hash, and dirty-flag as a plain dict."""
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=PROJECT_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    branch = _run(["git", "branch", "--show-current"]) or \
             _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty_files = _run(["git", "status", "--porcelain"])
    return {
        "git/branch": branch,
        "git/commit": commit,
        "git/dirty":  bool(dirty_files),
        "git/dirty_files": dirty_files or "clean",
    }


def _render_trajectory_gif(
    nav_manager,
    n_frames: int = 200,
    fps: float = 8.0,
    scale: float = 1.0,
) -> "Any | None":
    """Render the last *n_frames* trajectory steps as an animated GIF and
    return a ``wandb.Video`` wrapping it.

    Uses render_nav_gif_bytes (offline JSONL renderer) for PPO.
    """
    try:
        import wandb
        from CrowdSim.tools.render_navigation_fast import render_nav_gif_bytes
    except ImportError:
        return None

    out_dir = Path(nav_manager.config.output_dir if hasattr(nav_manager.config, "output_dir")
                   else "output/crowdsim_robot_ppo")
    gif_path = out_dir / "eval_video.gif"

    import numpy as np

    traj_path = getattr(nav_manager, "trajectory_log_path", None)
    cfg       = nav_manager.config

    gif_bytes = render_nav_gif_bytes(
        obstacle_map     = nav_manager.obstacle_map,
        resolution       = float(cfg.map_resolution),
        origin_xy        = tuple(cfg.map_origin_xy),
        num_humanoids    = int(cfg.num_humanoids),
        num_robots       = int(cfg.num_robots),
        paths_xy         = [np.asarray(p) for p in nav_manager.paths_xy],
        goals_xy         = np.asarray(nav_manager.goals_xy, dtype=np.float32),
        trajectory_log_path = traj_path,
        n_frames         = n_frames,
        fps              = fps,
        scale            = scale,
    )

    if gif_bytes is None:
        return None

    # Save to disk + wandb
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.write_bytes(gif_bytes)
    kb = len(gif_bytes) // 1024
    print(f"[GIF] Saved → {gif_path}  ({n_frames} frames @ {fps}fps, {kb} KB)")
    return wandb.Video(io.BytesIO(gif_bytes), format="gif", fps=int(fps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO local navigation policy for CrowdSim robots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-config", default="CrowdSim/config/env.yaml")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="IsaacLab parallel envs. Defaults to "
                             "max(num_humanoids, num_robots) from env.yaml. "
                             "num_robots/num_humanoids are taken from env.yaml.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scene-physics", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(cfg_path(args.env_config))

    # Create output_dir BEFORE build_env so we can bind the navigation
    # trajectory log to this run's directory.  Multiple concurrent runs
    # each get their own sub-directory and never overwrite each other's
    # trajectory_latest.jsonl / paths_latest.json.
    training_cfg = get_config_section(config, "training")
    network_cfg  = get_config_section(config, "network")
    env_config_path = cfg_path(args.env_config)
    base_output_dir = Path(
        str(config_value(args.output_dir, training_cfg, "output_dir", "output/crowdsim_robot_ppo"))
    )
    output_dir = make_training_output_dir(base_output_dir)
    copy_training_configs(output_dir, env_config_path)
    print(f"[CrowdSim][PPO] Output directory: {output_dir}")

    # Bind trajectory logs to this run's directory before the env starts writing.
    nav_rec = config.setdefault("navigation", {}).setdefault("recording", {})
    nav_rec["output_dir"] = str(output_dir / "navigation")

    # num_robots and num_humanoids come from env.yaml (independent of num_envs).
    # --num-envs (default None) is inferred as max(num_humanoids, num_robots, 1)
    # so the crowd fits; extra envs beyond the active counts hold frozen slots.
    from CrowdSim.world.builder import build_env
    result = build_env(config, num_envs=args.num_envs, headless=args.headless,
                       scene_physics=args.scene_physics)
    env, agent, nav_manager, runtime = result
    device = runtime.fabric.device
    depth_enabled = nav_manager.config.rl_depth_enabled
    depth_size = nav_manager.config.rl_depth_size if depth_enabled else 0
    map_enabled = nav_manager.config.rl_map_size > 0
    map_size = nav_manager.config.rl_map_size if map_enabled else 0

    ppo_cfg = RobotPPOConfig(
        obs_dim=nav_manager.robot_rl_obs_dim,
        vector_obs_dim=nav_manager.robot_rl_vector_obs_dim,
        # map_size / map_enabled are supplied by robot_network_kwargs; do not pass
        # them here too or Python raises "multiple values for keyword argument".
        **robot_network_kwargs(
            network_cfg,
            hidden_dim_override=args.hidden_dim,
            num_layers_override=args.num_layers,
            depth_enabled=depth_enabled,
            depth_size=depth_size,
            map_enabled=map_enabled,
            map_size=map_size,
            num_neighbors=nav_manager.config.rl_num_neighbors,
        ),
        lr=float(config_value(args.lr, training_cfg, "lr", 3.0e-4)),
        gamma=float(training_cfg.get("gamma", 0.99)),
        gae_lambda=float(training_cfg.get("gae_lambda", 0.95)),
        clip_ratio=float(training_cfg.get("clip_ratio", 0.2)),
        value_coef=float(training_cfg.get("value_coef", 0.5)),
        entropy_coef=float(training_cfg.get("entropy_coef", 0.005)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        ppo_epochs=int(config_value(args.ppo_epochs, training_cfg, "ppo_epochs", 4)),
        minibatch_size=int(config_value(args.minibatch_size, training_cfg, "minibatch_size", 256)),
    )
    ppo = RobotPPOTrainer(ppo_cfg, device)
    start_step = 0
    if args.resume is not None:
        start_step = ppo.load(Path(args.resume).expanduser().resolve())
        nav_manager.goal_curriculum.load_state_dict(
            ppo.extra_state.get("goal_curriculum")
        )
        print(f"[CrowdSim][PPO] Resumed from {args.resume} at robot_step={start_step}.")

    buffer = RobotRolloutBuffer(
        rollout_steps=int(config_value(args.rollout_steps, training_cfg, "rollout_steps", 256)),
        num_envs=nav_manager.config.num_robots,
        obs_dim=nav_manager.robot_rl_obs_dim,
        action_dim=2,
        device=device,
        depth_size=depth_size,
        map_size=map_size,
        num_neighbors=nav_manager.config.rl_num_neighbors,
        neighbor_dim=nav_manager.robot_rl_neighbor_dim,
    )
    # Build the wandb config capturing every runtime parameter.
    _resume_ckpt = str(Path(args.resume).expanduser().resolve()) if args.resume else "none"
    _total_steps  = int(config_value(args.total_steps,    training_cfg, "total_steps",    200_000))
    _rollout      = int(config_value(args.rollout_steps,  training_cfg, "rollout_steps",  256))
    _save_int     = int(config_value(args.save_interval,  training_cfg, "save_interval",  20_000))
    wandb_cfg: dict = {
        **_flatten_config(config,     prefix="env"),   # full env.yaml
        # CLI overrides — win over yaml defaults
        "args/num_envs":       args.num_envs,
        "args/headless":       args.headless,
        "args/total_steps":    _total_steps,
        "args/rollout_steps":  _rollout,
        "args/ppo_epochs":     int(config_value(args.ppo_epochs,     training_cfg, "ppo_epochs",     4)),
        "args/minibatch_size": int(config_value(args.minibatch_size, training_cfg, "minibatch_size", 256)),
        "args/lr":             float(config_value(args.lr,           training_cfg, "lr",             3.0e-4)),
        "args/gamma":          ppo_cfg.gamma,
        "args/gae_lambda":     ppo_cfg.gae_lambda,
        "args/clip_ratio":     ppo_cfg.clip_ratio,
        "args/value_coef":     ppo_cfg.value_coef,
        "args/entropy_coef":   ppo_cfg.entropy_coef,
        "args/max_grad_norm":  ppo_cfg.max_grad_norm,
        "args/save_interval":  _save_int,
        # checkpoint paths (absolute, resolved)
        "checkpoints/policy":  _resume_ckpt,
        # config file paths on disk
        "config_files/env_yaml": str(env_config_path),
        # derived model dims
        "model/obs_dim":        nav_manager.robot_rl_obs_dim,
        "model/vector_obs_dim": nav_manager.robot_rl_vector_obs_dim,
        "model/map_size":       nav_manager.config.rl_map_size,
        "model/num_robots":     nav_manager.config.num_robots,
        "model/depth_enabled":  depth_enabled,
        "checkpoints/policy_step": start_step,
    }

    train_loop(
        runtime=runtime,
        nav_manager=nav_manager,
        ppo=ppo,
        buffer=buffer,
        total_steps=_total_steps,
        start_step=start_step,
        save_interval=_save_int,
        output_dir=output_dir,
        use_wandb=not args.no_wandb,
        wandb_config=wandb_cfg,
        run_name=f"base_{output_dir.name}",
    )


def make_training_output_dir(base_output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = base_output_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    latest_link = base_output_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_dir.name, target_is_directory=True)
    except OSError:
        pass
    return output_dir


def copy_training_configs(output_dir: Path, env_config_path: Path) -> None:
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_config_path, config_dir / "env.yaml")


def train_loop(
    runtime,
    nav_manager: CrowdNavigationManager,
    ppo: RobotPPOTrainer,
    buffer: RobotRolloutBuffer,
    total_steps: int,
    start_step: int,
    save_interval: int,
    output_dir: Path,
    use_wandb: bool,
    wandb_config: dict | None = None,
    run_name: str | None = None,
) -> None:
    env = runtime.env
    agent = runtime.agent
    robot_steps = start_step
    done_indices = None
    episode_returns = torch.zeros(nav_manager.config.num_robots, device=ppo.device)
    episode_lengths = torch.zeros(nav_manager.config.num_robots, device=ppo.device)
    completed_returns: deque = deque(maxlen=100)
    completed_lengths: deque = deque(maxlen=100)
    next_save = ((robot_steps // save_interval) + 1) * save_interval if save_interval > 0 else None
    logger = RobotTrainingLogger(output_dir, use_wandb, wandb_config=wandb_config, run_name=run_name)

    depth_enabled = buffer.has_depth
    map_enabled = buffer.has_map
    headless = getattr(getattr(env, "simulator", None), "headless", False)
    print(
        f"[CrowdSim][PPO] Training robot policy: robots={nav_manager.config.num_robots}, "
        f"obs_dim={nav_manager.robot_rl_obs_dim}"
        + (f", depth_size={buffer.depth.shape[-1]}" if depth_enabled else "")
        + (f", map_size={buffer.map.shape[-1]}" if map_enabled else "")
        + f", total_robot_steps={total_steps}."
    )
    curriculum = nav_manager.goal_curriculum
    if curriculum.spec.enabled:
        initial = curriculum.metrics()
        print(
            "[CrowdSim][PPO] Goal curriculum enabled: "
            f"stage={initial['curriculum/stage_name']} "
            f"({initial['curriculum/stage']:.0f}/"
            f"{initial['curriculum/num_stages'] - 1:.0f}) "
            f"distance=[{initial['curriculum/distance_min_m']:.1f},"
            f"{initial['curriculum/distance_max_m']:.1f}]m "
            f"heading=[{initial['curriculum/heading_min_degrees']:.1f},"
            f"{initial['curriculum/heading_max_degrees']:.1f}]deg"
        )
    while robot_steps < total_steps:
        buffer.reset()
        # GPU accumulators — avoid 14 GPU→CPU syncs per step (×256 steps = 3584/rollout).
        # Converted to Python floats in one batch after the rollout loop.
        rollout_accum = {
            key: torch.zeros(1, device=ppo.device)
            for key in (
                "reward_total", "reached", "collision", "timeout",
                "goal_distance", "progress", "reward_progress",
                "goal_heading_abs", "goal_heading_over_30deg",
                "reward_velocity_direction", "reward_proximity", "reward_smoothness",
                "reward_static_proximity",
                "reward_angular_velocity", "reward_angular_change", "reward_linear_change",
                "reward_goal", "reward_collision", "reward_timeout",
                "stuck", "reward_stuck",
                "action_linear", "action_angular", "action_abs_angular",
                "action_linear_low", "action_linear_high",
                "action_angular_saturated",
            )
        }

        for _ in range(buffer.rollout_steps):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            with torch.no_grad():
                model_outs = agent.model(obs_td)
                humanoid_action = model_outs.get("mean_action", model_outs["action"])
                robot_obs, robot_neighbors, neighbor_mask, robot_depth, robot_map = nav_manager.get_robot_rl_observations()
                robot_action, raw_action, log_prob, value = ppo.act(
                    robot_obs, robot_depth, robot_map, robot_neighbors, neighbor_mask
                )

            nav_manager.set_robot_rl_actions(robot_action)
            _, _, humanoid_dones, _, _ = env.step(humanoid_action)
            if depth_enabled and headless:
                env.simulator._sim.render()
            _, robot_reward, robot_done, info, _depth_out, _map_out, _, _ = nav_manager.get_robot_rl_feedback()
            buffer.add(robot_obs, raw_action, log_prob, robot_reward, robot_done, value,
                       depth=robot_depth if depth_enabled else None,
                       map_patch=robot_map if map_enabled else None,
                       neighbors=robot_neighbors, neighbor_mask=neighbor_mask)
            episode_returns += robot_reward
            episode_lengths += 1

            zero_reward = torch.zeros(1, device=ppo.device)
            rollout_accum["reward_total"]             += info.get("reward_total",             robot_reward).sum()
            rollout_accum["reached"]                  += info.get("reached",                  robot_done).float().sum()
            rollout_accum["collision"]                += info.get("collision",                 robot_done).float().sum()
            rollout_accum["timeout"]                  += info.get("timeout",                   robot_done).float().sum()
            rollout_accum["goal_distance"]            += info.get("distance_to_goal",          zero_reward).sum()
            goal_heading = torch.atan2(robot_obs[:, 1], robot_obs[:, 2])
            rollout_accum["goal_heading_abs"]         += goal_heading.abs().sum()
            rollout_accum["goal_heading_over_30deg"]  += (
                goal_heading.abs() > (math.pi / 6.0)
            ).float().sum()
            rollout_accum["progress"]                 += info.get("progress",                  zero_reward).sum()
            rollout_accum["reward_progress"]          += info.get("reward_progress",           zero_reward).sum()
            rollout_accum["reward_velocity_direction"]+= info.get("reward_velocity_direction", zero_reward).sum()
            rollout_accum["reward_proximity"]         += info.get("reward_proximity",          zero_reward).sum()
            rollout_accum["reward_static_proximity"]  += info.get("reward_static_proximity",   zero_reward).sum()
            rollout_accum["reward_smoothness"]        += info.get("reward_smoothness",         zero_reward).sum()
            rollout_accum["reward_angular_velocity"] += info.get("reward_angular_velocity",  zero_reward).sum()
            rollout_accum["reward_angular_change"]   += info.get("reward_angular_change",    zero_reward).sum()
            rollout_accum["reward_linear_change"]    += info.get("reward_linear_change",     zero_reward).sum()
            rollout_accum["reward_goal"]              += info.get("reward_goal",               zero_reward).sum()
            rollout_accum["reward_collision"]         += info.get("reward_collision",          zero_reward).sum()
            rollout_accum["reward_timeout"]           += info.get("reward_timeout",            zero_reward).sum()
            rollout_accum["stuck"]                    += info.get("stuck",                     robot_done).float().sum()
            rollout_accum["reward_stuck"]             += info.get("reward_stuck",              zero_reward).sum()
            rollout_accum["action_linear"]             += robot_action[:, 0].sum()
            rollout_accum["action_angular"]            += robot_action[:, 1].sum()
            rollout_accum["action_abs_angular"]        += robot_action[:, 1].abs().sum()
            rollout_accum["action_linear_low"]         += (robot_action[:, 0] < 0.05).float().sum()
            rollout_accum["action_linear_high"]        += (robot_action[:, 0] > 0.95).float().sum()
            rollout_accum["action_angular_saturated"]  += (robot_action[:, 1].abs() > 0.95).float().sum()

            if robot_done.any():
                done_ids = robot_done.nonzero(as_tuple=False).squeeze(-1)
                completed_returns.extend(episode_returns[done_ids].detach().cpu().tolist())
                completed_lengths.extend(episode_lengths[done_ids].detach().cpu().tolist())
                episode_returns[done_ids] = 0.0
                episode_lengths[done_ids] = 0.0
                reached = info.get("reached", robot_done)
                changed = curriculum.observe(
                    reached[done_ids].detach().cpu().numpy()
                )
                if changed:
                    current = curriculum.metrics()
                    print(
                        "[CrowdSim][PPO] Goal curriculum updated: "
                        f"SR={current['curriculum/success_rate']:.1%} "
                        f"stage={current['curriculum/stage_name']} "
                        f"({current['curriculum/stage']:.0f}/"
                        f"{current['curriculum/num_stages'] - 1:.0f}) "
                        f"distance=[{current['curriculum/distance_min_m']:.1f},"
                        f"{current['curriculum/distance_max_m']:.1f}]m "
                        f"heading=[{current['curriculum/heading_min_degrees']:.1f},"
                        f"{current['curriculum/heading_max_degrees']:.1f}]deg"
                    )
                nav_manager.reset_robot_rl_episodes(robot_done)

            done_indices = humanoid_dones.nonzero(as_tuple=False).squeeze(-1)
            robot_steps += nav_manager.config.num_robots

        with torch.no_grad():
            last_obs, last_neighbors, last_neighbor_mask, last_depth, last_map = nav_manager.get_robot_rl_observations()
            last_value = ppo.value(
                last_obs,
                last_depth if depth_enabled else None,
                last_map if map_enabled else None,
                last_neighbors, last_neighbor_mask,
            )
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=ppo.config.gamma,
            gae_lambda=ppo.config.gae_lambda,
        )
        # Batch-convert GPU accumulators → Python floats (one sync per key, not per step).
        rollout_sums = {k: float(v.item()) for k, v in rollout_accum.items()}

        stats = ppo.update(buffer)

        recent_returns = list(completed_returns)
        recent_lengths = list(completed_lengths)
        mean_return = sum(recent_returns) / max(len(recent_returns), 1)
        mean_length = sum(recent_lengths) / max(len(recent_lengths), 1)
        total_samples = buffer.rollout_steps * nav_manager.config.num_robots
        denom = max(total_samples, 1)

        print(
            f"[CrowdSim][PPO] robot_steps={robot_steps} "
            f"return={mean_return:.3f} len={mean_length:.1f} "
            f"reached={rollout_sums['reached']:.0f} collision={rollout_sums['collision']:.0f} timeout={rollout_sums['timeout']:.0f} stuck={rollout_sums['stuck']:.0f} "
            f"goal_dist={rollout_sums['goal_distance'] / denom:.3f} progress={rollout_sums['progress'] / denom:.3f} "
            f"goal_angle={math.degrees(rollout_sums['goal_heading_abs'] / denom):.1f}deg "
            f"reward={rollout_sums['reward_total'] / denom:.3f} "
            f"r_progress={rollout_sums['reward_progress'] / denom:.3f} r_veldir={rollout_sums['reward_velocity_direction'] / denom:.3f} "
            f"r_prox={rollout_sums['reward_proximity'] / denom:.3f} "
            f"r_static={rollout_sums['reward_static_proximity'] / denom:.3f} "
            f"r_smooth={rollout_sums['reward_smoothness'] / denom:.3f} "
            f"r_w={rollout_sums['reward_angular_velocity'] / denom:.3f} "
            f"r_dw={rollout_sums['reward_angular_change'] / denom:.3f} "
            f"r_dv={rollout_sums['reward_linear_change'] / denom:.3f} "
            f"r_goal={rollout_sums['reward_goal'] / denom:.3f} "
            f"policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} "
            f"ratio={stats['ratio_mean']:.3f}+/-{stats['ratio_std']:.3f} "
            f"clip={stats['clip_fraction']:.3f} kl={stats['approx_kl']:.5f} "
            f"ev={stats['explained_variance']:.3f}"
            + (
                f" curriculum={curriculum.stage.name} "
                f"curriculum_SR={curriculum.success_rate:.1%}"
                if curriculum.spec.enabled else ""
            )
        )
        logger.log(
            step=robot_steps,
            total_samples=total_samples,
            mean_return=mean_return,
            mean_length=mean_length,
            reached=rollout_sums["reached"],
            collision=rollout_sums["collision"],
            timeout=rollout_sums["timeout"],
            stuck=rollout_sums["stuck"],
            goal_distance=rollout_sums["goal_distance"] / denom,
            goal_heading_abs_degrees=math.degrees(
                rollout_sums["goal_heading_abs"] / denom
            ),
            goal_heading_over_30deg_frac=(
                rollout_sums["goal_heading_over_30deg"] / denom
            ),
            progress=rollout_sums["progress"] / denom,
            reward_total=rollout_sums["reward_total"] / denom,
            reward_progress=rollout_sums["reward_progress"] / denom,
            reward_velocity_direction=rollout_sums["reward_velocity_direction"] / denom,
            reward_proximity=rollout_sums["reward_proximity"] / denom,
            reward_static_proximity=rollout_sums["reward_static_proximity"] / denom,
            reward_smoothness=rollout_sums["reward_smoothness"] / denom,
            reward_angular_velocity=rollout_sums["reward_angular_velocity"] / denom,
            reward_angular_change=rollout_sums["reward_angular_change"] / denom,
            reward_linear_change=rollout_sums["reward_linear_change"] / denom,
            reward_goal=rollout_sums["reward_goal"] / denom,
            reward_collision=rollout_sums["reward_collision"] / denom,
            reward_timeout=rollout_sums["reward_timeout"] / denom,
            reward_stuck=rollout_sums["reward_stuck"] / denom,
            policy_loss=stats["policy_loss"],
            value_loss=stats["value_loss"],
            entropy=stats["entropy"],
            ratio_mean=stats["ratio_mean"],
            ratio_std=stats["ratio_std"],
            clip_fraction=stats["clip_fraction"],
            approx_kl=stats["approx_kl"],
            grad_norm=stats["grad_norm"],
            explained_variance=stats["explained_variance"],
            advantage_mean=stats["advantage_mean"],
            advantage_std=stats["advantage_std"],
            return_mean=stats["return_mean"],
            return_std=stats["return_std"],
            value_mean=stats["value_mean"],
            value_std=stats["value_std"],
            latent_std_linear=stats["latent_std_linear"],
            latent_std_angular=stats["latent_std_angular"],
            action_linear_mean=rollout_sums["action_linear"] / denom,
            action_angular_mean=rollout_sums["action_angular"] / denom,
            action_abs_angular_mean=rollout_sums["action_abs_angular"] / denom,
            action_linear_low_frac=rollout_sums["action_linear_low"] / denom,
            action_linear_high_frac=rollout_sums["action_linear_high"] / denom,
            action_angular_saturated_frac=(
                rollout_sums["action_angular_saturated"] / denom
            ),
            **curriculum.metrics(),
        )

        if next_save is not None and robot_steps >= next_save:
            extra_state = {"goal_curriculum": curriculum.state_dict()}
            ppo.save(
                output_dir / "robot_ppo_latest.pt", robot_steps,
                extra_state=extra_state,
            )
            ppo.save(
                output_dir / f"robot_ppo_{robot_steps}.pt", robot_steps,
                extra_state=extra_state,
            )
            logger.log_trajectory_image(nav_manager, step=robot_steps)
            next_save += save_interval

    ppo.save(
        output_dir / "robot_ppo_latest.pt", robot_steps,
        extra_state={"goal_curriculum": curriculum.state_dict()},
    )
    logger.log_trajectory_image(nav_manager, step=robot_steps)
    logger.close()
    print(f"[CrowdSim][PPO] Saved final checkpoint to {output_dir / 'robot_ppo_latest.pt'}")
    print("[CrowdSim][PPO] Training complete. Shutting down...")
    import omni.kit.app
    omni.kit.app.get_app().post_quit()


class RobotTrainingLogger:
    def __init__(
        self,
        output_dir: Path,
        use_wandb: bool,
        wandb_config: dict | None = None,
        run_name: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._wandb      = None
        self._start_time = time.time()

        if use_wandb:
            import wandb
            self._wandb = wandb
            # Merge caller-supplied config with git info and output dir.
            full_config: dict = {"output_dir": str(output_dir)}
            full_config.update(_get_git_info())
            if wandb_config:
                full_config.update(wandb_config)
            self._wandb.init(
                project="crowdsim-ppo",
                name=run_name if run_name is not None else output_dir.name,
                dir=str(output_dir),
                config=full_config,
            )

    @staticmethod
    def _outcome_stats(metrics: dict) -> dict:
        """Return both raw counts and rates for every episode-outcome type."""
        reached   = float(metrics.get("reached",   0))
        collision = float(metrics.get("collision", 0))
        timeout   = float(metrics.get("timeout",   0))
        stuck     = float(metrics.get("stuck",     0))
        total     = max(reached + collision + timeout + stuck, 1)
        return {
            # raw counts (easy to compare absolute progress across runs)
            "outcome/reached":   reached,
            "outcome/collision": collision,
            "outcome/timeout":   timeout,
            "outcome/stuck":     stuck,
            "outcome/total_episodes": reached + collision + timeout + stuck,
            # normalised rates (sum ≈ 1)
            "outcome/reached_rate":   reached   / total,
            "outcome/collision_rate": collision / total,
            "outcome/timeout_rate":   timeout   / total,
            "outcome/stuck_rate":     stuck     / total,
        }

    def log(self, step: int, **metrics: float) -> None:
        """Log metrics to wandb.  Missing keys are filled with 0.0.  Extra
        keys not in the fixed set (e.g. curriculum/*) are passed through."""
        if self._wandb is not None:
            m = lambda k: float(metrics.get(k, 0.0))
            # Collect extra keys (curriculum/* etc.) not handled by the fixed
            # dict below — pass them through as-is (keeps string values like
            # stage_name intact for wandb categorical metrics).
            _fixed = {
                "policy_loss", "value_loss", "entropy",
                "reward_total", "reward_progress", "reward_velocity_direction",
                "reward_proximity", "reward_static_proximity",
                "reward_smoothness", "reward_angular_velocity",
                "reward_angular_change", "reward_linear_change", "reward_goal",
                "reward_collision", "reward_timeout", "reward_stuck",
                "goal_distance", "progress", "mean_return", "mean_length",
                "goal_heading_abs_degrees", "goal_heading_over_30deg_frac",
                "reached", "collision", "timeout", "stuck",
                "ratio_mean", "ratio_std", "clip_fraction", "approx_kl",
                "grad_norm", "explained_variance", "advantage_mean",
                "advantage_std", "return_mean", "return_std", "value_mean",
                "value_std", "latent_std_linear", "latent_std_angular",
                "action_linear_mean", "action_angular_mean",
                "action_abs_angular_mean", "action_linear_low_frac",
                "action_linear_high_frac", "action_angular_saturated_frac",
            }
            passthrough = {k: v for k, v in metrics.items() if k not in _fixed}
            self._wandb.log({
                # training step as a metric (so it can be used as Y-axis too)
                "train/step":           step,
                # losses
                "loss/policy_loss": m("policy_loss"),
                "loss/value_loss":  m("value_loss"),
                "loss/entropy":     m("entropy"),
                "loss/grad_norm":   m("grad_norm"),
                "policy/ratio_mean": m("ratio_mean"),
                "policy/ratio_std": m("ratio_std"),
                "policy/clip_fraction": m("clip_fraction"),
                "policy/approx_kl": m("approx_kl"),
                "policy/latent_std_linear": m("latent_std_linear"),
                "policy/latent_std_angular": m("latent_std_angular"),
                "value/explained_variance": m("explained_variance"),
                "value/prediction_mean": m("value_mean"),
                "value/prediction_std": m("value_std"),
                "value/return_mean": m("return_mean"),
                "value/return_std": m("return_std"),
                "advantage/mean": m("advantage_mean"),
                "advantage/std": m("advantage_std"),
                "action/linear_mean": m("action_linear_mean"),
                "action/angular_mean": m("action_angular_mean"),
                "action/angular_abs_mean": m("action_abs_angular_mean"),
                "action/linear_low_frac": m("action_linear_low_frac"),
                "action/linear_high_frac": m("action_linear_high_frac"),
                "action/angular_saturated_frac":
                    m("action_angular_saturated_frac"),
                "goal/heading_abs_degrees": m("goal_heading_abs_degrees"),
                "goal/heading_over_30deg_frac":
                    m("goal_heading_over_30deg_frac"),
                # per-step rewards
                "step/reward_total":              m("reward_total"),
                "step/reward_progress":           m("reward_progress"),
                "step/reward_velocity_direction": m("reward_velocity_direction"),
                "step/reward_proximity":          m("reward_proximity"),
                "step/reward_static_proximity":
                    m("reward_static_proximity"),
                "step/reward_smoothness":         m("reward_smoothness"),
                "step/reward_angular_velocity":   m("reward_angular_velocity"),
                "step/reward_angular_change":     m("reward_angular_change"),
                "step/reward_linear_change":      m("reward_linear_change"),
                "step/reward_goal":               m("reward_goal"),
                "step/reward_collision":          m("reward_collision"),
                "step/reward_timeout":            m("reward_timeout"),
                "step/reward_stuck":              m("reward_stuck"),
                "step/goal_distance":             m("goal_distance"),
                "step/progress":                  m("progress"),
                # episode-level summary
                "episode/return": m("mean_return"),
                "episode/length": m("mean_length"),
                # outcome counts + rates
                **self._outcome_stats(metrics),
                # curriculum/* and any other extra keys
                **passthrough,
            }, step=step)

    def log_trajectory_image(self, nav_manager, step: int) -> None:
        """Upload an animated GIF of the recent trajectory to wandb."""
        if self._wandb is None:
            return
        # Cover one full max-length episode at real simulation speed.
        n_frames = int(nav_manager.config.rl_max_episode_steps)*2
        fps      = float(nav_manager.config.update_hz)
        img = _render_trajectory_gif(nav_manager, n_frames=n_frames, fps=fps)
        if img is not None:
            self._wandb.log({"viz/trajectory": img}, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def get_config_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    section = cfg.get(key, {})
    return section if isinstance(section, dict) else {}


def config_value(cli_value, cfg: dict[str, Any], key: str, default):
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


if __name__ == "__main__":
    main()
