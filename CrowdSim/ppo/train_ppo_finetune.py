"""Closed-loop PPO finetuning with the offline view-clearance reward model.

The saved policy checkpoint uses the normal CrowdSim PPO schema and can be
passed directly to ``CrowdSim/tools/eval_policy.py``.
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
import json
from pathlib import Path
import shutil
import sys
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
from CrowdSim.ppo.reward_model_ppo import (  # noqa: E402
    OnlineRewardHistory,
    RewardCompositionConfig,
    compose_reward,
    reward_model_timing,
)
from CrowdSim.ppo.train_ppo import (  # noqa: E402
    RobotTrainingLogger,
    _flatten_config,
    cfg_path,
    config_value,
    copy_training_configs,
    get_config_section,
    make_training_output_dir,
)
from CrowdSim.pref.offline_reward_model import load_offline_reward_model  # noqa: E402
from CrowdSim.utils.config_loader import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune CrowdSim PPO in closed loop with an offline reward model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-config", default="CrowdSim/config/env.yaml")
    parser.add_argument(
        "--train-config", default="CrowdSim/config/reward_model_ppo.yaml",
        help="Reward-model PPO algorithm configuration",
    )
    parser.add_argument("--base-ppo", required=True, help="Base PPO checkpoint")
    parser.add_argument("--reward-ckpt", required=True, help="Offline reward checkpoint")
    parser.add_argument(
        "--resume", default=None,
        help="Resume a reward-model PPO run (base PPO is still used as KL reference)",
    )
    parser.add_argument("--num-envs", type=int, default=None)
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
    parser.add_argument("--kl-beta", type=float, default=None)
    parser.add_argument("--learned-weight", type=float, default=None)
    parser.add_argument("--progress-weight", type=float, default=None)
    parser.add_argument("--goal-bonus", type=float, default=None)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--timeout-penalty", type=float, default=None)
    parser.add_argument("--stuck-penalty", type=float, default=None)
    parser.add_argument("--time-penalty", type=float, default=None)
    parser.add_argument("--environment-reward-weight", type=float, default=None)
    parser.add_argument("--learned-reward-clip", type=float, default=None)
    parser.add_argument("--min-history-steps", type=int, default=None)
    parser.add_argument(
        "--reward-source-hz", type=float, default=None,
        help="Required only for legacy reward checkpoints without source_hz metadata",
    )
    parser.add_argument(
        "--reward-runtime-stride", type=int, default=None,
        help="Override automatic reward-model temporal alignment",
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def _cfg(cli_value: Any, section: dict[str, Any], key: str, default: Any) -> Any:
    return config_value(cli_value, section, key, default)


def _reward_config(args: argparse.Namespace, section: dict[str, Any]) -> RewardCompositionConfig:
    return RewardCompositionConfig(
        learned_weight=float(_cfg(args.learned_weight, section, "learned_weight", 1.0)),
        progress_weight=float(_cfg(args.progress_weight, section, "progress_weight", 2.0)),
        goal_bonus=float(_cfg(args.goal_bonus, section, "goal_bonus", 10.0)),
        collision_penalty=float(_cfg(args.collision_penalty, section, "collision_penalty", -10.0)),
        timeout_penalty=float(_cfg(args.timeout_penalty, section, "timeout_penalty", -3.0)),
        stuck_penalty=float(_cfg(args.stuck_penalty, section, "stuck_penalty", -3.0)),
        time_penalty=float(_cfg(args.time_penalty, section, "time_penalty", -0.005)),
        environment_reward_weight=float(
            _cfg(args.environment_reward_weight, section, "environment_reward_weight", 0.0)
        ),
        learned_reward_clip=float(
            _cfg(args.learned_reward_clip, section, "learned_reward_clip", 5.0)
        ),
    )


def _set_optimizer_lr(ppo: RobotPPOTrainer, learning_rate: float) -> None:
    ppo.config.lr = float(learning_rate)
    for group in ppo.optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _save_training_gif(nav_manager, output_dir: Path) -> Path | None:
    """Save a recent closed-loop rollout GIF even when W&B is disabled."""

    try:
        import numpy as np
        from CrowdSim.tools.render_navigation_fast import render_nav_gif_bytes

        cfg = nav_manager.config
        gif = render_nav_gif_bytes(
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
        if gif is None:
            return None
        path = output_dir / "training_rollout.gif"
        path.write_bytes(gif)
        print(f"[CrowdSim][RM-PPO] Saved rollout GIF: {path}")
        return path
    except Exception as error:
        print(f"[CrowdSim][RM-PPO] GIF rendering skipped: {error}")
        return None


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mean(values: deque[float]) -> float:
    return sum(values) / max(len(values), 1)


def _save_policy(
    ppo: RobotPPOTrainer,
    path: Path,
    step: int,
    *,
    nav_manager,
    base_ppo: Path,
    reward_checkpoint: Path,
    reward_config: RewardCompositionConfig,
    reward_stride: int,
) -> None:
    ppo.save(path, step, extra_state={
        "goal_curriculum": nav_manager.goal_curriculum.state_dict(),
        "reward_model_ppo": {
            "base_ppo": str(base_ppo),
            "reward_checkpoint": str(reward_checkpoint),
            "reward_config": reward_config.to_dict(),
            "reward_runtime_stride": reward_stride,
        },
    })


def train_loop(
    *,
    runtime,
    nav_manager,
    ppo: RobotPPOTrainer,
    reference_model,
    buffer: RobotRolloutBuffer,
    reward_history: OnlineRewardHistory,
    reward_config: RewardCompositionConfig,
    reward_stride: int,
    kl_beta: float,
    total_steps: int,
    start_step: int,
    save_interval: int,
    output_dir: Path,
    base_ppo: Path,
    reward_checkpoint: Path,
    use_wandb: bool,
    wandb_config: dict[str, Any],
) -> None:
    env, agent = runtime.env, runtime.agent
    num_robots = nav_manager.config.num_robots
    robot_steps = int(start_step)
    model_sample_step = 0
    done_indices = None
    depth_enabled, map_enabled = buffer.has_depth, buffer.has_map
    headless = bool(getattr(getattr(env, "simulator", None), "headless", False))
    episode_returns = torch.zeros(num_robots, device=ppo.device)
    episode_lengths = torch.zeros(num_robots, device=ppo.device)
    completed_returns: deque[float] = deque(maxlen=100)
    completed_lengths: deque[float] = deque(maxlen=100)
    cumulative = {key: 0 for key in ("reached", "collision", "timeout", "stuck")}
    metrics_path = output_dir / "metrics.jsonl"
    next_save = (
        ((robot_steps // save_interval) + 1) * save_interval if save_interval > 0 else None
    )
    logger = RobotTrainingLogger(
        output_dir, use_wandb, wandb_config=wandb_config,
        run_name=f"reward_model_ppo_{output_dir.name}",
    )
    curriculum = nav_manager.goal_curriculum
    print(
        f"[CrowdSim][RM-PPO] robots={num_robots} steps={robot_steps}->{total_steps} "
        f"reward_stride={reward_stride} kl_beta={kl_beta:g}"
    )

    while robot_steps < total_steps:
        buffer.reset()
        keys = (
            "reward_total", "reward_model_raw", "reward_learned",
            "reward_progress_task", "reward_goal_task", "reward_collision_task",
            "reward_timeout_task", "reward_stuck_task", "reward_time_task",
            "reward_environment", "goal_distance", "progress", "reached",
            "collision", "timeout", "stuck", "action_linear", "action_abs_angular",
            "reward_model_ticks",
        )
        accum = {key: torch.zeros(1, device=ppo.device) for key in keys}

        for _ in range(buffer.rollout_steps):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                model_outs = agent.model(obs_td)
                humanoid_action = model_outs.get("mean_action", model_outs["action"])
                robot_obs, neighbors, neighbor_mask, depth, map_patch = (
                    nav_manager.get_robot_rl_observations()
                )
                executed_action, raw_action, log_prob, value = ppo.act(
                    robot_obs, depth, map_patch, neighbors, neighbor_mask
                )

            reward_tick = model_sample_step % reward_stride == 0
            if reward_tick:
                learned_raw = reward_history.append_and_score(
                    robot_obs, executed_action, depth
                )
                accum["reward_model_ticks"] += num_robots
            else:
                learned_raw = torch.zeros(num_robots, device=ppo.device)

            nav_manager.set_robot_rl_actions(executed_action)
            _, _, humanoid_dones, _, _ = env.step(humanoid_action)
            if depth_enabled and headless:
                env.simulator._sim.render()
            _, environment_reward, robot_done, info, _, _, _, _ = (
                nav_manager.get_robot_rl_feedback()
            )
            reward, components = compose_reward(
                learned_raw, environment_reward, info, reward_config
            )
            buffer.add(
                robot_obs, raw_action, log_prob, reward, robot_done, value,
                depth=depth if depth_enabled else None,
                map_patch=map_patch if map_enabled else None,
                neighbors=neighbors, neighbor_mask=neighbor_mask,
            )
            episode_returns += reward
            episode_lengths += 1

            zero = torch.zeros_like(reward)
            accum["reward_total"] += reward.sum()
            accum["reward_model_raw"] += learned_raw.sum()
            for key, value_tensor in components.items():
                accum[key] += value_tensor.sum()
            accum["goal_distance"] += info.get("distance_to_goal", zero).sum()
            accum["progress"] += info.get("progress", zero).sum()
            for key in ("reached", "collision", "timeout", "stuck"):
                accum[key] += info.get(key, robot_done).float().sum()
            accum["action_linear"] += executed_action[:, 0].sum()
            accum["action_abs_angular"] += executed_action[:, 1].abs().sum()

            if robot_done.any():
                done_ids = robot_done.nonzero(as_tuple=False).squeeze(-1)
                completed_returns.extend(episode_returns[done_ids].detach().cpu().tolist())
                completed_lengths.extend(episode_lengths[done_ids].detach().cpu().tolist())
                episode_returns[done_ids] = 0
                episode_lengths[done_ids] = 0
                reached = info.get("reached", robot_done)
                curriculum.observe(reached[done_ids].detach().cpu().numpy())
                reward_history.reset(robot_done)
                nav_manager.reset_robot_rl_episodes(robot_done)

            done_indices = humanoid_dones.nonzero(as_tuple=False).squeeze(-1)
            robot_steps += num_robots
            model_sample_step += 1

        with torch.no_grad():
            last_obs, last_neighbors, last_mask, last_depth, last_map = (
                nav_manager.get_robot_rl_observations()
            )
            last_value = ppo.value(
                last_obs, last_depth if depth_enabled else None,
                last_map if map_enabled else None, last_neighbors, last_mask,
            )
        buffer.compute_returns_and_advantages(
            last_value, gamma=ppo.config.gamma, gae_lambda=ppo.config.gae_lambda
        )
        rollout = {key: float(value.item()) for key, value in accum.items()}
        stats = ppo.update_with_kl(buffer, reference_model, kl_beta)
        samples = buffer.rollout_steps * num_robots
        denom = max(samples, 1)
        episodes = sum(rollout[key] for key in cumulative)
        for key in cumulative:
            cumulative[key] += int(rollout[key])
        record: dict[str, Any] = {
            "step": robot_steps,
            "samples": samples,
            "episode_return": _mean(completed_returns),
            "episode_length": _mean(completed_lengths),
            **{key: rollout[key] for key in cumulative},
            "episodes": episodes,
            "success_rate": rollout["reached"] / max(episodes, 1),
            "collision_rate": rollout["collision"] / max(episodes, 1),
            "goal_distance": rollout["goal_distance"] / denom,
            "progress": rollout["progress"] / denom,
            "reward_total": rollout["reward_total"] / denom,
            "reward_model_raw": rollout["reward_model_raw"] /
                max(rollout["reward_model_ticks"], 1),
            "reward_model_tick_fraction": rollout["reward_model_ticks"] / denom,
            "action_linear_mean": rollout["action_linear"] / denom,
            "action_abs_angular_mean": rollout["action_abs_angular"] / denom,
            **{
                key: rollout[key] / denom
                for key in (
                    "reward_learned", "reward_progress_task", "reward_goal_task",
                    "reward_collision_task", "reward_timeout_task", "reward_stuck_task",
                    "reward_time_task", "reward_environment",
                )
            },
            **stats,
            **curriculum.metrics(),
        }
        _append_metrics(metrics_path, record)
        logger.log(
            step=robot_steps,
            mean_return=record["episode_return"],
            mean_length=record["episode_length"],
            goal_distance=record["goal_distance"],
            progress=record["progress"],
            reached=record["reached"],
            collision=record["collision"],
            timeout=record["timeout"],
            stuck=record["stuck"],
            reward_total=record["reward_total"],
            policy_loss=stats["policy_loss"],
            value_loss=stats["value_loss"],
            entropy=stats["entropy"],
            kl_loss=stats.get("kl_loss", 0.0),
            **{
                key: value for key, value in record.items()
                if key.startswith("reward_") and key != "reward_total"
            },
            **curriculum.metrics(),
        )
        print(
            f"[CrowdSim][RM-PPO] steps={robot_steps} "
            f"return={record['episode_return']:.3f} success={record['success_rate']:.1%} "
            f"collision={record['collision_rate']:.1%} progress={record['progress']:.4f} "
            f"reward={record['reward_total']:.4f} rm={record['reward_model_raw']:.4f} "
            f"p_loss={stats['policy_loss']:.4f} v_loss={stats['value_loss']:.4f} "
            f"kl={stats.get('kl_loss', 0.0):.5f}"
        )

        if next_save is not None and robot_steps >= next_save:
            _save_policy(
                ppo, output_dir / "robot_ppo_latest.pt", robot_steps,
                nav_manager=nav_manager, base_ppo=base_ppo,
                reward_checkpoint=reward_checkpoint, reward_config=reward_config,
                reward_stride=reward_stride,
            )
            _save_policy(
                ppo, output_dir / f"robot_ppo_rm_{robot_steps}.pt", robot_steps,
                nav_manager=nav_manager, base_ppo=base_ppo,
                reward_checkpoint=reward_checkpoint, reward_config=reward_config,
                reward_stride=reward_stride,
            )
            _save_training_gif(nav_manager, output_dir)
            next_save += save_interval

    _save_policy(
        ppo, output_dir / "robot_ppo_latest.pt", robot_steps,
        nav_manager=nav_manager, base_ppo=base_ppo,
        reward_checkpoint=reward_checkpoint, reward_config=reward_config,
        reward_stride=reward_stride,
    )
    gif_path = _save_training_gif(nav_manager, output_dir)
    total_episodes = sum(cumulative.values())
    summary = {
        "steps": robot_steps,
        "base_ppo": str(base_ppo),
        "reward_checkpoint": str(reward_checkpoint),
        "checkpoint": str(output_dir / "robot_ppo_latest.pt"),
        "metrics": str(metrics_path),
        "gif": None if gif_path is None else str(gif_path),
        "outcomes": cumulative,
        "episodes": total_episodes,
        "success_rate": cumulative["reached"] / max(total_episodes, 1),
        "collision_rate": cumulative["collision"] / max(total_episodes, 1),
        "timeout_rate": cumulative["timeout"] / max(total_episodes, 1),
        "stuck_rate": cumulative["stuck"] / max(total_episodes, 1),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.close()
    print(f"[CrowdSim][RM-PPO] Complete: {output_dir}")
    import omni.kit.app
    omni.kit.app.get_app().post_quit()


def main() -> None:
    args = parse_args()
    config_path = cfg_path(args.env_config)
    training_config_path = cfg_path(args.train_config)
    config = load_config(config_path)
    training_config = load_config(training_config_path)
    rm_training = get_config_section(training_config, "reward_model_ppo")
    if not rm_training:
        raise ValueError(
            f"Training config {training_config_path} lacks reward_model_ppo settings"
        )
    network = get_config_section(config, "network")
    base_output = Path(str(_cfg(
        args.output_dir, rm_training, "output_dir", "output/crowdsim_reward_model_ppo"
    )))
    output_dir = make_training_output_dir(base_output)
    copy_training_configs(output_dir, config_path)
    shutil.copy2(
        training_config_path,
        output_dir / "config" / "reward_model_ppo.yaml",
    )
    recording = config.setdefault("navigation", {}).setdefault("recording", {})
    recording["enabled"] = True
    recording["output_dir"] = str(output_dir / "navigation")

    from CrowdSim.world.builder import build_env
    _, _, nav_manager, runtime = build_env(
        config, num_envs=args.num_envs, headless=args.headless,
        scene_physics=args.scene_physics,
    )
    device = runtime.fabric.device
    depth_enabled = nav_manager.config.rl_depth_enabled
    depth_size = nav_manager.config.rl_depth_size if depth_enabled else 0
    map_enabled = nav_manager.config.rl_map_size > 0
    map_size = nav_manager.config.rl_map_size if map_enabled else 0
    learning_rate = float(_cfg(args.lr, rm_training, "lr", 3.0e-5))
    ppo_config = RobotPPOConfig(
        obs_dim=nav_manager.robot_rl_obs_dim,
        vector_obs_dim=nav_manager.robot_rl_vector_obs_dim,
        **robot_network_kwargs(
            network, hidden_dim_override=args.hidden_dim,
            num_layers_override=args.num_layers, depth_enabled=depth_enabled,
            depth_size=depth_size, map_enabled=map_enabled, map_size=map_size,
            num_neighbors=nav_manager.config.rl_num_neighbors,
        ),
        lr=learning_rate,
        gamma=float(rm_training.get("gamma", 0.99)),
        gae_lambda=float(rm_training.get("gae_lambda", 0.95)),
        clip_ratio=float(rm_training.get("clip_ratio", 0.2)),
        value_coef=float(rm_training.get("value_coef", 0.5)),
        entropy_coef=float(rm_training.get("entropy_coef", 0.005)),
        max_grad_norm=float(rm_training.get("max_grad_norm", 1.0)),
        ppo_epochs=int(_cfg(args.ppo_epochs, rm_training, "ppo_epochs", 4)),
        minibatch_size=int(_cfg(args.minibatch_size, rm_training, "minibatch_size", 256)),
    )
    ppo = RobotPPOTrainer(ppo_config, device)
    base_ppo = Path(args.base_ppo).expanduser().resolve()
    reward_checkpoint = Path(args.reward_ckpt).expanduser().resolve()
    ppo.load(base_ppo)
    nav_manager.goal_curriculum.load_state_dict(ppo.extra_state.get("goal_curriculum"))
    reference_model = copy.deepcopy(ppo.model).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    start_step = 0
    if args.resume:
        start_step = ppo.load(Path(args.resume).expanduser().resolve())
        nav_manager.goal_curriculum.load_state_dict(ppo.extra_state.get("goal_curriculum"))
    _set_optimizer_lr(ppo, learning_rate)

    reward_model, reward_payload = load_offline_reward_model(
        reward_checkpoint, device=device
    )
    if reward_model.obs_dim != nav_manager.robot_rl_obs_dim:
        raise ValueError(
            f"Reward model obs_dim={reward_model.obs_dim}, environment "
            f"obs_dim={nav_manager.robot_rl_obs_dim}"
        )
    if reward_model.action_dim != 2:
        raise ValueError(f"Reward model action_dim must be 2, got {reward_model.action_dim}")
    metadata = reward_payload.get("dataset_metadata", {})
    reward_depth_size = int(metadata.get("depth_size", 0))
    if reward_depth_size > 0 and not depth_enabled:
        raise ValueError("Reward model requires depth but environment depth is disabled")
    sequence_steps, reward_stride, reward_hz = reward_model_timing(
        reward_payload, runtime_hz=float(nav_manager.config.update_hz),
        source_hz_override=args.reward_source_hz,
        stride_override=args.reward_runtime_stride,
    )
    min_history = int(_cfg(
        args.min_history_steps, rm_training, "min_history_steps", 4
    ))
    reward_history = OnlineRewardHistory(
        reward_model, num_envs=nav_manager.config.num_robots,
        sequence_steps=sequence_steps, depth_size=reward_depth_size,
        min_history_steps=min_history, device=device,
    )
    reward_config = _reward_config(args, rm_training)
    rollout_steps = int(_cfg(args.rollout_steps, rm_training, "rollout_steps", 256))
    buffer = RobotRolloutBuffer(
        rollout_steps, nav_manager.config.num_robots, nav_manager.robot_rl_obs_dim, 2,
        device, depth_size=depth_size, map_size=map_size,
        num_neighbors=nav_manager.config.rl_num_neighbors,
        neighbor_dim=nav_manager.robot_rl_neighbor_dim,
    )
    total_steps = int(_cfg(args.total_steps, rm_training, "total_steps", 500_000))
    save_interval = int(_cfg(args.save_interval, rm_training, "save_interval", 20_000))
    kl_beta = float(_cfg(args.kl_beta, rm_training, "kl_beta", 0.02))
    run_config = {
        **_flatten_config(config, prefix="env"),
        **_flatten_config(rm_training, prefix="reward_model_ppo"),
        "config_files/env_yaml": str(config_path),
        "config_files/reward_model_ppo_yaml": str(training_config_path),
        "checkpoints/base_ppo": str(base_ppo),
        "checkpoints/reward_model": str(reward_checkpoint),
        "checkpoints/resume": str(args.resume or "none"),
        "reward/model_hz": reward_hz,
        "reward/runtime_stride": reward_stride,
        "reward/sequence_steps": sequence_steps,
        "reward/min_history_steps": min_history,
        "reward/kl_beta": kl_beta,
        **{f"reward/{key}": value for key, value in reward_config.to_dict().items()},
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(
        f"[CrowdSim][RM-PPO] base={base_ppo} reward={reward_checkpoint} "
        f"reward_history={sequence_steps}@{reward_hz:g}Hz output={output_dir}"
    )
    train_loop(
        runtime=runtime, nav_manager=nav_manager, ppo=ppo,
        reference_model=reference_model, buffer=buffer,
        reward_history=reward_history, reward_config=reward_config,
        reward_stride=reward_stride, kl_beta=kl_beta, total_steps=total_steps,
        start_step=start_step, save_interval=save_interval, output_dir=output_dir,
        base_ppo=base_ppo, reward_checkpoint=reward_checkpoint,
        use_wandb=not args.no_wandb, wandb_config=run_config,
    )


if __name__ == "__main__":
    main()
