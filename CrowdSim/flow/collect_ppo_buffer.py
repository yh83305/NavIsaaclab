"""Collect raw per-step rollout data from a frozen PPO policy.

Saves one timestamped file per collection run:
    output/collect_buffer/ppo/raw_YYYYMMDD_HHMMSS.pt

Buffer schema (per-episode, no interleaving)::

    {
        "episodes": [
            {"obs": (T_i, obs_dim), "depth": (T_i, 224, 224),
             "rgb": (T_i, 3, 224, 224) | None, "action": (T_i, 2)},
            ...
        ],
        "meta": {
            "num_episodes":    N,
            "total_steps":     sum(T_i),
            "obs_dim":         obs_dim,
            "depth_size":      224,
            "ppo_ckpt":        "...",
            "num_envs":        K,
            "min_episode_len": discarded_count,
        }
    }

An optional on-disk checkpoint (see ``--checkpoint-every`` / yaml
``collection.checkpoint_every``) atomically flushes completed episodes into
append-only shards every N episodes so RGB data does not accumulate in RAM.
Existing shards are never rewritten, so a full disk cannot corrupt earlier
recovery data. ``checkpoint_every <= 0`` disables checkpointing.

Usage
-----
    python CrowdSim/flow/collect_ppo_buffer.py \\
        --ppo-ckpt  output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt \\
        --headless \\
        --collect-steps 50000 \\
        --output-dir output/collect_buffer/ppo
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys, time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402
AppLauncher = import_simulator_before_torch("isaaclab")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_float32_matmul_precision("high")

from CrowdSim.ppo.ppo_policy import (  # noqa: E402
    RobotPPOConfig,
    RobotPPOTrainer,
    bounded_robot_action,
    robot_network_kwargs,
)
from CrowdSim.utils.config_loader import cfg_path as _cfg_path, load_config  # noqa: E402
from CrowdSim.world.builder import build_env  # noqa: E402
from CrowdSim.tools import panel_render  # noqa: E402


def cfg_path(p: str) -> Path:
    return _cfg_path(p, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────
# Checkpoint helpers — flush / merge episode shards
# ─────────────────────────────────────────────────────────────────

_CKPT_FILENAME = "ckpt_latest.pt"  # legacy single-file recovery checkpoint
_CKPT_PREFIX = "ckpt_"


def _payload_nbytes(value) -> int:
    """Conservative payload-size estimate used for preflight disk checks."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_payload_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_nbytes(item) for item in value)
    return 0


def _atomic_torch_save(payload, path: Path) -> None:
    """Save via a sibling temporary file without risking an existing shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    estimated = max(int(_payload_nbytes(payload) * 1.10), 64 * 1024**2)
    free = shutil.disk_usage(path.parent).free
    if free < estimated:
        raise RuntimeError(
            "Insufficient disk space for PPO buffer checkpoint: "
            f"need approximately {estimated / 1024**3:.2f} GiB, "
            f"available {free / 1024**3:.2f} GiB in {path.parent}. "
            "Existing checkpoint shards remain recoverable."
        )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"Failed to write PPO buffer checkpoint {path}. Existing shards "
            "were not overwritten; check disk space/quota and resume recovery."
        ) from error


def _flush_checkpoint(checkpoint_dir: Path, new_episodes: list[dict]) -> None:
    """Atomically append one recovery shard without rewriting prior data."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = [
        int(path.stem.removeprefix(_CKPT_PREFIX))
        for path in checkpoint_dir.glob(f"{_CKPT_PREFIX}[0-9]*.pt")
        if path.stem.removeprefix(_CKPT_PREFIX).isdigit()
    ]
    shard_id = max(existing_ids, default=0) + 1
    path = checkpoint_dir / f"{_CKPT_PREFIX}{shard_id:06d}.pt"
    _atomic_torch_save(new_episodes, path)
    print(
        f"[PPO] Checkpoint shard saved → {path.name}  "
        f"({len(new_episodes)} eps, {path.stat().st_size / 1024**2:.1f} MB)"
    )


def _load_latest_checkpoint(checkpoint_dir: Path) -> list[dict]:
    """Load legacy recovery data followed by every append-only shard."""
    episodes: list[dict] = []
    legacy = checkpoint_dir / _CKPT_FILENAME
    if legacy.exists():
        episodes.extend(torch.load(legacy, weights_only=False))
    for path in sorted(checkpoint_dir.glob(f"{_CKPT_PREFIX}[0-9]*.pt")):
        episodes.extend(torch.load(path, weights_only=False))
    return episodes


# ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect raw PPO rollout buffer for flow policy training.\n"
                    "Runtime params are read from the yaml; speed and output "
                    "directory may be overridden for targeted collection runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env-config", default="CrowdSim/config/collect/ppo.yaml",
                   help="Path to collection yaml (defines everything except ppo-ckpt).")
    p.add_argument("--ppo-ckpt", required=True,
                   help="Frozen PPO checkpoint to roll out.")
    p.add_argument(
        "--max-linear-velocity",
        type=float,
        default=None,
        help="Override navigation.rl.max_linear_velocity in m/s.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override collection.output_dir; use a separate directory per speed.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# GIF helpers (same pattern as crowd_sim.py / collect_cbf_buffer.py)
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
# GIF rendering is unified in CrowdSim.tools.panel_render (bird's-eye +
# RGB + depth + obs + action, per episode, every frame).  Real-time
# collection triggers one full-panel GIF every ``gif_every_episodes``
# completed episodes (see the episode-finalize block in collect()).
# ─────────────────────────────────────────────────────────────────


def collect(
    config: dict,
    ppo_ckpt: str | None,
    num_envs: int,
    headless: bool,
    collect_steps: int,
    min_episode_len: int,
    only_success: bool,
    device: str,
    wandb_run=None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 5,
    gif_out: "str | None" = None,
    gif_fps: float = 7.5,
    gif_every_episodes: int = 50,
    action_provider=None,
    algorithm: str = "ppo",
    repeat_robot_routes: bool = False,
    target_episodes: int = 0,
    route_repeats: int = 0,
    action_transform=None,
) -> dict:
    """Run PPO rollout and return per-episode data.

    PPO collection parallelism = number of active robots (one robot per env
    collects a trajectory).  ``num_envs`` (from --num-envs or inferred) sets
    the robot count; num_humanoids stays from yaml for crowd density.
    num_envs = max(num_robots, num_humanoids) so the crowd fits; extra envs
    beyond num_robots hold frozen humanoid slots.  All per-robot iteration
    uses num_robots (not num_envs).

    When *checkpoint_dir* is set, completed episodes are flushed to that
    directory every *checkpoint_every* episodes (mirrors the CBF collection
    script).  The final return value merges the on-disk shard with any
    remaining in-memory episodes.
    """
    # num_robots/num_humanoids come from the collection config; --num-envs is inferred.
    # PPO rollout action selection follows navigation.rl.deterministic, just
    # like crowd_sim.py evaluation.  Flow imitation buffers normally want the
    # mean action so policy sampling noise is not baked into the dataset.
    rl_cfg = config.get("navigation", {}).get("rl", {})
    deterministic = bool(rl_cfg.get("deterministic", True))
    result = build_env(config, num_envs=num_envs, headless=headless)
    env, agent, nav_manager, runtime = result

    # PPO collection iterates over active robots, one per env slot.
    num_envs = env.num_envs
    n_robots = nav_manager.config.num_robots

    ppo = None
    if action_provider is None:
        if not ppo_ckpt:
            raise ValueError("ppo_ckpt is required without an action_provider")
        network_cfg = config.get("network", {})
        map_size = nav_manager.config.rl_map_size
        ppo_cfg = RobotPPOConfig(
            obs_dim=nav_manager.robot_rl_obs_dim,
            vector_obs_dim=nav_manager.robot_rl_vector_obs_dim,
            **robot_network_kwargs(
                network_cfg,
                depth_enabled=nav_manager.config.rl_depth_enabled,
                depth_size=nav_manager.config.rl_depth_size,
                map_enabled=map_size > 0,
                map_size=map_size,
                num_neighbors=nav_manager.config.rl_num_neighbors,
            ),
        )
        ppo = RobotPPOTrainer(ppo_cfg, device)
        ppo.load(Path(ppo_ckpt).expanduser().resolve())
        action_mode = "deterministic mean" if deterministic else "stochastic sample"
        print(f"[Collect] PPO loaded from {ppo_ckpt}  action_mode={action_mode}")
    else:
        action_provider.initialize(nav_manager)
        print(f"[Collect] {algorithm.upper()} action provider initialized")
    if action_transform is not None and hasattr(action_transform, "initialize"):
        action_transform.initialize(nav_manager)

    # Per-robot current episode accumulator (one per ACTIVE robot, not per env)
    # Each entry: {"obs": [], "depth": [], "action": [], "world": {}}
    live_episodes: list[list[dict]] = [[] for _ in range(n_robots)]
    completed_episodes: list[dict] = []

    done_indices   = None
    total_episodes = 0
    route_episode_counts = np.zeros(n_robots, dtype=np.int64)
    total_goals    = 0
    discarded      = 0

    n_h = nav_manager.config.num_humanoids
    roff = n_h   # robot slots start after humanoid slots in the positions array

    # ── Per-episode GIF rendering (unified panel: bird's-eye + RGB + depth +
    # obs + action, every frame).  Built once; bird's-eye background loads once.
    # Triggered every ``gif_every_episodes`` completed episodes — renders the
    # just-finished episode (any robot), saves + uploads to wandb.
    gif_path = Path(gif_out) if gif_out else None
    record_gif = gif_path is not None
    birdseye = None
    if record_gif:
        birdseye = panel_render.BirdseyeRenderer(
            map_path=str(getattr(nav_manager.config, "map_path", "") or ""),
            origin=getattr(nav_manager.config, "map_origin_xy", (0.0, 0.0)),
            resolution=float(getattr(nav_manager.config, "map_resolution", 0.05)),
            tile_px=320,
            robot_radius=float(getattr(nav_manager.config, "agent_radius", 0.3)),
            goal_tolerance=float(getattr(nav_manager.config, "goal_tolerance", 0.75)),
        )
        if not birdseye.has_map:
            print(f"[Collect] GIF: occupancy map not found — bird's-eye panel "
                  f"will show a placeholder.")
    completed_episode_count = 0

    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    print(f"[Collect] Rolling out {collect_steps} steps "
          f"({n_robots} robots × {collect_steps} = up to {collect_steps * n_robots} samples, "
          f"{num_envs} envs) …")

    camera_enabled = bool(
        nav_manager.config.rl_depth_enabled
        or nav_manager.config.rl_rgb_enabled
    )

    def render_camera() -> None:
        """Publish current simulation transforms to the camera render product."""
        if not camera_enabled:
            return
        simulator = getattr(env, "simulator", None)
        sim_context = getattr(simulator, "_sim", None)
        if sim_context is None:
            raise RuntimeError(
                "RGB/depth collection requires an Isaac Sim rendering context."
            )
        sim_context.render()

    progress_total = target_episodes if target_episodes > 0 else collect_steps
    step_iter = tqdm(
        total=progress_total,
        desc="collect",
        unit="episode" if target_episodes > 0 else "step",
        dynamic_ncols=True,
        disable=not _tqdm_available,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| "
            "episode：{n_fmt}/{total_fmt} "
            "{postfix} [{elapsed}<{remaining}]"
            if target_episodes > 0 else None
        ),
    )

    robot_reset_since_render = False
    progress_episodes = 0
    t0 = time.time()
    for step in range(collect_steps):
        humanoid_reset = step == 0 or (
            done_indices is not None and len(done_indices) > 0
        )
        env_obs, _ = env.reset(done_indices)

        # RGB/depth are pre-step observations.  A reset writes new root poses
        # directly to the simulator, but CameraSensor still exposes the last
        # render product until SimulationContext.render() runs.  Render only
        # when the scene changed since the regular post-step render below.
        if humanoid_reset or robot_reset_since_render:
            render_camera()
        robot_reset_since_render = False

        env_obs    = agent.add_agent_info_to_obs(env_obs)
        obs_td     = agent.obs_dict_to_tensordict(env_obs)

        with torch.no_grad():
            model_outs      = agent.model(obs_td)
            humanoid_action = model_outs.get("mean_action", model_outs["action"])
            # nav_manager.get_robot_rl_observations returns (obs, depth, map);
            # RGB is cached on _robot_last_rgb (filled inside that call) and
            # read here for the buffer only — PPO never consumes rgb.
            robot_obs, robot_neighbors, neighbor_mask, robot_depth, robot_map = nav_manager.get_robot_rl_observations()
            robot_rgb = getattr(nav_manager, "_robot_last_rgb", None)

        # World-frame state synced with the pre-step observation (same instant
        # as obs/depth/rgb) for the bird's-eye panel.  One CPU sync per step.
        world_positions, world_velocities = nav_manager._read_agent_state()
        world_yaws = nav_manager.robot_yaws()
        world_goals = np.asarray(nav_manager.goals_xy)
        all_robot_xys = np.asarray(world_positions[roff:roff + n_robots], dtype=np.float32)
        all_robot_yaws = np.asarray(world_yaws[:n_robots], dtype=np.float32)
        hum_xys_world = np.asarray(world_positions[:n_h], dtype=np.float32)

        # Route-guided collection must replace the goal portion before PPO
        # inference.  Post-hoc action steering alone leaves PPO conditioned on
        # the final goal and makes the rendered A* waypoint purely cosmetic.
        ppo_robot_obs = robot_obs
        if action_transform is not None and hasattr(action_transform, "transform_observations"):
            ppo_robot_obs = action_transform.transform_observations(
                robot_obs, world_positions, all_robot_yaws, nav_manager
            )

        action_modes = None
        with torch.no_grad():
            if action_provider is None:
                if deterministic:
                    mean, _, _ = ppo.model(
                        ppo_robot_obs, robot_depth, robot_map,
                        robot_neighbors, neighbor_mask,
                    )
                    action = bounded_robot_action(mean)
                else:
                    action, *_ = ppo.act(
                        ppo_robot_obs, robot_depth, robot_map,
                        robot_neighbors, neighbor_mask,
                    )
            else:
                action, action_modes = action_provider.act(
                    positions=world_positions,
                    velocities=world_velocities,
                    robot_yaws=all_robot_yaws,
                    goals=world_goals[roff:roff + n_robots],
                    nav_manager=nav_manager,
                )
            if action_transform is not None:
                action = action_transform(
                    action, world_positions, all_robot_yaws,
                    world_goals[roff:roff + n_robots], nav_manager,
                )

        nav_manager.set_robot_rl_actions(action)
        _, _, dones, _, _ = env.step(humanoid_action)
        # _physics_step() uses SimulationContext.step(render=False) in headless
        # mode.  Render now so the next loop observes this post-step scene.
        render_camera()
        _, _, robot_done, info, _, _, _, _ = nav_manager.get_robot_rl_feedback()
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

        # 224² depth is stored as float16 to keep raw shards tractable; model
        # preprocessing restores float32 before the ViT.
        dep_cpu = (
            robot_depth.to(device="cpu", dtype=torch.float16)
            if robot_depth is not None else None
        )
        map_cpu = robot_map.cpu() if robot_map is not None else None
        # _read_camera_rgb already returns uint8 (N,3,S,S) [0,255]; just move
        # to CPU.  (No policy consumes rgb here — it is for the buffer / CFM.)
        rgb_cpu = robot_rgb.cpu() if robot_rgb is not None else None

        # ── Per-env: accumulate step, finalise on episode end ──
        robot_done_np = (
            robot_done.detach().cpu().numpy()
            if hasattr(robot_done, "detach")
            else np.asarray(robot_done)
        )
        # Transfer to CPU once per step (not per-agent).
        # Read actions from nav_manager._robot_rl_actions which has already
        # been clipped to [0,1] / [-1,1] by set_robot_rl_actions(), matching
        # drive.py execution exactly.  Using the raw PPO output (action.cpu())
        # would include v_lin < 0 values that the robot never actually executed.
        robot_obs_cpu = robot_obs.cpu()
        action_cpu    = torch.as_tensor(
            nav_manager._robot_rl_actions.copy(), dtype=torch.float32
        )

        for i in range(n_robots):
            step_payload = {
                "obs":    robot_obs_cpu[i],
                "neighbors": robot_neighbors[i].cpu(),
                "neighbor_mask": neighbor_mask[i].cpu(),
                "depth":  dep_cpu[i] if dep_cpu is not None else None,
                "map":    map_cpu[i] if map_cpu is not None else None,
                "rgb":    rgb_cpu[i] if rgb_cpu is not None else None,
                "action": action_cpu[i],
                # World state synced with obs (pre-step).  Per-robot: focal = i,
                # all_robot_* carry every active robot for the multi-robot
                # bird's-eye (focal highlighted with trail/barrier/goal).
                "world": {
                    "robot_xy":       all_robot_xys[i].copy(),
                    "yaw":            np.float32(all_robot_yaws[i]),
                    "goal_xy":        np.asarray(world_goals[roff + i], dtype=np.float32),
                    "hum_xys":        hum_xys_world.copy(),
                    "all_robot_xys":  all_robot_xys.copy(),
                    "all_robot_yaws": all_robot_yaws.copy(),
                    "focal_idx":      i,
                },
            }
            if action_transform is not None and hasattr(action_transform, "route_sparse_path"):
                step_payload["world"]["route_sparse_path"] = np.asarray(
                    action_transform.route_sparse_path(i), dtype=np.float32
                )
            if action_transform is not None and hasattr(action_transform, "route_sparse_target"):
                step_payload["world"]["route_sparse_target"] = np.asarray(
                    action_transform.route_sparse_target(i), dtype=np.float32
                )
            if action_modes is not None:
                step_payload["mode"] = torch.as_tensor(
                    int(action_modes[i]), dtype=torch.int8
                )
            if action_transform is not None and hasattr(action_transform, "route_mode"):
                step_payload["route_mode"] = torch.as_tensor(
                    action_transform.route_mode(i), dtype=torch.int8
                )
            if action_transform is not None and hasattr(action_transform, "route_robot_id"):
                step_payload["route_robot_id"] = torch.as_tensor(
                    action_transform.route_robot_id(i), dtype=torch.int16
                )
            live_episodes[i].append(step_payload)

        # ── Finalize done episodes ───────────────────────────────
        n_done_this_step = int(robot_done_np.sum())
        reached_np = info.get("reached", None) if isinstance(info, dict) else None
        for i in range(n_robots):
            if robot_done_np[i]:
                # Filter: only keep successful episodes (reached goal)
                is_success = reached_np is not None and bool(reached_np[i])
                if only_success and not is_success:
                    discarded += 1
                    live_episodes[i] = []
                    total_episodes += 1
                    route_episode_counts[i] += 1
                    continue

                steps = live_episodes[i]
                if len(steps) >= min_episode_len:
                    ep_obs = torch.stack([s["obs"] for s in steps])
                    ep_act = torch.stack([s["action"] for s in steps])
                    ep_dep = torch.stack([s["depth"] for s in steps]) if dep_cpu is not None else None
                    ep_map = torch.stack([s["map"] for s in steps]) if map_cpu is not None and steps[0].get("map") is not None else None
                    ep_rgb = torch.stack([s["rgb"] for s in steps]) if rgb_cpu is not None and steps[0].get("rgb") is not None else None
                    episode_dict = {
                        "obs":    ep_obs,
                        "neighbors": torch.stack([s["neighbors"] for s in steps]),
                        "neighbor_mask": torch.stack([s["neighbor_mask"] for s in steps]),
                        "depth":  ep_dep,
                        "map":    ep_map,
                        "rgb":    ep_rgb,
                        "action": ep_act,
                        "world":  panel_render.stack_world(steps),
                    }
                    if "mode" in steps[0]:
                        episode_dict["mode"] = torch.stack(
                            [s["mode"] for s in steps]
                        )
                    if "route_mode" in steps[0]:
                        episode_dict["route_mode"] = torch.stack(
                            [s["route_mode"] for s in steps]
                        )
                    if "route_robot_id" in steps[0]:
                        episode_dict["route_robot_id"] = steps[0]["route_robot_id"]
                    completed_episodes.append(episode_dict)
                    if action_transform is not None and hasattr(action_transform, "on_episode_saved"):
                        action_transform.on_episode_saved(i)
                    # ── Per-episode full-panel GIF (every N completed episodes) ─
                    # Renders the just-finished episode (any robot) with the
                    # unified layout + uploads to wandb.
                    completed_episode_count += 1
                    if record_gif and completed_episode_count % gif_every_episodes == 0:
                        panel_render.render_episode_gif(
                            episode_dict, birdseye, gif_path, gif_fps,
                            ep_idx=total_episodes, wandb_run=wandb_run,
                            wandb_step=step + 1,
                        )
                else:
                    discarded += 1
                live_episodes[i] = []
                total_episodes += 1
                route_episode_counts[i] += 1

        if action_provider is not None and robot_done_np.any():
            action_provider.reset(np.flatnonzero(robot_done_np))
        if route_repeats > 0:
            repeat_mask = (route_episode_counts % route_repeats) != 0
        else:
            repeat_mask = (
                np.ones(n_robots, dtype=bool) if repeat_robot_routes else None
            )
        nav_manager.reset_robot_rl_episodes(robot_done, repeat_mask=repeat_mask)
        if action_transform is not None and bool(robot_done_np.any()):
            if hasattr(action_transform, "reset"):
                action_transform.reset(np.flatnonzero(robot_done_np), nav_manager)
            elif hasattr(action_transform, "initialize"):
                action_transform.initialize(nav_manager)
        # reset_robot_rl_episodes() teleports robots after the post-step render.
        # Defer its extra render until after env.reset() at the top of the next
        # loop, so humanoid and robot resets are captured in one fresh frame.
        robot_reset_since_render = bool(robot_done_np.any())

        if target_episodes > 0 and completed_episode_count > progress_episodes:
            step_iter.update(completed_episode_count - progress_episodes)
            progress_episodes = completed_episode_count
        elif target_episodes <= 0:
            step_iter.update(1)

        if target_episodes > 0 and completed_episode_count >= target_episodes:
            print(
                f"[Collect] Reached target_episodes={target_episodes}; "
                "stopping rollout."
            )
            break
        if action_transform is not None and hasattr(action_transform, "collection_complete"):
            if action_transform.collection_complete():
                print("[Collect] Action transform completed its balanced collection target.")
                break

        # ── Checkpoint: flush completed episodes to disk ──────────
        # RGB data is ~150 KB/frame; with many episodes in memory this
        # quickly exhausts RAM + GPU VRAM. Flush an append-only atomic shard
        # every `checkpoint_every` episodes, then free the in-memory tensors.
        if (checkpoint_dir is not None
                and len(completed_episodes) >= checkpoint_every):
            # Move everything off the GPU before the flush frees memory.
            for _ep in completed_episodes:
                for _t in _ep.values():
                    if isinstance(_t, torch.Tensor):
                        _t.cpu()
            _flush_checkpoint(checkpoint_dir, completed_episodes)
            completed_episodes.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Goal counting: read ONCE per step, not per-agent ─────
        if n_done_this_step > 0 and reached_np is not None:
            total_goals += int(reached_np.sum())

        if _tqdm_available:
            samples_so_far = (step + 1) * n_robots
            sr = (total_goals / total_episodes * 100) if total_episodes > 0 else 0.0
            step_iter.set_postfix(
                samples=f"{samples_so_far:,}",
                episodes=total_episodes,
                pairs=len(completed_episodes),
                goal_rate=f"{sr:.1f}%",
                step=f"{step + 1:,}/{collect_steps:,}",
                step_rate=f"{(step + 1) / max(time.time() - t0, 1e-6):.2f}step/s",
                refresh=False,
            )
        elif (step + 1) % 10_000 == 0:
            sr = (total_goals / total_episodes * 100) if total_episodes > 0 else 0.0
            print(f"  step {step+1}/{collect_steps}  "
                  f"episodes={total_episodes}  pairs={len(completed_episodes)}  "
                  f"goal_rate={sr:.1f}%")

        # ── Wandb logging ────────────────────────────────────────
        if wandb_run is not None and (step + 1) % 100 == 0:
            wandb_run.log({
                "collect/samples":    (step + 1) * n_robots,
                "collect/episodes":   total_episodes,
                "collect/pairs":      len(completed_episodes),
                "collect/goal_rate":  (total_goals / total_episodes * 100) if total_episodes > 0 else 0.0,
            }, step=step + 1)

    # ── Finalize any remaining live episodes ────────────────────
    for i in range(n_robots):
        # A live tail did not terminate successfully.  Do not let it bypass
        # only_success merely because collection stopped mid-episode.
        if len(live_episodes[i]) >= min_episode_len and not only_success:
            steps = live_episodes[i]
            episode_dict = {
                "obs":    torch.stack([s["obs"] for s in steps]),
                "neighbors": torch.stack([s["neighbors"] for s in steps]),
                "neighbor_mask": torch.stack([s["neighbor_mask"] for s in steps]),
                "depth":  torch.stack([s["depth"] for s in steps]) if dep_cpu is not None else None,
                "map":    torch.stack([s["map"] for s in steps]) if map_cpu is not None and steps[0].get("map") is not None else None,
                "rgb":    torch.stack([s["rgb"] for s in steps]) if rgb_cpu is not None and steps[0].get("rgb") is not None else None,
                "action": torch.stack([s["action"] for s in steps]),
                "world":  panel_render.stack_world(steps),
            }
            if "mode" in steps[0]:
                episode_dict["mode"] = torch.stack(
                    [s["mode"] for s in steps]
                )
            completed_episodes.append(episode_dict)
        elif len(live_episodes[i]) > 0:
            discarded += 1

    # ── Merge with checkpoint shard (if any) + clean up ─────
    # Load every append-only recovery shard and append the in-memory tail.
    # Cleanup happens only after the final raw buffer is saved successfully.
    if checkpoint_dir is not None:
        merged_episodes = _load_latest_checkpoint(checkpoint_dir)
        merged_episodes.extend(completed_episodes)
    else:
        merged_episodes = completed_episodes

    total_steps = sum(ep["obs"].shape[0] for ep in merged_episodes)
    final_goal_rate = (total_goals / total_episodes * 100) if total_episodes > 0 else 0.0
    print(f"[Collect] Done: {len(merged_episodes)} episodes, "
          f"{total_steps:,} steps, {discarded} discarded (too short), "
          f"goal_rate={final_goal_rate:.1f}%")

    # (Per-episode GIFs were emitted during the rollout — nothing to render
    # at the end.  The merged buffer carries the ``world`` field so
    # visualize_flow_dataset.py can render any episode offline.)

    if wandb_run is not None:
        wandb_run.summary.update({
            "final/episodes":    len(merged_episodes),
            "final/steps":       total_steps,
            "final/discarded":   discarded,
            "final/goal_rate":   final_goal_rate,
            "final/obs_dim":     merged_episodes[0]["obs"].shape[1] if merged_episodes else 0,
            "final/depth_size":  merged_episodes[0]["depth"].shape[-1] if merged_episodes else 0,
        })
        wandb_run.finish()

    return {
        "episodes": merged_episodes,
        "meta": {
            "algorithm":       str(algorithm),
            "num_episodes":    len(merged_episodes),
            "total_steps":     total_steps,
            "discarded":       discarded,
            "total_episodes":  total_episodes,
            "goal_rate":       final_goal_rate,
            "obs_dim":         merged_episodes[0]["obs"].shape[1] if merged_episodes else 0,
            "depth_size":      (merged_episodes[0]["depth"].shape[-1]
                                if merged_episodes and merged_episodes[0]["depth"] is not None else 0),
            "depth_max_range": float(nav_manager.config.rl_depth_max_range),
            "max_linear_velocity": float(nav_manager.config.rl_max_linear_velocity),
            "max_angular_velocity": float(nav_manager.config.rl_max_angular_velocity),
            "source_hz":       float(config["navigation"]["update_hz"]),
            "camera_mount_pos": list(config["sensors"]["camera"]["pos"]),
            "camera_horizontal_fov_deg": float(
                config["sensors"]["camera"]["horizontal_fov"]
            ),
            "camera_convention": str(
                config["sensors"]["camera"].get("convention", "ros")
            ),
            "ppo_ckpt":        str(ppo_ckpt) if ppo_ckpt else None,
            "deterministic":   deterministic,
            "num_envs":        num_envs,
            "min_episode_len": min_episode_len,
            "num_neighbors":   int(nav_manager.config.rl_num_neighbors),
            "neighbor_dim":    int(nav_manager.robot_rl_neighbor_dim),
            "schema_version":  2,
            "repeat_robot_routes": bool(repeat_robot_routes),
            "target_episodes": int(target_episodes),
            "route_repeats": int(route_repeats),
            # Bird's-eye visualization metadata (same schema as CBF buffer) so
            # visualize_flow_dataset.py can load the occupancy-map background
            # + convert world↔pixel without a live nav_manager.
            "map_path":        str(getattr(nav_manager.config, "map_path", "") or ""),
            "map_resolution":  float(getattr(nav_manager.config, "map_resolution", 0.05)),
            "map_origin_xy":   list(getattr(nav_manager.config, "map_origin_xy", (0.0, 0.0))),
            "num_humanoids":   int(nav_manager.config.num_humanoids),
            "num_robots":      int(nav_manager.config.num_robots),
            "robot_radius":    float(getattr(nav_manager.config, "agent_radius", 0.3)),
        },
    }


def main() -> None:
    args   = parse_args()
    config = load_config(cfg_path(args.env_config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.num_robots is not None:
        if args.num_robots <= 0:
            raise ValueError("--num-robots must be positive")
        config["navigation"]["num_robots"] = int(args.num_robots)
    if args.stochastic:
        config.setdefault("navigation", {}).setdefault("rl", {})["deterministic"] = False
    if args.disable_rgb:
        config["navigation"]["rl"]["rgb_enabled"] = False
    if args.no_gif:
        config.setdefault("gif", {})["out"] = None
    if args.fixed_route_episodes is not None:
        if args.fixed_route_episodes <= 0:
            raise ValueError("--fixed-route-episodes must be positive")
        config.setdefault("collection", {})["target_episodes"] = int(args.fixed_route_episodes)
        config["collection"]["repeat_robot_routes"] = True
    if args.route_repeats is not None:
        if args.route_repeats <= 0:
            raise ValueError("--route-repeats must be positive")
        config.setdefault("collection", {})["route_repeats"] = int(args.route_repeats)
    if args.route_count is not None:
        if args.route_count <= 0:
            raise ValueError("--route-count must be positive")
        config.setdefault("collection", {})["route_count"] = int(args.route_count)
        repeats = int(config["collection"].get("route_repeats", 1))
        config["collection"]["target_episodes"] = int(args.route_count) * repeats

    if args.max_linear_velocity is not None:
        if args.max_linear_velocity <= 0.0:
            raise ValueError("--max-linear-velocity must be positive")
        config["navigation"]["rl"]["max_linear_velocity"] = float(
            args.max_linear_velocity
        )
    if args.output_dir is not None:
        config.setdefault("collection", {})["output_dir"] = args.output_dir

    # ── All runtime params come from the yaml's collection section ──
    c = config.get("collection", {})

    output_dir      = c.get("output_dir", "output/collect_buffer/ppo")
    headless        = c.get("headless", True)
    collect_steps   = int(c.get("collect_steps", 50_000))
    min_episode_len = int(c.get("min_episode_len", 16))
    only_success    = c.get("only_success", False)
    ckpt_every      = int(c.get("checkpoint_every", 0))
    max_linear_velocity = float(config["navigation"]["rl"]["max_linear_velocity"])

    print(
        f"[Collect] max_linear_velocity={max_linear_velocity:.3f}m/s "
        f"output_dir={output_dir}"
    )

    buf_dir = Path(output_dir)
    buf_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = buf_dir / "checkpoints" if ckpt_every > 0 else None

    # ── Wandb ──────────────────────────────────────────────────────
    wandb_run = None
    cfg_wandb = config.get("wandb", {})
    if cfg_wandb.get("enabled", True):
        try:
            import wandb
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run = wandb.init(
                project=cfg_wandb.get("project", "crowdsim-flow-data"),
                name=f"collect_ppo_{ts}",
                dir=str(buf_dir / "wandb_collect"),
                config={
                    "ppo_ckpt":         args.ppo_ckpt,
                    "deterministic":    bool(config.get("navigation", {}).get("rl", {}).get("deterministic", True)),
                    "collect_steps":    collect_steps,
                    "min_episode_len":  min_episode_len,
                    "only_success":     only_success,
                    "headless":         headless,
                    "max_linear_velocity": max_linear_velocity,
                },
            )
        except Exception as error:
            print(
                "[Collect][WARN] W&B initialization failed; "
                f"continuing without W&B: {error}"
            )
            wandb_run = None

    # ── GIF params (top-level `gif:` section, CLI > yaml > default) ──
    # gif_out is a directory/base: per-episode GIFs are written alongside it
    # with an episode index suffix.  gif_every_episodes controls how often a
    # full-panel GIF is emitted during the rollout.
    cfg_gif = config.get("gif", {})
    gif_out  = cfg_gif.get("out", None)
    gif_fps  = float(cfg_gif.get("fps", 7.5))
    gif_every_episodes = int(cfg_gif.get("every_episodes", 50))
    # Auto-create a timestamped sub-folder so each run lands in its own folder.
    if gif_out is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = Path(gif_out)
        gif_out = str(p.parent / ts / p.name)

    t0 = time.time()
    data = collect(
        config          = config,
        ppo_ckpt        = args.ppo_ckpt,
        num_envs        = None,
        headless        = headless,
        collect_steps   = collect_steps,
        min_episode_len = min_episode_len,
        only_success    = only_success,
        device          = device,
        wandb_run       = wandb_run,
        checkpoint_dir  = checkpoint_dir,
        checkpoint_every= ckpt_every,
        gif_out         = gif_out,
        gif_fps         = gif_fps,
        gif_every_episodes = gif_every_episodes,
        repeat_robot_routes=repeat_robot_routes,
        target_episodes=target_episodes,
        route_repeats=route_repeats,
    )
    elapsed = time.time() - t0
    print(f"[Collect] Elapsed: {elapsed:.0f}s  ({elapsed/60:.1f} min)")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = buf_dir / f"raw_{ts}.pt"
    _atomic_torch_save(data, out_path)
    print(f"[Collect] Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Remove recovery data only after the final raw buffer is durable.
    if checkpoint_dir is not None:
        recovery_files = [checkpoint_dir / _CKPT_FILENAME]
        recovery_files.extend(
            checkpoint_dir.glob(f"{_CKPT_PREFIX}[0-9]*.pt")
        )
        for checkpoint_path in recovery_files:
            try:
                checkpoint_path.unlink()
            except FileNotFoundError:
                pass
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
