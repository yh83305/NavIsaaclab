"""Config-driven CrowdSim entry point."""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch("isaaclab")

import torch  # noqa: E402

from CrowdSim.utils.humanoid_state_recorder import (  # noqa: E402
    HumanoidStateRecorderConfig,
    configure_humanoid_state_recorder,
)
from CrowdSim.utils.map_metadata import OccupancyMapMetadata, load_occupancy_map_metadata  # noqa: E402
from CrowdSim.nav_manager import CrowdNavigationConfig, CrowdNavigationManager  # noqa: E402
from CrowdSim.control.drive import DifferentialDriveConfig  # noqa: E402
from CrowdSim.utils.sensor_stream import (  # noqa: E402
    RobotCameraStreamConfig,
    configure_robot_camera_recorder,
)
from CrowdSim.protomotions_runtime import (  # noqa: E402
    build_runtime,
    configure_viewer_camera,
    create_fabric,
    enable_human_mesh,
    make_crowd_robot_config,
    resolve_robot_usd,
    suppress_known_isaaclab_warning_spam,
)
from CrowdSim.scene_setup import (  # noqa: E402
    add_global_usd_reference,
    apply_fixed_crowd_robot_spawns,
    apply_fixed_spawn_offsets,
    parent_camera_to_robot,
    parse_spawn_xy,
    parse_spawn_xy_yaw,
    patch_isaaclab_scene_with_crowdsim_assets,
    resolve_repo_path,
    sample_spawn_xy_from_map,
    spawn_scene_objects,
)
from CrowdSim.utils.config_loader import cfg_path as _cfg_path, load_config  # noqa: E402


def cfg_path(path_like: str) -> Path:
    """Module-local shim: resolve path relative to this file's PROJECT_ROOT."""
    return _cfg_path(path_like, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CrowdSim from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--env-config", default="CrowdSim/config/env.yaml")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="IsaacLab parallel envs. Defaults to "
                             "max(num_humanoids, num_robots) from env.yaml.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--scene-physics", action="store_true")
    parser.add_argument("--empty", action="store_true",
                        help="Load warehouse only (no robots/humanoids/sim) for occ map export.")
    parser.add_argument("--gif-out",     default=None,
                   help="Save bird's-eye GIF to this path.")
    parser.add_argument("--gif-every",   type=int, default=4,
                   help="Record one GIF frame every N sim steps.")
    parser.add_argument("--gif-fps",     type=float, default=7.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(cfg_path(args.env_config))

    from CrowdSim.world.builder import build_env
    result = build_env(config, num_envs=args.num_envs, headless=args.headless,
                       scene_physics=args.scene_physics, empty_mode=args.empty)
    if result is None:
        return  # empty mode
    env, agent, nav_manager, runtime = result
    if args.empty:
        print("[CrowdSim] Paused (no robots loaded) — use Tools > Robotics > Occupancy Map.")
        print("  Close window to exit.")
        # Keep GUI alive via Kit app loop
        import omni.kit.app
        try:
            omni.kit.app.get_app().run()
        except KeyboardInterrupt:
            pass
        return
    if args.full_eval:
        runtime.agent.evaluator.eval_count = 0
        print(runtime.agent.evaluator.evaluate())
    elif nav_manager is not None and nav_manager.config.car_rl_policy:
        run_masked_mimic_with_robot_ppo(
            runtime, nav_manager, config,
            gif_out=args.gif_out, gif_every=args.gif_every, gif_fps=args.gif_fps,
        )
    else:
        run_masked_mimic_policy_loop(runtime)




def run_masked_mimic_policy_loop(runtime) -> None:
    agent = runtime.agent
    env = runtime.env
    agent.eval()
    done_indices = None
    step = 0
    print("[CrowdSim] Running MaskedMimic inference loop... (Ctrl+C to stop)")
    try:
        while True:
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                model_outs = agent.model(obs_td)
                action = model_outs.get("mean_action", model_outs["action"])
            _, _, dones, _, _ = env.step(action)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            step += 1
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")


def run_masked_mimic_with_robot_ppo(
    runtime,
    nav_manager: CrowdNavigationManager,
    cfg: dict[str, Any],
    max_episodes: int | None = None,
    **kwargs,
) -> None:
    gif_out  = kwargs.get("gif_out", None)
    gif_every = kwargs.get("gif_every", 4)
    gif_fps  = kwargs.get("gif_fps", 7.5)
    car_cfg = cfg.get("car", {})
    checkpoint = car_cfg.get("policy_checkpoint")
    if checkpoint is None:
        raise RuntimeError(
            "car.rl_policy is true, but car.policy_checkpoint is not set in the environment config. "
            "Train a policy with CrowdSim/train_robot_ppo.py first, then point this field at the checkpoint."
        )

    from CrowdSim.ppo.ppo_policy import (
        RobotPPOConfig, RobotPPOTrainer, bounded_robot_action, robot_network_kwargs,
    )

    network_cfg = get_network_config(cfg)
    rl_cfg = cfg.get("navigation", {}).get("rl", {})
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
        ),
        nav_manager.config.device,
    )
    loaded_step = ppo.load(resolve_repo_path(str(checkpoint)))
    deterministic = bool(rl_cfg.get("deterministic", True))
    print(f"[CrowdSim] Loaded robot PPO policy from {checkpoint} at robot_step={loaded_step}.")

    agent = runtime.agent
    env = runtime.env
    agent.eval()

    # ── Preference collection (pref_collect.py only) ──────────────
    pref_collector = kwargs.pop("pref_collector", None)
    pref_save_path = kwargs.pop("pref_save_path", None)
    pref_save_freq_pairs = kwargs.pop("pref_save_freq_pairs", 100)
    if pref_collector is not None:
        from CrowdSim.pref.pref_buffer import pack_crowdsim_step
        print(f"\033[96m[CrowdSim] Preference collection enabled: "
              f"{nav_manager.config.num_robots} env(s), "
              f"save every {pref_save_freq_pairs} pairs\033[0m")

    done_indices = None
    step = 0
    total_car_episodes = 0
    last_saved_episodes = -1  # pair count at last save
    hint = f"(max {max_episodes} episodes)" if max_episodes else "(Ctrl+C to stop)"
    print(f"Evaluating MaskedMimic + robot PPO policy... {hint}")
    try:
        while True:
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                model_outs = agent.model(obs_td)
                humanoid_action = model_outs.get("mean_action", model_outs["action"])
                robot_obs, robot_neighbors, neighbor_mask, robot_depth, robot_map = nav_manager.get_robot_rl_observations()
                if deterministic:
                    mean, _, _ = ppo.model(robot_obs, robot_depth, robot_map, robot_neighbors, neighbor_mask)
                    robot_action = bounded_robot_action(mean)
                else:
                    robot_action, _, _, _ = ppo.act(robot_obs, robot_depth, robot_map, robot_neighbors, neighbor_mask)

            # ── Preference collection: pack & record every step ────
            if pref_collector is not None:
                robot_obs_np = robot_obs.detach().cpu().numpy()
                robot_act_np = robot_action.detach().cpu().numpy()
                robot_depth_np = robot_depth.detach().cpu().numpy() if robot_depth is not None else None
                positions = nav_manager.drive.positions_xy()
                yaws = nav_manager.robot_yaws()
                offset = nav_manager.config.num_humanoids
                _depth_size = nav_manager.config.rl_depth_size
                _prox_thresh = float(nav_manager.config.rl_proximity_penalty_threshold)
                for i in range(nav_manager.config.num_robots):
                    pref_collector.add_step(i, pack_crowdsim_step(
                        robot_obs=robot_obs_np[i],
                        robot_action=robot_act_np[i],
                        robot_xy=positions[i],
                        robot_yaw=float(yaws[i]),
                        goal_xy=nav_manager.goals_xy[offset + i],
                        min_dist=nav_manager.robot_min_dist(i),
                        neighbors_xy=nav_manager.robot_neighbors_xy(i),
                        depth=robot_depth_np[i] if robot_depth_np is not None else None,
                        depth_size=_depth_size,
                        min_dist_static=nav_manager._min_static_obstacle_dist(
                            positions[i], _prox_thresh,
                        ),
                    ))

            nav_manager.set_robot_rl_actions(robot_action)
            _, _, dones, _, _ = env.step(humanoid_action)
            _, _, robot_done, _, _, _, _, _ = nav_manager.get_robot_rl_feedback()

            # ── GIF frame ─────────────────────────────────────────
            record_gif_frame(nav_manager, gif_out, gif_every, step)

            # ── Count completed car episodes BEFORE reset (which zeros robot_done) ──
            robot_done_np = robot_done.detach().cpu().numpy() if hasattr(robot_done, "detach") else np.asarray(robot_done)
            total_car_episodes += int(robot_done_np.sum())

            # ── Preference collection: finalise episodes ──────────
            # end_episode() returns True when run-A just finished and the same
            # route should be replayed for run-B.  We collect those flags into
            # repeat_mask and pass it to reset so nav_manager teleports the robot
            # back to the same start + goal instead of sampling a fresh route.
            if pref_collector is not None:
                offset = nav_manager.config.num_humanoids
                repeat_mask = np.zeros(nav_manager.config.num_robots, dtype=bool)
                for i in range(nav_manager.config.num_robots):
                    if robot_done_np[i]:
                        should_repeat = pref_collector.end_episode(
                            i,
                            goal_xy=nav_manager.goals_xy[offset + i],
                        )
                        repeat_mask[i] = should_repeat
                nav_manager.reset_robot_rl_episodes(robot_done, repeat_mask=repeat_mask)
            else:
                nav_manager.reset_robot_rl_episodes(robot_done)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            step += 1

            # Periodic save (by pair count)
            if (pref_collector is not None
                    and len(pref_collector.pairs) > 0
                    and len(pref_collector.pairs) % pref_save_freq_pairs == 0
                    and last_saved_episodes != len(pref_collector.pairs)):
                pref_collector.save(pref_save_path, finish=False)
                last_saved_episodes = len(pref_collector.pairs)

            if max_episodes is not None and total_car_episodes >= max_episodes:
                print(f"\033[95m[CrowdSim] Reached {total_car_episodes} car episodes — stopping.\033[0m")
                break
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        if gif_out:
            save_gif(Path(gif_out), gif_fps)
        if pref_collector is not None:
            pref_collector.save(pref_save_path, finish=True)
            print(f"\033[93m[CrowdSim] Preference data saved: {pref_collector}\033[0m")




def get_network_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("network", {})


# ─────────────────────────────────────────────────────────────────
# GIF helpers — used by run_masked_mimic_with_robot_ppo
# ─────────────────────────────────────────────────────────────────

_AGENT_PALETTE = [
    (230, 40, 40), (40, 160, 230), (40, 210, 80), (230, 180, 30),
    (170, 50, 210), (230, 80, 170), (30, 200, 190), (210, 130, 40),
    (70, 130, 230), (190, 210, 30), (210, 60, 130), (50, 210, 150),
]


def _agent_color(aid: int):
    return _AGENT_PALETTE[aid % len(_AGENT_PALETTE)]


def _load_font():
    try: return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError: return ImageFont.load_default()


def record_gif_frame(
    nav_manager, out_path: Path, gif_every: int, step: int,
    circle_positions=None,
) -> bool:
    """Record one frame if step % gif_every == 0.  Returns True if frame saved."""
    if out_path is None or step % gif_every != 0:
        return False
    positions, _ = nav_manager._read_agent_state()
    yaws = nav_manager.robot_yaws()
    goals = nav_manager.goals_xy
    cfg = nav_manager.config
    n_robots = cfg.num_robots
    n_humanoids = cfg.num_humanoids
    roff = n_humanoids

    map_path = str(getattr(cfg, "map_path", "") or "")
    res = cfg.map_resolution
    ori = getattr(cfg, "map_origin_xy", None)
    ox, oy = (float(ori[0]), float(ori[1])) if ori else (0.0, 0.0)

    if map_path and Path(map_path).exists():
        from PIL import ImageOps
        bg = ImageOps.autocontrast(Image.open(map_path).convert("L")).convert("RGBA")
    else:
        side = max(128, int(40.0 / res * 2))
        bg = Image.new("RGBA", (side, side), (25, 25, 35, 255))
    fw, fh = bg.size
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    def w2p(xy):
        px = int(round((float(xy[0]) - ox) / res))
        py = int(round((fh - 1) - (float(xy[1]) - oy) / res))
        return px, py

    # Humanoids (positions[0:n_h] are humanoid positions)
    for h in range(min(n_humanoids, positions.shape[0])):
        c = _agent_color(h)
        px, py = w2p(positions[h])
        draw.ellipse((px-5, py-5, px+5, py+5), fill=(*c, 210), outline=(0,0,0,230), width=2)

    # Robots (positions[roff:roff+n_r] are robot positions)
    for i in range(min(n_robots, positions.shape[0] - roff)):
        c = _agent_color(roff + i)
        px, py = w2p(positions[roff + i])
        theta = float(yaws[i])
        draw.rectangle((px-6, py-6, px+6, py+6), fill=(*c, 245), outline=(0,0,0,230), width=2)
        arr = (px + int(12 * math.cos(theta)), py - int(12 * math.sin(theta)))
        draw.line([(px, py), arr], fill=(*c, 230), width=2)

    # Goals
    for i in range(min(n_robots, goals.shape[0] - roff)):
        c = _agent_color(roff + i)
        gx, gy = w2p(goals[roff + i])
        pts = [(gx + int(math.cos(math.pi*0.5*j)*6),
                gy + int(math.sin(math.pi*0.5*j)*6)) for j in range(4)]
        draw.polygon(pts, fill=(*c, 230))

    # Text
    font = _load_font()
    txt = f"step {step}  |  {n_robots}R  {n_humanoids}H"
    pad = 4
    bbox = draw.textbbox((10, 8), txt, font=font)
    draw.rectangle((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad), fill=(0,0,0,150))
    draw.text((10, 8), txt, fill=(255,255,255,235), font=font)

    frame = Image.alpha_composite(bg, overlay).convert("RGB")

    # Append to GIF frame list (initialise on first call)
    if not hasattr(record_gif_frame, "_frames"):
        record_gif_frame._frames = []  # type: ignore
    record_gif_frame._frames.append(frame)  # type: ignore
    return True


def save_gif(out_path: Path, gif_fps: float, wandb_run=None, wandb_step: int = 0):
    """Save accumulated frames to disk + optional wandb."""
    frames = getattr(record_gif_frame, "_frames", [])
    if not frames:
        return
    dur_ms = int(round(1000.0 / max(gif_fps, 1)))
    quantized = [f.convert("P", palette=Image.Palette.ADAPTIVE) for f in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        out_path, save_all=True, append_images=quantized[1:],
        duration=dur_ms, loop=0, optimize=False,
    )
    kb = out_path.stat().st_size // 1024
    print(f"[GIF] Saved → {out_path}  ({len(frames)} frames @ {gif_fps}fps, {kb} KB)")
    if wandb_run is not None:
        try:
            import wandb
            buf = io.BytesIO()
            quantized[0].save(buf, format="GIF", save_all=True,
                              append_images=quantized[1:], duration=dur_ms, loop=0)
            wandb_run.log({"eval/video": wandb.Video(io.BytesIO(buf.getvalue()),
                                                      format="gif", fps=int(gif_fps))},
                          step=wandb_step)
            print(f"[W&B] Uploaded GIF → eval/video (step {wandb_step})")
        except Exception as exc:
            print(f"[W&B] WARNING: GIF upload failed: {exc}")
    record_gif_frame._frames = []  #type: ignore


if __name__ == "__main__":
    main()
