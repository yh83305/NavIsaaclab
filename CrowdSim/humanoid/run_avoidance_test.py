"""Run a finite MaskedMimic humanoid-only avoidance diagnostic."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("isaaclab")

import torch  # noqa: E402

from CrowdSim.utils.config_loader import cfg_path, load_config  # noqa: E402
from CrowdSim.world.builder import build_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="CrowdSim/config/eval/humanoid_avoidance.yaml"
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def _pairwise_minimum(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("inf")
    delta = positions[:, None] - positions[None]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def _humanoid_root_yaws(nav, count: int) -> np.ndarray:
    quaternions = (
        nav.env.simulator.get_root_state().root_rot[:count]
        .detach().cpu().numpy()
    )
    w, x, y, z = quaternions.T
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": float(finite.mean()),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
    }


def _diagnose(report: dict) -> list[str]:
    findings: list[str] = []
    velocity = report["velocity_tracking_error_mps"]["mean"]
    heading = report["heading_tracking_error_deg"]["p95"]
    target = report["delayed_target_error_m"]["mean"]
    ttc = report["ttc_force"]["p95"]
    if velocity is not None and velocity > 0.4:
        findings.append("MaskedMimic velocity tracking is the primary bottleneck")
    if heading is not None and heading > 30.0:
        findings.append("MaskedMimic turning response is too slow for the requested targets")
    if target is not None and target > 0.5:
        findings.append("Pelvis misses the first future target by more than 0.5 m on average")
    if report["unsafe_frame_rate"] > 0.05 and (ttc is None or ttc < 0.1):
        findings.append("TTC rarely activates before unsafe proximity; tune TTC/SFM")
    if report["collision_frame_rate"] > 0.0:
        findings.append("Physical humanoid spacing still enters the collision threshold")
    if not findings:
        findings.append("Planner-to-humanoid tracking is within the configured thresholds")
    return findings


def run(config: dict, steps: int, headless: bool, render_gif: bool) -> Path:
    output_dir = cfg_path(config["evaluation"]["output_dir"], PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, _, nav, runtime = build_env(
        config, num_envs=None, headless=headless, scene_physics=False
    )
    if nav.config.num_robots != 0:
        raise ValueError("Humanoid avoidance diagnostic requires num_robots=0")

    agent = runtime.agent
    agent.eval()
    done_indices = None
    target_delay = max(1, int(round(nav.config.local_target_timestep)))
    pending_targets: deque[tuple[int, np.ndarray]] = deque()
    velocity_errors: list[float] = []
    heading_errors: list[float] = []
    target_errors: list[float] = []
    minimum_distances: list[float] = []
    ttc_magnitudes: list[float] = []
    collision_frames = 0

    print(
        f"[HumanoidEval] Running {steps} steps with "
        f"{nav.config.num_humanoids} humanoids; target delay={target_delay} steps"
    )
    try:
        for step in range(steps):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                outputs = agent.model(obs_td)
                action = outputs.get("mean_action", outputs["action"])
            _, _, dones, _, _ = env.step(action)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

            positions, velocities = nav._read_agent_state()
            count = nav.config.num_humanoids
            positions = positions[:count]
            actual = velocities[:count]
            desired = nav._sfm_desired_velocities[:count].copy()
            velocity_errors.extend(np.linalg.norm(actual - desired, axis=1).tolist())
            root_yaws = _humanoid_root_yaws(nav, count)
            target_yaws = nav._humanoid_future_first_yaws[:count]
            heading_errors.extend(np.abs(np.arctan2(
                np.sin(root_yaws - target_yaws),
                np.cos(root_yaws - target_yaws),
            )).tolist())
            minimum_distance = _pairwise_minimum(positions)
            minimum_distances.append(minimum_distance)
            ttc_magnitudes.extend(
                np.linalg.norm(nav._sfm_ttc_forces[:count], axis=1).tolist()
            )
            collision_frames += int(
                minimum_distance < nav.config.collision_distance
            )

            pending_targets.append((
                step + target_delay,
                nav._humanoid_future_first_targets[:count].copy(),
            ))
            while pending_targets and pending_targets[0][0] <= step:
                _, targets = pending_targets.popleft()
                target_errors.extend(np.linalg.norm(positions - targets, axis=1).tolist())

            if step and step % 100 == 0:
                print(
                    f"[HumanoidEval] step={step}/{steps} "
                    f"min_dist={minimum_distances[-1]:.3f}m "
                    f"velocity_error={np.mean(velocity_errors[-count:]):.3f}m/s"
                )
    finally:
        nav._close_trajectory_logs()

    report = {
        "steps": steps,
        "num_humanoids": nav.config.num_humanoids,
        "minimum_pairwise_distance_m": float(min(minimum_distances)),
        "unsafe_frame_rate": float(np.mean(
            np.asarray(minimum_distances) < nav.config.safe_distance
        )),
        "collision_frame_rate": collision_frames / max(steps, 1),
        "velocity_tracking_error_mps": _summary(velocity_errors),
        "heading_tracking_error_deg": _summary(
            [math.degrees(value) for value in heading_errors]
        ),
        "delayed_target_error_m": _summary(target_errors),
        "ttc_force": _summary(ttc_magnitudes),
    }
    report["diagnosis"] = _diagnose(report)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[HumanoidEval] Metrics saved: {metrics_path}")
    print(json.dumps(report, indent=2))

    if render_gif:
        gif_path = (output_dir / "humanoid_avoidance.gif").resolve()
        trajectory_path = Path(nav.trajectory_log_path).resolve()
        path_log_path = Path(nav.path_log_path).resolve()
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "CrowdSim/tools/render_navigation_fast.py"),
                str(trajectory_path),
                "--path-log", str(path_log_path),
                "--tail-frames", "0",
                "--stride", str(config["evaluation"]["gif_stride"]),
                "--fps", str(config["evaluation"]["gif_fps"]),
                "--show-yaw-source-labels",
                "--output", str(gif_path),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
    return metrics_path


def main() -> None:
    args = parse_args()
    config = load_config(cfg_path(args.config, PROJECT_ROOT))
    evaluation = config["evaluation"]
    steps = int(args.steps if args.steps is not None else evaluation["steps"])
    headless = bool(
        args.headless if args.headless is not None else evaluation["headless"]
    )
    run(config, steps, headless, not args.no_gif)


if __name__ == "__main__":
    main()
