#!/usr/bin/env python3
"""Validate CrowdSim robot kinematics inside the real Isaac/PhysX loop.

The test uses the production navigation wrapper and reports commanded versus
measured linear/angular motion.  It fails when a command is integrated twice
or when commanded root velocity leaks into PhysX.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

import_simulator_before_torch("isaaclab")

import torch  # noqa: E402

from CrowdSim.utils.config_loader import load_config  # noqa: E402
from CrowdSim.world.builder import build_env  # noqa: E402


def _yaw(quat: torch.Tensor) -> float:
    w, x, y, z = (float(value) for value in quat.detach().cpu())
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_delta(end: float, start: float) -> float:
    return math.atan2(math.sin(end - start), math.cos(end - start))


def _humanoid_action(runtime, observation):
    agent = runtime.agent
    observation = agent.add_agent_info_to_obs(observation)
    observation_td = agent.obs_dict_to_tensordict(observation)
    with torch.no_grad():
        outputs = agent.model(observation_td)
    return outputs.get("mean_action", outputs["action"])


def _advance(runtime, nav, action: np.ndarray, steps: int) -> tuple[float, float, float]:
    env = runtime.env
    observation, _ = env.reset(None)
    robot = nav.robot
    start_xy = robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
    start_yaw = _yaw(robot.data.root_quat_w[0])
    max_post_step_root_velocity = 0.0

    command = torch.as_tensor(action[None], dtype=torch.float32, device=nav.config.device)
    for _ in range(steps):
        humanoid_action = _humanoid_action(runtime, observation)
        nav.set_robot_rl_actions(command)
        observation, _, _, _, _ = env.step(humanoid_action)
        physical_linear = torch.linalg.vector_norm(robot.data.root_lin_vel_w[0, :2])
        physical_angular = torch.abs(robot.data.root_ang_vel_w[0, 2])
        max_post_step_root_velocity = max(
            max_post_step_root_velocity,
            float(physical_linear),
            float(physical_angular),
        )

    end_xy = robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
    end_yaw = _yaw(robot.data.root_quat_w[0])
    return (
        float(np.linalg.norm(end_xy - start_xy)),
        _angle_delta(end_yaw, start_yaw),
        max_post_step_root_velocity,
    )


def _track_root_velocity_writes(robot) -> dict[str, float]:
    """Record the largest velocity explicitly commanded through IsaacLab.

    PhysX may report a small non-zero root velocity after contact and gravity
    solving.  That is physical response, not duplicate command integration.
    The regression invariant is that DriveController itself only writes zero.
    """
    original = robot.write_root_velocity_to_sim
    tracker = {"maximum": 0.0}

    def tracked_write(velocities, *args, **kwargs):
        tracker["maximum"] = max(
            tracker["maximum"], float(torch.max(torch.abs(velocities)).item())
        )
        return original(velocities, *args, **kwargs)

    robot.write_root_velocity_to_sim = tracked_write
    return tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="CrowdSim/config/collect/ppo.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.12)
    args = parser.parse_args()

    config = copy.deepcopy(load_config(PROJECT_ROOT / args.config))
    navigation = config.setdefault("navigation", {})
    navigation["num_humanoids"] = 1
    navigation["num_robots"] = 1
    config.setdefault("sensors", {}).setdefault("camera", {})["enabled"] = False
    config.setdefault("humanoid", {})["human_mesh"] = False
    config.setdefault("car", {})["rl_policy"] = True

    env, _, nav, runtime = build_env(config, num_envs=1, headless=True)
    written_root_velocity = _track_root_velocity_writes(nav.robot)
    dt = float(nav._env_dt)
    steps = int(args.steps)
    tolerance = float(args.tolerance)

    linear_action = 0.5
    linear_command = linear_action * float(nav.config.rl_max_linear_velocity)
    distance, yaw_drift, straight_physx = _advance(
        runtime, nav, np.asarray([linear_action, 0.0], dtype=np.float32), steps
    )
    measured_linear = distance / (steps * dt)
    linear_ratio = measured_linear / max(linear_command, 1e-8)

    nav.reset_robot_rl_episodes(np.asarray([True]))
    angular_action = 0.5
    angular_command = angular_action * float(nav.config.rl_max_angular_velocity)
    drift, yaw_delta, turn_physx = _advance(
        runtime, nav, np.asarray([0.0, angular_action], dtype=np.float32), steps
    )
    measured_angular = yaw_delta / (steps * dt)
    angular_ratio = measured_angular / max(angular_command, 1e-8)
    max_post_step_root_velocity = max(straight_physx, turn_physx)

    print("[Drive Validation] production Isaac/PhysX integration")
    print(f"  dt={dt:.8f}s steps={steps}")
    print(
        f"  linear:  command={linear_command:.6f}m/s "
        f"measured={measured_linear:.6f}m/s ratio={linear_ratio:.6f} "
        f"yaw_drift={yaw_drift:.6f}rad"
    )
    print(
        f"  angular: command={angular_command:.6f}rad/s "
        f"measured={measured_angular:.6f}rad/s ratio={angular_ratio:.6f} "
        f"xy_drift={drift:.6f}m"
    )
    print(
        f"  max commanded root velocity written to PhysX="
        f"{written_root_velocity['maximum']:.8f}"
    )
    print(
        f"  max post-step physical root velocity="
        f"{max_post_step_root_velocity:.8f} (diagnostic only)"
    )

    failures = []
    if abs(linear_ratio - 1.0) > tolerance:
        failures.append(f"linear ratio {linear_ratio:.4f} is not within {tolerance:.3f} of 1")
    if abs(angular_ratio - 1.0) > tolerance:
        failures.append(f"angular ratio {angular_ratio:.4f} is not within {tolerance:.3f} of 1")
    if written_root_velocity["maximum"] > 1e-7:
        failures.append(
            "DriveController wrote non-zero root velocity "
            f"({written_root_velocity['maximum']:.6f})"
        )
    if failures:
        raise RuntimeError("Drive integration validation failed: " + "; ".join(failures))
    print("[Drive Validation] PASS: commands are integrated exactly once")


if __name__ == "__main__":
    main()
