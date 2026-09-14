#!/usr/bin/env python3
"""Keyboard-driven robot control for CrowdSim / IsaacLab.

Usage:
    python CrowdSim/tools/test_robot_kinematics.py \
        --env-config CrowdSim/config/env.yaml

Controls:
    W / ↑    : increase forward speed (+0.2 m/s)
    S / ↓    : decrease forward speed / reverse
    A / ←    : turn left  (+0.3 rad/s)
    D / →    : turn right (-0.3 rad/s)
    Space    : emergency stop
    R        : reset robot to origin
    Q / Esc  : quit
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CrowdSim.control.drive import (
    DifferentialDriveConfig,
    ManualDifferentialController,
)

# ---------------------------------------------------------------------------
# YAML config loader (no PyYAML dependency)
# ---------------------------------------------------------------------------

def _load_yaml_config(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(raw_value)
    return root


def _parse_yaml_scalar(value: str):
    text = value.strip()
    quote: str | None = None
    for idx, char in enumerate(text):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            text = text[:idx].strip()
            break
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    if "," in text:
        return [_parse_yaml_scalar(item) for item in text.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# Non-blocking keyboard input
# ---------------------------------------------------------------------------

class KeyboardDriver:
    """Non-blocking keyboard input using raw terminal mode.

    Raw mode disables output processing (\\n → \\r\\n), so all output during
    the control loop must use write_raw() which compensates with explicit \\r.
    """

    LINEAR_STEP = 0.2
    ANGULAR_STEP = 0.3
    MAX_LINEAR = 2.0
    MAX_ANGULAR = 3.0

    def __init__(self) -> None:
        self._stdin_fd = sys.stdin.fileno()
        self._stdout_fd = sys.stdout.fileno()
        self._old_settings: list[object] | None = None
        self._use_termios = hasattr(termios, "TCSADRAIN")
        self.v = 0.0
        self.w = 0.0
        self.quit = False
        self.reset_pose = False

    def start(self) -> None:
        if self._use_termios:
            try:
                self._old_settings = termios.tcgetattr(self._stdin_fd)
                tty.setraw(self._stdin_fd)
            except (termios.error, OSError):
                self._use_termios = False

    def stop(self) -> None:
        if self._use_termios and self._old_settings is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_settings)
            except (termios.error, OSError):
                pass
            self._old_settings = None

    def write_raw(self, text: str) -> None:
        fixed = text.replace("\n", "\r\n")
        if not fixed.endswith("\r\n"):
            fixed += "\r\n"
        os.write(self._stdout_fd, fixed.encode("utf-8"))

    def poll(self) -> None:
        while True:
            ch = self._read_key()
            if ch is None:
                break
            if ch in ("q", "Q", "\x1b"):
                self.quit = True
                return
            if ch in ("r", "R"):
                self.v = 0.0
                self.w = 0.0
                self.reset_pose = True
                continue
            if ch == " ":
                self.v = 0.0
                self.w = 0.0
                continue
            if ch == "\x1b":
                ch2 = self._read_key()
                if ch2 == "[":
                    ch3 = self._read_key()
                    if ch3 == "A":
                        self.v = min(self.v + self.LINEAR_STEP, self.MAX_LINEAR)
                    elif ch3 == "B":
                        self.v = max(self.v - self.LINEAR_STEP, -self.MAX_LINEAR)
                    elif ch3 == "C":
                        self.w = max(self.w - self.ANGULAR_STEP, -self.MAX_ANGULAR)
                    elif ch3 == "D":
                        self.w = min(self.w + self.ANGULAR_STEP, self.MAX_ANGULAR)
                continue
            if ch in ("w", "W"):
                self.v = min(self.v + self.LINEAR_STEP, self.MAX_LINEAR)
            elif ch in ("s", "S"):
                self.v = max(self.v - self.LINEAR_STEP, -self.MAX_LINEAR)
            elif ch in ("a", "A"):
                self.w = min(self.w + self.ANGULAR_STEP, self.MAX_ANGULAR)
            elif ch in ("d", "D"):
                self.w = max(self.w - self.ANGULAR_STEP, -self.MAX_ANGULAR)

    def _read_key(self) -> str | None:
        ready, _, _ = select.select([self._stdin_fd], [], [], 0.0)
        if ready:
            try:
                data = os.read(self._stdin_fd, 1)
                if data:
                    return data.decode("utf-8", errors="replace")
            except (OSError, ValueError):
                pass
        return None


# ---------------------------------------------------------------------------
# Keyboard control loop
# ---------------------------------------------------------------------------

HELP = """
╔══════════════════════════════════════════════════════╗
║         Keyboard Robot Control                       ║
╠══════════════════════════════════════════════════════╣
║  W / ↑    : +0.2 m/s      Space : stop              ║
║  S / ↓    : -0.2 m/s      R     : reset to origin   ║
║  A / ←    : turn left      Q/Esc : quit             ║
║  D / →    : turn right                              ║
╚══════════════════════════════════════════════════════╝
"""


def _run_keyboard_loop(env, agent, robot, ctrl, wheel_joint_ids, drive_mode,
                       device, num_envs, env_dt, status_interval,
                       recorded_positions: list) -> None:
    import torch

    kbd = KeyboardDriver()

    print(HELP)
    print(f"[keyboard] env_dt={env_dt:.4f}s  mode={drive_mode}  num_envs={num_envs}")
    print()

    kbd.start()

    header = (f"{'time':>7s} {'v_cmd':>7s} {'v_act':>7s} "
              f"{'ω_cmd':>7s} {'ω_act':>7s} "
              f"{'x':>8s} {'y':>8s} {'yaw°':>7s}")
    kbd.write_raw(header)
    kbd.write_raw("-" * len(header))

    step = 0
    done_indices = None

    try:
        while not kbd.quit:
            kbd.poll()

            if kbd.reset_pose:
                _reset_robot(robot, device, num_envs)
                recorded_positions.clear()
                step = 0
                kbd.reset_pose = False
                kbd.write_raw("[reset]")

            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            with torch.no_grad():
                model_outs = agent.model(obs_td)
                humanoid_action = model_outs.get("mean_action", model_outs["action"])

            _apply_command(robot, ctrl, kbd.v, kbd.w, drive_mode,
                           device, num_envs, wheel_joint_ids)

            _, _, dones, _, _ = env.step(humanoid_action)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            step += 1

            t = step * env_dt
            x = float(robot.data.root_pos_w[0, 0].cpu().item())
            y = float(robot.data.root_pos_w[0, 1].cpu().item())
            yaw = _yaw_from_quat(robot.data.root_quat_w[0].cpu().numpy())
            recorded_positions.append((t, x, y, yaw))

            if step % status_interval == 0:
                # Read actual velocity from simulator
                lin_vel = robot.data.root_lin_vel_w[0, :2].cpu().numpy()
                v_act = float(np.linalg.norm(lin_vel))
                ω_act = float(robot.data.root_ang_vel_w[0, 2].cpu().item())
                kbd.write_raw(
                    f"{t:7.2f} {kbd.v:7.3f} {v_act:7.3f} "
                    f"{kbd.w:7.3f} {ω_act:7.3f} "
                    f"{x:8.3f} {y:8.3f} {math.degrees(yaw):7.1f}"
                )

    except KeyboardInterrupt:
        kbd.write_raw("[interrupted]")
    finally:
        kbd.stop()

    print(f"\n[keyboard] Done. {step} steps ({step * env_dt:.1f}s).")


# ---------------------------------------------------------------------------
# Robot control helpers
# ---------------------------------------------------------------------------

def _apply_command(robot, ctrl: ManualDifferentialController,
                   v: float, w: float, drive_mode: str,
                   device, num_envs: int, joint_ids: list[int]) -> None:
    import torch

    if drive_mode == "wheel":
        wheels = ctrl.forward(np.array([v, w], dtype=np.float32))
        targets = torch.zeros(num_envs, 2, dtype=torch.float32, device=device)
        targets[:, 0] = float(wheels[0])
        targets[:, 1] = float(wheels[1])
        robot.set_joint_velocity_target(targets, joint_ids=joint_ids)
    elif drive_mode == "kinematic":
        # Zero wheel targets so actuators don't fight root velocity writes
        robot.set_joint_velocity_target(
            torch.zeros(num_envs, 2, dtype=torch.float32, device=device),
            joint_ids=joint_ids)
        vel = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        yaws = _batch_yaw_from_quat(robot.data.root_quat_w.cpu().numpy())
        for i in range(num_envs):
            vel[i, 0] = v * math.cos(float(yaws[i]))
            vel[i, 1] = v * math.sin(float(yaws[i]))
            vel[i, 5] = w
        robot.write_root_velocity_to_sim(vel)


def _reset_robot(robot, device, num_envs: int) -> None:
    import torch

    poses = torch.zeros(num_envs, 7, dtype=torch.float32, device=device)
    poses[:, 0] = 1.0
    robot.write_root_pose_to_sim(poses)
    robot.write_root_velocity_to_sim(
        torch.zeros(num_envs, 6, dtype=torch.float32, device=device))
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        torch.zeros_like(robot.data.default_joint_vel))


def _read_env_dt(env) -> float:
    dt = float(getattr(env, "dt", 0.0) or 0.0)
    if dt > 0.0:
        return dt
    sim = getattr(env, "simulator", None)
    sim_dt = float(getattr(sim, "dt", 0.0) or 0.0)
    if sim_dt > 0.0:
        return sim_dt
    return 1.0 / 30.0


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _batch_yaw_from_quat(quats: np.ndarray) -> np.ndarray:
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config_path: Path, ppo_config_path: Path | None, num_envs: int,
        headless: bool, drive_mode: str) -> None:
    # --- Setup ---
    print("[setup] Starting Isaac Sim...")
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    AppLauncher = import_simulator_before_torch("isaaclab")
    import torch
    torch.set_float32_matmul_precision("high")

    from CrowdSim.protomotions_runtime import (
        build_runtime, configure_viewer_camera, create_fabric,
        make_crowd_robot_config, resolve_robot_usd,
        suppress_known_isaaclab_warning_spam,
    )
    from CrowdSim.scene_setup import (
        apply_fixed_crowd_robot_spawns, apply_fixed_spawn_offsets,
        patch_isaaclab_scene_with_crowdsim_assets, resolve_repo_path,
    )
    from CrowdSim.utils.map_metadata import load_occupancy_map_metadata

    print("[setup] Loading config...")
    config = _load_yaml_config(config_path)
    if ppo_config_path and ppo_config_path.exists():
        ppo_cfg = _load_yaml_config(ppo_config_path)
        rl = ppo_cfg.get("rl", ppo_cfg)
        if isinstance(rl, dict):
            config.setdefault("navigation", {})["rl"] = rl

    sc = config.get("scene", {})
    hc = config.get("humanoid", {})
    cc = config.get("car", {})

    checkpoint = resolve_repo_path(hc["checkpoint"])
    motion_file = resolve_repo_path(hc["motion_file"])
    scene_usd = resolve_repo_path(sc["scene_usd"])
    scene_map = resolve_repo_path(sc["scene_map"])
    map_md = load_occupancy_map_metadata(scene_map)

    for label, p in [("checkpoint", checkpoint), ("motion", motion_file),
                       ("scene", scene_usd), ("map", map_md.image_path)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    fabric = create_fabric()
    launcher = AppLauncher({"headless": headless, "device": str(fabric.device)})
    suppress_known_isaaclab_warning_spam()

    # --- Import robot ---
    print("[setup] Importing robot...")
    robot_usd = resolve_robot_usd(cc.get("usd"))
    if robot_usd is None:
        raise RuntimeError("car.usd not set in env.yaml (e.g. nova_carter)")
    crowd_robot_cfg = make_crowd_robot_config(cc, config.get("sensors", {}), robot_usd)
    patch_isaaclab_scene_with_crowdsim_assets(
        scene_usd_path=scene_usd,
        scene_z_offset=float(sc.get("z_offset", 0.0)),
        scene_prim_path=str(sc.get("prim_path", "/World/Scene")),
        crowd_robot=crowd_robot_cfg,
    )

    # --- Build runtime ---
    print("[setup] Building ProtoMotions runtime...")
    runtime = build_runtime(checkpoint=checkpoint, motion_file=motion_file,
                            num_envs=num_envs, headless=headless,
                            simulation_app=launcher.app, fabric=fabric)
    env = runtime.env
    agent = runtime.agent
    agent.eval()
    configure_viewer_camera(env, config.get("viewer", {}), headless)

    # --- Place robot ---
    print("[setup] Placing robot...")
    robot_spawn = torch.zeros(num_envs, 3, dtype=torch.float32, device=fabric.device)
    humanoid_spawn = torch.tensor(
        [[float(i + 1) * 2.0, 0.0] for i in range(num_envs)],
        dtype=torch.float32, device=fabric.device)
    apply_fixed_spawn_offsets(env, humanoid_spawn)
    apply_fixed_crowd_robot_spawns(env, robot_spawn)
    robot = env.crowdsim_robot

    left_ids, _ = robot.find_joints(
        list(DifferentialDriveConfig().left_wheel_joint_names), preserve_order=True)
    right_ids, _ = robot.find_joints(
        list(DifferentialDriveConfig().right_wheel_joint_names), preserve_order=True)
    wheel_joint_ids = [int(left_ids[0]), int(right_ids[0])]
    print(f"[setup] Wheel joints: left={left_ids}, right={right_ids}")
    print(f"[setup] Robot joints: {list(robot.data.joint_names)}")

    # --- Keyboard loop ---
    ctrl = ManualDifferentialController()
    env_dt = _read_env_dt(env)
    interval = max(1, int(round(0.5 / env_dt)))
    recorded: list[tuple[float, float, float, float]] = []

    _run_keyboard_loop(env, agent, robot, ctrl, wheel_joint_ids,
                       drive_mode, fabric.device, num_envs, env_dt, interval,
                       recorded)

    # --- Save trajectory ---
    if recorded:
        out = Path("output/crowdsim_kinematics_test")
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        import json
        path = out / f"trajectory_{ts}.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"type": "metadata", "timestamp": ts,
                                "drive_mode": drive_mode,
                                "num_frames": len(recorded),
                                "duration_s": recorded[-1][0]}) + "\n")
            for t, x, y, yaw in recorded:
                f.write(json.dumps(
                    {"t": t, "x": x, "y": y, "yaw_deg": math.degrees(yaw)}) + "\n")
        print(f"[save] {path} ({len(recorded)} frames)")
    print("[done]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyboard-driven robot control for CrowdSim.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP,
    )
    parser.add_argument("--env-config", default="CrowdSim/config/env.yaml")
    parser.add_argument("--ppo-config", default="CrowdSim/config/env.yaml")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", choices=["wheel", "kinematic"], default="kinematic")
    args = parser.parse_args()

    env_config = _resolve_path(args.env_config)
    if not env_config.exists():
        print(f"ERROR: {env_config} not found")
        sys.exit(1)
    ppo_config = _resolve_path(args.ppo_config)
    if not ppo_config.exists():
        print(f"WARNING: {ppo_config} not found, using defaults")
        ppo_config = None

    run(env_config, ppo_config, args.num_envs, args.headless, args.mode)


if __name__ == "__main__":
    main()
