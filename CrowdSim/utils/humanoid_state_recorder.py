"""Humanoid state recording utilities for CrowdSim."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class HumanoidStateRecorderConfig:
    output_dir: Path
    fps: float = 30.0
    env_ids: str = "0"
    auto_record: bool = False
    key: str = "H"


class HumanoidStateRecorder:
    def __init__(
        self,
        env,
        output_dir: Path,
        fps: float,
        env_ids: list[int],
    ) -> None:
        if fps <= 0:
            raise ValueError(f"humanoid state recording fps must be positive, got {fps}")

        self.env = env
        self.output_dir = output_dir
        self.fps = fps
        self.env_ids = env_ids
        self.env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=env.device)
        self.enabled = False
        self.frame_idx = 0
        self.sim_time = 0.0
        self.next_capture_time = 0.0
        self.session_dir: Path | None = None

    def toggle(self) -> None:
        if self.enabled:
            self.enabled = False
            print(f"[CrowdSim] Humanoid state recording stopped: {self.session_dir}")
            return

        self.session_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True
        self.frame_idx = 0
        self.sim_time = 0.0
        self.next_capture_time = 0.0
        self._write_metadata()
        print(
            f"[CrowdSim] Humanoid state recording started at {self.fps:g} fps: "
            f"{self.session_dir}"
        )

    def step(self, actions: torch.Tensor | None = None) -> None:
        self.sim_time += float(getattr(self.env, "dt", 0.0))
        if not self.enabled:
            return
        if self.sim_time + 1e-9 < self.next_capture_time:
            return

        self._save_frame(actions)
        self.frame_idx += 1
        self.next_capture_time += 1.0 / self.fps

    def _write_metadata(self) -> None:
        control_dt = float(getattr(self.env, "dt", 0.0))
        torch.save(
            {
                "fps": self.fps,
                "env_dt": control_dt,
                "control_hz": 1.0 / control_dt if control_dt > 0 else None,
                "env_ids": self.env_ids,
                "dof_names": list(getattr(self.env.robot_config.kinematic_info, "dof_names", [])),
                "body_names": list(getattr(self.env.robot_config.kinematic_info, "body_names", [])),
            },
            self.session_dir / "metadata.pt",
        )

    def _save_frame(self, actions: torch.Tensor | None) -> None:
        root_state = self.env.simulator.get_root_state(self.env_ids_tensor)
        dof_state = self.env.simulator.get_dof_state(self.env_ids_tensor)
        frame = {
            "frame": self.frame_idx,
            "time": self.sim_time,
            "root_pos": root_state.root_pos.detach().cpu(),
            "root_rot": root_state.root_rot.detach().cpu(),
            "root_vel": root_state.root_vel.detach().cpu(),
            "root_ang_vel": root_state.root_ang_vel.detach().cpu(),
            "dof_pos": dof_state.dof_pos.detach().cpu(),
            "dof_vel": dof_state.dof_vel.detach().cpu(),
        }
        if actions is not None:
            frame["actions"] = actions.detach().cpu()[self.env_ids_tensor.cpu()]
        torch.save(frame, self.session_dir / f"frame_{self.frame_idx:06d}.pt")


def configure_humanoid_state_recorder(
    env,
    config: HumanoidStateRecorderConfig,
) -> HumanoidStateRecorder | None:
    recorder = HumanoidStateRecorder(
        env=env,
        output_dir=config.output_dir.expanduser().resolve(),
        fps=config.fps,
        env_ids=parse_record_env_ids(config.env_ids, env.num_envs),
    )
    env.crowdsim_humanoid_state_recorder = recorder

    import types

    original_step = env.step

    def step_with_humanoid_state_recording(self, action):
        result = original_step(action)
        recorder.step(action)
        return result

    env.step = types.MethodType(step_with_humanoid_state_recording, env)

    keyboard = getattr(env.simulator, "keyboard_interface", None)
    if keyboard is not None:
        try:
            keyboard.add_callback(config.key, recorder.toggle)
            print(f"[CrowdSim] Press {config.key} to start/stop humanoid state recording.")
        except Exception as exc:
            print(f"[CrowdSim] Warning: failed to register {config.key} recording key: {exc}")

    if config.auto_record:
        recorder.toggle()

    return recorder


def parse_record_env_ids(value: str, num_envs: int) -> list[int]:
    normalized = str(value).strip().lower()
    if normalized == "all":
        return list(range(num_envs))

    env_ids = [int(item.strip()) for item in normalized.split(",") if item.strip()]
    if not env_ids:
        raise ValueError("humanoid state record env ids must contain at least one env id")
    for env_id in env_ids:
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(f"Humanoid state record env id {env_id} outside [0, {num_envs})")
    return env_ids
