"""Sensor recording utilities for CrowdSim."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class RobotCameraStreamConfig:
    output_dir: Path
    fps: float = 10.0
    env_ids: str = "0"
    auto_record: bool = False


class RobotCameraRecorder:
    def __init__(
        self,
        env,
        output_dir: Path,
        fps: float,
        env_ids: list[int],
    ) -> None:
        if fps <= 0:
            raise ValueError(f"robot camera fps must be positive, got {fps}")

        self.env = env
        self.output_dir = output_dir
        self.fps = fps
        self.env_ids = env_ids
        self.enabled = False
        self.frame_idx = 0
        self.sim_time = 0.0
        self.next_capture_time = 0.0
        self.session_dir: Path | None = None

    def toggle(self) -> None:
        if self.enabled:
            self.enabled = False
            print(f"[CrowdSim] Robot camera recording stopped: {self.session_dir}")
            return

        self.session_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True
        self.frame_idx = 0
        self.sim_time = 0.0
        self.next_capture_time = 0.0
        for env_id in self.env_ids:
            env_dir = self._env_dir(env_id)
            env_dir.mkdir(parents=True, exist_ok=True)
        self._write_metadata()
        print(
            f"[CrowdSim] Robot camera recording started at {self.fps:g} fps: "
            f"{self.session_dir}"
        )

    def step(self) -> None:
        self.sim_time += float(getattr(self.env, "dt", 0.0))
        if not self.enabled:
            return
        if self.sim_time + 1e-9 < self.next_capture_time:
            return

        self._render_for_camera()
        self._save_frame()
        self.frame_idx += 1
        self.next_capture_time += 1.0 / self.fps

    def _render_for_camera(self) -> None:
        simulator = getattr(self.env, "simulator", None)
        if simulator is None or not getattr(simulator, "headless", False):
            return

        sim_context = getattr(simulator, "_sim", None)
        if sim_context is None:
            return
        sim_context.render()

    def _write_metadata(self) -> None:
        camera = self.env.crowdsim_robot_camera
        torch.save(
            {
                "fps": self.fps,
                "env_ids": self.env_ids,
                "layout": "per_env",
                "image_shape": camera.data.image_shape,
                "intrinsic_matrices": None
                if camera.data.intrinsic_matrices is None
                else camera.data.intrinsic_matrices.detach().cpu(),
            },
            self.session_dir / "metadata.pt",
        )

    def _env_dir(self, env_id: int) -> Path:
        if self.session_dir is None:
            raise RuntimeError("Camera recording session has not been started.")
        return self.session_dir / f"env_{env_id:04d}"

    def _save_frame(self) -> None:
        from PIL import Image

        camera = self.env.crowdsim_robot_camera
        output = camera.data.output
        if "rgb" not in output or "distance_to_image_plane" not in output:
            print("[CrowdSim] Camera output is not ready yet; skipping frame.")
            return

        rgb = output["rgb"].detach().cpu()
        depth = output["distance_to_image_plane"].detach().cpu()

        for env_id in self.env_ids:
            env_dir = self._env_dir(env_id)
            rgb_np = rgb[env_id].numpy()
            if rgb_np.dtype != "uint8":
                rgb_np = rgb_np.clip(0, 255).astype("uint8")
            Image.fromarray(rgb_np[..., :3]).save(env_dir / f"rgb_{self.frame_idx:06d}.png")
            torch.save(depth[env_id], env_dir / f"depth_{self.frame_idx:06d}.pt")


def configure_robot_camera_recorder(
    env,
    config: RobotCameraStreamConfig,
) -> RobotCameraRecorder | None:
    if not hasattr(env, "crowdsim_robot_camera"):
        return None

    recorder = RobotCameraRecorder(
        env=env,
        output_dir=config.output_dir.expanduser().resolve(),
        fps=config.fps,
        env_ids=parse_record_env_ids(config.env_ids, env.num_envs),
    )
    env.crowdsim_robot_camera_recorder = recorder

    import types

    original_step = env.step

    def step_with_camera_recording(self, action):
        result = original_step(action)
        recorder.step()
        return result

    env.step = types.MethodType(step_with_camera_recording, env)

    keyboard = getattr(env.simulator, "keyboard_interface", None)
    if keyboard is not None:
        try:
            keyboard.add_callback("Y", recorder.toggle)
            print("[CrowdSim] Press Y to start/stop robot camera recording.")
        except Exception as exc:
            print(f"[CrowdSim] Warning: failed to register Y recording key: {exc}")

    if config.auto_record:
        recorder.toggle()

    return recorder


def parse_record_env_ids(value: str, num_envs: int) -> list[int]:
    normalized = str(value).strip().lower()
    if normalized == "all":
        return list(range(num_envs))

    env_ids = [int(item.strip()) for item in normalized.split(",") if item.strip()]
    if not env_ids:
        raise ValueError("camera record env ids must contain at least one env id")
    for env_id in env_ids:
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(f"Camera record env id {env_id} outside [0, {num_envs})")
    return env_ids
