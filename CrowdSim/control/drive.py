"""Physical drive layer for CrowdSim robots: kinematics, camera sync."""

from __future__ import annotations

import math

import numpy as np
import torch

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Differential drive config and controller
# ---------------------------------------------------------------------------
MAX_LINEAR_SPEED = 1.0
MAX_ANGULAR_SPEED = 1.0
WHEEL_BASE = 0.413
WHEEL_RADIUS = 0.14
MAX_WHEEL_SPEED = (MAX_LINEAR_SPEED + 0.5 * WHEEL_BASE * MAX_ANGULAR_SPEED) / WHEEL_RADIUS
LEFT_WHEEL_JOINT_NAMES = ("joint_wheel_left",)
RIGHT_WHEEL_JOINT_NAMES = ("joint_wheel_right",)


@dataclass(frozen=True)
class DifferentialDriveConfig:
    max_linear_speed: float = MAX_LINEAR_SPEED
    max_angular_speed: float = MAX_ANGULAR_SPEED
    wheel_radius: float = WHEEL_RADIUS
    wheel_base: float = WHEEL_BASE
    max_wheel_speed: float = MAX_WHEEL_SPEED
    left_wheel_joint_names: tuple[str, ...] = LEFT_WHEEL_JOINT_NAMES
    right_wheel_joint_names: tuple[str, ...] = RIGHT_WHEEL_JOINT_NAMES


# ---------------------------------------------------------------------------

DEBUG_DRIVE = False
DEBUG_CONSTANT_COMMAND: tuple[float, float] | None = None


class DriveController:
    """Encapsulates all low-level robot articulation interaction."""

    def __init__(
        self,
        robot,                          # IsaacLab Articulation
        config,                         # CrowdNavigationConfig
        device: torch.device,
    ) -> None:
        self.robot = robot
        self.config = config
        self.device = device
        self._wheel_targets_tensor: torch.Tensor | None = None
        self._wheel_joint_ids: list[int] | None = None
        self._env_dt = 1.0 / 25.0
        self._actions = np.zeros((config.num_robots, 2), dtype=np.float32)
        self._prev_actions = np.zeros((config.num_robots, 2), dtype=np.float32)
        # Executed planar velocity from the deterministic kinematic update.
        # The robot root velocity written to PhysX is deliberately zero: the
        # next pose has already been integrated below, so a non-zero root
        # velocity would make PhysX integrate the same command a second time.
        # Observations, rewards and neighbour features must therefore read
        # these caches rather than robot.data.root_*_vel_w.
        self._executed_velocities_xy = np.zeros(
            (config.num_robots, 2), dtype=np.float32
        )
        self._executed_angular_velocities = np.zeros(
            config.num_robots, dtype=np.float32
        )
        # Optional collision-check callback set by CrowdNavigationConfig.attach().
        # Signature: (positions_np: (N, 2) world xy) -> bool ndarray (N,),
        # True for robots whose next pose would collide with a wall.
        self.collision_check: callable | None = None
        # Per-robot flag set each step by the wall guard: True for robots whose
        # requested move was blocked by a wall (i.e. they tried to translate
        # into an obstacle cell).  nav_manager reads this in post_step to flag
        # a collision — "guard blocked the move" IS the collision signal, more
        # reliable than re-querying the (now parked) pose via hits_wall, which
        # is noisy because the parked pose may or may not overlap a wall cell
        # depending on approach angle / grid discretisation.
        self.wall_blocked: np.ndarray = np.zeros(config.num_robots, dtype=bool)
        # Most recent pose written by _apply_kinematic (n_r, 7: x,y,z,qw,qx,qy,qz).
        # Used by _sync_prim_transforms so the camera-parented Car prim tracks
        # the written pose immediately, instead of the stale root_pos_w.
        self._last_written_poses: torch.Tensor | None = None

    # ---- setup ----

    def attach(self, env_dt: float) -> None:
        self._env_dt = env_dt
        if self.config.num_robots == 0:
            return
        available = list(getattr(self.robot, "joint_names", None)
                         or getattr(self.robot.data, "joint_names", []) or [])
        left_ids, left_names = self.robot.find_joints(
            list(self.config.differential_drive.left_wheel_joint_names), preserve_order=True)
        right_ids, right_names = self.robot.find_joints(
            list(self.config.differential_drive.right_wheel_joint_names), preserve_order=True)
        if not left_ids or not right_ids or len(left_ids) != 1 or len(right_ids) != 1:
            raise RuntimeError(
                f"Failed to resolve wheel joints. "
                f"left={list(zip(left_ids, left_names))} right={list(zip(right_ids, right_names))}")
        self._wheel_joint_ids = [int(left_ids[0]), int(right_ids[0])]
        self._wheel_targets_tensor = torch.zeros(
            self.config.num_robots, 2, dtype=torch.float32, device=self.device)
        print(f"[CrowdSim] Drive wheels: left={list(zip(left_ids, left_names))}, "
              f"right={list(zip(right_ids, right_names))}")

    # ---- action API ----

    def set_action(self, actions: torch.Tensor | np.ndarray) -> None:
        if self.config.num_robots == 0:
            return
        values = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions, dtype=np.float32)
        if values.shape != (self.config.num_robots, 2):
            raise ValueError(f"Expected ({self.config.num_robots}, 2) actions, got {values.shape}")
        self._prev_actions[:] = self._actions
        self._actions[:] = np.clip(values, -1.0, 1.0)

    def override_with_constant(self) -> None:
        if DEBUG_CONSTANT_COMMAND is None or self.config.num_robots == 0:
            return
        cmd = np.asarray(DEBUG_CONSTANT_COMMAND, dtype=np.float32)
        action = np.array([
            float(cmd[0]) / max(float(self.config.rl_max_linear_velocity), 1e-6),
            float(cmd[1]) / max(float(self.config.rl_max_angular_velocity), 1e-6),
        ], dtype=np.float32)
        self._actions[:] = np.clip(action, -1.0, 1.0)

    # ---- pre-step: apply to simulation ----

    def pre_step(self) -> None:
        if self.config.num_robots == 0:
            return
        self.override_with_constant()
        self._apply_kinematic()
        self._sync_prim_transforms()

    def post_step_cleanup(self) -> None:
        if self.config.num_robots == 0:
            return
        self._clear_wheel_targets()

    # ---- robot state queries ----

    def yaws(self) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros(0, dtype=np.float32)
        quats = self.robot.data.root_quat_w[:self.config.num_robots]
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).detach().cpu().numpy().astype(np.float32)

    def angular_velocities(self) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros(0, dtype=np.float32)
        return self._executed_angular_velocities.copy()

    def positions_xy(self) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self.robot.data.root_pos_w[:self.config.num_robots, :2].detach().cpu().numpy()

    def velocities_xy(self) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self._executed_velocities_xy.copy()

    def clear_state(self, robot_ids: np.ndarray) -> None:
        self._actions[robot_ids] = 0.0
        self._prev_actions[robot_ids] = 0.0
        self._executed_velocities_xy[robot_ids] = 0.0
        self._executed_angular_velocities[robot_ids] = 0.0

    def sync_teleported_poses(
        self, poses: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Publish externally teleported robot poses to the camera parent.

        Reset code writes articulation roots directly to PhysX.  IsaacLab's
        root-state buffers and the USD ``/Car`` prim do not necessarily reflect
        those writes until a later simulation update, while collectors may
        render immediately after reset.  Cache the exact written poses and
        synchronize the USD prim before that render so the camera never emits
        one frame from the pre-reset location.
        """
        value = torch.as_tensor(poses, dtype=torch.float32, device=self.device)
        if value.ndim != 2 or value.shape[1] != 7:
            raise ValueError(f"Expected teleported poses [N,7], got {tuple(value.shape)}")
        if env_ids is None:
            ids = torch.arange(value.shape[0], dtype=torch.long, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if len(ids) != len(value):
            raise ValueError("Teleported poses and env_ids must have matching lengths")
        if len(ids) and (int(ids.min()) < 0 or int(ids.max()) >= self.config.num_robots):
            raise ValueError("Teleported env_ids must refer to active CrowdSim robots")

        if self._last_written_poses is None:
            self._last_written_poses = torch.cat(
                (
                    self.robot.data.root_pos_w[: self.config.num_robots, :3].clone(),
                    self.robot.data.root_quat_w[: self.config.num_robots, :4].clone(),
                ),
                dim=1,
            ).to(self.device)
        self._last_written_poses[ids] = value
        self._sync_prim_transforms()

    # ---- low-level kinematics ----

    def _commands_from_actions(self) -> np.ndarray:
        cmds = np.zeros_like(self._actions, dtype=np.float32)
        cmds[:, 0] = np.clip(self._actions[:, 0], 0.0, 1.0) * float(self.config.rl_max_linear_velocity)
        cmds[:, 1] = np.clip(self._actions[:, 1], -1.0, 1.0) * float(self.config.rl_max_angular_velocity)
        return cmds

    def _apply_kinematic(self) -> None:
        self._clear_wheel_targets()
        cmds = self._commands_from_actions()
        yaws = self.yaws()
        dt = self._env_dt
        linear, angular = cmds[:, 0], cmds[:, 1]
        yaw_mid = yaws + 0.5 * angular * dt
        next_yaws = yaws + angular * dt
        n_r = self.config.num_robots
        # Only the first n_r envs hold active robots; the rest are frozen
        # filler slots that must keep their parked pose (don't move them).
        env_ids = torch.arange(n_r, dtype=torch.long, device=self.device)
        poses = torch.cat((self.robot.data.root_pos_w[:n_r, :3].clone(),
                           self.robot.data.root_quat_w[:n_r, :4].clone()), dim=1)

        # Compute tentative next xy for every active robot.
        cur_x = poses[:, 0].cpu().numpy()
        cur_y = poses[:, 1].cpu().numpy()
        next_x = cur_x + linear * np.cos(yaw_mid) * dt
        next_y = cur_y + linear * np.sin(yaw_mid) * dt

        # ── Wall-projection guard (path sweep) ─────────────────────
        # Sample along the cur→next segment (not just the endpoint) so a fast
        # robot can't skip over a thin wall in one step.  If ANY sample hits a
        # wall cell, freeze the robot at cur this step (yaw still updates so it
        # can turn toward free space) and zero its linear velocity so velocity-
        # based observations don't see phantom motion.
        eff_linear = linear.copy()
        cb = getattr(self, "collision_check", None)
        # Reset the blocked flag each step; set True below for any robot the
        # guard freezes.  Only counts as a collision when the robot actually
        # tried to move (linear > 0) — a stationary robot blocked at a wall
        # it's already touching shouldn't re-trigger collision every step.
        self.wall_blocked = np.zeros(n_r, dtype=bool)
        if cb is not None and n_r > 0:
            # Per-robot step length in metres; sample at ~half a map cell so no
            # wall thinner than one cell is skipped.
            cell = float(self.config.map_resolution) if hasattr(self.config, "map_resolution") else 0.05
            dx = next_x - cur_x
            dy = next_y - cur_y
            seg_len = np.sqrt(dx * dx + dy * dy)
            n_samples = np.maximum(2, np.ceil(seg_len / max(cell * 0.5, 1e-6)).astype(int))
            for i in range(n_r):
                if seg_len[i] <= 1e-8:
                    continue  # not moving — no wall collision this step
                ts = np.linspace(0.0, 1.0, n_samples[i])
                pts = np.stack([cur_x[i] + ts * dx[i], cur_y[i] + ts * dy[i]], axis=1)
                if np.any(cb(pts)):
                    next_x[i] = cur_x[i]
                    next_y[i] = cur_y[i]
                    eff_linear[i] = 0.0
                    self.wall_blocked[i] = True

        poses[:, 0] = torch.as_tensor(next_x, dtype=torch.float32, device=self.device)
        poses[:, 1] = torch.as_tensor(next_y, dtype=torch.float32, device=self.device)
        poses[:, 3:7] = self._yaw_to_quat(torch.as_tensor(next_yaws, dtype=torch.float32, device=self.device))
        self.robot.write_root_pose_to_sim(poses, env_ids=env_ids)

        # Cache the command that was actually executed.  A wall-blocked robot
        # has zero linear velocity but may still rotate away from the wall.
        self._executed_velocities_xy[:, 0] = eff_linear * np.cos(next_yaws)
        self._executed_velocities_xy[:, 1] = eff_linear * np.sin(next_yaws)
        self._executed_angular_velocities[:] = angular

        # The planar pose was already advanced exactly once above.  Keep the
        # PhysX root velocity at zero so env.step() cannot integrate the same
        # (v, omega) command again.  Physical contacts are still solved during
        # the simulation step; only commanded root motion is removed here.
        velocities = torch.zeros((n_r, 6), dtype=torch.float32, device=self.device)
        self.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)

        # Cache the pose just written so _sync_prim_transforms can propagate it
        # to the USD Car prim IMMEDIATELY.  Reading robot.data.root_pos_w here
        # would return the PRE-step value (it only refreshes after the next
        # physics step + robot.update()), leaving the camera — which parents to
        # the Car prim — one frame behind the robot's true pose.  This mismatch
        # is why the RGB camera view didn't line up with the robot's position.
        self._last_written_poses = poses.detach()

    def _clear_wheel_targets(self, env_ids: torch.Tensor | None = None) -> None:
        if self._wheel_targets_tensor is None or self._wheel_joint_ids is None:
            return
        if env_ids is None:
            # Default to active robots only — frozen filler slots (>= num_robots)
            # must keep their parked state, and the _wheel_targets_tensor itself
            # is sized (num_robots, 2), so writing all-envs would be out of range.
            env_ids = torch.arange(self.config.num_robots, dtype=torch.long, device=self.device)
        self._wheel_targets_tensor[env_ids] = 0.0
        targets = torch.zeros((int(env_ids.numel()), len(self._wheel_joint_ids)),
                              dtype=torch.float32, device=self.device)
        self.robot.set_joint_velocity_target(targets, joint_ids=self._wheel_joint_ids, env_ids=env_ids)

    def enable_prim_sync(self, enabled: bool = True) -> None:
        """Toggle per-step USD prim sync (default: True).

        Must stay True when depth camera is active: the camera prim
        (/Car/<mount>/front_cam) is a USD descendant of /Car and only follows
        the robot if _sync_prim_transforms() propagates the physics pose to USD
        each step.
        """
        self._prim_sync_enabled = enabled

    def _sync_prim_transforms(self) -> None:
        if not getattr(self, "_prim_sync_enabled", True):
            return
        try:
            from pxr import UsdGeom, Gf
            import omni.usd
        except ImportError:
            return
        # Prefer the pose just written by _apply_kinematic.  robot.data.root_pos_w
        # lags by one physics step (refreshed only after robot.update()), so
        # reading it here would sync the Car prim — and the camera parented to
        # it — to the PREVIOUS pose, making the RGB view mismatch the robot's
        # true position.  Fall back to root_pos_w only before the first step.
        if self._last_written_poses is not None:
            poses_np = self._last_written_poses.detach().cpu().numpy()
        else:
            poses_np = None
            root_pos = self.robot.data.root_pos_w.detach().cpu().numpy()
            root_quat = self.robot.data.root_quat_w.detach().cpu().numpy()
        stage = omni.usd.get_context().get_stage()
        for env_id in range(self.config.num_robots):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Car")
            if not prim.IsValid():
                continue
            xform = UsdGeom.Xformable(prim)
            if poses_np is not None:
                p = poses_np[env_id]          # [x, y, z, qw, qx, qy, qz]
                tx, ty, tz = float(p[0]), float(p[1]), float(p[2])
                qw, qx, qy, qz = float(p[3]), float(p[4]), float(p[5]), float(p[6])
            else:
                rp = root_pos[env_id]
                rq = root_quat[env_id]
                tx, ty, tz = float(rp[0]), float(rp[1]), float(rp[2])
                qw, qx, qy, qz = float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
            for op in xform.GetOrderedXformOps():
                name = op.GetName()
                if name == "xformOp:translate":
                    op.Set(Gf.Vec3d(tx, ty, tz))
                elif "orient" in name:
                    op.Set(Gf.Quatd(qw, qx, qy, qz))

    # ---- coordinate utilities ----

    @staticmethod
    def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), dtype=yaw.dtype, device=yaw.device)
        half = 0.5 * yaw
        quat[:, 0] = torch.cos(half)
        quat[:, 3] = torch.sin(half)
        return quat
