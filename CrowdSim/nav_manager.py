"""Navigation helpers for CrowdSim humanoid/robot scenes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from CrowdSim.control.drive import DifferentialDriveConfig, DriveController
from CrowdSim.ppo.goal_curriculum import PPOGoalCurriculum
from CrowdSim.utils.vision import resize_with_letterbox
from CrowdSim.world.map import (
    CollisionDetector, NavigationTask,
    build_agent_marker_prototypes,
)
from CrowdSim.world.sfm import Social_Force


@dataclass
class CrowdNavigationConfig:
    map_path: Path
    map_resolution: float
    free_threshold: int
    num_humanoids: int
    num_robots: int
    device: torch.device
    map_origin_xy: tuple[float, float] = (0.0, 0.0)
    seed: int = 7
    agent_radius: float = 0.35
    safe_distance: float = 0.9
    max_speed: float = 1.5
    waypoint_tolerance: float = 0.45
    goal_tolerance: float = 0.75
    min_start_goal_distance: float = 5.0
    max_start_goal_distance: float = 10.0
    min_spawn_spacing: float = 1.2
    planning_step_size: float = 0.5
    planning_clearance: float = 0.2
    neighbor_radius: float = 4.0
    humanoid_interaction_radius: float = 4.0
    collision_distance: float = 0.7
    log_interval: int = 120
    update_hz: float = 25.0
    trajectory_recording_enabled: bool = True
    trajectory_output_dir: Path = Path("output/crowdsim_navigation")
    differential_drive: DifferentialDriveConfig = field(default_factory=DifferentialDriveConfig)
    visual_markers_enabled: bool = False
    humanoid_target_enabled: bool = True
    local_target_timestep: float = 10.0
    humanoid_target_min_heading_speed: float = 0.05
    humanoid_velocity_smoothing: float = 0.3
    humanoid_max_acceleration: float = 1.5
    humanoid_max_yaw_rate: float = 1.5
    humanoid_prediction_horizon: float = 2.0
    humanoid_ttc_threshold: float = 1.5
    humanoid_ttc_gain: float = 12.0
    car_rl_policy: bool = True
    rl_num_neighbors: int = 4
    rl_max_linear_velocity: float = 1.0
    rl_max_angular_velocity: float = 1.0
    rl_initial_heading_min_offset_degrees: float = 0.0
    rl_initial_heading_max_offset_degrees: float = 0.0
    rl_goal_observation_max_distance: float = 10.0
    rl_goal_curriculum: dict = field(default_factory=dict)
    rl_progress_reward_scale: float = 4.0
    rl_goal_reward: float = 10.0
    rl_collision_penalty: float = -10.0
    rl_timeout_penalty: float = -5.0
    rl_time_penalty: float = -0.01
    rl_velocity_direction_reward_scale: float = 0.5
    rl_proximity_penalty_start: float = -1.5      # penalty at outer edge (negative, linear start)
    rl_proximity_penalty_threshold: float = 1.5   # outer boundary (m)
    rl_proximity_danger_threshold: float = 0.5    # inner danger boundary (m)
    rl_proximity_danger_penalty: float = -4.0     # penalty at inner edge (negative, linear end)
    rl_static_proximity_threshold: float = 0.8
    rl_static_proximity_danger_threshold: float = 0.25
    rl_static_proximity_danger_penalty: float = 0.0
    # Legacy combined L1 action-difference penalty for non-PPO configs.
    rl_action_smoothness_scale: float = 0.02
    rl_angular_velocity_scale: float = 0.0
    rl_angular_change_scale: float = 0.0
    rl_linear_change_scale: float = 0.0
    rl_max_episode_steps: int = 600
    rl_stuck_window: int = 300         # steps to look back for stuck detection
    rl_stuck_threshold: float = 0.1    # metres — robot is stuck if it moved less than this
    rl_stuck_penalty: float = -30.0    # high cost: stopping must not be cheaper than navigating
    rl_map_size: int = 24
    rl_map_extent: float = 8.0
    scene_objects: dict = field(default_factory=dict)
    rl_depth_enabled: bool = True
    rl_depth_size: int = 224
    rl_depth_max_range: float = 5.0
    rl_depth_use_vit: bool = False
    rl_rgb_enabled: bool = True
    rl_rgb_size: int = 224
    rl_rgb_use_vit: bool = False
    scenario: dict = field(default_factory=dict)


class CrowdNavigationManager:
    """Plan and control CrowdSim agents in the shared Office map.

    Humanoids are still controlled by the loaded MaskedMimic policy. This manager
    samples their starts/goals and monitors their progress/collisions. Navigation
    wheel commands are applied only to CrowdRobot/Jetbot agents.
    """

    def __init__(self, config: CrowdNavigationConfig) -> None:
        self.config = config
        if config.rl_depth_enabled and int(config.rl_depth_size) != 224:
            raise ValueError(
                "CrowdSim depth input is fixed at 224x224 for ViT compatibility; "
                f"got rl_depth_size={config.rl_depth_size}"
            )
        if not 0.0 <= config.humanoid_velocity_smoothing <= 1.0:
            raise ValueError("humanoid_velocity_smoothing must be in [0, 1]")
        if config.humanoid_max_acceleration <= 0.0:
            raise ValueError("humanoid_max_acceleration must be positive")
        if config.humanoid_max_yaw_rate <= 0.0:
            raise ValueError("humanoid_max_yaw_rate must be positive")
        if config.humanoid_prediction_horizon <= 0.0:
            raise ValueError("humanoid_prediction_horizon must be positive")
        if config.humanoid_ttc_threshold <= 0.0:
            raise ValueError("humanoid_ttc_threshold must be positive")
        if not (
            0.0 <= config.rl_initial_heading_min_offset_degrees
            <= config.rl_initial_heading_max_offset_degrees <= 180.0
        ):
            raise ValueError(
                "Robot initial heading offsets must satisfy "
                "0 <= min <= max <= 180 degrees"
            )
        if config.rl_goal_observation_max_distance <= 0.0:
            raise ValueError("Robot goal observation max distance must be positive")
        self.goal_curriculum = PPOGoalCurriculum(
            config, config.rl_goal_curriculum
        )
        self.num_agents = config.num_humanoids + config.num_robots
        self.task = NavigationTask(config, self.num_agents)
        self._heading_rng = np.random.default_rng(int(config.seed) + 104_729)

        self.free_mask = self.task.free_mask
        self.obstacle_map = self.task.obstacle_map
        self.height = self.task.height
        self.width = self.task.width
        self.pixels_per_meter = 1.0 / config.map_resolution
        self.starts_px = self.task.starts_px
        self.goals_px = self.task.goals_px
        self.starts_xy = self.task.starts_xy
        self.goals_xy = self.task.goals_xy
        self.paths_xy = self.task.paths_xy
        self._robot_spawn_yaws = np.zeros(config.num_robots, dtype=np.float32)
        for robot_id in range(config.num_robots):
            self._resample_robot_spawn_yaw(robot_id)
        self.waypoint_ids = np.ones(self.num_agents, dtype=np.int64)
        self.reached = np.zeros(self.num_agents, dtype=bool)
        self.collision_pairs: set[tuple[int, int]] = set()
        self.collision = CollisionDetector(
            self.obstacle_map, config.map_resolution,
            config.collision_distance, config.agent_radius,
            self.task.world_to_pixel, self.height, self.width)
        self.env_step_count = 0
        self.step_count = 0
        self._env_dt = 1.0 / 25.0
        self._update_interval_steps = self._compute_update_interval_steps(self._env_dt)
        self._last_positions = self.starts_xy.copy()
        self._has_completed_initial_reset = False
        self.drive: DriveController | None = None  # created in attach()
        self._sfm_waypoints = self.starts_xy.copy().astype(np.float32)
        self._humanoid_sfm_waypoints = self.starts_xy[: config.num_humanoids].copy().astype(np.float32)
        self._robot_progress_targets = self.goals_xy[
            config.num_humanoids : config.num_humanoids + config.num_robots
        ].copy().astype(np.float32)
        self._robot_prev_progress_dist = np.zeros(config.num_robots, dtype=np.float32)
        self._sfm_desired_velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._sfm_interact_forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._sfm_repulsive_forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._sfm_d_vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._sfm_ttc_forces = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._humanoid_smoothed_velocities = np.zeros(
            (config.num_humanoids, 2), dtype=np.float32
        )
        self._humanoid_actual_velocities = np.zeros(
            (config.num_humanoids, 2), dtype=np.float32
        )
        self._humanoid_future_first_targets = self.starts_xy[
            : config.num_humanoids
        ].copy().astype(np.float32)
        self._humanoid_future_first_yaws = np.zeros(
            config.num_humanoids, dtype=np.float32
        )
        self._humanoid_velocity_initialized = np.zeros(
            config.num_humanoids, dtype=bool
        )
        self._humanoid_target_yaws = np.zeros(config.num_humanoids, dtype=np.float32)
        self._humanoid_yaw_source = np.full(
            config.num_humanoids,
            "initial",
            dtype=object,
        )
        self._pending_path_updates: list[dict[str, object]] = []
        self._path_log_dirty = False
        self._pending_humanoid_reset_reasons: dict[int, str] = {}
        self._printed_masked_mimic_target_warning = False
        self._local_target_marker = None
        self.path_log_path = self._write_navigation_path_log()
        self.trajectory_log_path = self._open_trajectory_log()

        self.sfm_controller = self._make_sfm_controller(config.agent_radius)

        if config.scene_objects:
            self._paint_scene_objects_to_map(config.scene_objects)

        # Cached free pixel for parking frozen filler slots (envs beyond the
        # active humanoid/robot count).  Sampled once after scene objects are
        # painted so the MaskedMimic target and the physical spawn agree on
        # the same point — avoids frozen slots drifting from their spawn into
        # a wall or an active agent's path.
        self._frozen_slot_xy_cache: np.ndarray | None = None
        self._frozen_robot_park_cache: np.ndarray | None = None

        self._robot_rl_actions = np.zeros((config.num_robots, 2), dtype=np.float32)
        self._robot_rl_prev_actions = np.zeros((config.num_robots, 2), dtype=np.float32)
        self._robot_episode_steps = np.zeros(config.num_robots, dtype=np.int64)
        # Stuck detection: ring buffer of the last rl_stuck_window positions per robot.
        # Shape (num_robots, window, 2).  Slot index = episode_step % window.
        _stuck_w = max(1, int(config.rl_stuck_window))
        self._stuck_pos_buf = np.zeros((config.num_robots, _stuck_w, 2), dtype=np.float32)
        self._robot_last_obs: torch.Tensor | None = None
        self._robot_last_neighbors: torch.Tensor | None = None
        self._robot_last_neighbor_mask: torch.Tensor | None = None
        self._robot_last_depth: torch.Tensor | None = None
        self._robot_last_map: torch.Tensor | None = None
        self._robot_last_rgb: torch.Tensor | None = None
        self._robot_last_rewards = torch.zeros(config.num_robots, device=config.device)
        self._robot_last_dones = torch.zeros(config.num_robots, dtype=torch.bool, device=config.device)
        self._robot_last_info: dict[str, torch.Tensor] = {}

    @property
    def humanoid_starts_xy(self) -> torch.Tensor:
        values = self.starts_xy[: self.config.num_humanoids]
        return torch.tensor(values, dtype=torch.float32, device=self.config.device)

    def _frozen_slot_xy(self) -> np.ndarray:
        """A free map pixel for parking frozen filler slots (cached).

        Sampled once (after scene objects are painted) so the MaskedMimic
        target and the physical spawn agree on the same point.  Rejects
        candidates within ``min_spawn_spacing`` of any active agent's
        start/goal so the parked slot doesn't sit on a navigation route.
        """
        if self._frozen_slot_xy_cache is None:
            # Active agent anchor points to avoid (starts + goals).
            anchors = np.concatenate([
                self.starts_xy[:self.num_agents],
                self.goals_xy[:self.num_agents],
            ], axis=0) if self.num_agents > 0 else np.zeros((0, 2), dtype=np.float32)
            min_dist = float(self.config.min_spawn_spacing)
            best = None
            best_dist = -1.0
            for _ in range(100):
                px = self.task._sample_free_pixel()
                if px is None:
                    continue
                cand = np.asarray(self.task.pixel_to_world(px), dtype=np.float32)
                if anchors.shape[0] > 0:
                    d = float(np.linalg.norm(anchors - cand, axis=1).min())
                else:
                    d = float("inf")
                if d >= min_dist:
                    best = cand
                    break
                if d > best_dist:
                    best_dist, best = d, cand
            self._frozen_slot_xy_cache = best if best is not None else np.array(
                [float(self.config.map_origin_xy[0]),
                 float(self.config.map_origin_xy[1])],
                dtype=np.float32,
            )
        return self._frozen_slot_xy_cache

    def humanoid_spawn_xy_for_all_envs(self, num_envs: int) -> torch.Tensor:
        """Return (num_envs, 2) humanoid spawn XY for every IsaacLab env slot.

        Active humanoids (0..num_humanoids-1) get their real start; frozen
        filler slots are parked at a sampled free pixel.  ``apply_fixed_spawn_offsets``
        requires one XY per env.
        """
        n_h = self.config.num_humanoids
        poses = np.zeros((num_envs, 2), dtype=np.float32)
        if n_h > 0:
            poses[:n_h] = self.starts_xy[:n_h]
        if num_envs > n_h:
            poses[n_h:] = self._frozen_slot_xy()
        return torch.tensor(poses, dtype=torch.float32, device=self.config.device)

    @property
    def robot_starts_xy_yaw(self) -> torch.Tensor:
        start = self.config.num_humanoids
        values = np.zeros((self.config.num_robots, 3), dtype=np.float32)
        values[:, :2] = self.starts_xy[start : start + self.config.num_robots]
        values[:, 2] = self._initial_robot_yaws()
        return torch.tensor(values, dtype=torch.float32, device=self.config.device)

    def _frozen_robot_park_xy(self) -> np.ndarray:
        """A free map pixel for parking frozen robot filler slots (cached).

        Sampled to avoid active agents' starts/goals AND the frozen-humanoid
        park point (``_frozen_slot_xy``) so a frozen robot doesn't sit on top
        of a frozen humanoid when an env holds both.  Falls back to the
        humanoid park point offset by 2*agent_radius if no separate free
        pixel is found.
        """
        if self._frozen_robot_park_cache is None:
            anchors = np.concatenate([
                self.starts_xy[:self.num_agents],
                self.goals_xy[:self.num_agents],
            ], axis=0) if self.num_agents > 0 else np.zeros((0, 2), dtype=np.float32)
            # Also avoid the frozen-humanoid park point.
            hum_park = self._frozen_slot_xy()
            anchors = np.concatenate([anchors, hum_park[None, :]], axis=0)
            min_dist = float(self.config.min_spawn_spacing)
            best = None
            best_dist = -1.0
            for _ in range(100):
                px = self.task._sample_free_pixel()
                if px is None:
                    continue
                cand = np.asarray(self.task.pixel_to_world(px), dtype=np.float32)
                d = float(np.linalg.norm(anchors - cand, axis=1).min())
                if d >= min_dist:
                    best = cand
                    break
                if d > best_dist:
                    best_dist, best = d, cand
            if best is None:
                best = hum_park + np.array(
                    [2.0 * float(self.config.agent_radius), 0.0], dtype=np.float32)
            self._frozen_robot_park_cache = best
        return self._frozen_robot_park_cache

    def robot_spawn_poses_for_all_envs(self, num_envs: int) -> torch.Tensor:
        """Return (num_envs, 3) robot spawn poses for every IsaacLab env slot.

        Active robots (indices 0..num_robots-1) get their real start pose;
        frozen filler slots (num_robots..num_envs-1) are parked at a sampled
        free pixel (``_frozen_robot_park_xy``) so they sit out the episode
        without colliding with active agents, the frozen humanoid, or walls.
        ``apply_fixed_crowd_robot_spawns`` requires one pose per env.
        """
        n_r = self.config.num_robots
        poses = np.zeros((num_envs, 3), dtype=np.float32)
        if n_r > 0:
            active = self.robot_starts_xy_yaw.cpu().numpy()
            poses[:n_r] = active
        if num_envs > n_r:
            park = self._frozen_robot_park_xy()
            poses[n_r:, :2] = park
            poses[n_r:, 2] = 0.0
        return torch.tensor(poses, dtype=torch.float32, device=self.config.device)

    def attach(self, env) -> None:
        self.env = env
        self.robot = getattr(env, "crowdsim_robot", None)
        self._env_dt = self._read_env_dt(env)
        self._update_interval_steps = self._compute_update_interval_steps(self._env_dt)
        self._refresh_controller_dt()
        if self.config.num_robots > 0 and self.robot is None:
            raise RuntimeError("Navigation requested robot control, but crowdsim_robot is missing.")
        if self.config.num_robots > 0:
            self.drive = DriveController(self.robot, self.config, self.config.device)
            self.drive.attach(self._env_dt)
            # Hook the occupancy-map collision checker into the drive so
            # _apply_kinematic refuses to translate a robot into a wall cell.
            # Reuses CollisionDetector.hits_wall (radius-inflated wall query)
            # so the scan logic lives in one place.
            if self.collision is not None:
                self.drive.collision_check = self.collision.hits_wall

        import types

        original_reset = env.reset
        original_step = env.step

        def reset_with_navigation(env_self, *args, **kwargs):
            env_ids = self._reset_env_ids_from_args(args, kwargs)

            # Decide whether this is the very first full reset
            initial_full_reset = (
                len(env_ids) > 0
                and not self._has_completed_initial_reset
                and len(env_ids) == self.env.num_envs
                and np.array_equal(env_ids, np.arange(self.env.num_envs))
            )

            # For non-initial resets, replan BEFORE original_reset() so that
            # _crowdsim_desired_root_xy is already set when the env teleports.
            # This ensures the humanoid's physical position matches the new path start.
            if len(env_ids) > 0 and not initial_full_reset:
                self._reset_navigation_agents(env_ids)

            result = original_reset(*args, **kwargs)

            if len(env_ids) > 0:
                if initial_full_reset:
                    self._has_completed_initial_reset = True
                    if self.config.num_robots > 0 and hasattr(env, "_crowdsim_robot_spawn_poses"):
                        # _crowdsim_robot_spawn_poses is (num_envs, 7); write all env
                        # slots (frozen ones are parked at a corner).  Velocity must
                        # match num_envs too — using num_robots here crashes when
                        # num_envs != num_robots (write_root_velocity_to_sim with
                        # env_ids=None indexes all instances).
                        self.robot.write_root_pose_to_sim(env._crowdsim_robot_spawn_poses)
                        self.robot.write_root_velocity_to_sim(
                            torch.zeros(env.num_envs, 6, device=self.config.device))
                        self.drive.sync_teleported_poses(
                            env._crowdsim_robot_spawn_poses[: self.config.num_robots]
                        )
                    # 机器人位姿写入完成后，统一在下方读取一次，不提前记录

                positions, velocities = self._read_agent_state()
                self._compute_sfm_reference_waypoints(positions, velocities)
                self._update_local_target_markers()
                self._last_positions = positions
                # 无论初始还是后续 reset，都只记录一次轨迹帧。
                # Record immediately: agent is now at the (new) start, and
                # _pending_path_updates has the matching path data.
                self._record_trajectory_frame(positions, velocities)
            return result

        def step_with_navigation(env_self, action):
            should_update_navigation = self._should_update_navigation()
            if should_update_navigation:
                self.pre_step()
            if self.drive is not None:
                self.drive.pre_step()
            result = original_step(action)
            self.env_step_count += 1
            # Suppress resets on frozen filler humanoid slots (env_id >=
            # num_humanoids).  ProtoMotions sets reset_buf on these slots via
            # max_episode_length / tracking_error / fall-detection, which would
            # teleport the parked humanoid away from its corner and pollute the
            # active robot's physics + observations.  no_done_tracks only blocks
            # clip-done; this clears the remaining termination sources so frozen
            # slots stay parked for the whole run.
            n_h = self.config.num_humanoids
            if n_h < self.env.num_envs:
                self.env.reset_buf[n_h:] = False
                if hasattr(self.env, "progress_buf"):
                    # Keep progress_buf from hitting max_episode_length again
                    # immediately by clamping frozen slots under the limit.
                    self.env.progress_buf[n_h:] = 0
            if self.drive is not None:
                self.drive.post_step_cleanup()
            if should_update_navigation:
                self.post_step()
            else:
                self._clear_robot_rl_env_step_feedback()
            # Update the SMPL human mesh overlay (if enabled) so the depth/rgb
            # cameras see textured humans matching the current pose.  Runs every
            # step after physics; visual only, no effect on training dynamics.
            adapter = getattr(self, "_human_mesh_adapter", None)
            if adapter is not None:
                try:
                    adapter.update()
                except Exception:  # noqa: BLE001
                    pass
            return result

        env.reset = types.MethodType(reset_with_navigation, env)
        env.step = types.MethodType(step_with_navigation, env)
        env.crowdsim_navigation = self
        self._attach_masked_mimic_navigation_targets(env)
        self.task.create_visualization_markers(
            num_humanoids=self.config.num_humanoids,
            enabled=self.config.visual_markers_enabled
            and not getattr(env.simulator, "headless", True),
        )
        self._create_local_target_markers(
            enabled=self.config.visual_markers_enabled
            and not getattr(env.simulator, "headless", True)
        )
        print(
            f"[CrowdSim] Navigation enabled: {self.config.num_humanoids} humanoid(s), "
            f"{self.config.num_robots} robot(s), car_rl_policy={self.config.car_rl_policy}."
        )
        print(
            "[CrowdSim] Navigation update rate: "
            f"{self._navigation_update_hz():.3g} Hz "
            f"(env_dt={self._env_dt:.6f}s, every {self._update_interval_steps} env step(s))."
        )
        if self.config.humanoid_target_enabled:
            print(
                "[CrowdSim] Humanoid navigation targets enabled: "
                "controller=sfm, "
                "format=Pelvis translation + Pelvis rotation."
            )
        print(f"[CrowdSim] Navigation path log: {self.path_log_path}")
        if self._trajectory_log_file is not None:
            print(f"[CrowdSim] Navigation trajectory log: {self.trajectory_log_path}")

    def pre_step(self) -> None:
        positions, velocities = self._read_agent_state()
        self._compute_sfm_reference_waypoints(positions, velocities)
        self._update_robot_progress_targets(positions)
        self._update_local_target_markers()

    def post_step(self) -> None:
        self.step_count += 1
        positions, velocities = self._read_agent_state()
        self._update_waypoints_and_goals(positions)
        new_pairs = self._detect_collisions(positions)
        nav_done_ids = self._navigation_done_agent_ids(new_pairs)
        self._request_humanoid_resets(nav_done_ids)
        self._update_robot_rl_feedback(positions, velocities, new_pairs)
        self._debug_drive()
        self._last_positions = positions
        self.env.extras["crowdsim_navigation"] = {
            "reached": int(self.reached.sum()),
            "num_agents": self.num_agents,
            "collision_pairs": len(self.collision_pairs),
            "new_collision_pairs": len(new_pairs),
            "navigation_done_agents": len(nav_done_ids),
            "update_hz": self._navigation_update_hz(),
        }
        if self.config.num_robots > 0:
            n_h = self.config.num_humanoids
            n_r = self.config.num_robots
            goal_distances = np.linalg.norm(
                self.goals_xy[n_h : n_h + n_r] - positions[n_h :],
                axis=1,
            )
            self.env.extras["crowdsim_navigation"]["robot_goal_distance_mean"] = float(
                goal_distances.mean()
            )

        self._record_trajectory_frame(positions, velocities)

    def _initial_robot_yaws(self) -> np.ndarray:
        """Return the cached yaw sampled when each robot route was created."""
        return self._robot_spawn_yaws.copy()

    def _resample_robot_spawn_yaw(self, robot_id: int) -> None:
        """Face the goal with a balanced signed offset for turning diversity."""
        offset = self.config.num_humanoids
        start = self.starts_xy[offset + robot_id]
        goal = self.goals_xy[offset + robot_id]
        delta = goal - start
        goal_yaw = (
            math.atan2(float(delta[1]), float(delta[0]))
            if float(np.linalg.norm(delta)) > 1e-4 else 0.0
        )
        minimum = math.radians(
            self.config.rl_initial_heading_min_offset_degrees
        )
        maximum = math.radians(
            self.config.rl_initial_heading_max_offset_degrees
        )
        magnitude = float(self._heading_rng.uniform(minimum, maximum))
        sign = -1.0 if int(self._heading_rng.integers(0, 2)) == 0 else 1.0
        self._robot_spawn_yaws[robot_id] = math.atan2(
            math.sin(goal_yaw + sign * magnitude),
            math.cos(goal_yaw + sign * magnitude),
        )

    def _write_navigation_path_log(self) -> Path:
        # Use the same directory as the trajectory log so both files are
        # co-located under the training run's output folder.
        output_dir = Path(self.config.trajectory_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = output_dir / "paths_latest.json"
        timestamp_path = output_dir / f"paths_{timestamp}.json"

        records = [self._navigation_path_record(agent_id) for agent_id in range(self.num_agents)]
        self._path_log_dirty = False
        self.path_log_path = latest_path

        payload = {
            "created_at": timestamp,
            "map_path": str(self.config.map_path),
            "map_resolution": self.config.map_resolution,
            "map_origin_xy": list(self.config.map_origin_xy),
            "free_threshold": self.config.free_threshold,
            "planning_step_size": self.config.planning_step_size,
            "planning_clearance": self.config.planning_clearance,
            "navigation_update_hz": self._navigation_update_hz(),
            "local_target_timestep": self.config.local_target_timestep,
            "num_humanoids": self.config.num_humanoids,
            "num_cars": self.config.num_robots,
            "agents": records,
        }
        text = json.dumps(payload, indent=2)
        latest_path.write_text(text, encoding="utf-8")
        timestamp_path.write_text(text, encoding="utf-8")
        return latest_path

    def _open_trajectory_log(self) -> Path | None:
        self._trajectory_log_file = None
        self._trajectory_timestamp_file = None
        if not self.config.trajectory_recording_enabled:
            return None

        output_dir = Path(self.config.trajectory_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = output_dir / "trajectory_latest.jsonl"
        timestamp_path = output_dir / f"trajectory_{timestamp}.jsonl"
        self._trajectory_log_file = latest_path.open("w", encoding="utf-8")
        self._trajectory_timestamp_file = timestamp_path.open("w", encoding="utf-8")
        metadata = {
            "type": "metadata",
            "created_at": timestamp,
            "path_log": str(self.path_log_path),
            "timestamp_path": str(timestamp_path),
            "map_path": str(self.config.map_path),
            "map_resolution": self.config.map_resolution,
            "map_origin_xy": list(self.config.map_origin_xy),
            "num_humanoids": self.config.num_humanoids,
            "num_cars": self.config.num_robots,
            "num_agents": self.num_agents,
            "navigation_update_hz": self._navigation_update_hz(),
            "local_target_timestep": self.config.local_target_timestep,
        }
        line = json.dumps(metadata) + "\n"
        self._trajectory_log_file.write(line)
        self._trajectory_timestamp_file.write(line)
        self._trajectory_log_file.flush()
        self._trajectory_timestamp_file.flush()
        return latest_path

    def _close_trajectory_logs(self) -> None:
        """Close trajectory log file handles (call on normal/abnormal exit)."""
        for f in (self._trajectory_log_file, self._trajectory_timestamp_file):
            if f is not None and not f.closed:
                f.flush()
                f.close()

    def __del__(self) -> None:
        self._close_trajectory_logs()

    def _navigation_path_record(self, agent_id: int) -> dict[str, object]:
        agent_type = "humanoid" if agent_id < self.config.num_humanoids else "car"
        local_id = agent_id if agent_type == "humanoid" else agent_id - self.config.num_humanoids
        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "local_id": local_id,
            "start_xy": self.starts_xy[agent_id].astype(float).tolist(),
            "goal_xy": self.goals_xy[agent_id].astype(float).tolist(),
            "path_xy": self.paths_xy[agent_id].astype(float).tolist(),
        }

    def _consume_pending_path_updates(self) -> list[dict[str, object]]:
        if not self._pending_path_updates:
            return []
        updates = self._pending_path_updates
        self._pending_path_updates = []
        return updates

    def _flush_navigation_path_log_if_dirty(self) -> None:
        if not self._path_log_dirty:
            return
        self._write_navigation_path_log()

    def _record_trajectory_frame(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        if self._trajectory_log_file is None:
            return

        current_waypoints = np.asarray(
            [self._current_waypoint(agent_id) for agent_id in range(self.num_agents)],
            dtype=np.float32,
        )
        frame = {
            "type": "frame",
            "step": int(self.step_count),
            "env_step": int(self.env_step_count),
            "time": float(self.step_count * self._dt()),
            "positions_xy": positions.astype(float).tolist(),
            "velocities_xy": velocities.astype(float).tolist(),
            "current_waypoints_xy": current_waypoints.astype(float).tolist(),
            "goals_xy": self.goals_xy.astype(float).tolist(),
            "waypoint_ids": self.waypoint_ids.astype(int).tolist(),
            "local_targets_xy": self._sfm_waypoints.astype(float).tolist(),
            "sfm_desired_velocities_xy": self._sfm_desired_velocities.astype(float).tolist(),
            "sfm_ttc_forces_xy": self._sfm_ttc_forces.astype(float).tolist(),
            "humanoid_smoothed_velocities_xy": self._humanoid_smoothed_velocities.astype(float).tolist(),
            "humanoid_future_first_targets_xy": self._humanoid_future_first_targets.astype(float).tolist(),
            "humanoid_future_first_yaws": self._humanoid_future_first_yaws.astype(float).tolist(),
            "sfm_interact_forces_xy": self._sfm_interact_forces.astype(float).tolist(),
            "sfm_repulsive_forces_xy": self._sfm_repulsive_forces.astype(float).tolist(),
            "sfm_d_vel_xy": self._sfm_d_vel.astype(float).tolist(),
            "humanoid_yaw_source": self._humanoid_yaw_source.astype(str).tolist(),
            "reached": self.reached.astype(bool).tolist(),
            "collision_pairs": [list(pair) for pair in sorted(self.collision_pairs)],
        }
        pending_before = len(self._pending_path_updates)
        path_updates = self._consume_pending_path_updates()
        if path_updates:
            frame["path_updates"] = path_updates
        elif pending_before > 0:
            print(f"[CrowdSim] TRAJECTORY MISS: {pending_before} pending NOT written step={int(self.step_count)}")
        line = json.dumps(frame) + "\n"
        self._trajectory_log_file.write(line)
        if self._trajectory_timestamp_file is not None:
            self._trajectory_timestamp_file.write(line)
            self._trajectory_timestamp_file.flush()
        self._trajectory_log_file.flush()

    def _read_agent_state(self) -> tuple[np.ndarray, np.ndarray]:
        humanoid_state = self.env.simulator.get_root_state()
        # root_pos is (num_envs, 3) — only the first num_humanoids slots are
        # active; the rest are frozen filler slots that must not leak into the
        # shared positions array (they would pollute SFM / obs / collision).
        n_h = self.config.num_humanoids
        humanoid_pos = humanoid_state.root_pos[:n_h, :2].detach().cpu().numpy()
        humanoid_vel = humanoid_state.root_vel[:n_h, :2].detach().cpu().numpy()
        if self.config.num_robots == 0:
            return humanoid_pos, humanoid_vel
        # drive.positions_xy() already slices [:num_robots] internally.
        if self.drive is not None:
            return (
                np.concatenate([humanoid_pos, self.drive.positions_xy()], axis=0),
                np.concatenate([humanoid_vel, self.drive.velocities_xy()], axis=0),
            )
        robot_pos = self.robot.data.root_pos_w[:self.config.num_robots, :2].detach().cpu().numpy()
        robot_vel = self.robot.data.root_lin_vel_w[:self.config.num_robots, :2].detach().cpu().numpy()
        return (
            np.concatenate([humanoid_pos, robot_pos], axis=0),
            np.concatenate([humanoid_vel, robot_vel], axis=0),
        )

    def _update_robot_progress_targets(self, positions: np.ndarray) -> None:
        if self.config.num_robots == 0:
            return
        robot_offset = self.config.num_humanoids
        robot_agent_ids = np.arange(robot_offset, robot_offset + self.config.num_robots)
        robot_positions = positions[robot_offset : robot_offset + self.config.num_robots]
        # Fixed goal target — RL policy directly outputs velocity commands, no SFM needed.
        self._robot_progress_targets[:] = self.goals_xy[robot_agent_ids]
        self._robot_prev_progress_dist[:] = np.linalg.norm(
            robot_positions - self._robot_progress_targets,
            axis=1,
        ).astype(np.float32)

    def _compute_sfm_reference_waypoints(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        self._sfm_waypoints[: self.num_agents] = positions[: self.num_agents]
        # local_target_timestep is consistently a simulation-step multiplier.
        # The configured multiplier preserves a 0.333 s locomotion target.
        target_timestep = max(float(self.config.local_target_timestep) * self._dt(), self._dt())
        self._sfm_desired_velocities.fill(0.0)
        self._sfm_interact_forces.fill(0.0)
        self._sfm_repulsive_forces.fill(0.0)
        self._sfm_d_vel.fill(0.0)
        self._sfm_ttc_forces.fill(0.0)
        if self.config.num_humanoids > 0:
            self._humanoid_sfm_waypoints[: self.config.num_humanoids] = positions[: self.config.num_humanoids]
            self._humanoid_actual_velocities[:] = velocities[
                : self.config.num_humanoids
            ]

        # SFM only for humanoids — cars use RL policy with fixed goal targets.
        for agent_id in range(self.config.num_humanoids):
            if self.reached[agent_id]:
                self._humanoid_yaw_source[agent_id] = "reached"
                continue

            pos = positions[agent_id]
            vel = velocities[agent_id]
            goal = self._current_waypoint(agent_id)
            nbr_state = self._neighbor_state(agent_id, positions, velocities)
            cord_int = self._clip_pixel_yx(self.world_to_pixel(pos))
            desired_vel, force_terms = self.sfm_controller.get_action(
                (pos, cord_int, vel, goal),
                (nbr_state[0], nbr_state[1], nbr_state[2], nbr_state[3]),
            )
            desired_vel = np.asarray(desired_vel, dtype=np.float32)
            if not np.all(np.isfinite(desired_vel)):
                desired_vel = np.zeros(2, dtype=np.float32)
            previous = (
                self._humanoid_smoothed_velocities[agent_id]
                if self._humanoid_velocity_initialized[agent_id]
                else np.asarray(vel, dtype=np.float32)
            )
            desired_vel = self._filter_humanoid_velocity(
                previous,
                desired_vel,
                smoothing=self.config.humanoid_velocity_smoothing,
                max_acceleration=self.config.humanoid_max_acceleration,
                dt=self._dt(),
            )
            self._humanoid_smoothed_velocities[agent_id] = desired_vel
            self._humanoid_velocity_initialized[agent_id] = True
            self._sfm_desired_velocities[agent_id] = desired_vel
            self._sfm_interact_forces[agent_id] = np.asarray(force_terms[0], dtype=np.float32)
            self._sfm_repulsive_forces[agent_id] = np.asarray(force_terms[1], dtype=np.float32)
            self._sfm_d_vel[agent_id] = np.asarray(force_terms[2], dtype=np.float32)
            self._sfm_ttc_forces[agent_id] = np.asarray(
                force_terms[3], dtype=np.float32
            )
            sfm_waypoint = (pos + desired_vel * target_timestep).astype(np.float32)
            self._sfm_waypoints[agent_id] = sfm_waypoint
            self._humanoid_sfm_waypoints[agent_id] = sfm_waypoint

            heading_delta = sfm_waypoint - pos
            if np.linalg.norm(heading_delta) >= self.config.humanoid_target_min_heading_speed:
                self._humanoid_target_yaws[agent_id] = math.atan2(
                    float(heading_delta[1]), float(heading_delta[0])
                )
                self._humanoid_yaw_source[agent_id] = "sfm_target"
            else:
                fallback = self._waypoint_desired_velocity(pos, goal)
                if np.linalg.norm(fallback) >= 1e-5:
                    self._humanoid_target_yaws[agent_id] = math.atan2(
                        float(fallback[1]), float(fallback[0])
                    )
                    self._humanoid_yaw_source[agent_id] = "waypoint_fallback"
                else:
                    self._humanoid_yaw_source[agent_id] = "previous"

    def _attach_masked_mimic_navigation_targets(self, env) -> None:
        # Install hooks when there are active humanoids to steer OR frozen
        # filler slots to pin (num_envs > num_humanoids).  The latter matters
        # even when num_humanoids == 0: a frozen humanoid slot still exists
        # (num_envs = max(num_robots, 1) >= 1) and would otherwise wander
        # under default MaskedMimic replay, colliding with the active robot.
        has_active = self.config.humanoid_target_enabled and self.config.num_humanoids > 0
        has_frozen = self.env is not None and self.env.num_envs > self.config.num_humanoids
        if not has_active and not has_frozen:
            return
        control_manager = getattr(env, "control_manager", None)
        component = getattr(control_manager, "components", {}).get("masked_mimic")
        if component is None:
            print("[CrowdSim] Humanoid navigation targets skipped: masked_mimic control not found.")
            return

        import types

        original_populate_context = component.populate_context

        def populate_context_with_navigation_targets(component_self, ctx):
            original_populate_context(ctx)
            self._override_masked_mimic_context(component_self, ctx)

        component.populate_context = types.MethodType(populate_context_with_navigation_targets, component)

        # Disable motion-clip-done resets so humanoids loop forever.
        motion_manager = getattr(env, "motion_manager", None)
        if motion_manager is not None:
            num_humanoids = self.config.num_humanoids
            device = self.config.device

            original_get_done = motion_manager.get_done_tracks

            def no_done_tracks(_self, env_ids=None):
                done = original_get_done(env_ids=env_ids)
                # Suppress clip-done for ALL humanoid slots (active + frozen):
                # active slots loop their navigation motion forever, frozen
                # slots must stay parked at the corner instead of being
                # teleported around by clip-done resets.
                done[:] = False
                return done

            motion_manager.get_done_tracks = types.MethodType(
                no_done_tracks, motion_manager,
            )
            print("[CrowdSim] Humanoid motion_clip_done disabled (motion loops forever).")

    def _override_masked_mimic_context(self, component, ctx) -> None:
        base = getattr(ctx, "masked_mimic", None)
        if base is None:
            return

        try:
            from protomotions.envs.context_views import MaskedMimicContext
        except ImportError:
            return

        conditionable_body_ids = getattr(component, "conditionable_body_ids", None)
        if conditionable_body_ids is None:
            self._warn_masked_mimic_target_once("conditionable_body_ids missing.")
            return

        pelvis_body_id = int(getattr(component.env.robot_config, "anchor_body_index", 0))
        pelvis_matches = (conditionable_body_ids == pelvis_body_id).nonzero(as_tuple=False)
        if pelvis_matches.numel() == 0:
            self._warn_masked_mimic_target_once(
                f"pelvis body id {pelvis_body_id} is not conditionable."
            )
            return

        num_envs, num_future_steps = base.ref_pos.shape[:2]
        num_humanoids = min(self.config.num_humanoids, num_envs)
        n_frozen = num_envs - num_humanoids
        # Need to install targets when there are active humanoids to steer OR
        # frozen filler slots to pin.  When both are zero (num_envs == 0, never
        # happens in practice) there's nothing to do.
        if num_humanoids <= 0 and n_frozen <= 0:
            return

        device = base.ref_pos.device
        dtype = base.ref_pos.dtype
        offsets_np = self._humanoid_target_offsets(num_future_steps)
        offsets = torch.as_tensor(offsets_np, dtype=dtype, device=device)

        ref_pos = base.ref_pos.clone()
        ref_rot = base.ref_rot.clone()
        target_times = base.target_times.clone()
        time_offsets = base.time_offsets.clone()
        target_bodies_masks = torch.zeros_like(base.target_bodies_masks)
        target_poses_masks = torch.zeros_like(base.target_poses_masks)

        pelvis_condition_index = int(pelvis_matches[0].item())
        masks = target_bodies_masks.view(
            num_envs,
            num_future_steps,
            int(getattr(component, "num_conditionable_bodies")),
            2,
        )

        # ── Active humanoid slots (0..num_humanoids-1): steer to path ──
        if num_humanoids > 0:
            current_pelvis = ctx.current.rigid_body_pos[:num_humanoids, pelvis_body_id, :]
            xy_targets_np, target_yaws_np = self._humanoid_future_targets_from_path(
                current_pelvis[:, :2].detach().cpu().numpy(),
                offsets_np,
            )
            xy_targets = torch.as_tensor(xy_targets_np, dtype=dtype, device=device)
            yaws = torch.as_tensor(target_yaws_np, dtype=dtype, device=device)
            active = torch.as_tensor(
                ~self.reached[:num_humanoids],
                dtype=torch.bool,
                device=device,
            )

            ref_pos[:num_humanoids, :, pelvis_body_id, :2] = xy_targets
            ref_pos[:num_humanoids, :, pelvis_body_id, 2] = current_pelvis[:, None, 2]

            yaw_quat = self._yaw_to_quat_xyzw_tensor(yaws.reshape(-1)).to(dtype=dtype)
            ref_rot[:num_humanoids, :, pelvis_body_id, :] = yaw_quat.view(
                num_humanoids, num_future_steps, 4
            )

            masks[:num_humanoids, :, pelvis_condition_index, 0] = active[:, None]
            masks[:num_humanoids, :, pelvis_condition_index, 1] = active[:, None]
            target_poses_masks[:num_humanoids, :] = active[:, None]

            if hasattr(component.env, "motion_manager"):
                motion_times = component.env.motion_manager.motion_times.to(device=device, dtype=dtype)
                target_times[:num_humanoids, :] = motion_times[:num_humanoids, None] + offsets[None, :]
            else:
                target_times[:num_humanoids, :] = offsets[None, :]
            time_offsets[:num_humanoids, :] = offsets[None, :]

        # ── Freeze filler humanoid slots (env_id >= num_humanoids) ────
        # These slots have no navigation goal; without a conditioning target
        # MaskedMimic falls back to random motion-clip replay and wanders out
        # of the corner, physically colliding with the active robot sharing
        # the env.  Pin them to the same free pixel used as their physical
        # spawn (see _frozen_slot_xy) so target and spawn agree, with an
        # active mask so the policy holds them in place.
        if n_frozen > 0:
            park_xy = self._frozen_slot_xy()
            corner_xy = torch.as_tensor(
                [float(park_xy[0]), float(park_xy[1])],
                dtype=dtype, device=device,
            )
            slot = slice(num_humanoids, num_envs)
            # Use the frozen slots' CURRENT pelvis position/rotation so the
            # MaskedMimic target matches their actual state (Δ≈0).  Hard-coding
            # yaw=0 or z=0.16 makes the policy see a non-zero pose error and
            # generate corrective turn/crouch motion instead of holding still.
            fr_pelvis = ctx.current.rigid_body_pos[num_humanoids:num_envs, pelvis_body_id, :]
            fr_pelvis_rot = ctx.current.rigid_body_rot[num_humanoids:num_envs, pelvis_body_id, :]
            # Pin xy to the park point (keep them in the corner) but use the
            # actual pelvis height so no vertical correction is generated.
            ref_pos[slot, :, pelvis_body_id, :2] = corner_xy[None, None, :]
            ref_pos[slot, :, pelvis_body_id, 2] = fr_pelvis[:, None, 2]
            # Reuse the current rotation as the target (broadcast over future steps).
            ref_rot[slot, :, pelvis_body_id, :] = fr_pelvis_rot[:, None, :]
            masks[slot, :, pelvis_condition_index, 0] = 1.0
            masks[slot, :, pelvis_condition_index, 1] = 1.0
            target_poses_masks[slot, :] = 1.0
            if hasattr(component.env, "motion_manager"):
                fr_times = component.env.motion_manager.motion_times.to(
                    device=device, dtype=dtype)
                target_times[slot, :] = fr_times[num_humanoids:num_envs, None] + offsets[None, :]
            else:
                target_times[slot, :] = offsets[None, :]
            time_offsets[slot, :] = offsets[None, :]

        ctx.masked_mimic = MaskedMimicContext(
            mimic=base.mimic,
            ref_pos=ref_pos,
            ref_rot=ref_rot,
            target_times=target_times,
            time_offsets=time_offsets,
            target_poses_masks=target_poses_masks,
            target_bodies_masks=target_bodies_masks,
        )

    def _warn_masked_mimic_target_once(self, reason: str) -> None:
        if self._printed_masked_mimic_target_warning:
            return
        self._printed_masked_mimic_target_warning = True
        print(f"[CrowdSim] Humanoid navigation targets disabled for this run: {reason}")

    def _create_local_target_markers(self, enabled: bool) -> None:
        if not enabled:
            return

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._local_target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/local_targets",
                markers=build_agent_marker_prototypes(
                    sim_utils,
                    num_humanoids=self.config.num_humanoids,
                    num_robots=self.config.num_robots,
                ),
            )
        )

    def _update_local_target_markers(self) -> None:
        if self._local_target_marker is None or self.num_agents == 0:
            return
        # Humanoids: show SFM lookahead target.
        # Cars: show fixed goal position (RL policy target).
        heights = self._agent_root_heights()
        translations = np.zeros((self.num_agents, 3), dtype=np.float32)
        orientations = np.zeros((self.num_agents, 4), dtype=np.float32)
        orientations[:, 0] = 1.0
        scales = np.zeros((self.num_agents, 3), dtype=np.float32)

        for agent_id in range(self.num_agents):
            if agent_id < self.config.num_humanoids:
                translations[agent_id, :2] = self._sfm_waypoints[agent_id]
                translations[agent_id, 2] = heights[agent_id]
                scales[agent_id] = [0.16, 0.16, 0.16]
            else:
                translations[agent_id, :2] = self.goals_xy[agent_id]
                translations[agent_id, 2] = heights[agent_id]
                scales[agent_id] = [0.25, 0.25, 0.10]

        marker_indices = np.arange(self.num_agents, dtype=np.int32)
        self._local_target_marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )

    def _agent_root_heights(self) -> np.ndarray:
        heights = np.full(self.num_agents, 0.16, dtype=np.float32)
        if self.config.num_humanoids > 0:
            humanoid_state = self.env.simulator.get_root_state()
            # root_pos is (num_envs, 3); slice to active humanoids only.
            heights[: self.config.num_humanoids] = (
                humanoid_state.root_pos[: self.config.num_humanoids, 2]
                .detach().cpu().numpy().astype(np.float32)
            )
        if self.config.num_robots > 0 and self.robot is not None:
            start = self.config.num_humanoids
            # root_pos_w is (num_envs, 3); slice to active robots only.
            heights[start : start + self.config.num_robots] = (
                self.robot.data.root_pos_w[: self.config.num_robots, 2]
                .detach().cpu().numpy().astype(np.float32)
            )
        return heights

    def _neighbor_state(
        self, agent_id: int, positions: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        relpos = positions - positions[agent_id]
        reldis = np.linalg.norm(relpos, axis=1)
        mask = (np.arange(self.num_agents) != agent_id) & (reldis < self.config.neighbor_radius)
        nbrs_idx = np.nonzero(mask)[0]
        return (
            nbrs_idx,
            reldis[nbrs_idx],
            relpos[nbrs_idx],
            velocities[nbrs_idx] - velocities[agent_id],
        )

    def set_robot_rl_actions(self, actions):
        values = actions.detach().cpu().numpy() if hasattr(actions, 'detach') else np.asarray(actions, dtype=np.float32)
        # Clip once here before touching drive or storage so that
        # drive._actions, _robot_rl_actions, rewards, and buffers all see
        # the same values — no silent secondary clip downstream.
        #   v_lin in [0, 1]  (forward-only, matches drive._commands_from_actions)
        #   v_ang in [-1, 1] (symmetric turning)
        clipped = np.empty_like(values)
        clipped[:, 0] = np.clip(values[:, 0],  0.0, 1.0)
        clipped[:, 1] = np.clip(values[:, 1], -1.0, 1.0)
        if self.drive is not None:
            self.drive.set_action(clipped)
        self._robot_rl_prev_actions[:] = self._robot_rl_actions
        self._robot_rl_actions[:] = clipped

    def _debug_drive(self) -> None:
        """Debug print drive state (no-op in production)."""
        if self.drive is None or self.config.num_robots == 0:
            return
        from CrowdSim.control.drive import DEBUG_DRIVE, DEBUG_CONSTANT_COMMAND
        if not DEBUG_DRIVE and DEBUG_CONSTANT_COMMAND is None:
            return
        if self.drive._wheel_targets_tensor is None:
            return
        actual = None
        if self.drive._wheel_joint_ids is not None and hasattr(self.drive.robot.data, "joint_vel"):
            actual = self.drive.robot.data.joint_vel[:, self.drive._wheel_joint_ids].detach().cpu().numpy()
        print(f"[CrowdSim][RobotDrive] "
              f"command_vw={self.drive._commands_from_actions().tolist()} "
              f"wheel_targets={self.drive._wheel_targets_tensor.detach().cpu().numpy().tolist()} "
              f"actual_wheel_velocities={actual.tolist() if actual is not None else None}")

    def _update_waypoints_and_goals(self, positions: np.ndarray) -> None:
        for agent_id, pos in enumerate(positions):
            path = self.paths_xy[agent_id]
            if self.reached[agent_id]:
                continue
            while self.waypoint_ids[agent_id] < len(path) - 1:
                waypoint = path[self.waypoint_ids[agent_id]]
                if np.linalg.norm(pos - waypoint) > self.config.waypoint_tolerance:
                    break
                self.waypoint_ids[agent_id] += 1
            if np.linalg.norm(pos - self.goals_xy[agent_id]) <= self.config.goal_tolerance:
                if (
                    self.task.is_fixed_scenario
                    and agent_id < self.config.num_humanoids
                ):
                    self._reverse_fixed_humanoid_route(agent_id)
                else:
                    self.reached[agent_id] = True

    def _reverse_fixed_humanoid_route(self, agent_id: int) -> None:
        """Turn a fixed-flow pedestrian around without teleporting it.

        Initial pedestrians are phase-spaced along their lanes.  Once one
        reaches its first destination, it traverses the complete fixed route
        in the opposite direction and then keeps ping-ponging between the two
        endpoints.  The immutable route stored by ``NavigationTask`` is used
        instead of reversing the phase-spaced initial path, which would make a
        pedestrian turn around in the middle of the corridor on its next leg.
        """
        (
            fixed_start_px,
            fixed_goal_px,
            fixed_start_xy,
            fixed_goal_xy,
            fixed_path_xy,
        ) = self.task.fixed_agent_route(agent_id)

        heading_to_fixed_goal = (
            np.linalg.norm(self.goals_xy[agent_id] - fixed_goal_xy)
            <= np.linalg.norm(self.goals_xy[agent_id] - fixed_start_xy)
        )
        if heading_to_fixed_goal:
            start_px, goal_px = fixed_goal_px, fixed_start_px
            start_xy, goal_xy = fixed_goal_xy, fixed_start_xy
            path_xy = fixed_path_xy[::-1].copy()
        else:
            start_px, goal_px = fixed_start_px, fixed_goal_px
            start_xy, goal_xy = fixed_start_xy, fixed_goal_xy
            path_xy = fixed_path_xy.copy()

        self.starts_px[agent_id] = start_px
        self.goals_px[agent_id] = goal_px
        self.starts_xy[agent_id] = start_xy
        self.goals_xy[agent_id] = goal_xy
        self.paths_xy[agent_id] = path_xy
        self.task.starts_px[agent_id] = start_px
        self.task.goals_px[agent_id] = goal_px
        self.task.starts_xy[agent_id] = start_xy
        self.task.goals_xy[agent_id] = goal_xy
        self.task.paths_xy[agent_id] = path_xy
        self.waypoint_ids[agent_id] = 1 if len(path_xy) > 1 else 0
        self.reached[agent_id] = False
        self.task.refresh_visualization_markers()

        if self._trajectory_log_file is not None:
            self._pending_path_updates.append(
                self._navigation_path_record(agent_id)
            )
            self._path_log_dirty = True

    def _navigation_done_agent_ids(self, new_pairs: set[tuple[int, int]]) -> np.ndarray:
        done: set[int] = set(int(agent_id) for agent_id in np.nonzero(self.reached)[0])
        for pair in new_pairs:
            for agent_id in pair:
                if agent_id < 0:
                    continue
                # In a continuous pedestrian stream, contact is an SFM event,
                # not a reason to teleport a person back to the entry.  Doing
                # that creates repeated entry collisions and apparent
                # "standing" agents.  Robot collision termination is kept.
                if self.task.is_fixed_scenario and agent_id < self.config.num_humanoids:
                    continue
                done.add(int(agent_id))
        return np.asarray(sorted(done), dtype=np.int64)

    def _request_humanoid_resets(self, agent_ids: np.ndarray) -> None:
        humanoid_ids = agent_ids[(0 <= agent_ids) & (agent_ids < self.config.num_humanoids)]
        if len(humanoid_ids) == 0:
            return

        reset_ids: list[int] = []
        positions = None
        reserved_entries: list[np.ndarray] = []
        if self.task.is_fixed_scenario:
            positions, _ = self._read_agent_state()

        for raw_agent_id in humanoid_ids:
            agent_id = int(raw_agent_id)
            if self.task.is_fixed_scenario:
                # Fixed-flow pedestrians recycle only after reaching the end.
                # If the entry is occupied, keep the pedestrian at the exit
                # and retry next navigation update instead of spawning agents
                # on top of one another.
                if not self.reached[agent_id]:
                    continue
                _, _, entry_xy, _, _ = self.task.fixed_agent_route(agent_id)
                assert positions is not None
                other_ids = np.arange(self.num_agents) != agent_id
                clearance = max(
                    float(self.config.collision_distance),
                    float(self.config.min_spawn_spacing),
                )
                if (
                    np.any(
                        np.linalg.norm(
                            positions[other_ids] - entry_xy[None, :], axis=1
                        )
                        < clearance
                    )
                    or any(
                        np.linalg.norm(entry_xy - reserved) < clearance
                        for reserved in reserved_entries
                    )
                ):
                    continue
                reserved_entries.append(entry_xy)

            self._pending_humanoid_reset_reasons[int(agent_id)] = self._humanoid_reset_reason(
                int(agent_id)
            )
            reset_ids.append(agent_id)

        if not reset_ids:
            return
        ids = torch.as_tensor(reset_ids, dtype=torch.long, device=self.config.device)
        self.env.reset_buf[ids] = True

    def _humanoid_reset_reason(self, agent_id: int) -> str:
        reasons = []
        if bool(self.reached[agent_id]):
            reasons.append("reached")
        if any(agent_id in pair for pair in self.collision_pairs):
            reasons.append("collision")
        if (agent_id, -1) in self.collision_pairs:
            reasons.append("wall")
        return "+".join(reasons) if reasons else "external"

    def _detect_collisions(self, positions: np.ndarray) -> set[tuple[int, int]]:
        return self.collision.detect(positions, self.num_agents, self.collision_pairs)

    def _reset_env_ids_from_args(self, args: tuple, kwargs: dict) -> np.ndarray:
        if "env_ids" in kwargs:
            env_ids = kwargs["env_ids"]
            if env_ids is None:
                # Full reset covers every env slot so frozen filler slots
                # (>= num_humanoids) are also teleported to their parked corner.
                return np.arange(self.env.num_envs, dtype=np.int64)
            if isinstance(env_ids, torch.Tensor):
                return np.atleast_1d(
                    env_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                )
            return np.atleast_1d(np.asarray(env_ids, dtype=np.int64))

        if not args or args[0] is None:
            return np.arange(self.env.num_envs, dtype=np.int64)

        env_ids = args[0]
        if isinstance(env_ids, torch.Tensor):
            return np.atleast_1d(
                env_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            )
        return np.atleast_1d(np.asarray(env_ids, dtype=np.int64))

    def _reset_navigation_agents(self, env_ids: np.ndarray) -> None:
        if len(env_ids) == 0:
            return

        humanoid_ids = env_ids[
            (0 <= env_ids) & (env_ids < self.config.num_humanoids)
        ]
        if len(humanoid_ids) > 0:
            positions, _ = self._read_agent_state()
            for agent_id in humanoid_ids:
                self._pending_humanoid_reset_reasons.pop(int(agent_id), None)
                if self.task.is_fixed_scenario:
                    self._restore_fixed_scenario_agent(int(agent_id))
                else:
                    sampled = self._sample_spaced_free_xy(positions, int(agent_id))
                    start_xy = sampled if sampled is not None else positions[int(agent_id)]
                    self._replan_agent_from_xy(int(agent_id), start_xy)
                self._humanoid_yaw_source[int(agent_id)] = "reset"
                self._humanoid_smoothed_velocities[int(agent_id)] = 0.0
                self._humanoid_velocity_initialized[int(agent_id)] = False
            self._flush_navigation_path_log_if_dirty()

    def _restore_fixed_scenario_agent(self, agent_id: int) -> None:
        """Restore one agent to its configured scenario start and path."""
        start_px, goal_px, start_xy, goal_xy, path_xy = self.task.fixed_agent_route(agent_id)
        self.starts_px[agent_id] = start_px
        self.goals_px[agent_id] = goal_px
        self.starts_xy[agent_id] = start_xy
        self.goals_xy[agent_id] = goal_xy
        self.paths_xy[agent_id] = path_xy
        self.waypoint_ids[agent_id] = 1 if len(path_xy) > 1 else 0
        self.reached[agent_id] = False
        self.collision_pairs = {pair for pair in self.collision_pairs if agent_id not in pair}
        if agent_id < self.config.num_humanoids and hasattr(self, "env"):
            new_xy = torch.as_tensor(start_xy, dtype=torch.float32, device=self.config.device)
            if hasattr(self.env, "_crowdsim_desired_root_xy"):
                self.env._crowdsim_desired_root_xy[agent_id] = new_xy
            self.env.respawn_root_offset[agent_id, :2] = new_xy
        self.task.refresh_visualization_markers()

    def _replan_agent_from_xy(self, agent_id: int, start_xy: np.ndarray) -> None:
        # H1: guard against planning failures — fall back to current start/goal with
        # a straight-line path so training continues rather than crashing.
        try:
            start_px, goal_px, path_xy = self.task.sample_goal_and_plan_path(start_xy)
        except RuntimeError as exc:
            print(f"[CrowdSim][WARN] path planning failed for agent {agent_id}: {exc}"
                  " — keeping current start/goal with straight-line path.")
            start_px = self.starts_px[agent_id]
            goal_px  = self.goals_px[agent_id]
            path_xy  = np.array(
                [self.starts_xy[agent_id], self.goals_xy[agent_id]], dtype=np.float32
            )
        self.starts_px[agent_id] = start_px
        self.goals_px[agent_id] = goal_px
        self.starts_xy[agent_id] = self.task.pixel_to_world(start_px)
        self.goals_xy[agent_id] = self.task.pixel_to_world(goal_px)
        self.paths_xy[agent_id] = path_xy
        self.task.starts_px[agent_id] = start_px
        self.task.goals_px[agent_id] = goal_px
        self.task.starts_xy[agent_id] = self.starts_xy[agent_id]
        self.task.goals_xy[agent_id] = self.goals_xy[agent_id]
        self.task.paths_xy[agent_id] = path_xy
        self.waypoint_ids[agent_id] = 1 if len(path_xy) > 1 else 0
        self.reached[agent_id] = False
        self.collision_pairs = {
            pair for pair in self.collision_pairs if agent_id not in pair
        }
        # Sync humanoid spawn position so physical reset goes to the new start
        if agent_id < self.config.num_humanoids and hasattr(self, "env"):
            env = self.env
            new_xy = torch.as_tensor(
                self.starts_xy[agent_id], dtype=torch.float32, device=self.config.device,
            )
            # Update the mutable tensor that respawn_offset reads from
            if hasattr(env, "_crowdsim_desired_root_xy"):
                env._crowdsim_desired_root_xy[agent_id] = new_xy
            env.respawn_root_offset[agent_id, :2] = new_xy
        self.task.refresh_visualization_markers()
        # Only queue path updates if trajectory recording is active (avoids unbounded
        # list growth when recording is disabled — _consume only runs in _record_trajectory_frame).
        if self._trajectory_log_file is not None:
            self._pending_path_updates.append(self._navigation_path_record(agent_id))
            self._path_log_dirty = True

    def _current_waypoint(self, agent_id: int) -> np.ndarray:
        path = self.paths_xy[agent_id]
        idx = min(int(self.waypoint_ids[agent_id]), len(path) - 1)
        return path[idx]

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        return self.task.world_to_pixel(xy)

    def pixel_to_world(self, pixel_yx: np.ndarray) -> np.ndarray:
        return self.task.pixel_to_world(pixel_yx)

    def _min_static_obstacle_dist(self, robot_xy: np.ndarray, radius: float) -> float:
        """Min world-frame distance from *robot_xy* to the nearest static obstacle.

        Scans a square window of side ``2*radius`` (metres) around the robot's
        pixel in ``self.obstacle_map`` and returns the distance to the closest
        occupied cell.  Out-of-map cells are treated as obstacles (matches the
        ``_local_obstacle_patch`` convention).  Returns ``radius`` (the cap)
        when no obstacle is found within the window — callers that want a
        "no obstacle nearby" sentinel should treat any value ≥ radius as safe.

        This is used by the proximity penalty so that approaching walls /
        shelves / boxes penalises the robot even though the RL observation no
        longer carries an occupancy map.  The map is still loaded (via the
        scene's ``MapTask``) and used only for reward / collision here.
        """
        if self.obstacle_map is None:
            raise RuntimeError(
                "obstacle_map is None — static-obstacle proximity/collision "
                "unavailable. The map is required for reward/collision even "
                "though it no longer enters the RL observation. Check the "
                "scene_map config (it must point to a valid occupancy PNG)."
            )
        if radius <= 0:
            return float(radius)
        res = float(self.config.map_resolution)
        r_px = max(1, int(math.ceil(radius / res)))
        pyx = self.world_to_pixel(np.asarray(robot_xy, dtype=np.float32))
        cy, cx = int(pyx[0]), int(pyx[1])

        y0, y1 = cy - r_px, cy + r_px + 1
        x0, x1 = cx - r_px, cx + r_px + 1
        # Clip to map bounds; remember which rows/cols were out-of-range so
        # we can flag them as obstacles (out-of-map = occupied).
        y_lo, y_hi = max(0, y0), min(self.height, y1)
        x_lo, x_hi = max(0, x0), min(self.width, x1)
        if y_lo >= y_hi or x_lo >= x_hi:
            # Robot is entirely outside the map → treat as touching an obstacle.
            return 0.0

        sub = self.obstacle_map[y_lo:y_hi, x_lo:x_hi]
        # Relative pixel coords of occupied cells within the sub-window
        occ_y, occ_x = np.nonzero(sub > 0)

        # Distance from robot centre to map edges (in world metres).
        edge_d = float("inf")
        edge_d = min(edge_d, cy * res)                      # top edge
        edge_d = min(edge_d, (self.height - 1 - cy) * res)  # bottom edge
        edge_d = min(edge_d, cx * res)                      # left edge
        edge_d = min(edge_d, (self.width - 1 - cx) * res)  # right edge

        if occ_y.size == 0:
            # No occupied cell inside the window; only map edges matter.
            d = edge_d - float(self.config.agent_radius)
            return max(min(d, radius), 0.0) if np.isfinite(edge_d) else float(radius)

        # Cell-centre offsets (in pixels) from the robot pixel
        dy = (occ_y + y_lo) - cy
        dx = (occ_x + x_lo) - cx
        dists_px = np.sqrt(dy * dy + dx * dx)
        # Nearest obstacle distance, considering both cells and map edges.
        min_obs_d = float(dists_px.min() - 0.5) * res
        d = min(min_obs_d, edge_d) - float(self.config.agent_radius)
        return max(d, 0.0)

    def _planner_cfg(self, radius: float) -> dict:
        return {
            "map": {
                "resolution": self.config.map_resolution,
                "resolution_viz": self.pixels_per_meter,
            },
            "env": {
                "dt": self._dt(),
                "safe_distance": self.config.safe_distance,
                "neighbor_radius": self.config.humanoid_interaction_radius,
                "reach_distance": self.config.waypoint_tolerance,
                "prediction_horizon": self.config.humanoid_prediction_horizon,
                "ttc_threshold": self.config.humanoid_ttc_threshold,
                "ttc_gain": self.config.humanoid_ttc_gain,
            },
            "agent": {"radius": radius, "max_vel": self.config.max_speed},
        }

    def get_robot_rl_observations(self):
        import logging
        _log = logging.getLogger("CrowdSim")
        positions, velocities = self._read_agent_state()
        obs = self._build_robot_rl_observations(positions, velocities)
        neighbors, neighbor_mask = self._build_robot_neighbor_observations(positions, velocities)
        map_patch = self._read_map_patch(positions)
        depth = self._read_camera_depth() if self.config.rl_depth_enabled else None
        if self.config.rl_depth_enabled and depth is None:
            cached = getattr(self, "_robot_last_depth", None)
            if cached is not None:
                # Use last valid frame; warn once so the issue is still visible.
                if not getattr(self, "_warned_depth_fallback", False):
                    _log.warning(
                        "rl_depth_enabled=True but current depth read failed — "
                        "falling back to last valid depth frame. "
                        "Check _read_camera_depth warnings for root cause."
                    )
                    self._warned_depth_fallback = True
                depth = cached
            else:
                if not getattr(self, "_warned_depth_unavailable", False):
                    _log.warning(
                        "rl_depth_enabled=True but no depth is available (camera not ready on first step?) — "
                        "buffer depth falls back to a zero frame for this step. "
                        "Check that sensors.camera.enabled is true and the robot has a camera sensor."
                    )
                    self._warned_depth_unavailable = True
                n_r = self.config.num_robots
                depth = torch.zeros(
                    n_r, int(self.config.rl_depth_size), int(self.config.rl_depth_size),
                    dtype=torch.float32, device=obs.device,
                )
        # map: _build_robot_rl_observations returns None when rl_map_size==0;
        # otherwise a (num_robots, map_size, map_size) patch.  No camera-style
        # fallback needed — the patch is always computable from the occupancy
        # map when enabled.
        self._robot_last_obs = obs
        self._robot_last_neighbors = neighbors
        self._robot_last_neighbor_mask = neighbor_mask
        self._robot_last_depth = depth
        self._robot_last_map = map_patch
        rgb = self._read_camera_rgb() if self.config.rl_rgb_enabled else None
        if self.config.rl_rgb_enabled and rgb is None:
            cached_rgb = getattr(self, "_robot_last_rgb", None)
            if cached_rgb is not None:
                rgb = cached_rgb
            else:
                if not getattr(self, "_warned_rgb_unavailable", False):
                    _log.warning(
                        "rl_rgb_enabled=True but no RGB is available (camera not ready on first step?) — "
                        "buffer RGB falls back to a black frame for this step. "
                        "Check that sensors.camera.enabled is true and the robot has a camera sensor."
                    )
                    self._warned_rgb_unavailable = True
                n_r = self.config.num_robots
                rgb = torch.zeros(
                    n_r, 3, int(self.config.rl_rgb_size), int(self.config.rl_rgb_size),
                    dtype=torch.uint8, device=obs.device,
                )
        self._robot_last_rgb = rgb
        return obs, neighbors, neighbor_mask, depth, map_patch

    def get_robot_rl_feedback(self):
        if self._robot_last_obs is None:
            self.get_robot_rl_observations()
        # Return clones of rewards / dones / info-tensors so callers can safely
        # read them AFTER calling reset_robot_rl_episodes(), which zeroes the
        # internal _robot_last_{rewards,dones,info[*]} in place.  Without these
        # clones, a caller that holds the returned reference sees its values
        # flipped to 0/False mid-step — e.g. collect_cbf_buffer read
        # robot_done[0] as False right after reset, dropping every completed
        # episode.  info is shallow-copied and its per-robot tensor values are
        # cloned so callers can read reached/collision/timeout/stuck reliably.
        info_copy = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in self._robot_last_info.items()
        }
        return (
            self._robot_last_obs,
            self._robot_last_rewards.clone(),
            self._robot_last_dones.clone(),
            info_copy,
            getattr(self, "_robot_last_depth", None),
            getattr(self, "_robot_last_map", None),
            getattr(self, "_robot_last_neighbors", None),
            getattr(self, "_robot_last_neighbor_mask", None),
        )

    def _clear_robot_rl_env_step_feedback(self) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return

        self._robot_last_rewards.zero_()
        self._robot_last_dones.zero_()
        zero_float = torch.zeros(self.config.num_robots, dtype=torch.float32, device=self.config.device)
        zero_bool = torch.zeros(self.config.num_robots, dtype=torch.bool, device=self.config.device)
        self._robot_last_info = {
            **self._robot_last_info,
            "is_navigation_step": zero_bool,
            "reached": zero_bool,
            "collision": zero_bool,
            "timeout": zero_bool,
            "stuck": zero_bool,
            "progress": zero_float,
            "reward_total": zero_float,
            "reward_time": zero_float,
            "reward_progress": zero_float,
            "reward_velocity_direction": zero_float,
            "reward_proximity": zero_float,
            "reward_smoothness": zero_float,
            "reward_angular_velocity": zero_float,
            "reward_angular_change": zero_float,
            "reward_linear_change": zero_float,
            "reward_goal": zero_float,
            "reward_collision": zero_float,
            "reward_timeout": zero_float,
            "reward_stuck": zero_float,
            "distance_to_goal": zero_float,
            "distance_to_progress_target": zero_float,
        }

    @property
    def robot_rl_depth_obs_dim(self) -> int:
        return (
            self.config.rl_depth_size * self.config.rl_depth_size
            if self.config.rl_depth_enabled else 0
        )

    def _read_camera_depth(self) -> torch.Tensor | None:
        """Read depth and resize the raw camera frame to square (N, S, S).

        Returns ``None`` on any failure so callers can fall back to the cached
        last-valid depth.  All errors are logged once per session to avoid
        flooding the console.
        """
        import logging
        _log = logging.getLogger("CrowdSim")
        try:
            if not hasattr(self, "env") or not hasattr(self.env, "crowdsim_robot_camera"):
                return None
            camera = self.env.crowdsim_robot_camera
            output = camera.data.output
            raw = output.get("distance_to_image_plane")
            if raw is None:
                return None
            # raw: (N, H, W, 1) or (N, H, W)
            if raw.dim() == 4:
                raw = raw.squeeze(-1)  # → (N, H, W)
            elif raw.dim() != 3:
                if not getattr(self, "_warned_depth_dim", False):
                    _log.warning(
                        "_read_camera_depth: unexpected tensor rank %d (expected 3 or 4) — skipping.",
                        raw.dim(),
                    )
                    self._warned_depth_dim = True
                return None
            depth = raw[: self.config.num_robots]  # only robot envs
            if depth.shape[0] == 0:
                return None
            # Match FLUX/NavDP preprocessing in metric space: no-hit pixels are
            # black, aspect ratio is preserved, and padding produces 224x224.
            max_range = float(self.config.rl_depth_max_range)
            if max_range <= 0.0:
                raise ValueError(
                    f"rl_depth_max_range must be positive, got {max_range}"
                )
            depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
            depth = resize_with_letterbox(
                depth.unsqueeze(1), int(self.config.rl_depth_size)
            ).squeeze(1)
            invalid = (depth < 0.1) | (depth > max_range)
            depth = depth.masked_fill(invalid, 0.0)
            return depth.div(max_range)  # valid metric depth -> [0,1], invalid -> 0
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_warned_depth_read_error", False):
                _log.warning(
                    "_read_camera_depth failed: %s — will use fallback depth. "
                    "Check that sensors.camera.enabled is true and the robot has a camera sensor.",
                    exc,
                )
                self._warned_depth_read_error = True
            return None

    def _read_camera_rgb(self) -> torch.Tensor | None:
        """Read RGB and resize the raw frame to (N, 3, S, S) uint8 [0,255].

        Returns ``None`` on any failure so callers can fall back to the cached
        last-valid RGB (mirrors ``_read_camera_depth``).  Output is uint8 to
        keep the data buffer compact (~150 KB/frame at 224×224) for CFM
        training; PPO never consumes this tensor.
        """
        import logging
        _log = logging.getLogger("CrowdSim")
        try:
            if not hasattr(self, "env") or not hasattr(self.env, "crowdsim_robot_camera"):
                return None
            camera = self.env.crowdsim_robot_camera
            output = camera.data.output
            raw = output.get("rgb")
            if raw is None:
                return None
            # raw: (N, H, W, 3) — IsaacLab CameraSensor delivers float [0,1] uint8-agnostic.
            if raw.dim() != 4:
                if not getattr(self, "_warned_rgb_dim", False):
                    _log.warning(
                        "_read_camera_rgb: unexpected tensor rank %d (expected 4) — skipping.",
                        raw.dim(),
                    )
                    self._warned_rgb_dim = True
                return None
            rgb = raw[: self.config.num_robots]  # only robot envs
            if rgb.shape[0] == 0:
                return None
            # Normalise to uint8 [0, 255].  Tolerate both float [0,1] and
            # already-uint8 inputs (defensive; IsaacLab normally gives float).
            if rgb.dtype != torch.uint8:
                rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            # (N, H, W, 3) -> (N, 3, H, W) for interpolate
            rgb = rgb.permute(0, 3, 1, 2).contiguous()
            target = int(self.config.rl_rgb_size)
            rgb = resize_with_letterbox(rgb.float(), target).round().to(torch.uint8)
            return rgb  # (N, 3, target, target) uint8
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_warned_rgb_read_error", False):
                _log.warning(
                    "_read_camera_rgb failed: %s — will use fallback RGB. "
                    "Check that sensors.camera.enabled is true and the robot has a camera sensor.",
                    exc,
                )
                self._warned_rgb_read_error = True
            return None

    def _read_map_patch(self, positions: np.ndarray | None = None) -> torch.Tensor | None:
        """Build the ego-centric occupancy-map patch for every active robot.

        Returns a ``(num_robots, map_size, map_size)`` float32 tensor (1.0 =
        obstacle / unknown, 0.0 = free), or ``None`` when ``rl_map_size == 0``.
        Mirrors ``_read_camera_depth`` / ``_read_camera_rgb`` — map is a
        standalone channel consumed by MapEncoder, not built inside the vector
        obs constructor.

        ``positions`` (all-agent xy from ``_read_agent_state``) may be passed
        in to avoid a redundant GPU→CPU read when the caller already fetched
        it for the vector obs.
        """
        if self.config.rl_map_size <= 0 or self.config.num_robots == 0:
            return None
        if positions is None:
            positions, _ = self._read_agent_state()
        robot_offset = self.config.num_humanoids
        robot_range = np.arange(self.config.num_robots)
        robot_agent_ids = robot_offset + robot_range
        robot_pos = positions[robot_agent_ids, :2]
        yaws = self._robot_yaws()
        return torch.as_tensor(
            np.stack(
                [self._local_obstacle_patch(robot_pos[i], yaws[i])
                 for i in range(self.config.num_robots)],
                axis=0,
            ),
            dtype=torch.float32, device=self.config.device,
        )

    @property
    def robot_rl_obs_dim(self) -> int:
        # obs now contains ONLY the vector part; map is a separate channel
        # (like depth/rgb).  robot_rl_map_obs_dim is kept for buffer/dataset
        # sizing of the standalone map tensor.
        return self.robot_rl_vector_obs_dim

    @property
    def robot_rl_vector_obs_dim(self) -> int:
        # goal(3) + self motion(2). Neighbors are a standalone set channel.
        return 5

    @property
    def robot_rl_neighbor_dim(self) -> int:
        return 5

    @property
    def robot_rl_map_obs_dim(self) -> int:
        map_size = max(0, int(self.config.rl_map_size))
        return map_size * map_size

    def reset_robot_rl_episodes(
        self,
        done: torch.Tensor | np.ndarray,
        repeat_mask: np.ndarray | None = None,
    ) -> None:
        """Reset done robot episodes.

        Parameters
        ----------
        done:
            Boolean mask of shape (num_robots,) indicating which robots finished.
        repeat_mask:
            Optional boolean mask of shape (num_robots,).  When ``repeat_mask[i]``
            is True the robot is teleported back to its *current* start position
            with the *same* goal — no new route is sampled.  This supports
            paired-run preference collection where the same route is attempted
            twice in a row.
        """
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return
        if isinstance(done, torch.Tensor):
            done_np = done.detach().cpu().numpy().astype(bool)
        else:
            done_np = np.asarray(done, dtype=bool)
        robot_ids = np.nonzero(done_np)[0]
        if len(robot_ids) == 0:
            return

        agent_offset = self.config.num_humanoids
        agent_ids = agent_offset + robot_ids
        env_ids = torch.as_tensor(robot_ids, dtype=torch.long, device=self.config.device)

        # ① Sample new starts and replan FIRST so starts_xy is up to date.
        #    For robots in repeat_mask, keep current start + goal (run-B setup).
        self._robot_rl_actions[robot_ids] = 0.0
        self._robot_rl_prev_actions[robot_ids] = 0.0
        if self.drive is not None:
            self.drive.clear_state(robot_ids)
        all_pos, _ = self._read_agent_state()
        # H9: use a mutable position snapshot so successive agents in the same
        # reset batch see each other's new spawn locations during spacing checks,
        # preventing two robots from being placed at the same position.
        planned_positions = all_pos.copy()
        for agent_id in agent_ids:
            robot_id = int(agent_id) - agent_offset
            _repeat = (
                repeat_mask is not None
                and robot_id < len(repeat_mask)
                and bool(repeat_mask[robot_id])
            )
            if _repeat:
                # Keep starts_xy / goals_xy unchanged — teleport back to same start.
                # Must clear reached flag so run-B doesn't instantly terminate.
                self.reached[int(agent_id)] = False
                self.waypoint_ids[int(agent_id)] = 1 if len(self.paths_xy[int(agent_id)]) > 1 else 0
            elif self.task.is_fixed_scenario:
                self._restore_fixed_scenario_agent(int(agent_id))
            else:
                sampled = self._sample_spaced_free_xy(planned_positions, int(agent_id))
                start_xy = sampled if sampled is not None else self.starts_xy[int(agent_id)]
                self._replan_agent_from_xy(int(agent_id), start_xy)
            if not _repeat:
                self._resample_robot_spawn_yaw(robot_id)
            # Record the (new or repeated) spawn so spacing checks see it.
            planned_positions[int(agent_id), :2] = self.starts_xy[int(agent_id)]

        # ② Build teleport poses from the NEW starts_xy
        yaw = self._initial_robot_yaws()[robot_ids]
        poses = self.robot.data.default_root_state[env_ids, :7].clone()
        poses[:, 0:2] = torch.as_tensor(
            self.starts_xy[agent_ids], dtype=torch.float32, device=self.config.device,
        )
        poses[:, 3:7] = self._yaw_to_quat_tensor(torch.as_tensor(yaw, device=self.config.device))

        self.robot.write_root_pose_to_sim(poses, env_ids=env_ids)
        if self.drive is not None:
            self.drive.sync_teleported_poses(poses, env_ids)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((len(robot_ids), 6), dtype=torch.float32, device=self.config.device),
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids].clone(),
            torch.zeros_like(self.robot.data.default_joint_vel[env_ids]),
            env_ids=env_ids,
        )
        self.robot.reset(env_ids=env_ids)
        if self.drive is not None:
            self.drive._clear_wheel_targets(env_ids)

        # ③ Record trajectory + flush path log (replan already set _path_log_dirty)
        if self._trajectory_log_file is not None:
            positions, velocities = self._read_agent_state()
            for i, robot_id in enumerate(robot_ids):
                positions[agent_offset + robot_id] = poses[i, :2].detach().cpu().numpy()
            self._record_trajectory_frame(positions, velocities)
        if hasattr(self, "_flush_navigation_path_log_if_dirty"):
            self._flush_navigation_path_log_if_dirty()

        self._robot_episode_steps[robot_ids] = 0
        self._stuck_pos_buf[robot_ids] = 0.0   # clear stuck history so old positions don't bleed into new episode
        self._robot_progress_targets[robot_ids] = self.starts_xy[agent_ids]
        # Set prev_dist to actual start→goal distance so the first step
        # after reset has zero progress (not a large negative penalty).
        self._robot_prev_progress_dist[robot_ids] = np.linalg.norm(
            self.starts_xy[agent_ids] - self.goals_xy[agent_ids], axis=1
        ).astype(np.float32)
        agent_id_set = {int(agent_id) for agent_id in agent_ids}
        self.collision_pairs = {
            pair
            for pair in self.collision_pairs
            if pair[0] not in agent_id_set and pair[1] not in agent_id_set
        }
        self._robot_last_rewards[env_ids] = 0.0
        self._robot_last_dones[env_ids] = False
        for value in self._robot_last_info.values():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.config.num_robots,):
                if value.dtype == torch.bool:
                    value[env_ids] = False
                else:
                    value[env_ids] = 0.0

    def _robot_reset_reason(self, robot_id: int) -> str:
        reasons = []
        for key in ("reached", "collision", "timeout"):
            value = self._robot_last_info.get(key)
            if isinstance(value, torch.Tensor) and value.numel() > robot_id:
                if bool(value[robot_id].detach().cpu().item()):
                    reasons.append(key)
        agent_id = self.config.num_humanoids + robot_id
        if (agent_id, -1) in self.collision_pairs:
            reasons.append("wall")
        return "+".join(reasons) if reasons else "external"

    def _update_robot_rl_feedback(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        new_pairs: set[tuple[int, int]],
    ) -> None:
        if not self.config.car_rl_policy or self.config.num_robots == 0:
            return

        robot_offset = self.config.num_humanoids
        robot_agent_ids = np.arange(robot_offset, robot_offset + self.config.num_robots)
        current_goal_dist = np.linalg.norm(
            positions[robot_agent_ids] - self.goals_xy[robot_agent_ids],
            axis=1,
        )
        current_progress_dist = np.linalg.norm(
            positions[robot_agent_ids] - self._robot_progress_targets,
            axis=1,
        )
        progress = self._robot_prev_progress_dist - current_progress_dist
        self._robot_episode_steps += 1

        # Robot collisions come from two sources, OR'd together:
        #   (1) detect()'s new_pairs — any pair (agent-agent OR agent-wall)
        #       that involves a robot.
        #   (2) the drive guard's wall_blocked flag — set when a robot tried to
        #       translate into a wall cell.  Catches wall hits that detect's
        #       parked-pose query misses (the guard parks the robot at cur,
        #       which may not overlap a wall cell depending on approach angle).
        robot_ids_in_pairs = {
            a for pair in new_pairs for a in pair
            if robot_offset <= a < robot_offset + self.config.num_robots
        }
        collision = np.zeros(self.config.num_robots, dtype=bool)
        for rid in robot_ids_in_pairs:
            collision[rid - robot_offset] = True
        blocked = getattr(self.drive, "wall_blocked", None) if self.drive is not None else None
        if blocked is not None:
            collision |= blocked

        reached = self.reached[robot_agent_ids].copy()
        timeout = self._robot_episode_steps >= self.config.rl_max_episode_steps

        # Stuck detection: compare current position against position _window steps ago.
        _window      = max(1, int(self.config.rl_stuck_window))
        robot_pos_xy = positions[robot_agent_ids, :2]
        slot_now     = (self._robot_episode_steps - 1) % _window
        slot_old     = self._robot_episode_steps % _window
        _ridx        = np.arange(self.config.num_robots)
        self._stuck_pos_buf[_ridx, slot_now] = robot_pos_xy
        dist_moved = np.linalg.norm(robot_pos_xy - self._stuck_pos_buf[_ridx, slot_old], axis=1)
        # Exempt terminal states: a robot that just reached the goal (and is
        # decelerating) or just collided should NOT also be penalised for
        # "stuck".  Otherwise reached(+50)+stuck(-30)=+20 pollutes the goal
        # signal, and collision(-50)+stuck(-30)=-80 over-penalises.  Stuck is
        # meant to catch non-terminal stagnation only.
        terminal = reached | collision | timeout
        stuck = (
            (dist_moved < float(self.config.rl_stuck_threshold))
            & (self._robot_episode_steps >= _window)
            & (~terminal)
        )

        time_reward = np.full(self.config.num_robots, self.config.rl_time_penalty, dtype=np.float32)
        progress_reward = (self.config.rl_progress_reward_scale * progress).astype(np.float32)

        # Velocity direction reward: reward velocity component toward the goal.
        robot_vels = velocities[robot_agent_ids]
        goal_vecs = self.goals_xy[robot_agent_ids] - positions[robot_agent_ids]
        goal_dists_safe = np.maximum(np.linalg.norm(goal_vecs, axis=1), 1e-4)
        velocity_toward_goal = np.sum(robot_vels * (goal_vecs / goal_dists_safe[:, None]), axis=1)
        velocity_direction_reward = (
            self.config.rl_velocity_direction_reward_scale * velocity_toward_goal
        ).astype(np.float32)

        linear_action = self._robot_rl_actions[:, 0]
        angular_action = self._robot_rl_actions[:, 1]
        previous_linear = self._robot_rl_prev_actions[:, 0]
        previous_angular = self._robot_rl_prev_actions[:, 1]
        legacy_smoothness_penalty = (
            -self.config.rl_action_smoothness_scale
            * np.abs(self._robot_rl_actions - self._robot_rl_prev_actions).sum(axis=1)
        ).astype(np.float32)
        angular_velocity_penalty = (
            -self.config.rl_angular_velocity_scale * np.square(angular_action)
        ).astype(np.float32)
        angular_change_penalty = (
            -self.config.rl_angular_change_scale
            * np.square(angular_action - previous_angular)
        ).astype(np.float32)
        linear_change_penalty = (
            -self.config.rl_linear_change_scale
            * np.square(linear_action - previous_linear)
        ).astype(np.float32)
        action_smoothness_penalty = (
            legacy_smoothness_penalty
            + angular_velocity_penalty
            + angular_change_penalty
            + linear_change_penalty
        ).astype(np.float32)

        # Proximity penalty — three-zone design so safety dominates at close range:
        #   d ≥ outer_threshold          : 0             (safe, no penalty)
        #   danger_threshold ≤ d < outer : linear 0 → -scale  (approach gradient)
        #   d < danger_threshold         : flat danger_penalty  (hard deterrent)
        #
        # ``d`` is the min distance to *any* obstacle — dynamic agents (humanoids
        # / other robots) AND static obstacles (walls / shelves / boxes).  The RL
        # observation no longer carries an occupancy map, so this penalty is the
        # only signal that teaches the policy to keep clear of static geometry.
        proximity_penalty = np.zeros(self.config.num_robots, dtype=np.float32)
        outer_thresh  = float(self.config.rl_proximity_penalty_threshold)
        danger_thresh = float(self.config.rl_proximity_danger_threshold)
        if outer_thresh > 0:
            all_dists = np.linalg.norm(
                positions[robot_agent_ids, :2][:, None, :] - positions[None, :, :2], axis=2
            )  # (N, A)  — dynamic agents only (positions = humanoid + robot)
            all_dists[_ridx, robot_agent_ids] = np.inf   # exclude self
            min_dists = all_dists.min(axis=1)
            # Two-point linear proximity penalty.  yaml stores NEGATIVE values.
            #   d ≥ outer_thresh              → 0       (outside penalty zone)
            #   danger ≤ d < outer            → linear interp between penalty_outer and penalty_inner
            #   d < danger (clamped to danger)→ penalty_inner (flat danger zone)
            span = max(outer_thresh - danger_thresh, 1e-4)
            penalty_outer = float(self.config.rl_proximity_penalty_start)    # negative, e.g. -1.5
            penalty_inner = float(self.config.rl_proximity_danger_penalty)   # negative, e.g. -5.0
            # Only penalise robots within the outer threshold.
            proximity_penalty = np.zeros(self.config.num_robots, dtype=np.float32)
            close = min_dists < outer_thresh
            if close.any():
                clamped = np.clip(min_dists[close], danger_thresh, outer_thresh)
                # t = 0 at outer → penalty_outer, t = 1 at inner → penalty_inner
                t = (outer_thresh - clamped) / span
                proximity_penalty[close] = (
                    penalty_outer + t * (penalty_inner - penalty_outer)
                ).astype(np.float32)

        # Static obstacles use a separate, deliberately weak bounded penalty.
        # Reusing the dynamic-agent scale previously overwhelmed progress.
        static_proximity_penalty = np.zeros(
            self.config.num_robots, dtype=np.float32
        )
        static_outer = float(self.config.rl_static_proximity_threshold)
        static_inner = float(np.clip(
            self.config.rl_static_proximity_danger_threshold,
            0.0,
            max(static_outer, 0.0),
        ))
        static_floor = min(
            float(self.config.rl_static_proximity_danger_penalty), 0.0
        )
        if static_outer > 0.0 and static_floor < 0.0:
            static_distances = np.asarray(
                [
                    self._min_static_obstacle_dist(position, static_outer)
                    for position in robot_pos_xy
                ],
                dtype=np.float32,
            )
            close = static_distances < static_outer
            if close.any():
                span = max(static_outer - static_inner, 1.0e-4)
                clamped = np.clip(
                    static_distances[close], static_inner, static_outer
                )
                strength = (static_outer - clamped) / span
                static_proximity_penalty[close] = static_floor * strength

        goal_reward = (self.config.rl_goal_reward * reached.astype(np.float32)).astype(np.float32)
        collision_reward = (
            self.config.rl_collision_penalty * collision.astype(np.float32)
        ).astype(np.float32)
        timeout_reward = (self.config.rl_timeout_penalty * timeout.astype(np.float32)).astype(np.float32)
        stuck_reward = (self.config.rl_stuck_penalty * stuck.astype(np.float32)).astype(np.float32)
        rewards = (
            time_reward
            + progress_reward
            + velocity_direction_reward
            + action_smoothness_penalty
            + proximity_penalty
            + static_proximity_penalty
            + goal_reward
            + collision_reward
            + timeout_reward
            + stuck_reward
        ).astype(np.float32)
        dones = reached | collision | timeout | stuck

        self._robot_last_obs = self._build_robot_rl_observations(positions, velocities)
        self._robot_last_neighbors, self._robot_last_neighbor_mask = (
            self._build_robot_neighbor_observations(positions, velocities)
        )
        self._robot_last_rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.config.device)
        self._robot_last_dones = torch.as_tensor(dones, dtype=torch.bool, device=self.config.device)
        self._robot_last_info = {
            "reached": torch.as_tensor(reached, dtype=torch.bool, device=self.config.device),
            "collision": torch.as_tensor(collision, dtype=torch.bool, device=self.config.device),
            "timeout": torch.as_tensor(timeout, dtype=torch.bool, device=self.config.device),
            "distance_to_goal": torch.as_tensor(current_goal_dist, dtype=torch.float32, device=self.config.device),
            "distance_to_progress_target": torch.as_tensor(
                current_progress_dist, dtype=torch.float32, device=self.config.device
            ),
            "progress": torch.as_tensor(progress, dtype=torch.float32, device=self.config.device),
            "reward_total": torch.as_tensor(rewards, dtype=torch.float32, device=self.config.device),
            "reward_time": torch.as_tensor(time_reward, dtype=torch.float32, device=self.config.device),
            "reward_progress": torch.as_tensor(progress_reward, dtype=torch.float32, device=self.config.device),
            "reward_velocity_direction": torch.as_tensor(
                velocity_direction_reward, dtype=torch.float32, device=self.config.device
            ),
            "reward_proximity": torch.as_tensor(
                proximity_penalty, dtype=torch.float32, device=self.config.device
            ),
            "reward_static_proximity": torch.as_tensor(
                static_proximity_penalty,
                dtype=torch.float32,
                device=self.config.device,
            ),
            "reward_smoothness": torch.as_tensor(
                action_smoothness_penalty, dtype=torch.float32, device=self.config.device
            ),
            "reward_angular_velocity": torch.as_tensor(
                angular_velocity_penalty, dtype=torch.float32, device=self.config.device
            ),
            "reward_angular_change": torch.as_tensor(
                angular_change_penalty, dtype=torch.float32, device=self.config.device
            ),
            "reward_linear_change": torch.as_tensor(
                linear_change_penalty, dtype=torch.float32, device=self.config.device
            ),
            "reward_goal": torch.as_tensor(goal_reward, dtype=torch.float32, device=self.config.device),
            "reward_collision": torch.as_tensor(collision_reward, dtype=torch.float32, device=self.config.device),
            "reward_timeout": torch.as_tensor(timeout_reward, dtype=torch.float32, device=self.config.device),
            "stuck": torch.as_tensor(stuck, dtype=torch.bool, device=self.config.device),
            "reward_stuck": torch.as_tensor(stuck_reward, dtype=torch.float32, device=self.config.device),
            "is_navigation_step": torch.ones(
                self.config.num_robots,
                dtype=torch.bool,
                device=self.config.device,
            ),
        }

    def _build_robot_rl_observations(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> torch.Tensor:
        if self.config.num_robots == 0:
            return torch.zeros((0, self.robot_rl_vector_obs_dim), dtype=torch.float32, device=self.config.device)

        robot_offset     = self.config.num_humanoids
        robot_agent_ids  = robot_offset + np.arange(self.config.num_robots)
        yaws             = self._robot_yaws()                # (N,)
        angular_vels     = self._robot_angular_velocities()  # (N,)
        max_linear_speed = max(float(self.config.rl_max_linear_velocity), 1e-4)
        max_angular_speed= max(float(self.config.rl_max_angular_velocity), 1e-4)
        # Keep the observation scale fixed while curriculum changes only the
        # reset sampling range. Otherwise every curriculum update would alter
        # observations for robots already in the middle of an episode.
        max_goal_dist = float(self.config.rl_goal_observation_max_distance)

        robot_pos = positions[robot_agent_ids, :2]   # (N, 2)
        robot_vel = velocities[robot_agent_ids, :2]  # (N, 2)

        # ── Goal: dist + sin/cos of heading-relative angle ───────────
        goal_vecs         = self.goals_xy[robot_agent_ids] - robot_pos  # (N, 2)
        goal_dists        = np.linalg.norm(goal_vecs, axis=1)           # (N,)
        goal_world_angles = np.arctan2(goal_vecs[:, 1], goal_vecs[:, 0])
        goal_heading_angles = np.arctan2(
            np.sin(goal_world_angles - yaws), np.cos(goal_world_angles - yaws)
        )
        goal_obs = np.stack(
            [goal_dists / max_goal_dist, np.sin(goal_heading_angles), np.cos(goal_heading_angles)],
            axis=1,
        )  # (N, 3)

        # ── Self motion: forward speed + angular velocity ─────────────
        heading_vecs = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)   # (N, 2)
        forward_speed = np.sum(robot_vel * heading_vecs, axis=1)         # (N,)
        motion_obs = np.stack(
            [forward_speed / max_linear_speed, angular_vels / max_angular_speed], axis=1
        )  # (N, 2)

        # Neighbor agents are exposed as a standalone set channel.  Keeping
        # the base vector to goal + ego motion prevents slot ordering from
        # leaking into the generic MLP input.
        return torch.as_tensor(
            np.concatenate([goal_obs, motion_obs], axis=1),
            dtype=torch.float32, device=self.config.device,
        )

    def _build_robot_neighbor_observations(
        self, positions: np.ndarray, velocities: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Top-K neighbors as an unordered set and a validity mask."""
        n_r, k_max = self.config.num_robots, self.config.rl_num_neighbors
        feat = np.zeros((n_r, k_max, 5), dtype=np.float32)
        valid_mask = np.zeros((n_r, k_max), dtype=bool)
        if n_r == 0 or k_max == 0:
            return (
                torch.as_tensor(feat, device=self.config.device),
                torch.as_tensor(valid_mask, device=self.config.device),
            )
        offset = self.config.num_humanoids
        robot_ids = offset + np.arange(n_r)
        robot_pos, robot_vel = positions[robot_ids, :2], velocities[robot_ids, :2]
        yaws = self._robot_yaws()
        headings = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
        laterals = np.stack([-headings[:, 1], headings[:, 0]], axis=1)
        radius = max(float(self.config.neighbor_radius), 1e-4)
        max_speed = max(float(self.config.rl_max_linear_velocity), 1e-4)
        distances = np.linalg.norm(robot_pos[:, None, :] - positions[None, :, :2], axis=2)
        distances[np.arange(n_r), robot_ids] = np.inf
        for rid in range(n_r):
            candidate_ids = np.flatnonzero(distances[rid] <= radius)
            if candidate_ids.size == 0:
                continue
            ids = candidate_ids[
                np.argsort(distances[rid, candidate_ids], kind="stable")[:k_max]
            ]
            count = len(ids)
            rel_pos = positions[ids, :2] - robot_pos[rid]
            angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0]) - yaws[rid]
            angles = np.arctan2(np.sin(angles), np.cos(angles))
            rel_vel = velocities[ids, :2] - robot_vel[rid]
            feat[rid, :count] = np.stack([
                distances[rid, ids] / radius,
                np.sin(angles), np.cos(angles),
                rel_vel @ headings[rid] / max_speed,
                rel_vel @ laterals[rid] / max_speed,
            ], axis=1)
            valid_mask[rid, :count] = True
        return (
            torch.as_tensor(feat, dtype=torch.float32, device=self.config.device),
            torch.as_tensor(valid_mask, dtype=torch.bool, device=self.config.device),
        )

    @staticmethod
    def _polar_agent_obs(
        self_pos: np.ndarray, self_yaw: float, self_vel: np.ndarray,
        positions: np.ndarray, velocities: np.ndarray,
        exclude_agent_id: int, nb_radius: float, max_lin: float,
    ) -> list[float]:
        """Polar position + relative velocity for the closest agent."""
        dists = np.linalg.norm(positions - self_pos, axis=1)
        dists[exclude_agent_id] = np.inf
        best = int(np.argmin(dists))
        dist = float(dists[best])
        if dist >= nb_radius:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        rel = positions[best] - self_pos
        angle = math.atan2(float(rel[1]), float(rel[0]))
        rel_angle = math.atan2(math.sin(angle - self_yaw), math.cos(angle - self_yaw))
        rel_vel = velocities[best] - self_vel
        return [
            dist / nb_radius,
            math.sin(rel_angle),
            math.cos(rel_angle),
            float(rel_vel[0]) / max_lin,
            float(rel_vel[1]) / max_lin,
        ]

    def _polar_neighbor_observations(
        self, self_pos: np.ndarray, self_yaw: float, self_vel: np.ndarray,
        positions: np.ndarray, velocities: np.ndarray, exclude_id: int,
    ) -> list[float]:
        """Polar position + relative velocity for N nearest neighbors."""
        dists = np.linalg.norm(positions - self_pos, axis=1)
        dists[exclude_id] = np.inf
        valid = (dists <= self.config.neighbor_radius)
        sorted_ids = np.argsort(np.where(valid, dists, np.inf))[: self.config.rl_num_neighbors]
        nb_radius = max(float(self.config.neighbor_radius), 1e-4)
        max_lin = max(float(self.config.rl_max_linear_velocity), 1e-4)

        values: list[float] = []
        used = 0
        for nid in sorted_ids:
            if not valid[nid]:
                continue
            d = float(dists[nid])
            rel = positions[nid] - self_pos
            angle = math.atan2(float(rel[1]), float(rel[0]))
            rel_angle = math.atan2(math.sin(angle - self_yaw), math.cos(angle - self_yaw))
            rel_vel = velocities[nid] - self_vel
            values.extend([
                d / nb_radius,
                math.sin(rel_angle),
                math.cos(rel_angle),
                float(rel_vel[0]) / max_lin,
                float(rel_vel[1]) / max_lin,
            ])
            used += 1
        for _ in range(self.config.rl_num_neighbors - used):
            values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        return values

    def _local_obstacle_patch(self, xy: np.ndarray, yaw: float) -> np.ndarray:
        """Build an ego-centric obstacle patch centred on *xy* with heading *yaw*.

        Returns a ``(size, size)`` float32 array where 1.0 = obstacle / unknown
        and 0.0 = free.  Out-of-map cells are treated as obstacles (1.0).

        Replaces an equivalent double Python loop with fully vectorised NumPy
        operations (≈20× faster; output is bit-for-bit identical).
        """
        size = max(0, int(self.config.rl_map_size))
        if size <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        extent = max(float(self.config.rl_map_extent), self.config.map_resolution)
        cell = extent / float(size)
        # coords[i] = centre of the i-th grid cell along one local axis
        coords = (np.arange(size, dtype=np.float32) + 0.5) * cell - 0.5 * extent
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        origin_x, origin_y = self.config.map_origin_xy

        # local_x: row axis (forward),  shape (size, 1)  — coords reversed so row 0 = most forward
        # local_y: col axis (lateral),  shape (1, size)
        local_x = coords[::-1][:, None]   # (size, 1)
        local_y = coords[None, :]          # (1, size)

        # World-frame XY for every grid cell,  shape (size, size)
        world_x = xy[0] + forward[0] * local_x + lateral[0] * local_y
        world_y = xy[1] + forward[1] * local_x + lateral[1] * local_y

        # Map → pixel indices (same rounding as the original int(round(...)))
        pixel_x = np.round((world_x - origin_x) / self.config.map_resolution).astype(np.int32)
        pixel_y = np.round(
            (self.height - 1) - (world_y - origin_y) / self.config.map_resolution
        ).astype(np.int32)

        valid = (
            (pixel_y >= 0) & (pixel_y < self.height) &
            (pixel_x >= 0) & (pixel_x < self.width)
        )
        # Default: 1.0 (obstacle) for out-of-bounds cells — matches original behaviour
        patch = np.ones((size, size), dtype=np.float32)
        py_c = np.clip(pixel_y, 0, self.height - 1)
        px_c = np.clip(pixel_x, 0, self.width - 1)
        patch[valid] = (self.obstacle_map[py_c[valid], px_c[valid]] > 0).astype(np.float32)
        return patch

    def _robot_angular_velocities(self) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros(0, dtype=np.float32)
        if self.drive is not None:
            return self.drive.angular_velocities()
        ang_vel = self.robot.data.root_ang_vel_w[: self.config.num_robots]
        return ang_vel[:, 2].detach().cpu().numpy().astype(np.float32)

    def _robot_yaws(self) -> np.ndarray:
        if self.config.num_robots == 0:
            return np.zeros(0, dtype=np.float32)
        quats = self.robot.data.root_quat_w[: self.config.num_robots]
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)).detach().cpu().numpy().astype(np.float32)

    def robot_min_dist(self, robot_id: int) -> float:
        """Distance from car *robot_id* to its nearest neighbour agent."""
        offset = self.config.num_humanoids
        pos = self.drive.positions_xy()[robot_id]
        best = float("inf")
        for i in range(self.num_agents):
            if i == offset + robot_id:
                continue
            d = float(np.linalg.norm(pos - self._last_positions[i]))
            if d < best:
                best = d
        return best

    def robot_neighbors_xy(self, robot_id: int) -> list[list[float]]:
        """Return ``[[x,y], ...]`` of agents within *neighbor_radius* of car *robot_id*."""
        offset = self.config.num_humanoids
        pos = self.drive.positions_xy()[robot_id]
        result: list[list[float]] = []
        for i in range(self.num_agents):
            if i == offset + robot_id:
                continue
            d = float(np.linalg.norm(pos - self._last_positions[i]))
            if d < self.config.neighbor_radius:
                result.append(self._last_positions[i].tolist())
        return result

    def robot_yaws(self) -> np.ndarray:
        """Yaw angles for all cars."""
        return self.drive.yaws() if self.drive is not None else np.zeros(0, dtype=np.float32)

    def _sample_spaced_free_xy(self, positions: np.ndarray, exclude_agent_id: int) -> np.ndarray | None:
        """Sample a free pixel at least min_spawn_spacing from all other agents.

        Also avoids frozen filler slots' parked positions (not in ``positions``,
        which only holds active agents) so a reset robot doesn't spawn on top
        of a parked humanoid/robot and physically interpenetrate.
        """
        # Collect parked positions of frozen filler slots.  Frozen humanoid
        # slots exist when num_envs > num_humanoids; frozen robot slots when
        # num_envs > num_robots.  (num_envs > num_agents is impossible since
        # max(n_h,n_r) <= n_h+n_r, so that earlier guard was dead code.)
        frozen_points: list[np.ndarray] = []
        if self.env is not None:
            n_h = self.config.num_humanoids
            n_r = self.config.num_robots
            if self.env.num_envs > n_h:
                frozen_points.append(self._frozen_slot_xy())
            if self.env.num_envs > n_r:
                frozen_points.append(self._frozen_robot_park_xy())
        for attempt in range(100):  # max attempts
            px = self.task._sample_free_pixel()
            if px is None:
                return None
            candidate_xy = self.task.pixel_to_world(px)
            too_close = False
            for other_id in range(self.num_agents):
                if other_id == exclude_agent_id:
                    continue
                if np.linalg.norm(candidate_xy - positions[other_id]) < self.config.min_spawn_spacing:
                    too_close = True
                    break
            if not too_close:
                for fp in frozen_points:
                    if np.linalg.norm(candidate_xy - fp) < self.config.min_spawn_spacing:
                        too_close = True
                        break
            if not too_close:
                return candidate_xy
        print(f"[CrowdSim] WARNING: _sample_spaced_free_xy failed after 100 attempts for agent {exclude_agent_id}")
        return None

    def _sync_robot_prim_transforms(self) -> None:
        """Sync Car USD prim transforms from articulation data (for camera tracking)."""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
        except ImportError:
            return
        stage = omni.usd.get_context().get_stage()
        root_pos = self.robot.data.root_pos_w.detach().cpu().numpy()
        root_quat = self.robot.data.root_quat_w.detach().cpu().numpy()
        for env_id in range(self.config.num_robots):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Car")
            if not prim.IsValid():
                continue
            xform = UsdGeom.Xformable(prim)
            for op in xform.GetOrderedXformOps():
                name = op.GetName()
                if name == "xformOp:translate":
                    op.Set(Gf.Vec3d(float(root_pos[env_id, 0]),
                                    float(root_pos[env_id, 1]),
                                    float(root_pos[env_id, 2])))
                elif "orient" in name:
                    q = root_quat[env_id]  # wxyz
                    op.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    def _paint_scene_objects_to_map(self, objects: dict) -> None:
        """Paint scene object footprints onto the obstacle map so A* avoids them.

        Args:
            objects: Dict mapping an integer key to a flat list
                     ``[x, y, z, w, d, h]`` where ``(x, y)`` is the object
                     center in world coordinates and ``(w, d)`` are its width
                     and depth (the ``z`` and ``h`` components are ignored here).
                     Example: ``{0: [1.0, 2.0, 0.0, 0.8, 1.2, 1.5], ...}``
        """
        import cv2
        for _k, v in objects.items():
            cx, cy = float(v[0]), float(v[1])
            hw, hd = float(v[3]) / 2, float(v[4]) / 2
            obj_type = v[6] if len(v) > 6 else "box"
            # Object centre in pixel coordinates (world_to_pixel returns [py, px])
            pc = self.world_to_pixel(np.array([cx, cy]))
            py_c, px_c = int(pc[0]), int(pc[1])
            res = float(self.config.map_resolution)

            if obj_type == "cylinder":
                # Cylinder footprint: a filled circle of radius = max(hw, hd).
                # The old code painted it as an axis-aligned box AABB, which
                # over-inflated the four corners — the policy was penalised /
                # A* detoured around free space next to the cylinder's sides.
                r_px = max(1, int(round(max(hw, hd) / res)))
                cv2.circle(self.obstacle_map, (px_c, py_c), r_px, 1, -1)
                # free_mask is bool; cv2.circle needs uint8 — use a persistent
                # view so the modification is in-place (astype creates a copy).
                free_u8 = self.free_mask.view(np.uint8)
                cv2.circle(free_u8, (px_c, py_c), r_px, 0, -1)
            else:
                # Box footprint: axis-aligned bounding box (unchanged behaviour).
                corners = [
                    (cx - hw, cy - hd),
                    (cx + hw, cy - hd),
                    (cx + hw, cy + hd),
                    (cx - hw, cy + hd),
                ]
                px_corners = [self.world_to_pixel(np.array([x, y])) for x, y in corners]
                ys = sorted(p[0] for p in px_corners)
                xs = sorted(p[1] for p in px_corners)
                y_min, y_max = max(0, ys[0]), min(self.height - 1, ys[-1])
                x_min, x_max = max(0, xs[0]), min(self.width - 1, xs[-1])
                self.obstacle_map[y_min:y_max + 1, x_min:x_max + 1] = 1
                self.free_mask[y_min:y_max + 1, x_min:x_max + 1] = 0
        # Rebuild SFM controller with updated distance transform
        self.sfm_controller = self._make_sfm_controller(self.config.agent_radius)
        # H2+H12: rebuild A* planner's dilated map so new obstacles are respected
        # (previously the planner still used the original map without scene objects).
        self.task.planner.map_dialate = self.task.planner.post_proc_map(self.obstacle_map)
        self.task.planner_free_mask   = self.task.planner.map_dialate == 0
        self.task._refresh_free_pixels_cache()
        print("[CrowdSim] A* planner map rebuilt after scene object update.")

    def _make_sfm_controller(self, radius: float) -> Social_Force:
        import cv2

        free_uint8 = self.free_mask.astype(np.uint8)
        distance_px = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
        return Social_Force(distance_px * self.config.map_resolution, self._planner_cfg(radius))

    def _waypoint_desired_velocity(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - pos
        distance = float(np.linalg.norm(delta))
        if distance < 1e-5:
            return np.zeros(2, dtype=np.float32)
        speed = min(self.config.max_speed, distance / max(self._dt(), 1e-5))
        return (delta / distance * speed).astype(np.float32)

    def _clip_pixel_yx(self, pixel_yx: np.ndarray) -> np.ndarray:
        return np.array(
            [
                int(np.clip(pixel_yx[0], 0, self.height - 1)),
                int(np.clip(pixel_yx[1], 0, self.width - 1)),
            ],
            dtype=np.int64,
        )

    def _humanoid_target_offsets(self, num_future_steps: int) -> np.ndarray:
        # local_target_timestep is a dt multiplier, as in the online SFM target.
        # Ensure at least one simulation step so offsets are never zero.
        timestep = max(float(self.config.local_target_timestep) * self._dt(), self._dt())
        return timestep * np.arange(1, num_future_steps + 1, dtype=np.float32)

    def _humanoid_future_targets_from_path(
        self, current_xy: np.ndarray, offsets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the complete MaskedMimic target horizon along avoidance velocity."""
        num_humanoids = min(self.config.num_humanoids, current_xy.shape[0])
        num_steps = len(offsets)
        targets = np.zeros((num_humanoids, num_steps, 2), dtype=np.float32)
        yaws = np.zeros((num_humanoids, num_steps), dtype=np.float32)

        for agent_id in range(num_humanoids):
            pos = np.asarray(current_xy[agent_id], dtype=np.float32)
            desired_velocity = self._sfm_desired_velocities[agent_id]
            actual_velocity = self._humanoid_actual_velocities[agent_id]
            desired_speed = float(np.linalg.norm(desired_velocity))
            actual_speed = float(np.linalg.norm(actual_velocity))
            desired_yaw = (
                math.atan2(float(desired_velocity[1]), float(desired_velocity[0]))
                if desired_speed >= 1e-5 else float(self._humanoid_target_yaws[agent_id])
            )
            yaw = (
                math.atan2(float(actual_velocity[1]), float(actual_velocity[0]))
                if actual_speed >= self.config.humanoid_target_min_heading_speed
                else float(self._humanoid_target_yaws[agent_id])
            )
            speed = actual_speed
            previous_time = 0.0
            previous = pos.copy()
            last_yaw = float(self._humanoid_target_yaws[agent_id])

            for step_id, offset in enumerate(offsets):
                interval = max(float(offset) - previous_time, 0.0)
                yaw_error = math.atan2(
                    math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw)
                )
                yaw += float(np.clip(
                    yaw_error,
                    -self.config.humanoid_max_yaw_rate * interval,
                    self.config.humanoid_max_yaw_rate * interval,
                ))
                speed += float(np.clip(
                    desired_speed - speed,
                    -self.config.humanoid_max_acceleration * interval,
                    self.config.humanoid_max_acceleration * interval,
                ))
                target = previous + interval * speed * np.asarray(
                    [math.cos(yaw), math.sin(yaw)], dtype=np.float32
                )
                targets[agent_id, step_id] = target
                direction = target - previous
                if np.linalg.norm(direction) < 1e-5:
                    direction = target - pos
                if np.linalg.norm(direction) >= 1e-5:
                    last_yaw = math.atan2(float(direction[1]), float(direction[0]))
                yaws[agent_id, step_id] = last_yaw
                previous = target
                previous_time = float(offset)

        if num_steps > 0:
            self._humanoid_future_first_targets[:num_humanoids] = targets[:, 0]
            self._humanoid_future_first_yaws[:num_humanoids] = yaws[:, 0]

        return targets, yaws

    @staticmethod
    def _filter_humanoid_velocity(
        previous: np.ndarray,
        desired: np.ndarray,
        *,
        smoothing: float,
        max_acceleration: float,
        dt: float,
    ) -> np.ndarray:
        """Low-pass an SFM command, then enforce a metric acceleration bound."""
        previous = np.asarray(previous, dtype=np.float32)
        desired = np.asarray(desired, dtype=np.float32)
        filtered = previous + float(smoothing) * (desired - previous)
        delta = filtered - previous
        delta_norm = float(np.linalg.norm(delta))
        max_delta = float(max_acceleration) * float(dt)
        if delta_norm > max_delta:
            filtered = previous + delta * (max_delta / max(delta_norm, 1e-8))
        return filtered.astype(np.float32)

    def _dt(self) -> float:
        return self._update_interval_steps * self._env_dt

    def _should_update_navigation(self) -> bool:
        return self.env_step_count % self._update_interval_steps == 0

    def _navigation_update_hz(self) -> float:
        return 1.0 / max(self._dt(), 1e-6)

    def _compute_update_interval_steps(self, env_dt: float) -> int:
        update_period = 1.0 / max(float(self.config.update_hz), 1e-6)
        return max(1, int(round(update_period / max(env_dt, 1e-6))))

    def _refresh_controller_dt(self) -> None:
        if hasattr(self.sfm_controller, "dt"):
            self.sfm_controller.dt = self._dt()

    @staticmethod
    def _read_env_dt(env) -> float:
        env_dt = float(getattr(env, "dt", 0.0) or 0.0)
        if env_dt > 0.0:
            return env_dt
        simulator = getattr(env, "simulator", None)
        sim_dt = float(getattr(simulator, "dt", 0.0) or 0.0)
        if sim_dt > 0.0:
            return sim_dt
        return 1.0 / 25.0

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in quat]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_to_quat_tensor(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), dtype=yaw.dtype, device=yaw.device)
        half_yaw = 0.5 * yaw
        quat[:, 0] = torch.cos(half_yaw)
        quat[:, 3] = torch.sin(half_yaw)
        return quat

    @staticmethod
    def _yaw_to_quat_xyzw_tensor(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), dtype=yaw.dtype, device=yaw.device)
        half_yaw = 0.5 * yaw
        quat[:, 2] = torch.sin(half_yaw)
        quat[:, 3] = torch.cos(half_yaw)
        return quat
