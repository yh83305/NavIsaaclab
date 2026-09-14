"""Shared CrowdSim scene + runtime + navigation initialization.

Called by both train_ppo.py and crowd_sim.py to avoid ~80 lines of duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CrowdSim.protomotions_runtime import (  # noqa: E402
    build_runtime, configure_viewer_camera, create_fabric,
    enable_human_mesh, make_crowd_robot_config, resolve_robot_usd,
    suppress_known_isaaclab_warning_spam,
)
from CrowdSim.scene_setup import (  # noqa: E402
    apply_fixed_crowd_robot_spawns, apply_fixed_spawn_offsets,
    hide_inactive_crowd_robot_visuals,
    parent_camera_to_robot, patch_isaaclab_scene_with_crowdsim_assets,
    remove_scene_prims, resolve_repo_path, spawn_scene_objects,
)
from CrowdSim.utils.sensor_stream import (  # noqa: E402
    RobotCameraStreamConfig, configure_robot_camera_recorder,
)
from CrowdSim.utils.humanoid_state_recorder import (  # noqa: E402
    HumanoidStateRecorderConfig, configure_humanoid_state_recorder,
)
from CrowdSim.utils.map_metadata import load_occupancy_map_metadata  # noqa: E402
from CrowdSim.nav_manager import CrowdNavigationConfig, CrowdNavigationManager  # noqa: E402
from CrowdSim.control.drive import DifferentialDriveConfig  # noqa: E402


def _resolve_scene(scene: Any) -> dict:
    """Resolve a scene config that may be a string (scene name) or a dict.

    String values are looked up in ``CrowdSim/config/scenes/<name>.yaml``.
    """
    if isinstance(scene, dict):
        return scene
    name = str(scene)
    scenes_dir = Path(__file__).resolve().parent.parent / "config" / "scenes"
    path = scenes_dir / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene '{name}' not found — no such file {path}. "
            f"Available: {sorted(p.stem for p in scenes_dir.glob('*.yaml'))}"
        )
    from CrowdSim.utils.config_loader import load_config
    cfg = load_config(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Scene config {path} did not parse to a dict")
    return cfg


def build_env(config: dict, *,
              num_envs: int | None = None, headless: bool = False,
              scene_physics: bool = False, empty_mode: bool = False):
    """Initialize scene, runtime, and navigation. Returns (env, agent, nav).

    ``num_envs`` (IsaacLab scene clone count) is inferred from the active
    agent counts when not given: ``max(num_humanoids, num_robots, 1)``.
    Each env holds one humanoid slot + one robot slot; slots beyond the
    active counts are frozen so they don't participate in navigation.
    """

    config.setdefault("navigation", {})["enabled"] = True
    config.setdefault("car", {})

    scene_cfg = _resolve_scene(config["scene"])
    humanoid_cfg = config["humanoid"]
    car_cfg = config["car"]
    sensor_cfg = config.get("sensors", {})

    # Infer num_envs from active agent counts if not explicitly provided.
    # Default 20 matches env.yaml's num_humanoids/num_robots so a missing
    # config still yields a runnable crowd; _make_nav_config uses the same
    # default (capped to num_envs) for consistency.
    nav_cfg = config["navigation"]
    n_h = int(nav_cfg.get("num_humanoids", 20))
    n_r = int(nav_cfg.get("num_robots", 20))
    min_envs = max(n_h, n_r, 1)

    if num_envs is None:
        num_envs = min_envs
    elif num_envs < min_envs:
        print(f"[CrowdSim] WARNING: num_envs={num_envs} < max(num_humanoids={n_h}, "
              f"num_robots={n_r}); capping active counts to {num_envs}")

    checkpoint = resolve_repo_path(humanoid_cfg["checkpoint"])
    motion_file = resolve_repo_path(humanoid_cfg["motion_file"])
    scene_usd = resolve_repo_path(scene_cfg["scene_usd"])
    map_meta = load_occupancy_map_metadata(resolve_repo_path(scene_cfg["scene_map"]))

    _validate(checkpoint, motion_file, scene_usd, map_meta)

    if humanoid_cfg.get("human_mesh"):
        enable_human_mesh(
            model_dir=humanoid_cfg.get("smpl_model_dir"),
            hide_humanoid=bool(humanoid_cfg.get("hide_humanoid", False)),
            texture_dir=humanoid_cfg.get("texture_dir"),
            appearance_seed=int(humanoid_cfg.get("appearance_seed", 42)),
            shape_std=float(humanoid_cfg.get("shape_std", 0.55)),
            shape_clip=float(humanoid_cfg.get("shape_clip", 1.25)),
        )

    fabric = create_fabric()
    launcher = {"headless": headless, "device": str(fabric.device)}
    if sensor_cfg.get("camera", {}).get("enabled"):
        launcher["enable_cameras"] = True
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    AppLauncher = import_simulator_before_torch("isaaclab")
    app = AppLauncher(launcher)
    suppress_known_isaaclab_warning_spam()

    if empty_mode:
        from CrowdSim.scene_setup import add_global_usd_reference
        add_global_usd_reference(scene_usd, prim_path=scene_cfg.get("prim_path", "/World/Scene"),
                                 z_offset=float(scene_cfg.get("z_offset", 0.0)))

        # Remove unwanted prims so the occupancy map captures the real layout.
        rm_patterns = scene_cfg.get("remove_prims", [])
        if rm_patterns:
            remove_scene_prims(list(rm_patterns))

        # Spawn static obstacles so they appear in the viewport and are baked
        # into the occupancy map when the user exports it.
        # No build_runtime / env here — spawn_scene_objects only needs the
        # USD stage which is already live after add_global_usd_reference.
        objs = scene_cfg.get("objects")
        if objs and isinstance(objs, dict):
            pos_list, size_list, shape_list, color_list = [], [], [], []
            for v in objs.values():
                pos_list.append((float(v[0]), float(v[1]), float(v[2])))
                size_list.append((float(v[3]), float(v[4]), float(v[5])))
                shape_list.append(str(v[6]) if len(v) > 6 else "box")
                color_list.append(
                    (float(v[7]), float(v[8]), float(v[9])) if len(v) >= 10 else None
                )
            spawn_scene_objects(
                pos_list, size_list,
                shapes=shape_list,
                colors=color_list,
                static=bool(scene_cfg.get("static_obstacles", True)),
                parent_path="/World/SceneObjects",
            )

        print("[CrowdSim] Warehouse + static obstacles loaded. "
              "Use Tools > Robotics > Occupancy Map to export.")
        import omni.kit.app, time
        kit = omni.kit.app.get_app()
        while kit.is_running():
            kit.update()
            time.sleep(0.01)
        return None

    robot_usd = resolve_robot_usd(car_cfg.get("usd"))
    robot_cfg = make_crowd_robot_config(car_cfg, sensor_cfg, robot_usd)
    scene_loaded = scene_physics or robot_cfg is not None

    if scene_loaded:
        terrain = car_cfg.get("terrain_xy_offset")
        if terrain:
            terrain = (float(terrain[0]), float(terrain[1])) if isinstance(terrain, (list, tuple)) else None
        patch_isaaclab_scene_with_crowdsim_assets(
            scene_usd_path=scene_usd,
            scene_z_offset=float(scene_cfg.get("z_offset", 0.0)),
            scene_prim_path=str(scene_cfg.get("prim_path", "/World/Scene")),
            crowd_robot=robot_cfg,
            terrain_xy_offset=terrain,
        )

    # Expose the active (post-cap) humanoid count BEFORE build_runtime: the SMPL
    # mesh visualizer reads CROWDSIM_NUM_ACTIVE_HUMANOIDS during env construction
    # (inside build_runtime) to decide which humanoid slots to hide.  Frozen
    # filler slots (env_id >= num_active_humanoids) must stay visible so the
    # active robot's depth camera can perceive them.  Setting this after
    # build_runtime would be too late — the visualizer has already read None.
    n_active_humanoids = min(n_h, num_envs)
    os.environ["CROWDSIM_NUM_ACTIVE_HUMANOIDS"] = str(n_active_humanoids)

    runtime = build_runtime(
        checkpoint=checkpoint, motion_file=motion_file,
        num_envs=num_envs, headless=headless,
        simulation_app=app.app, fabric=fabric,
        control_hz=float(nav_cfg.get("update_hz", 25.0)),
    )
    env = runtime.env
    configure_viewer_camera(env, config.get("viewer", {}), headless)
    rec_cfg = config.get("humanoid", {}).get("state_recording", {})
    if rec_cfg and rec_cfg.get("enabled"):
        configure_humanoid_state_recorder(env, HumanoidStateRecorderConfig(
            output_dir=resolve_repo_path(rec_cfg.get("record_dir", "output/crowdsim_humanoid_state")),
            fps=float(rec_cfg.get("record_fps", 30)),
            env_ids=str(rec_cfg.get("record_envs", "0")),
            auto_record=bool(rec_cfg.get("auto_record", False)),
            key=str(rec_cfg.get("key", "H")),
        ))

    if not scene_loaded:
        from CrowdSim.scene_setup import add_global_usd_reference
        add_global_usd_reference(scene_usd, prim_path=scene_cfg.get("prim_path", "/World/Scene"),
                                 z_offset=float(scene_cfg.get("z_offset", 0.0)))

    # Navigation
    nav_cfg = _make_nav_config(map_meta, num_envs, fabric.device, config, scene_cfg)
    nav = CrowdNavigationManager(nav_cfg)

    apply_fixed_spawn_offsets(env, nav.humanoid_spawn_xy_for_all_envs(num_envs))
    apply_fixed_crowd_robot_spawns(env, nav.robot_spawn_poses_for_all_envs(num_envs))
    hidden_cars = hide_inactive_crowd_robot_visuals(env, nav_cfg.num_robots)
    if hidden_cars:
        print(f"[CrowdSim] Hidden {hidden_cars} inactive cloned car visual(s).")

    if sensor_cfg.get("camera", {}).get("enabled"):
        parent_camera_to_robot(env)
        camera_cfg = sensor_cfg["camera"]
        print(
            "[CrowdSim] Robot camera: "
            f"raw={int(camera_cfg.get('width', 640))}x"
            f"{int(camera_cfg.get('height', 480))}, "
            f"horizontal_fov={float(camera_cfg.get('horizontal_fov', 90.0)):.1f}deg, "
            f"RGB input={nav_cfg.rl_rgb_size}x{nav_cfg.rl_rgb_size}, "
            f"depth input={nav_cfg.rl_depth_size}x{nav_cfg.rl_depth_size}."
        )
        configure_robot_camera_recorder(env, RobotCameraStreamConfig(
            output_dir=resolve_repo_path(sensor_cfg["camera"].get("record_dir", "output/crowdsim_camera")),
            fps=float(sensor_cfg["camera"].get("record_fps", 10)),
            env_ids=str(sensor_cfg["camera"].get("record_envs", "0")),
            auto_record=bool(sensor_cfg["camera"].get("auto_record", False)),
        ))

    # Remove unwanted prims from the warehouse USD (can run before or after attach).
    rm_patterns = scene_cfg.get("remove_prims", [])
    if rm_patterns:
        remove_scene_prims(list(rm_patterns))

    # nav.attach patches env.reset() and must run before the main loop.
    # spawn_scene_objects is called AFTER this so the USD stage is in the
    # stable post-build_runtime state (build_runtime already ran the first
    # sim.reset internally).  Adding prims before build_runtime completes
    # risks them being wiped by the framework's own initialisation resets.
    nav.attach(env)

    # ── SMPL mesh overlay for humanoids ──────────────────────────
    # enable_human_mesh() only sets env vars; the actual UsdGeom.Mesh overlay
    # (with texture) is created here and updated each step from nav_manager so
    # the depth/rgb cameras see textured humans instead of the bare skeleton.
    if humanoid_cfg.get("human_mesh"):
        try:
            from CrowdSim.tools.smpl_mesh_visualizer import ProtoMotionsHumanMeshAdapter
            nav._human_mesh_adapter = ProtoMotionsHumanMeshAdapter.from_simulator(env.simulator)
            nav._human_mesh_adapter.create()
            print("[CrowdSim] SMPL human mesh overlay enabled (textured).")
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: training works with the bare skeleton; the overlay is
            # visual only.  Log once and continue.
            print(f"[CrowdSim] SMPL human mesh overlay FAILED ({exc}); "
                  f"falling back to bare skeleton (visual only, training unaffected).")
            nav._human_mesh_adapter = None
    else:
        nav._human_mesh_adapter = None

    # Spawn static obstacle geometry AFTER the stage is stable.
    # parent_path must live under /World directly, NOT under /World/Scene
    # (which is a USD reference prim — dynamic children are dropped on reload).
    objs = scene_cfg.get("objects")
    if objs and isinstance(objs, dict):
        pos_list, size_list, shape_list, color_list = [], [], [], []
        for v in objs.values():
            pos_list.append((float(v[0]), float(v[1]), float(v[2])))
            size_list.append((float(v[3]), float(v[4]), float(v[5])))
            shape_list.append(str(v[6]) if len(v) > 6 else "box")
            if len(v) >= 10:
                color_list.append((float(v[7]), float(v[8]), float(v[9])))
            else:
                color_list.append(None)   # auto-assign via HSV
        static_obstacles = bool(scene_cfg.get("static_obstacles", True))
        spawn_scene_objects(
            pos_list, size_list,
            shapes=shape_list,
            colors=color_list,
            static=static_obstacles,
            parent_path="/World/SceneObjects",
        )

    print("[CrowdSim] Scene ready.")
    return env, runtime.agent, nav, runtime


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------

def _validate(checkpoint, motion_file, scene_usd, map_meta):
    for label, path in [("Checkpoint", checkpoint), ("Motion file", motion_file),
                         ("Scene USD", scene_usd), ("Scene map", map_meta.image_path)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def _make_nav_config(map_meta, num_envs, device, config, scene_cfg):
    nav = config.get("navigation", {})
    path = nav.get("path", {})
    local = nav.get("local", {})
    recording = nav.get("recording", {})
    car = config.get("car", {})
    humanoid = config.get("humanoid", {})
    rl = nav.get("rl", {})
    markers = config.get("markers", {})
    reward = rl.get("reward", {})
    split_smoothness = any(
        key in reward
        for key in ("angular_velocity_scale", "angular_change_scale", "linear_change_scale")
    )
    return CrowdNavigationConfig(
        map_path=map_meta.image_path, map_resolution=map_meta.resolution,
        free_threshold=map_meta.free_threshold, map_origin_xy=map_meta.origin_xy,
        # Cap active counts to num_envs so frozen slots beyond num_envs are
        # never created (e.g. CBF forces num_envs=1 → 1 humanoid + 1 robot).
        # Default 20 matches builder's inference default.
        num_humanoids=min(int(nav.get("num_humanoids", 20)), num_envs),
        num_robots=min(int(nav.get("num_robots", 20)), num_envs),
        device=device,
        seed=int(path.get("seed", 7)),
        agent_radius=float(local.get("agent_radius", 0.35)),
        safe_distance=float(local.get("safe_distance", 0.9)),
        max_speed=float(local.get("max_speed", 1.5)),
        waypoint_tolerance=float(nav.get("waypoint_tolerance", 0.45)),
        goal_tolerance=float(nav.get("goal_tolerance", 0.75)),
        min_start_goal_distance=float(path.get("min_start_goal_distance", 7)),
        max_start_goal_distance=float(path.get("max_start_goal_distance", 10)),
        min_spawn_spacing=float(path.get("min_spawn_spacing", 1.5)),
        planning_step_size=float(path.get("planning_step_size", 0.5)),
        planning_clearance=float(path.get("planning_clearance", 0.2)),
        neighbor_radius=float(local.get("neighbor_radius", 4)),
        humanoid_interaction_radius=float(
            local.get("interaction_radius", local.get("neighbor_radius", 4))
        ),
        collision_distance=float(nav.get("collision_distance", 0.7)),
        log_interval=int(nav.get("log_interval", 120)),
        update_hz=float(nav.get("update_hz", 25.0)),
        trajectory_recording_enabled=bool(recording.get("enabled", True)),
        trajectory_output_dir=Path(str(recording.get("output_dir", "output/crowdsim_navigation"))),
        local_target_timestep=float(local.get("target_timestep", 10.0)),
        humanoid_velocity_smoothing=float(local.get("velocity_smoothing", 0.3)),
        humanoid_max_acceleration=float(local.get("max_acceleration", 1.5)),
        humanoid_max_yaw_rate=float(local.get("max_yaw_rate", 1.5)),
        humanoid_prediction_horizon=float(local.get("prediction_horizon", 2.0)),
        humanoid_ttc_threshold=float(local.get("ttc_threshold", 1.5)),
        humanoid_ttc_gain=float(local.get("ttc_gain", 12.0)),
        humanoid_target_min_heading_speed=float(
            humanoid.get("min_heading_speed", 0.05)
        ),
        car_rl_policy=bool(car.get("rl_policy", True)),
        rl_num_neighbors=int(rl.get("num_neighbors", 4)),
        rl_max_linear_velocity=float(rl.get("max_linear_velocity", 1.0)),
        rl_max_angular_velocity=float(rl.get("max_angular_velocity", 2.0)),
        rl_initial_heading_min_offset_degrees=float(
            rl.get("initial_heading_min_offset_degrees", 0.0)
        ),
        rl_initial_heading_max_offset_degrees=float(
            rl.get("initial_heading_max_offset_degrees", 0.0)
        ),
        rl_goal_observation_max_distance=float(
            rl.get(
                "goal_observation_max_distance",
                path.get("max_start_goal_distance", 10.0),
            )
        ),
        rl_goal_curriculum=dict(rl.get("goal_curriculum", {}) or {}),
        rl_progress_reward_scale=float(reward.get("progress_reward_scale", 4.0)),
        rl_goal_reward=float(reward.get("goal_reward", 10.0)),
        rl_collision_penalty=float(reward.get("collision_penalty", -10.0)),
        rl_timeout_penalty=float(reward.get("timeout_penalty", -5.0)),
        rl_time_penalty=float(reward.get("time_penalty", -0.01)),
        rl_velocity_direction_reward_scale=float(reward.get("velocity_direction_reward_scale", 0.1)),
        rl_proximity_penalty_start=float(reward.get("proximity_penalty_start", -1.5)),
        rl_proximity_penalty_threshold=float(reward.get("proximity_penalty_threshold", 1.5)),
        rl_proximity_danger_threshold=float(reward.get("proximity_danger_threshold", 0.5)),
        rl_proximity_danger_penalty=float(reward.get("proximity_danger_penalty", -5.0)),
        rl_static_proximity_threshold=float(
            reward.get("static_proximity_threshold", 0.8)
        ),
        rl_static_proximity_danger_threshold=float(
            reward.get("static_proximity_danger_threshold", 0.25)
        ),
        rl_static_proximity_danger_penalty=float(
            reward.get("static_proximity_danger_penalty", 0.0)
        ),
        rl_stuck_penalty=float(reward.get("stuck_penalty", -30.0)),
        # stuck_window / stuck_threshold may live under rl.reward (training)
        # or directly under rl (collection configs without a reward block).
        rl_stuck_window=int(reward.get("stuck_window", rl.get("stuck_window", 1000))),
        rl_stuck_threshold=float(reward.get("stuck_threshold", rl.get("stuck_threshold", 0.1))),
        rl_action_smoothness_scale=(
            0.0 if split_smoothness
            else float(reward.get("action_smoothness_scale", 0.02))
        ),
        rl_angular_velocity_scale=float(reward.get("angular_velocity_scale", 0.0)),
        rl_angular_change_scale=float(reward.get("angular_change_scale", 0.0)),
        rl_linear_change_scale=float(reward.get("linear_change_scale", 0.0)),
        # max_episode_steps may live under rl.reward (training env.yaml) or
        # directly under rl (collection configs that drop the reward block).
        # Fall back to the rl-level value so collection yamls' 800 isn't
        # silently overridden by the default 600.
        rl_max_episode_steps=int(reward.get("max_episode_steps",
                                            rl.get("max_episode_steps", 600))),
        rl_map_size=int(rl.get("map_size", 24)),
        rl_map_extent=float(rl.get("map_extent", 8.0)),
        scene_objects=scene_cfg.get("objects", {}),
        rl_depth_enabled=bool(rl.get("depth_enabled", True)),
        rl_depth_size=int(rl.get("depth_size", 224)),
        rl_depth_max_range=float(rl.get("depth_max_range", 5.0)),
        rl_depth_use_vit=bool(rl.get("depth_use_vit", False)),
        rl_rgb_enabled=bool(rl.get("rgb_enabled", False)),
        rl_rgb_size=int(rl.get("rgb_size", 224)),
        rl_rgb_use_vit=bool(rl.get("rgb_use_vit", False)),
        scenario=dict(nav.get("scenario", {}) or {}),
        visual_markers_enabled=bool(markers.get("enabled", False)),
    )
