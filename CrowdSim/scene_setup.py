"""Shared helpers for CrowdSim global USD scenes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from CrowdSim.utils.map_metadata import DEFAULT_FREE_THRESHOLD, DEFAULT_MAP_RESOLUTION


DEFAULT_SCENE_MAP_RESOLUTION = DEFAULT_MAP_RESOLUTION
DEFAULT_OFFICE_MAP_RESOLUTION = DEFAULT_SCENE_MAP_RESOLUTION


@dataclass(frozen=True)
class CrowdRobotSceneConfig:
    """Configuration for an optional navigation robot in each ProtoMotions env."""

    usd_path: str
    prim_name: str = "CrowdRobot"
    init_z: float = 0.0
    enable_camera: bool = False
    camera_height: int = 480
    camera_width: int = 640
    camera_horizontal_fov: float = 90.0
    camera_update_period: float = 0.1
    # Relative prim below the articulation root that physically carries the
    # sensor.  Nova Carter exposes its main rigid body as ``chassis_link``.
    # Set this to an empty string only for assets whose root prim is itself the
    # rigid body.
    camera_mount_prim_path: str = "chassis_link"
    # Rigid camera mount in the robot (Car) local frame.  Local +x is forward
    # and +z is up.  Keeping the mount close to the chassis avoids the wide
    # orbit of a rear chase-camera offset when the vehicle steers.  ``rot`` is
    # a ROS-convention quaternion whose optical axis points along local +x.
    camera_pos: tuple[float, float, float] = (0.25, 0.0, 0.6)
    camera_rot: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)
    camera_convention: str = "ros"
    enable_lidar: bool = False
    lidar_update_period: float = 1.0 / 20.0
    lidar_channels: int = 1
    lidar_horizontal_fov: float = 360.0
    lidar_horizontal_res: float = 1.0
    lidar_vertical_fov_min: float = 0.0
    lidar_vertical_fov_max: float = 0.0
    lidar_z_offset: float = 0.35
    lidar_mesh_prim_paths: tuple[str, ...] = ("/World/Scene",)
    debug_vis: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def parse_spawn_xy(value: str, num_envs: int, device: torch.device) -> torch.Tensor:
    pairs: list[tuple[float, float]] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        x_str, y_str = item.split(",", maxsplit=1)
        pairs.append((float(x_str), float(y_str)))

    if not pairs:
        raise ValueError("spawn_xy must contain at least one x,y pair")

    while len(pairs) < num_envs:
        pairs.append(pairs[len(pairs) % len(pairs)])

    return torch.tensor(pairs[:num_envs], dtype=torch.float32, device=device)


def parse_spawn_xy_yaw(value: str, num_envs: int, device: torch.device) -> torch.Tensor:
    """Parse x,y[,yaw] entries separated by semicolons."""
    poses: list[tuple[float, float, float]] = []
    for item in value.split(";"):
        if not item.strip():
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError("robot spawn entries must be x,y or x,y,yaw")
        yaw = float(parts[2]) if len(parts) == 3 else 0.0
        poses.append((float(parts[0]), float(parts[1]), yaw))

    if not poses:
        raise ValueError("robot spawn poses must contain at least one x,y[,yaw] entry")

    while len(poses) < num_envs:
        poses.append(poses[len(poses) % len(poses)])

    return torch.tensor(poses[:num_envs], dtype=torch.float32, device=device)


def sample_spawn_xy_from_map(
    map_path: Path,
    num_envs: int,
    device: torch.device,
    map_resolution: float = DEFAULT_SCENE_MAP_RESOLUTION,
    map_origin_xy: tuple[float, float] | None = None,
    humanoid_radius: float = 0.45,
    min_spacing: float = 0.9,
    free_threshold: int = DEFAULT_FREE_THRESHOLD,
    seed: int = 0,
) -> torch.Tensor:
    """Sample initial XY positions from an occupancy image.

    Image +x maps to world +x, and image -y maps to world +y. If a map origin is
    provided from an occupancy-map YAML, it is used as the lower-left pixel
    center. Otherwise the historical centered-map convention is used.
    """
    if map_resolution <= 0:
        raise ValueError(f"map_resolution must be positive, got {map_resolution}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for CrowdSim PNG map sampling.") from exc

    image = Image.open(map_path).convert("L")
    grid = np.asarray(image, dtype=np.uint8)
    free = grid >= int(free_threshold)
    height, width = free.shape
    origin_xy = (
        map_origin_xy
        if map_origin_xy is not None
        else (
            -0.5 * (width - 1) * map_resolution,
            -0.5 * (height - 1) * map_resolution,
        )
    )
    if not free.any():
        print(f"[CrowdSim] Warning: no free pixels found in {map_path}; using fallback grid.")
        return _fallback_spawn_grid(num_envs, device, min_spacing)

    radius_px = max(1, int(round(humanoid_radius / map_resolution)))
    spacing_px = max(1, int(round(min_spacing / map_resolution)))
    free_integral = _integral_image(free.astype(np.uint8))

    ys, xs = np.nonzero(free)
    stride = max(1, radius_px // 2)
    candidate_pixels = [
        (int(x), int(y))
        for x, y in zip(xs[::stride], ys[::stride])
        if _is_clear_square(free_integral, int(x), int(y), radius_px, width, height)
    ]

    rng = random.Random(seed)
    rng.shuffle(candidate_pixels)

    selected_pixels: list[tuple[int, int]] = []
    min_dist_sq = spacing_px * spacing_px
    for pixel in candidate_pixels:
        if all(
            (pixel[0] - other[0]) ** 2 + (pixel[1] - other[1]) ** 2 >= min_dist_sq
            for other in selected_pixels
        ):
            selected_pixels.append(pixel)
            if len(selected_pixels) == num_envs:
                break

    if len(selected_pixels) < num_envs:
        print(
            f"[CrowdSim] Warning: only found {len(selected_pixels)}/{num_envs} "
            f"free spawn points in {map_path}; filling remaining points with fallback grid."
        )
        fallback = _fallback_spawn_grid(num_envs - len(selected_pixels), device, min_spacing)
        selected_xy = [
            _pixel_to_world(px, py, width, height, map_resolution, origin_xy)
            for px, py in selected_pixels
        ]
        selected_xy.extend((float(x), float(y)) for x, y in fallback.cpu().tolist())
        return torch.tensor(selected_xy[:num_envs], dtype=torch.float32, device=device)

    selected_xy = [
        _pixel_to_world(px, py, width, height, map_resolution, origin_xy)
        for px, py in selected_pixels[:num_envs]
    ]
    print(
        f"[CrowdSim] PNG map spawn: {len(candidate_pixels)} clear candidate pixel(s), "
        f"resolution={map_resolution} m/px, free_threshold={free_threshold}."
    )
    return torch.tensor(selected_xy, dtype=torch.float32, device=device)


def _pixel_to_world(
    pixel_x: int,
    pixel_y: int,
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
) -> tuple[float, float]:
    del width
    origin_x, origin_y = origin_xy
    world_x = origin_x + pixel_x * resolution
    world_y = origin_y + (height - 1 - pixel_y) * resolution
    return world_x, world_y


def _integral_image(mask: np.ndarray) -> np.ndarray:
    values = mask.astype(np.int64, copy=False)
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _is_clear_square(
    integral: np.ndarray,
    x: int,
    y: int,
    radius: int,
    width: int,
    height: int,
) -> bool:
    x0 = max(0, x - radius)
    x1 = min(width - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(height - 1, y + radius)
    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    free_count = (
        integral[y1 + 1, x1 + 1]
        - integral[y0, x1 + 1]
        - integral[y1 + 1, x0]
        + integral[y0, x0]
    )
    return int(free_count) == area


def _fallback_spawn_grid(
    num_envs: int, device: torch.device, spacing: float = 1.0
) -> torch.Tensor:
    side = int(num_envs**0.5)
    if side * side < num_envs:
        side += 1
    origin = -0.5 * spacing * (side - 1)
    points = []
    for i in range(num_envs):
        row = i // side
        col = i % side
        points.append((origin + col * spacing, origin + row * spacing))
    return torch.tensor(points, dtype=torch.float32, device=device)


def add_global_usd_reference(
    usd_path: Path,
    prim_path: str = "/World/Scene",
    z_offset: float = 0.0,
) -> None:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd_path))
    UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, 0.0, z_offset))


def hide_prims_matching_keywords(
    root_prim_path: str = "/World/Scene",
    keywords: tuple[str, ...] = ("ceiling", "cube", "building", "door"),
    deactivate: bool = False,
) -> list[str]:
    """Hide or deactivate prims below a root prim when their path contains any keyword."""
    import omni.usd
    from pxr import Usd, UsdGeom

    normalized_keywords = tuple(keyword.lower() for keyword in keywords if keyword)
    if not normalized_keywords:
        return []

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        return []

    matched_paths: list[str] = []
    for prim in Usd.PrimRange(root_prim):
        path = str(prim.GetPath())
        if path == root_prim_path:
            continue
        path_lower = path.lower()
        if any(keyword in path_lower for keyword in normalized_keywords):
            matched_paths.append(path)

    hidden_paths: list[str] = []
    for path in matched_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        if deactivate:
            prim.SetActive(False)
            hidden_paths.append(path)
        else:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden_paths.append(path)
    return hidden_paths


def patch_isaaclab_scene_with_global_usd(
    usd_path: Path,
    z_offset: float = 0.0,
    prim_path: str = "/World/Scene",
    terrain_xy_offset: tuple[float, float] | None = None,
) -> None:
    """Add a global USD asset to IsaacLab SceneCfg before simulator construction."""
    patch_isaaclab_scene_with_crowdsim_assets(
        scene_usd_path=usd_path,
        scene_z_offset=z_offset,
        scene_prim_path=prim_path,
        terrain_xy_offset=terrain_xy_offset,
    )


def patch_isaaclab_scene_with_crowdsim_assets(
    scene_usd_path: Path | None = None,
    scene_z_offset: float = 0.0,
    scene_prim_path: str = "/World/Scene",
    crowd_robot: CrowdRobotSceneConfig | None = None,
    terrain_xy_offset: tuple[float, float] | None = None,
    office_usd_path: Path | None = None,
    office_z_offset: float | None = None,
    office_prim_path: str | None = None,
) -> None:
    """Add shared CrowdSim assets to IsaacLab SceneCfg before simulator construction."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
    import protomotions.simulator.isaaclab.simulator as simulator_module
    from protomotions.simulator.isaaclab.utils.scene import SceneCfg as BaseSceneCfg

    usd_path = scene_usd_path if scene_usd_path is not None else office_usd_path
    z_offset = scene_z_offset if office_z_offset is None else office_z_offset
    prim_path = scene_prim_path if office_prim_path is None else office_prim_path

    class CrowdSimSceneCfg(BaseSceneCfg):
        def __init__(self, *scene_args, **scene_kwargs):
            super().__init__(*scene_args, **scene_kwargs)
            if terrain_xy_offset is not None:
                _offset_trimesh_terrain_vertices_cfg(self.terrain, terrain_xy_offset)

            if usd_path is not None:
                self.global_usd_asset = AssetBaseCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=str(usd_path),
                        activate_contact_sensors=False,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z_offset)),
                    collision_group=-1,
                )

            if crowd_robot is None:
                return

            robot_prim_path = f"/World/envs/env_.*/{crowd_robot.prim_name}"
            self.crowdsim_robot = ArticulationCfg(
                prim_path=robot_prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=crowd_robot.usd_path,
                    activate_contact_sensors=False,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, crowd_robot.init_z),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "all_joints": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=None,
                        damping=None,
                    )
                },
            )

            if crowd_robot.enable_camera:
                camera_parent_path = robot_prim_path
                camera_mount_path = crowd_robot.camera_mount_prim_path.strip("/")
                if camera_mount_path:
                    camera_parent_path = f"{camera_parent_path}/{camera_mount_path}"
                self.crowdsim_robot_camera = CameraCfg(
                    prim_path=f"{camera_parent_path}/front_cam",
                    update_period=0.0,  # update every step
                    height=crowd_robot.camera_height,
                    width=crowd_robot.camera_width,
                    data_types=["rgb", "distance_to_image_plane"],
                    update_latest_camera_pose=True,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=(
                            2.0 * 24.0 * math.tan(
                                math.radians(crowd_robot.camera_horizontal_fov) / 2.0
                            )
                        ),
                        clipping_range=(0.1, 100.0),
                    ),
                    offset=CameraCfg.OffsetCfg(
                        pos=tuple(crowd_robot.camera_pos),
                        rot=tuple(crowd_robot.camera_rot),
                        convention=crowd_robot.camera_convention,
                    ),
                )

            if crowd_robot.enable_lidar:
                self.crowdsim_robot_lidar = RayCasterCfg(
                    prim_path=robot_prim_path,
                    update_period=crowd_robot.lidar_update_period,
                    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, crowd_robot.lidar_z_offset)),
                    mesh_prim_paths=list(crowd_robot.lidar_mesh_prim_paths),
                    ray_alignment="yaw",
                    pattern_cfg=patterns.LidarPatternCfg(
                        channels=crowd_robot.lidar_channels,
                        vertical_fov_range=[
                            crowd_robot.lidar_vertical_fov_min,
                            crowd_robot.lidar_vertical_fov_max,
                        ],
                        horizontal_fov_range=[
                            -0.5 * crowd_robot.lidar_horizontal_fov,
                            0.5 * crowd_robot.lidar_horizontal_fov,
                        ],
                        horizontal_res=crowd_robot.lidar_horizontal_res,
                    ),
                    debug_vis=crowd_robot.debug_vis,
                )

    simulator_module.SceneCfg = CrowdSimSceneCfg


def _offset_trimesh_terrain_vertices_cfg(
    terrain_cfg,
    xy_offset: tuple[float, float],
) -> None:
    if terrain_cfg is None or not hasattr(terrain_cfg, "terrain_vertices"):
        return
    vertices = getattr(terrain_cfg, "terrain_vertices", None)
    if vertices is None:
        return
    offset_x, offset_y = float(xy_offset[0]), float(xy_offset[1])
    for vertex in vertices:
        vertex[0] = float(vertex[0]) + offset_x
        vertex[1] = float(vertex[1]) + offset_y


def remove_scene_prims(patterns: list[str]) -> int:
    """Deactivate prims whose **name** starts with any of the given *patterns*.

    Matching is case-insensitive and applies to the prim's own name (the last
    component of its USD path), NOT the full path.  This avoids accidentally
    deactivating parent containers that happen to have a pattern string
    somewhere in their path.

    Example
    -------
    ``patterns=["Shelf", "PalletBin"]`` deactivates every prim whose name
    starts with "Shelf" or "PalletBin" (e.g. Shelf_1, Shelf_2, PalletBinI …)
    without touching the /World/Scene parent prim itself.
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    removed = 0
    total = 0
    for prim in stage.TraverseAll():
        total += 1
        name = prim.GetName()                          # just the leaf name
        if any(name.lower().startswith(p.lower()) for p in patterns):
            prim.SetActive(False)
            removed += 1
            print(f"  [CrowdSim] Deactivated: {prim.GetPath()}")
    if removed:
        print(f"[CrowdSim] Deactivated {removed}/{total} prims "
              f"with names starting with {patterns}")
    else:
        print(f"[CrowdSim] No prims matched {patterns} (scanned {total} prims)")
    return removed


def spawn_scene_objects(
    positions: list[tuple[float, float, float]],
    sizes: list[tuple[float, float, float]] | None = None,
    parent_path: str = "/World/SceneObjects",
    shapes: list[str] | None = None,
    colors: list[tuple[float, float, float] | None] | None = None,
    static: bool = True,
) -> None:
    """Spawn coloured boxes / cylinders as static obstacles in the scene.

    Parameters
    ----------
    shapes : list of ``"box"`` or ``"cylinder"`` per object (default all ``"box"``).
    colors : optional per-object RGB tuple in [0, 1]³.  Pass ``None`` for an
             entry (or omit the list entirely) to auto-assign a colour from a
             perceptually-uniform HSV palette so every obstacle looks distinct.
    static : if ``True``, obstacles are kinematic rigid bodies (immovable).
    """
    import colorsys
    import omni.usd
    from pxr import UsdGeom, Gf, Sdf, UsdShade, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim(parent_path, "Xform")

    n = len(positions)
    if sizes is None:
        sizes = [(0.5, 0.5, 0.8)] * n
    if shapes is None:
        shapes = ["box"] * n
    if colors is None:
        colors = [None] * n

    # Pre-build an HSV palette: hues spread evenly around the colour wheel,
    # high saturation and value so every obstacle is bright and distinct.
    auto_colors = [
        colorsys.hsv_to_rgb(k / max(n, 1), 0.82, 0.92)
        for k in range(n)
    ]

    for i, (pos, size, shape) in enumerate(zip(positions, sizes, shapes)):
        color = colors[i] if (colors[i] is not None) else auto_colors[i]
        label = f"{shape}_{i:02d}"
        prim_path = f"{parent_path}/{label}"
        if shape == "cylinder":
            prim = stage.DefinePrim(prim_path, "Cylinder")
            geom = UsdGeom.Cylinder(prim)
            geom.GetRadiusAttr().Set(size[0] / 2.0)
            geom.GetHeightAttr().Set(size[2])
            # Set axis=Z so the cylinder stands upright in a Z-up world.
            # This avoids the fragile rotation-matrix approach entirely.
            geom.GetAxisAttr().Set(UsdGeom.Tokens.z)
            xform = UsdGeom.Xformable(prim)
            xform_op = xform.AddXformOp(
                UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
            xform_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2] + size[2] / 2))
        else:
            prim = stage.DefinePrim(prim_path, "Cube")
            geom = UsdGeom.Cube(prim)
            geom.GetSizeAttr().Set(1.0)
            xform = UsdGeom.Xformable(prim)
            xform_op = xform.AddXformOp(
                UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
            mat = Gf.Matrix4d().SetScale(Gf.Vec3d(size[0], size[1], size[2]))
            mat.SetTranslateOnly(Gf.Vec3d(pos[0], pos[1], pos[2] + size[2] / 2))
            xform_op.Set(mat)

        # Static collider only — no RigidBodyAPI.
        # Adding RigidBodyAPI (even with kinematic=True) forces IsaacLab to
        # register a dynamic actor, which fails when prims are added after
        # sim.reset() and prevents the prim from appearing in the viewport.
        # A plain CollisionAPI creates a PhysX static body automatically:
        # it is immovable by definition and requires no registration step.
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)

        # Ensure the prim is visible in the viewport
        UsdGeom.Imageable(prim).MakeVisible()

        # Material — colour resolved at the top of the loop
        material_path = f"{prim_path}/Material"
        material = UsdShade.Material.Define(stage, material_path)
        pbr = UsdShade.Shader.Define(stage, f"{material_path}/PBR")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        pbr.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.5)
        pbr.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)

        print(f"  [CrowdSim] {label}: shape={shape} "
              f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
              f"size=({size[0]:.1f},{size[1]:.1f},{size[2]:.1f}) "
              f"rgb=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f}) static={static}")

    print(f"[CrowdSim] Spawned {len(positions)} scene objects under {parent_path}")

    # Force one Omniverse app update so the viewport renders the new prims
    # immediately rather than waiting for the next simulation step.
    try:
        import omni.kit.app
        omni.kit.app.get_app().update()
    except Exception:
        pass


def parent_camera_to_robot(env) -> None:
    """Verify the camera prim is a child of the configured Car rigid body.

    For Nova Carter the camera is spawned below ``/Car/chassis_link``, so the
    sensor inherits the physical chassis transform as well as the articulation
    root transform published by ``DriveController``.  ``CameraCfg.offset`` is
    the single source of truth for its local mount pose.
    """
    if not hasattr(env, "crowdsim_robot_camera"):
        return
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    camera_cfg = getattr(env.crowdsim_robot_camera, "cfg", None)
    camera_pattern = str(getattr(camera_cfg, "prim_path", ""))
    valid = 0
    for env_id in range(env.num_envs):
        camera_path = camera_pattern.replace("env_.*", f"env_{env_id}")
        cam = stage.GetPrimAtPath(camera_path)
        if cam.IsValid():
            valid += 1
    print(f"[CrowdSim] Camera prims verified at {camera_pattern}: "
          f"{valid}/{env.num_envs} (mount pose from sensors.camera.pos/rot).")


def apply_fixed_spawn_offsets(env, spawn_xy: torch.Tensor) -> None:
    """Pin ProtoMotions respawn offsets to fixed XY positions."""
    import types

    desired_root_xy = spawn_xy.to(device=env.device, dtype=torch.float32)
    if desired_root_xy.shape != (env.num_envs, 2):
        raise ValueError(
            f"Expected humanoid spawn XY with shape ({env.num_envs}, 2), "
            f"got {tuple(desired_root_xy.shape)}"
        )

    # Store a mutable reference so navigation can update spawn positions on the fly
    env._crowdsim_desired_root_xy = desired_root_xy

    def update_respawn_root_offset_by_env_ids(self, env_ids, ref_state=None, sample_flat=False):
        offset = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        offset[:, :2] = self._crowdsim_desired_root_xy[env_ids]
        if ref_state is not None:
            offset[:, :2] -= ref_state.root_pos[:, :2]
        offset[:, 2] += self.config.ref_respawn_offset
        self.respawn_root_offset[env_ids] = offset

    env.update_respawn_root_offset_by_env_ids = types.MethodType(
        update_respawn_root_offset_by_env_ids, env
    )
    env.respawn_root_offset[:, :2] = desired_root_xy
    env.respawn_root_offset[:, 2] = float(env.config.ref_respawn_offset)


def apply_fixed_crowd_robot_spawns(
    env,
    spawn_xy_yaw: torch.Tensor,
    scene_key: str = "crowdsim_robot",
) -> None:
    """Place optional CrowdSim navigation robots at fixed global XY/yaw poses."""
    scene = getattr(env.simulator, "_scene", None)
    if scene is None or scene_key not in scene.keys():
        raise KeyError(f"Scene entity '{scene_key}' was not created.")

    robot = scene[scene_key]
    poses = spawn_xy_yaw.to(device=env.device, dtype=torch.float32)
    if poses.shape[1] != 3:
        raise ValueError(
            f"Expected robot spawn poses with shape (N, 3), got {tuple(poses.shape)}"
        )
    # IsaacLab replicates the robot articulation across all envs, so the
    # root-state buffer has ``env.num_envs`` slots even when fewer robots are
    # active (e.g. num_robots=1 but num_envs=20).  Pad the trailing slots by
    # repeating the last active pose — they are frozen / non-participating and
    # only the first ``poses.shape[0]`` are ever read back by navigation.
    if poses.shape[0] < env.num_envs:
        pad = poses[-1:].repeat(env.num_envs - poses.shape[0], 1)
        poses = torch.cat([poses, pad], dim=0)
    if poses.shape[0] != env.num_envs:
        raise ValueError(
            f"Got {spawn_xy_yaw.shape[0]} robot spawn poses but only "
            f"{env.num_envs} envs."
        )

    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] = poses[:, 0]
    root_state[:, 1] = poses[:, 1]
    root_state[:, 3:7] = _yaw_to_quat_wxyz(poses[:, 2])
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        torch.zeros_like(robot.data.default_joint_vel),
    )
    scene.reset()
    # Force articulation data to reflect teleported poses immediately
    robot.write_root_pose_to_sim(root_state[:, :7])
    # Store spawn poses so navigation can re-apply after env.reset()
    env._crowdsim_robot_spawn_poses = root_state[:, :7].clone()

    env.crowdsim_robot = robot
    for sensor_name in ("crowdsim_robot_camera", "crowdsim_robot_lidar"):
        if sensor_name in scene.keys():
            setattr(env, sensor_name, scene[sensor_name])


def hide_inactive_crowd_robot_visuals(env, num_active: int) -> int:
    """Hide cloned car visuals for inactive IsaacLab environment slots."""
    import omni.usd
    from pxr import UsdGeom

    start = max(0, min(int(num_active), int(env.num_envs)))
    stage = omni.usd.get_context().get_stage()
    hidden = 0
    for env_id in range(start, int(env.num_envs)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Car")
        if not prim.IsValid():
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden += 1
    return hidden


def _yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((yaw.shape[0], 4), dtype=yaw.dtype, device=yaw.device)
    half_yaw = 0.5 * yaw
    quat[:, 0] = torch.cos(half_yaw)
    quat[:, 3] = torch.sin(half_yaw)
    return quat
