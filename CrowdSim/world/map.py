"""Map loading, start/goal sampling, and A* task planning for CrowdSim."""

from __future__ import annotations

import colorsys
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from CrowdSim.world.planning import Path_Planner

if TYPE_CHECKING:
    from CrowdSim.nav_manager import CrowdNavigationConfig


class NavigationTask:
    """Shared navigation task state independent of the local controller.

    Accepts a ``CrowdNavigationConfig`` directly — no separate config dataclass needed.
    """

    def __init__(self, config: CrowdNavigationConfig, num_agents: int) -> None:
        self.config: Any = config
        self._num_agents = int(num_agents)
        self.rng = random.Random(int(config.seed))
        self.free_mask, self.obstacle_map = self._load_map(config.map_path)
        if bool(config.scenario.get("ignore_static_obstacles", False)):
            # Controlled social-navigation scenarios may reuse a scene map
            # solely for its coordinate frame. In that case both planning and
            # rule rewards must see an obstacle-free map; otherwise removing
            # the USD geometry leaves invisible obstacles in the occupancy map.
            self.free_mask = np.ones_like(self.free_mask, dtype=bool)
            self.obstacle_map = np.zeros_like(self.obstacle_map, dtype=np.uint8)
        self.height, self.width = self.free_mask.shape
        self.planner = Path_Planner(
            self.obstacle_map,
            map_resolution=float(config.map_resolution),
            step_size_m=float(config.planning_step_size),
            clearance_m=float(config.planning_clearance),
            smooth=False,
            viz=False,
            verbose=False,
        )
        self.planner_free_mask = self.planner.map_dialate == 0
        self.component_labels, self.component_sizes = self._label_planner_free_space()
        self._component_pixel_cache: dict[int, np.ndarray] = {}
        self._markers: NavigationTaskMarkers | None = None
        self._marker_num_humanoids = 0
        # H8: cache free pixels to avoid O(H×W) scan on every reset call
        self._free_pixels_cache: np.ndarray = self._build_free_pixels_cache()
        self._fixed_routes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
        self._fixed_reset_starts_xy: list[np.ndarray] | None = None
        if self.is_fixed_scenario:
            self.starts_px, self.goals_px, self.paths_xy = self._build_bidirectional_flow_scenario()
        else:
            self.starts_px, self.goals_px, self.paths_xy = self._sample_start_goal_paths()
        self.starts_xy = np.asarray(
            [self.pixel_to_world(px) for px in self.starts_px], dtype=np.float32
        )
        self.goals_xy = np.asarray(
            [self.pixel_to_world(px) for px in self.goals_px], dtype=np.float32
        )
        if self.is_fixed_scenario:
            reset_starts_xy = self._fixed_reset_starts_xy
            if reset_starts_xy is None or len(reset_starts_xy) != self._num_agents:
                raise RuntimeError("Fixed scenario did not provide reset start points")
            self._fixed_routes = [
                (
                    self.world_to_pixel(reset_starts_xy[i]),
                    self.goals_px[i].copy(),
                    reset_starts_xy[i].copy(),
                    self.goals_xy[i].copy(),
                    np.stack((reset_starts_xy[i], self.goals_xy[i])).astype(np.float32),
                )
                for i in range(self._num_agents)
            ]

    @property
    def is_fixed_scenario(self) -> bool:
        return str(self.config.scenario.get("type", "random")).lower() == "bidirectional_flow"

    def fixed_agent_route(
        self, agent_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._fixed_routes is None:
            raise RuntimeError("No fixed scenario route is available.")
        route = self._fixed_routes[int(agent_id)]
        return tuple(value.copy() for value in route)  # type: ignore[return-value]

    def _build_bidirectional_flow_scenario(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Build two cyclic pedestrian streams and a middle B-to-C robot task.

        Humanoids are ordered first, followed by robots. Half of the humanoids
        circulate A-to-D in one lane and the remainder D-to-A in the other.
        They are phase-spaced at startup, then reverse at each endpoint and
        walk back along the same lane. Robots travel only between the inner
        B/C points.
        """
        cfg = self.config.scenario
        point_a = np.asarray(cfg.get("point_a", [-5.8, -8.0]), dtype=np.float32)
        point_b = np.asarray(cfg.get("point_b", [-5.8, -4.0]), dtype=np.float32)
        point_c = np.asarray(cfg.get("point_c", [-5.8, 4.0]), dtype=np.float32)
        point_d = np.asarray(cfg.get("point_d", [-5.8, 8.0]), dtype=np.float32)
        lane_offset = float(cfg.get("lane_offset", 1.2))
        endpoint_margin = float(cfg.get("endpoint_margin", 0.8))
        longitudinal_jitter = float(cfg.get("longitudinal_jitter", 0.0))
        lateral_jitter = float(cfg.get("lateral_jitter", 0.0))
        if longitudinal_jitter < 0.0 or lateral_jitter < 0.0:
            raise ValueError("bidirectional_flow jitter values must be non-negative")
        mirror_value = cfg.get("mirror", False)
        if isinstance(mirror_value, str) and mirror_value.lower() == "random":
            mirror = bool(self.rng.randrange(2))
        else:
            mirror = bool(mirror_value)

        axis = point_d - point_a
        length = float(np.linalg.norm(axis))
        if length <= 2.0 * endpoint_margin:
            raise ValueError("bidirectional_flow point_a/point_d are too close")
        forward = axis / length
        b_progress = float(np.dot(point_b - point_a, forward))
        c_progress = float(np.dot(point_c - point_a, forward))
        b_off_axis = float(np.linalg.norm((point_b - point_a) - b_progress * forward))
        c_off_axis = float(np.linalg.norm((point_c - point_a) - c_progress * forward))
        if not (0.0 < b_progress < c_progress < length):
            raise ValueError("bidirectional_flow points must be ordered A-B-C-D")
        if max(b_off_axis, c_off_axis) > 1.0e-3:
            raise ValueError("bidirectional_flow points A, B, C and D must be collinear")

        left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
        n_h = int(self.config.num_humanoids)
        if 0 < n_h < 4:
            raise ValueError(
                "bidirectional_flow requires at least four humanoids "
                "(one for each pedestrian lane)"
            )

        # Four parallel lanes centred on the robot route.  The two lanes on
        # one side move D->A and the two lanes on the other side move A->D;
        # ``mirror`` swaps these directions.  Distribute any remainder from
        # odd crowd sizes over the first lanes deterministically.
        lane_multipliers = (1.5, 0.5, -0.5, -1.5)
        lane_shifts = [left * lane_offset * value for value in lane_multipliers]
        base_count, remainder = divmod(n_h, len(lane_shifts))
        lane_counts = [
            base_count + (1 if lane_index < remainder else 0)
            for lane_index in range(len(lane_shifts))
        ]

        starts_xy: list[np.ndarray] = []
        goals_xy: list[np.ndarray] = []
        reset_starts_xy: list[np.ndarray] = []
        for lane_index, (count, lane_shift) in enumerate(
            zip(lane_counts, lane_shifts)
        ):
            same_direction = (lane_index >= 2) ^ mirror
            if count > 1:
                nominal_spacing = (
                    length - 2.0 * endpoint_margin
                ) / float(count - 1)
                minimum_spacing = nominal_spacing - 2.0 * longitudinal_jitter
                if minimum_spacing <= float(self.config.collision_distance):
                    raise ValueError(
                        "bidirectional_flow is too dense: the configured "
                        f"minimum longitudinal spacing ({minimum_spacing:.3f} m) "
                        "must exceed collision_distance "
                        f"({self.config.collision_distance:.3f} m); lengthen A-D "
                        "or reduce num_humanoids/longitudinal_jitter"
                    )
            fractions = np.linspace(
                endpoint_margin / length,
                1.0 - endpoint_margin / length,
                max(count, 1),
                dtype=np.float32,
            )
            for fraction in fractions[:count]:
                if longitudinal_jitter > 0.0:
                    fraction = float(np.clip(
                        fraction
                        + self.rng.uniform(
                            -longitudinal_jitter, longitudinal_jitter
                        )
                        / length,
                        endpoint_margin / length,
                        1.0 - endpoint_margin / length,
                    ))
                lateral_noise = (
                    self.rng.uniform(-lateral_jitter, lateral_jitter)
                    if lateral_jitter > 0.0
                    else 0.0
                )
                position = (
                    point_a
                    + float(fraction) * axis
                    + lane_shift
                    + lateral_noise * left
                )
                entry = (point_a if same_direction else point_d) + lane_shift
                destination = (point_d if same_direction else point_a) + lane_shift
                starts_xy.append(position.astype(np.float32))
                reset_starts_xy.append(entry.astype(np.float32))
                goals_xy.append(destination.astype(np.float32))

        robot_route = str(cfg.get("robot_route", "random")).lower()
        if robot_route not in {"random", "b_to_c", "c_to_b"}:
            raise ValueError("scenario.robot_route must be random, b_to_c or c_to_b")
        n_r = int(self.config.num_robots)
        for robot_id in range(n_r):
            lateral = (robot_id - (n_r - 1) / 2.0) * 0.5
            route = robot_route
            if route == "random":
                route = "b_to_c" if self.rng.randrange(2) == 0 else "c_to_b"
            route_start, route_goal = (
                (point_b, point_c) if route == "b_to_c" else (point_c, point_b)
            )
            robot_start = (route_start + lateral * left).astype(np.float32)
            starts_xy.append(robot_start)
            reset_starts_xy.append(robot_start.copy())
            goals_xy.append((route_goal + lateral * left).astype(np.float32))

        if len(starts_xy) != self._num_agents:
            raise RuntimeError("bidirectional_flow generated an unexpected agent count")
        self._fixed_reset_starts_xy = reset_starts_xy

        starts_px = np.asarray([self.world_to_pixel(xy) for xy in starts_xy], dtype=np.int64)
        goals_px = np.asarray([self.world_to_pixel(xy) for xy in goals_xy], dtype=np.int64)
        reset_starts_px = np.asarray(
            [self.world_to_pixel(xy) for xy in reset_starts_xy], dtype=np.int64
        )
        for label, pixels in (
            ("initial start", starts_px),
            ("reset start", reset_starts_px),
            ("goal", goals_px),
        ):
            invalid = [i for i, px in enumerate(pixels) if not self._is_white_traversable(px)]
            if invalid:
                raise ValueError(
                    f"bidirectional_flow {label} points for agents {invalid} are not traversable; "
                    "adjust points A/B/C/D or lane_offset"
                )
        paths = [
            np.stack([start, goal]).astype(np.float32)
            for start, goal in zip(starts_xy, goals_xy)
        ]
        return starts_px, goals_px, paths

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        origin_x, origin_y = self.config.map_origin_xy
        pixel_x = int(round((float(xy[0]) - origin_x) / self.config.map_resolution))
        pixel_y = int(
            round((self.height - 1) - (float(xy[1]) - origin_y) / self.config.map_resolution)
        )
        pixel_x = max(0, min(self.width - 1, pixel_x))
        pixel_y = max(0, min(self.height - 1, pixel_y))
        return np.array([pixel_y, pixel_x], dtype=np.int64)

    def pixel_to_world(self, pixel_yx: np.ndarray) -> np.ndarray:
        pixel_y = float(pixel_yx[0])
        pixel_x = float(pixel_yx[1])
        origin_x, origin_y = self.config.map_origin_xy
        return np.array(
            [
                origin_x + pixel_x * self.config.map_resolution,
                origin_y + (self.height - 1 - pixel_y) * self.config.map_resolution,
            ],
            dtype=np.float32,
        )

    def _load_map(self, map_path: Path) -> tuple[np.ndarray, np.ndarray]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for CrowdSim navigation maps.") from exc

        image = Image.open(map_path).convert("L")
        grid = np.asarray(image, dtype=np.uint8)
        free_mask = grid >= int(self.config.free_threshold)
        obstacle_map = (~free_mask).astype(np.uint8)
        return free_mask, obstacle_map

    def _sample_start_goal_paths(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        free_pixels = [tuple(pixel) for pixel in np.column_stack(np.nonzero(self.planner_free_mask))]
        self.rng.shuffle(free_pixels)

        starts: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        paths: list[np.ndarray] = []
        min_spacing_px = self.config.min_spawn_spacing / self.config.map_resolution
        min_goal_px = self.config.min_start_goal_distance / self.config.map_resolution
        max_goal_px = self.config.max_start_goal_distance / self.config.map_resolution

        for candidate in free_pixels:
            if len(starts) == self._num_agents:
                break
            candidate_array = np.asarray(candidate, dtype=np.int64)
            if not self._is_white_traversable(candidate_array):
                continue
            if self.component_labels[candidate] == 0:
                continue
            if starts and min(
                np.linalg.norm(candidate_array - np.asarray(point)) for point in starts
            ) < min_spacing_px:
                continue
            result = self._sample_goal_and_path_for_start(
                candidate_array,
                min_goal_px,
                max_goal_px,
                max_attempts=120,
            )
            if result is None:
                continue
            goal, path_xy = result
            starts.append(candidate_array.copy())
            goals.append(goal)
            paths.append(path_xy)

        if len(starts) < self._num_agents:
            raise RuntimeError(
                f"Only sampled {len(starts)}/{self._num_agents} navigation starts "
                "with valid A* paths from the traversable white map area. Check scene_map, "
                "free_threshold, planning_clearance, planning_step_size, or map connectivity."
            )

        return np.asarray(starts, dtype=np.int64), np.asarray(goals, dtype=np.int64), paths

    def _sample_goal_for_start(
        self, start: np.ndarray, min_goal_px: float, max_goal_px: float
    ) -> np.ndarray | None:
        result = self._sample_goal_and_path_for_start(
            start,
            min_goal_px,
            max_goal_px,
            max_attempts=2000,
            plan_path=False,
        )
        return None if result is None else result[0]

    def _sample_goal_and_path_for_start(
        self,
        start: np.ndarray,
        min_goal_px: float,
        max_goal_px: float,
        max_attempts: int,
        plan_path: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        component_id = int(self.component_labels[tuple(start)])
        component_pixels = self._pixels_for_component(component_id)
        if len(component_pixels) == 0:
            return None

        for _ in range(max_attempts):
            idx = self.rng.randrange(len(component_pixels))
            goal = component_pixels[idx]
            distance = np.linalg.norm(goal - start)
            if not min_goal_px <= distance <= max_goal_px:
                continue
            if not self._is_white_traversable(goal):
                continue
            if not plan_path:
                return goal.copy(), np.zeros((0, 2), dtype=np.float32)
            path_px = self.planner.get_astar_path(start, goal)
            if path_px is None or len(path_px) == 0:
                continue
            path_xy = np.asarray([self.pixel_to_world(px) for px in path_px], dtype=np.float32)
            return goal.copy(), path_xy
        return None

    def _plan_paths(self) -> list[np.ndarray]:
        paths: list[np.ndarray] = []
        for agent_id, (start, goal) in enumerate(zip(self.starts_px, self.goals_px)):
            path_px = self.planner.get_astar_path(start, goal)
            if path_px is None or len(path_px) == 0:
                raise RuntimeError(
                    f"A* failed for agent {agent_id}: start_px={start.tolist()}, "
                    f"goal_px={goal.tolist()}, start_xy={self.pixel_to_world(start).tolist()}, "
                    f"goal_xy={self.pixel_to_world(goal).tolist()}. "
                    "No straight-line fallback is used."
                )
            path_xy = np.asarray(
                [self.pixel_to_world(px) for px in path_px], dtype=np.float32
            )
            paths.append(path_xy)
        return paths

    def sample_goal_and_plan_path(self, start_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start_px = self.world_to_pixel(start_xy)
        if not self._is_white_traversable(start_px):
            start_px = self._nearest_traversable_pixel(start_px)

        min_goal_px = self.config.min_start_goal_distance / self.config.map_resolution
        max_goal_px = self.config.max_start_goal_distance / self.config.map_resolution
        result = self._sample_goal_and_path_for_start(
            start_px,
            min_goal_px,
            max_goal_px,
            max_attempts=200,
        )
        if result is not None:
            goal_px, path_xy = result
            return start_px, goal_px, path_xy

        raise RuntimeError(
            f"Failed to sample and plan a local goal near start_xy={np.asarray(start_xy).tolist()} "
            f"within [{self.config.min_start_goal_distance}, {self.config.max_start_goal_distance}] m."
        )

    def _nearest_traversable_pixel(self, pixel_yx: np.ndarray) -> np.ndarray:
        traversable = np.column_stack(np.nonzero(self.planner_free_mask))
        if len(traversable) == 0:
            raise RuntimeError("No traversable pixels available for navigation reset.")
        dists = np.linalg.norm(traversable - pixel_yx[None, :], axis=1)
        return traversable[int(np.argmin(dists))].astype(np.int64)

    # ── free-pixel cache (H8) ─────────────────────────────────────
    def _build_free_pixels_cache(self) -> np.ndarray:
        """Build (and return) an array of all traversable pixel coordinates."""
        free = np.column_stack(np.nonzero(self.planner_free_mask))
        return free.astype(np.int64) if len(free) else np.zeros((0, 2), dtype=np.int64)

    def _refresh_free_pixels_cache(self) -> None:
        """Rebuild the free-pixel cache after the map changes (e.g. scene objects added)."""
        self._free_pixels_cache = self._build_free_pixels_cache()
        self._component_pixel_cache.clear()   # component cache may also be stale

    def _sample_free_pixel(self) -> np.ndarray | None:
        """Return a random pixel from the cached free-pixel list, or None if empty."""
        if len(self._free_pixels_cache) == 0:
            return None
        idx = self.rng.randrange(len(self._free_pixels_cache))
        return self._free_pixels_cache[idx]

    def _is_white_traversable(self, pixel_yx: np.ndarray) -> bool:
        pixel = tuple(int(value) for value in pixel_yx)
        return bool(self.free_mask[pixel] and self.planner_free_mask[pixel])

    def _label_planner_free_space(self) -> tuple[np.ndarray, np.ndarray]:
        num_labels, labels = cv2.connectedComponents(
            self.planner_free_mask.astype(np.uint8), connectivity=8
        )
        sizes = np.bincount(labels.reshape(-1), minlength=num_labels)
        return labels.astype(np.int32, copy=False), sizes

    def _pixels_for_component(self, component_id: int) -> np.ndarray:
        if component_id not in self._component_pixel_cache:
            self._component_pixel_cache[component_id] = np.column_stack(
                np.nonzero(self.component_labels == component_id)
            ).astype(np.int64)
        return self._component_pixel_cache[component_id]

    def create_visualization_markers(self, num_humanoids: int, enabled: bool) -> None:
        self._marker_num_humanoids = int(num_humanoids)
        self._markers = NavigationTaskMarkers(enabled)
        self._markers.create(self, num_humanoids)

    def refresh_visualization_markers(self) -> None:
        if self._markers is None:
            return
        self._markers.update(self, self._marker_num_humanoids)


def agent_marker_color(agent_id: int) -> tuple[float, float, float]:
    hue = (0.08 + 0.61803398875 * float(agent_id)) % 1.0
    return tuple(float(value) for value in colorsys.hsv_to_rgb(hue, 0.78, 0.95))


def build_agent_marker_prototypes(sim_utils, num_humanoids: int, num_robots: int) -> dict:
    prototypes = {}
    total = int(num_humanoids) + int(num_robots)
    for agent_id in range(total):
        color = agent_marker_color(agent_id)
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        if agent_id < num_humanoids:
            prototypes[f"humanoid_{agent_id}"] = sim_utils.SphereCfg(
                radius=1.0,
                visual_material=material,
            )
        else:
            robot_id = agent_id - num_humanoids
            prototypes[f"car_{robot_id}"] = sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=material,
            )
    return prototypes


class NavigationTaskMarkers:
    """Static IsaacLab markers for A* paths and final goals."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.agent_marker = None

    def create(self, task: NavigationTask, num_humanoids: int) -> None:
        if not self.enabled:
            return

        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.agent_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/CrowdSim/nav_paths_and_goals",
                markers=build_agent_marker_prototypes(
                    sim_utils,
                    num_humanoids=num_humanoids,
                    num_robots=max(0, len(task.paths_xy) - num_humanoids),
                ),
            )
        )

        self.update(task, num_humanoids)

    def update(self, task: NavigationTask, num_humanoids: int) -> None:
        if not self.enabled or self.agent_marker is None:
            return

        translations, orientations, scales, marker_indices = self._static_marker_arrays(
            paths_xy=task.paths_xy,
            goals_xy=task.goals_xy,
            num_humanoids=num_humanoids,
        )
        self.agent_marker.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )

    def _static_marker_arrays(
        self,
        paths_xy: list[np.ndarray],
        goals_xy: np.ndarray,
        num_humanoids: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        translations: list[list[float]] = []
        scales: list[list[float]] = []
        marker_indices: list[int] = []

        for agent_id, path in enumerate(paths_xy):
            # Only show A* path waypoints for humanoids — cars use RL policy.
            if agent_id < num_humanoids:
                path_points = path[1:-1] if len(path) > 2 else np.zeros((0, 2), dtype=np.float32)
                for point in path_points:
                    translations.append([float(point[0]), float(point[1]), 0.04])
                    scales.append(self._path_scale(agent_id, num_humanoids))
                    marker_indices.append(agent_id)

            goal = goals_xy[agent_id]
            translations.append([float(goal[0]), float(goal[1]), 0.10])
            scales.append(self._goal_scale(agent_id, num_humanoids))
            marker_indices.append(agent_id)

        if not translations:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        translations_array = np.asarray(translations, dtype=np.float32)
        orientations = np.zeros((len(translations_array), 4), dtype=np.float32)
        orientations[:, 0] = 1.0
        return (
            translations_array,
            orientations,
            np.asarray(scales, dtype=np.float32),
            np.asarray(marker_indices, dtype=np.int32),
        )

    @staticmethod
    def _path_scale(agent_id: int, num_humanoids: int) -> list[float]:
        if agent_id < num_humanoids:
            return [0.055, 0.055, 0.055]
        return [0.075, 0.075, 0.035]

    @staticmethod
    def _goal_scale(agent_id: int, num_humanoids: int) -> list[float]:
        if agent_id < num_humanoids:
            return [0.22, 0.22, 0.22]
        return [0.26, 0.26, 0.10]
# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------


class CollisionDetector:
    """Detects agent-agent and agent-wall collisions on a shared occupancy map."""

    def __init__(
        self,
        obstacle_map: np.ndarray,
        map_resolution: float,
        collision_distance: float,
        agent_radius: float,
        world_to_pixel,          # callable: (xy) -> (py, px)
        height: int,
        width: int,
    ) -> None:
        self.obstacle_map = obstacle_map
        self.map_resolution = map_resolution
        self.collision_distance = collision_distance
        self.agent_radius = agent_radius
        self._world_to_pixel = world_to_pixel
        self.height = height
        self.width = width

    def hits_wall(self, xy: np.ndarray) -> np.ndarray:
        """Pure wall-collision query — no agent-agent, no side effects.

        Given world-frame xy of shape (N, 2), return a boolean (N,) array where
        True means the point (inflated by ``agent_radius``) overlaps an
        occupancy-map obstacle cell.  Out-of-map points are treated as walls
        (True) — stricter than ``detect``, which silently skips them.  This is
        the query the drive's wall guard calls each step to keep kinematic
        robots from translating into walls.
        """
        import math as _math
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        out = np.zeros(xy.shape[0], dtype=bool)
        radius_px = max(1, int(_math.ceil(self.agent_radius / self.map_resolution)))
        for idx in range(xy.shape[0]):
            pix = self._world_to_pixel(xy[idx])
            cy, cx = int(pix[0]), int(pix[1])
            if not (0 <= cy < self.height and 0 <= cx < self.width):
                out[idx] = True     # out-of-map ≡ wall
                continue
            hit = False
            for dy in range(-radius_px, radius_px + 1):
                if hit:
                    break
                for dx in range(-radius_px, radius_px + 1):
                    if dx * dx + dy * dy > radius_px * radius_px:
                        continue
                    y, x = cy + dy, cx + dx
                    if 0 <= y < self.height and 0 <= x < self.width and self.obstacle_map[y, x]:
                        hit = True
                        break
            out[idx] = hit
        return out

    def detect(self, positions: np.ndarray, num_agents: int,
               collision_pairs: set) -> set:
        """Return new collision pairs. Also adds to collision_pairs in-place."""
        new_pairs: set = set()
        # Agent-agent
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if float(np.linalg.norm(positions[i] - positions[j])) < self.collision_distance:
                    pair = (i, j)
                    if pair not in collision_pairs:
                        new_pairs.add(pair)
                    collision_pairs.add(pair)
        # Agent-wall — delegate the radius-inflated cell scan to hits_wall so
        # the wall-query logic lives in one place (also used by the drive guard).
        wall_hits = self.hits_wall(positions[:num_agents])
        for agent_id in range(num_agents):
            if wall_hits[agent_id]:
                pair = (agent_id, -1)
                if pair not in collision_pairs:
                    new_pairs.add(pair)
                collision_pairs.add(pair)
        return new_pairs
