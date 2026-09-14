from types import SimpleNamespace

import numpy as np

from CrowdSim.nav_manager import CrowdNavigationManager
from CrowdSim.world.sfm import Social_Force


def _sfm() -> Social_Force:
    cfg = {
        "env": {
            "dt": 1.0 / 30.0,
            "safe_distance": 0.9,
            "neighbor_radius": 4.0,
            "reach_distance": 0.7,
            "prediction_horizon": 2.0,
            "ttc_threshold": 1.5,
            "ttc_gain": 12.0,
        },
        "agent": {"radius": 0.35, "max_vel": 1.5},
        "map": {"resolution": 0.05},
    }
    return Social_Force(np.full((16, 16), 10.0, dtype=np.float32), cfg)


def test_ttc_adds_early_lateral_force_for_head_on_agents() -> None:
    sfm = _sfm()
    _, terms = sfm.get_action(
        (
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([8, 8]),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([10.0, 0.0], dtype=np.float32),
        ),
        (
            np.array([1]),
            np.array([2.0], dtype=np.float32),
            np.array([[2.0, 0.0]], dtype=np.float32),
            np.array([[-2.0, 0.0]], dtype=np.float32),
        ),
    )

    ttc_force = terms[3]
    assert np.linalg.norm(ttc_force) > 0.0
    assert ttc_force[1] < 0.0  # deterministic pass-right convention


def test_ttc_is_zero_for_separating_agents() -> None:
    sfm = _sfm()
    _, terms = sfm.get_action(
        (
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([8, 8]),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([10.0, 0.0], dtype=np.float32),
        ),
        (
            np.array([1]),
            np.array([2.0], dtype=np.float32),
            np.array([[2.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ),
    )

    np.testing.assert_allclose(terms[3], 0.0)


def test_velocity_filter_respects_acceleration_limit() -> None:
    previous = np.array([1.0, 0.0], dtype=np.float32)
    filtered = CrowdNavigationManager._filter_humanoid_velocity(
        previous,
        np.array([0.0, 1.5], dtype=np.float32),
        smoothing=0.3,
        max_acceleration=1.5,
        dt=1.0 / 30.0,
    )

    assert np.linalg.norm(filtered - previous) <= 1.5 / 30.0 + 1e-6


def test_all_future_targets_follow_avoidance_velocity() -> None:
    manager = CrowdNavigationManager.__new__(CrowdNavigationManager)
    manager.config = SimpleNamespace(
        num_humanoids=1,
        humanoid_target_min_heading_speed=0.05,
        humanoid_max_yaw_rate=1.5,
        humanoid_max_acceleration=1.5,
    )
    manager._trajectory_log_file = None
    manager._trajectory_timestamp_file = None
    manager._sfm_desired_velocities = np.array([[1.0, -0.5]], dtype=np.float32)
    manager._humanoid_actual_velocities = np.array([[1.0, -0.5]], dtype=np.float32)
    manager._humanoid_target_yaws = np.zeros(1, dtype=np.float32)
    manager._humanoid_future_first_targets = np.zeros((1, 2), dtype=np.float32)
    manager._humanoid_future_first_yaws = np.zeros(1, dtype=np.float32)
    offsets = np.array([0.3, 0.6, 0.9], dtype=np.float32)

    targets, yaws = manager._humanoid_future_targets_from_path(
        np.array([[2.0, 3.0]], dtype=np.float32), offsets
    )

    expected = np.array(
        [[[2.3, 2.85], [2.6, 2.7], [2.9, 2.55]]], dtype=np.float32
    )
    np.testing.assert_allclose(targets, expected, atol=1e-6)
    np.testing.assert_allclose(yaws, np.arctan2(-0.5, 1.0), atol=1e-6)


def test_future_targets_respect_humanoid_yaw_rate() -> None:
    manager = CrowdNavigationManager.__new__(CrowdNavigationManager)
    manager.config = SimpleNamespace(
        num_humanoids=1,
        humanoid_target_min_heading_speed=0.05,
        humanoid_max_yaw_rate=1.5,
        humanoid_max_acceleration=1.5,
    )
    manager._trajectory_log_file = None
    manager._trajectory_timestamp_file = None
    manager._sfm_desired_velocities = np.array([[0.0, 1.0]], dtype=np.float32)
    manager._humanoid_actual_velocities = np.array([[1.0, 0.0]], dtype=np.float32)
    manager._humanoid_target_yaws = np.zeros(1, dtype=np.float32)
    manager._humanoid_future_first_targets = np.zeros((1, 2), dtype=np.float32)
    manager._humanoid_future_first_yaws = np.zeros(1, dtype=np.float32)

    targets, yaws = manager._humanoid_future_targets_from_path(
        np.zeros((1, 2), dtype=np.float32), np.array([0.1, 0.2], dtype=np.float32)
    )

    assert 0.0 < yaws[0, 0] <= 0.15 + 1e-6
    assert yaws[0, 1] <= 0.30 + 1e-6
    assert targets[0, 0, 0] > targets[0, 0, 1] > 0.0


class _FixedRouteTask:
    is_fixed_scenario = True

    def __init__(self) -> None:
        self.starts_px = np.array([[10, 20]], dtype=np.int64)
        self.goals_px = np.array([[10, 80]], dtype=np.int64)
        self.starts_xy = np.array([[0.0, 0.0]], dtype=np.float32)
        self.goals_xy = np.array([[6.0, 0.0]], dtype=np.float32)
        self.paths_xy = [
            np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype=np.float32)
        ]
        self.refresh_count = 0

    def fixed_agent_route(self, _agent_id: int):
        return (
            np.array([10, 20], dtype=np.int64),
            np.array([10, 80], dtype=np.int64),
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([6.0, 0.0], dtype=np.float32),
            np.array(
                [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
                dtype=np.float32,
            ),
        )

    def refresh_visualization_markers(self) -> None:
        self.refresh_count += 1


def _fixed_route_manager() -> CrowdNavigationManager:
    manager = CrowdNavigationManager.__new__(CrowdNavigationManager)
    manager.config = SimpleNamespace(
        num_humanoids=1,
        goal_tolerance=0.2,
        waypoint_tolerance=0.2,
    )
    manager.task = _FixedRouteTask()
    # The initial position is phase-spaced, but the immutable fixed route
    # still spans the complete entry-to-exit corridor.
    manager.starts_px = np.array([[10, 50]], dtype=np.int64)
    manager.goals_px = np.array([[10, 80]], dtype=np.int64)
    manager.starts_xy = np.array([[3.0, 0.0]], dtype=np.float32)
    manager.goals_xy = np.array([[6.0, 0.0]], dtype=np.float32)
    manager.paths_xy = [
        np.array([[3.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    ]
    manager.waypoint_ids = np.array([1], dtype=np.int64)
    manager.reached = np.array([False])
    manager._trajectory_log_file = None
    manager._pending_path_updates = []
    manager._path_log_dirty = False
    return manager


def test_fixed_humanoid_reverses_at_goal_without_becoming_done() -> None:
    manager = _fixed_route_manager()

    manager._update_waypoints_and_goals(
        np.array([[6.0, 0.0]], dtype=np.float32)
    )

    assert not manager.reached[0]
    np.testing.assert_allclose(manager.starts_xy[0], [6.0, 0.0])
    np.testing.assert_allclose(manager.goals_xy[0], [0.0, 0.0])
    np.testing.assert_allclose(
        manager.paths_xy[0],
        [[6.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
    )
    assert manager.waypoint_ids[0] == 1
    assert manager.task.refresh_count == 1
    assert manager._navigation_done_agent_ids(set()).size == 0


def test_fixed_humanoid_reverses_again_at_opposite_endpoint() -> None:
    manager = _fixed_route_manager()
    manager._update_waypoints_and_goals(
        np.array([[6.0, 0.0]], dtype=np.float32)
    )

    manager._update_waypoints_and_goals(
        np.array([[0.0, 0.0]], dtype=np.float32)
    )

    assert not manager.reached[0]
    np.testing.assert_allclose(manager.starts_xy[0], [0.0, 0.0])
    np.testing.assert_allclose(manager.goals_xy[0], [6.0, 0.0])
    np.testing.assert_allclose(
        manager.paths_xy[0],
        [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
    )
    assert manager.task.refresh_count == 2
