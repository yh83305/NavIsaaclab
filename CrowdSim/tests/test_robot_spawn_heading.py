from types import SimpleNamespace

import numpy as np

from CrowdSim.nav_manager import CrowdNavigationManager


def _manager() -> CrowdNavigationManager:
    manager = object.__new__(CrowdNavigationManager)
    manager.config = SimpleNamespace(
        num_humanoids=0,
        num_robots=1,
        rl_initial_heading_min_offset_degrees=30.0,
        rl_initial_heading_max_offset_degrees=120.0,
    )
    manager.starts_xy = np.asarray([[0.0, 0.0]], dtype=np.float32)
    manager.goals_xy = np.asarray([[5.0, 0.0]], dtype=np.float32)
    manager._heading_rng = np.random.default_rng(42)
    manager._robot_spawn_yaws = np.zeros(1, dtype=np.float32)
    manager._trajectory_log_file = None
    manager._trajectory_timestamp_file = None
    return manager


def test_spawn_heading_has_large_balanced_goal_offsets() -> None:
    manager = _manager()
    offsets = []
    for _ in range(200):
        manager._resample_robot_spawn_yaw(0)
        offsets.append(float(manager._robot_spawn_yaws[0]))
    offsets = np.asarray(offsets)
    magnitudes = np.abs(np.rad2deg(offsets))
    assert magnitudes.min() >= 30.0 - 1e-4
    assert magnitudes.max() <= 120.0 + 1e-4
    assert np.any(offsets < 0.0)
    assert np.any(offsets > 0.0)


def test_initial_robot_yaws_returns_cached_copy() -> None:
    manager = _manager()
    manager._resample_robot_spawn_yaw(0)
    cached = manager._initial_robot_yaws()
    cached[0] = 0.0
    assert manager._robot_spawn_yaws[0] != 0.0
