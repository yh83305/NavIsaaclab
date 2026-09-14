from types import SimpleNamespace

import numpy as np
import torch

from CrowdSim.control.drive import DriveController


class _RobotStub:
    def __init__(self, count: int) -> None:
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros((count, 3), dtype=torch.float32),
            root_quat_w=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * count, dtype=torch.float32
            ),
        )
        self.written_pose = None
        self.written_velocity = None

    def write_root_pose_to_sim(self, poses, env_ids=None) -> None:
        self.written_pose = poses.clone()

    def write_root_velocity_to_sim(self, velocities, env_ids=None) -> None:
        self.written_velocity = velocities.clone()


def _controller(
    *, linear: float = 1.0, angular: float = 1.0, count: int = 1
) -> DriveController:
    config = SimpleNamespace(
        num_robots=count,
        rl_max_linear_velocity=linear,
        rl_max_angular_velocity=angular,
        map_resolution=0.05,
    )
    controller = DriveController(_RobotStub(count), config, torch.device("cpu"))
    controller._env_dt = 1.0 / 30.0
    return controller


def test_kinematic_pose_is_integrated_once_and_physx_velocity_is_zero() -> None:
    drive = _controller()
    drive.set_action(np.asarray([[1.0, 0.0]], dtype=np.float32))

    drive._apply_kinematic()

    np.testing.assert_allclose(
        drive.robot.written_pose[0, :2].numpy(), [1.0 / 30.0, 0.0], atol=1e-7
    )
    torch.testing.assert_close(
        drive.robot.written_velocity, torch.zeros((1, 6), dtype=torch.float32)
    )
    np.testing.assert_allclose(drive.velocities_xy(), [[1.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(drive.angular_velocities(), [0.0], atol=1e-7)


def test_executed_velocity_follows_midpoint_kinematics() -> None:
    drive = _controller()
    drive.set_action(np.asarray([[0.6, 0.9]], dtype=np.float32))

    drive._apply_kinematic()

    expected_yaw = 0.9 / 30.0
    expected_xy = np.asarray(
        [0.6 * np.cos(expected_yaw), 0.6 * np.sin(expected_yaw)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(drive.velocities_xy()[0], expected_xy, atol=1e-7)
    np.testing.assert_allclose(drive.angular_velocities(), [0.9], atol=1e-7)


def test_wall_guard_stops_translation_but_preserves_turning() -> None:
    drive = _controller()
    drive.collision_check = lambda points: np.ones(len(points), dtype=bool)
    drive.set_action(np.asarray([[1.0, 0.5]], dtype=np.float32))

    drive._apply_kinematic()

    np.testing.assert_allclose(drive.robot.written_pose[0, :2].numpy(), [0.0, 0.0])
    np.testing.assert_allclose(drive.velocities_xy(), [[0.0, 0.0]])
    np.testing.assert_allclose(drive.angular_velocities(), [0.5])
    assert drive.wall_blocked.tolist() == [True]


def test_clear_state_zeros_executed_velocity_cache() -> None:
    drive = _controller()
    drive.set_action(np.asarray([[0.8, -0.4]], dtype=np.float32))
    drive._apply_kinematic()

    drive.clear_state(np.asarray([0], dtype=np.int64))

    np.testing.assert_array_equal(drive.velocities_xy(), np.zeros((1, 2)))
    np.testing.assert_array_equal(drive.angular_velocities(), np.zeros(1))


def test_partial_teleport_updates_camera_pose_cache_without_touching_other_robot() -> None:
    drive = _controller(count=2)
    drive._last_written_poses = torch.tensor(
        [
            [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    sync_calls = []
    drive._sync_prim_transforms = lambda: sync_calls.append(True)
    teleported = torch.tensor([[8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    drive.sync_teleported_poses(teleported, torch.tensor([1]))

    torch.testing.assert_close(
        drive._last_written_poses[0],
        torch.tensor([1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(drive._last_written_poses[1], teleported[0])
    assert sync_calls == [True]
