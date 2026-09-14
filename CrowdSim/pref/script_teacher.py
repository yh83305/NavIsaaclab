# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic view-clearance teacher for offline preference labels.

The teacher is intentionally simple and auditable: it projects all pedestrians
into the robot camera's horizontal field of view and rewards clips whose
nearest visible pedestrian stays farther away.  It does not use learned
features and therefore provides reproducible labels for reward-model training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ViewClearanceTeacherConfig:
    horizontal_fov_deg: float = 90.0
    max_view_distance_m: float = 5.0
    interaction_distance_m: float = 3.0
    safe_distance_m: float = 1.2
    collision_distance_m: float = 0.7
    temperature_m: float = 0.5
    min_interaction_fraction: float = 0.25
    min_distance_change_m: float = 0.15
    collision_penalty: float = 2.0
    lower_quantile_weight: float = 0.5
    camera_mount_pos: tuple[float, float, float] = (0.25, 0.0, 0.6)

    def validate(self) -> None:
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError("horizontal_fov_deg must be in (0, 180)")
        if self.max_view_distance_m <= 0.0:
            raise ValueError("max_view_distance_m must be positive")
        if not 0.0 < self.interaction_distance_m <= self.max_view_distance_m:
            raise ValueError(
                "interaction_distance_m must be in (0, max_view_distance_m]"
            )
        if self.temperature_m <= 0.0:
            raise ValueError("temperature_m must be positive")
        if not 0.0 <= self.min_interaction_fraction <= 1.0:
            raise ValueError("min_interaction_fraction must be in [0, 1]")
        if not 0.0 <= self.lower_quantile_weight <= 1.0:
            raise ValueError("lower_quantile_weight must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TeacherResult:
    score: float
    nearest_visible_distance: np.ndarray
    visible_count: np.ndarray
    interaction_fraction: float
    collision_fraction: float
    distance_change_m: float
    mean_visible_distance_m: float
    lower_visible_distance_m: float
    is_interaction: bool


class ViewClearanceTeacher:
    """Score fixed-length clips by distance to pedestrians in camera view."""

    def __init__(self, config: ViewClearanceTeacherConfig):
        config.validate()
        self.config = config

    def camera_xy(self, robot_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        yaw = np.asarray(yaw, dtype=np.float32)
        mount_x, mount_y, _ = self.config.camera_mount_pos
        return robot_xy + np.stack(
            (
                np.cos(yaw) * mount_x - np.sin(yaw) * mount_y,
                np.sin(yaw) * mount_x + np.cos(yaw) * mount_y,
            ),
            axis=-1,
        )

    def visible_geometry(
        self,
        robot_xy: np.ndarray,
        yaw: np.ndarray,
        human_xy: np.ndarray,
        human_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return distance, visibility mask and robot-local human positions.

        Inputs are ``robot_xy[T,2]``, ``yaw[T]`` and ``human_xy[T,H,2]``.
        Local coordinates use ``+x`` forward and ``+y`` left.
        """
        robot_xy = np.asarray(robot_xy, dtype=np.float32)
        yaw = np.asarray(yaw, dtype=np.float32)
        humans = np.asarray(human_xy, dtype=np.float32)
        if robot_xy.ndim != 2 or robot_xy.shape[-1] != 2:
            raise ValueError(f"robot_xy must be [T,2], got {robot_xy.shape}")
        if yaw.shape != robot_xy.shape[:1]:
            raise ValueError(f"yaw must be [T], got {yaw.shape}")
        if humans.ndim != 3 or humans.shape[0] != len(robot_xy) or humans.shape[-1] != 2:
            raise ValueError(f"human_xy must be [T,H,2], got {humans.shape}")

        camera = self.camera_xy(robot_xy, yaw)
        delta = humans - camera[:, None, :]
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        forward = c * delta[..., 0] + s * delta[..., 1]
        left = -s * delta[..., 0] + c * delta[..., 1]
        local = np.stack((forward, left), axis=-1)
        distance = np.linalg.norm(local, axis=-1)
        angle = np.abs(np.arctan2(left, forward))
        visible = (
            np.isfinite(distance)
            & (forward > 0.0)
            & (angle <= np.deg2rad(self.config.horizontal_fov_deg) / 2.0)
            & (distance <= self.config.max_view_distance_m)
        )
        if human_mask is not None:
            mask = np.asarray(human_mask, dtype=bool)
            if mask.shape != visible.shape:
                raise ValueError(f"human_mask must be {visible.shape}, got {mask.shape}")
            visible &= mask
        return distance.astype(np.float32), visible, local.astype(np.float32)

    def evaluate(
        self,
        robot_xy: np.ndarray,
        yaw: np.ndarray,
        human_xy: np.ndarray,
        human_mask: np.ndarray | None = None,
    ) -> TeacherResult:
        distances, visible, _ = self.visible_geometry(
            robot_xy, yaw, human_xy, human_mask
        )
        if distances.shape[1] == 0:
            nearest = np.full(
                len(robot_xy), self.config.max_view_distance_m, dtype=np.float32
            )
        else:
            nearest = np.where(visible, distances, np.inf).min(axis=1)
        nearest = np.where(
            np.isfinite(nearest), nearest, self.config.max_view_distance_m
        ).astype(np.float32)
        counts = visible.sum(axis=1).astype(np.int16)
        interaction = nearest <= self.config.interaction_distance_m
        interaction_fraction = float(interaction.mean())
        collision_fraction = float(
            (nearest <= self.config.collision_distance_m).mean()
        )
        lower = float(np.quantile(nearest, 0.2))
        mean = float(nearest.mean())
        distance_change = float(np.quantile(nearest, 0.9) - np.quantile(nearest, 0.1))

        # Both terms are monotonic in clearance.  The lower-tail term makes a
        # brief near collision matter even if the rest of the clip is clear.
        utility = np.tanh(
            (nearest - self.config.safe_distance_m) / self.config.temperature_m
        )
        lower_utility = float(np.quantile(utility, 0.2))
        mean_utility = float(utility.mean())
        weight = self.config.lower_quantile_weight
        score = (
            (1.0 - weight) * mean_utility
            + weight * lower_utility
            - self.config.collision_penalty * collision_fraction
        )
        dynamic_enough = (
            distance_change >= self.config.min_distance_change_m
            or interaction_fraction >= 0.5
            or collision_fraction > 0.0
        )
        return TeacherResult(
            score=float(score),
            nearest_visible_distance=nearest,
            visible_count=counts,
            interaction_fraction=interaction_fraction,
            collision_fraction=collision_fraction,
            distance_change_m=distance_change,
            mean_visible_distance_m=mean,
            lower_visible_distance_m=lower,
            is_interaction=(
                interaction_fraction >= self.config.min_interaction_fraction
                and dynamic_enough
            ),
        )
