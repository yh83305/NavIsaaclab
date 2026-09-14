"""Multi-stage success-rate curriculum for PPO robot reset goals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GoalCurriculumStage:
    name: str
    distance: tuple[float, float]
    heading_degrees: tuple[float, float]


@dataclass(frozen=True)
class GoalCurriculumSpec:
    enabled: bool
    window_size: int
    min_episodes: int
    update_interval_episodes: int
    promote_success_rate: float
    demote_success_rate: float
    initial_stage: int
    stages: tuple[GoalCurriculumStage, ...]

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "GoalCurriculumSpec":
        values = values or {}
        raw_stages = values.get("stages", {})
        if not isinstance(raw_stages, dict) or not raw_stages:
            raw_stages = {
                "short_straight": {
                    "distance": [2.0, 4.0],
                    "heading_degrees": [0.0, 15.0],
                },
                "short_turn": {
                    "distance": [3.0, 5.0],
                    "heading_degrees": [10.0, 45.0],
                },
                "medium_turn": {
                    "distance": [4.0, 7.0],
                    "heading_degrees": [20.0, 75.0],
                },
                "long_turn": {
                    "distance": [5.5, 8.5],
                    "heading_degrees": [30.0, 105.0],
                },
                "full": {
                    "distance": [7.0, 10.0],
                    "heading_degrees": [30.0, 120.0],
                },
            }

        def pair(stage_name: str, field: str, raw) -> tuple[float, float]:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(
                    f"PPO goal curriculum stage {stage_name}.{field} must contain two values"
                )
            return float(raw[0]), float(raw[1])

        stages = []
        for name, stage_values in raw_stages.items():
            if not isinstance(stage_values, dict):
                raise ValueError(f"PPO goal curriculum stage {name} must be a mapping")
            stages.append(GoalCurriculumStage(
                name=str(name),
                distance=pair(str(name), "distance", stage_values.get("distance")),
                heading_degrees=pair(
                    str(name), "heading_degrees",
                    stage_values.get("heading_degrees"),
                ),
            ))
        spec = cls(
            enabled=bool(values.get("enabled", False)),
            window_size=int(values.get("window_size", 200)),
            min_episodes=int(values.get("min_episodes", 100)),
            update_interval_episodes=int(values.get("update_interval_episodes", 50)),
            promote_success_rate=float(values.get("promote_success_rate", 0.75)),
            demote_success_rate=float(values.get("demote_success_rate", 0.45)),
            initial_stage=int(values.get("initial_stage", 0)),
            stages=tuple(stages),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.window_size <= 0 or self.min_episodes <= 0:
            raise ValueError("Curriculum window_size and min_episodes must be positive")
        if self.update_interval_episodes <= 0:
            raise ValueError("Curriculum update_interval_episodes must be positive")
        if self.min_episodes > self.window_size:
            raise ValueError("Curriculum min_episodes cannot exceed window_size")
        if not 0.0 <= self.demote_success_rate < self.promote_success_rate <= 1.0:
            raise ValueError("Curriculum success thresholds must satisfy 0 <= demote < promote <= 1")
        if not 0 <= self.initial_stage < len(self.stages):
            raise ValueError("Curriculum initial_stage is outside the configured stages")
        for stage in self.stages:
            for field, bounds in (
                ("distance", stage.distance),
                ("heading_degrees", stage.heading_degrees),
            ):
                if bounds[0] < 0.0 or bounds[0] > bounds[1]:
                    raise ValueError(
                        f"Invalid curriculum stage {stage.name}.{field}: {bounds}"
                    )
            if stage.heading_degrees[1] > 180.0:
                raise ValueError("Curriculum heading offsets cannot exceed 180 degrees")


class PPOGoalCurriculum:
    """Mutate navigation reset ranges according to recent episode success."""

    def __init__(self, navigation_config, values: dict[str, Any] | None) -> None:
        self.navigation_config = navigation_config
        self.spec = GoalCurriculumSpec.from_dict(values)
        self.stage_index = self.spec.initial_stage
        self.outcomes: deque[float] = deque(maxlen=self.spec.window_size)
        self.completed_episodes = 0
        self._episodes_since_update = 0
        if self.spec.enabled:
            self._apply_stage()

    @property
    def stage(self) -> GoalCurriculumStage:
        return self.spec.stages[self.stage_index]

    @property
    def level(self) -> float:
        return self.stage_index / max(len(self.spec.stages) - 1, 1)

    def _apply_stage(self) -> None:
        stage = self.stage
        self.navigation_config.min_start_goal_distance = stage.distance[0]
        self.navigation_config.max_start_goal_distance = stage.distance[1]
        self.navigation_config.rl_initial_heading_min_offset_degrees = (
            stage.heading_degrees[0]
        )
        self.navigation_config.rl_initial_heading_max_offset_degrees = (
            stage.heading_degrees[1]
        )

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.outcomes)) if self.outcomes else 0.0

    def observe(self, successes) -> bool:
        """Record terminal outcomes and move at most one stage when due."""
        if not self.spec.enabled:
            return False
        values = np.asarray(successes, dtype=bool).reshape(-1)
        if len(values) == 0:
            return False
        self.outcomes.extend(values.astype(np.float32).tolist())
        self.completed_episodes += len(values)
        self._episodes_since_update += len(values)
        if (
            len(self.outcomes) < self.spec.min_episodes
            or self._episodes_since_update < self.spec.update_interval_episodes
        ):
            return False
        self._episodes_since_update = 0
        previous = self.stage_index
        if self.success_rate >= self.spec.promote_success_rate:
            self.stage_index = min(len(self.spec.stages) - 1, self.stage_index + 1)
        elif self.success_rate <= self.spec.demote_success_rate:
            self.stage_index = max(0, self.stage_index - 1)
        if self.stage_index != previous:
            self._apply_stage()
            return True
        return False

    def metrics(self) -> dict[str, float | str]:
        if not self.spec.enabled:
            return {}
        cfg = self.navigation_config
        return {
            "curriculum/stage": float(self.stage_index),
            "curriculum/stage_name": self.stage.name,
            "curriculum/num_stages": float(len(self.spec.stages)),
            "curriculum/level": self.level,
            "curriculum/success_rate": self.success_rate,
            "curriculum/completed_episodes": float(self.completed_episodes),
            "curriculum/distance_min_m": cfg.min_start_goal_distance,
            "curriculum/distance_max_m": cfg.max_start_goal_distance,
            "curriculum/heading_min_degrees": cfg.rl_initial_heading_min_offset_degrees,
            "curriculum/heading_max_degrees": cfg.rl_initial_heading_max_offset_degrees,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "outcomes": list(self.outcomes),
            "completed_episodes": self.completed_episodes,
            "episodes_since_update": self._episodes_since_update,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not self.spec.enabled or not state:
            return
        stage_index = int(state.get("stage_index", self.spec.initial_stage))
        if not 0 <= stage_index < len(self.spec.stages):
            raise ValueError(f"Invalid saved curriculum stage: {stage_index}")
        self.stage_index = stage_index
        self.outcomes.clear()
        self.outcomes.extend(float(value) for value in state.get("outcomes", []))
        self.completed_episodes = int(state.get("completed_episodes", 0))
        self._episodes_since_update = int(state.get("episodes_since_update", 0))
        self._apply_stage()
