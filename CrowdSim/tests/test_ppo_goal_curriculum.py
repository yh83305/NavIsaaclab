from types import SimpleNamespace

import pytest

from CrowdSim.ppo.goal_curriculum import GoalCurriculumSpec, PPOGoalCurriculum


def _navigation_config():
    return SimpleNamespace(
        min_start_goal_distance=7.0,
        max_start_goal_distance=10.0,
        rl_initial_heading_min_offset_degrees=30.0,
        rl_initial_heading_max_offset_degrees=120.0,
    )


def _settings(**overrides):
    settings = {
        "enabled": True,
        "window_size": 10,
        "min_episodes": 4,
        "update_interval_episodes": 4,
        "promote_success_rate": 0.75,
        "demote_success_rate": 0.25,
        "initial_stage": 0,
        "stages": {
            "easy": {"distance": [2.0, 4.0], "heading_degrees": [0.0, 15.0]},
            "medium": {"distance": [4.0, 7.0], "heading_degrees": [20.0, 75.0]},
            "hard": {"distance": [7.0, 10.0], "heading_degrees": [30.0, 120.0]},
        },
    }
    settings.update(overrides)
    return settings


def test_curriculum_applies_first_stage_before_initial_routes():
    config = _navigation_config()
    curriculum = PPOGoalCurriculum(config, _settings())
    assert curriculum.stage.name == "easy"
    assert curriculum.level == 0.0
    assert config.min_start_goal_distance == 2.0
    assert config.max_start_goal_distance == 4.0
    assert config.rl_initial_heading_min_offset_degrees == 0.0
    assert config.rl_initial_heading_max_offset_degrees == 15.0


def test_curriculum_advances_one_stage_per_success_window():
    config = _navigation_config()
    curriculum = PPOGoalCurriculum(config, _settings())
    assert curriculum.observe([True, True, True, True])
    assert curriculum.stage.name == "medium"
    assert curriculum.level == 0.5
    assert config.min_start_goal_distance == 4.0
    assert config.max_start_goal_distance == 7.0
    assert config.rl_initial_heading_min_offset_degrees == 20.0
    assert config.rl_initial_heading_max_offset_degrees == 75.0

    assert curriculum.observe([True, True, True, True])
    assert curriculum.stage.name == "hard"
    assert curriculum.level == 1.0


def test_curriculum_demotes_only_below_lower_hysteresis_threshold():
    curriculum = PPOGoalCurriculum(
        _navigation_config(), _settings(initial_stage=1)
    )
    # 50% lies between the promote and demote thresholds.
    assert not curriculum.observe([True, True, False, False])
    assert curriculum.stage.name == "medium"
    # The rolling rate is now 20%, below the 25% demotion threshold.
    assert curriculum.observe([False, False, False, False, False, False])
    assert curriculum.stage.name == "easy"


def test_curriculum_rejects_overlapping_success_thresholds():
    with pytest.raises(ValueError, match="demote < promote"):
        GoalCurriculumSpec.from_dict(
            _settings(demote_success_rate=0.8, promote_success_rate=0.7)
        )


def test_curriculum_state_restores_stage_history_and_ranges():
    first = PPOGoalCurriculum(_navigation_config(), _settings())
    first.observe([True, True, True, True])
    state = first.state_dict()
    config = _navigation_config()
    restored = PPOGoalCurriculum(config, _settings())
    restored.load_state_dict(state)
    assert restored.stage.name == "medium"
    assert restored.success_rate == 1.0
    assert restored.completed_episodes == 4
    assert config.min_start_goal_distance == 4.0
    assert config.rl_initial_heading_max_offset_degrees == 75.0
