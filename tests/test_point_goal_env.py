import torch

from crowdsim_ppo.point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig


def test_environment_shapes_and_step():
    env = PointGoalBatchEnv(
        PointGoalEnvConfig(num_envs=3, num_neighbors=2), torch.device("cpu")
    )
    obs, neighbors, mask = env.observe()
    assert obs.shape == (3, 5)
    assert neighbors.shape == (3, 2, 5)
    assert mask.shape == (3, 2)
    next_state, reward, done, info = env.step(torch.zeros(3, 2))
    assert next_state[0].shape == obs.shape
    assert reward.shape == done.shape == (3,)
    assert set(info) == {
        "reached", "collision", "timeout", "episode_return",
        "episode_length", "goal_distance",
    }
