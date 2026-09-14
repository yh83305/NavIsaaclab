import torch

from crowdsim_ppo import RobotActorCritic, bounded_robot_action


def test_bounded_robot_action_ranges():
    action = bounded_robot_action(torch.tensor([[-100.0, 100.0], [0.0, 0.0]]))
    assert torch.all((0.0 <= action[:, 0]) & (action[:, 0] <= 1.0))
    assert torch.all((-1.0 <= action[:, 1]) & (action[:, 1] <= 1.0))
    torch.testing.assert_close(action[1], torch.tensor([0.5, 0.0]))


def test_vector_only_actor_critic_smoke():
    model = RobotActorCritic(
        obs_dim=4,
        action_dim=2,
        hidden_dims=(16,),
        map_enabled=False,
        depth_enabled=False,
        num_neighbors=2,
    )
    obs = torch.zeros(3, 4)
    neighbors = torch.zeros(3, 2, 5)
    neighbor_mask = torch.ones(3, 2, dtype=torch.bool)
    mean, log_std, value = model(obs, neighbors=neighbors, neighbor_mask=neighbor_mask)
    assert mean.shape == (3, 2)
    assert log_std.shape == (3, 2)
    assert value.shape == (3,)
