import torch
from torch.distributions import Normal

from CrowdSim.ppo.ppo_policy import (
    RobotActorCritic,
    bounded_robot_action,
    bounded_robot_action_log_prob,
)


def test_ppo_action_transform_has_correct_ranges_and_jacobian():
    raw = torch.tensor(
        [[-8.0, -3.0], [0.0, 0.0], [8.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    action = bounded_robot_action(raw)

    assert torch.all((action[:, 0] > 0.0) & (action[:, 0] < 1.0))
    assert torch.all((action[:, 1] > -1.0) & (action[:, 1] < 1.0))

    linear_grad = torch.autograd.grad(action[:, 0].sum(), raw, retain_graph=True)[0][:, 0]
    angular_grad = torch.autograd.grad(action[:, 1].sum(), raw)[0][:, 1]
    dist = Normal(torch.zeros_like(raw), torch.ones_like(raw))
    transformed_log_prob = bounded_robot_action_log_prob(dist, raw, action)
    expected = (
        dist.log_prob(raw).sum(dim=-1)
        - linear_grad.log()
        - angular_grad.log()
    )
    torch.testing.assert_close(transformed_log_prob, expected)


def test_ppo_initial_mean_linear_action_is_half_speed():
    model = RobotActorCritic(
        obs_dim=5,
        action_dim=2,
        vector_obs_dim=5,
        num_neighbors=1,
        depth_enabled=False,
        map_enabled=False,
    )
    obs = torch.zeros(2, 5)
    neighbors = torch.zeros(2, 1, 5)
    neighbor_mask = torch.ones(2, 1, dtype=torch.bool)
    with torch.no_grad():
        raw_mean, _, _ = model(
            obs, neighbors=neighbors, neighbor_mask=neighbor_mask
        )
        mean_action = bounded_robot_action(raw_mean)
    torch.testing.assert_close(
        mean_action[:, 0],
        torch.full((2,), 0.5),
    )
