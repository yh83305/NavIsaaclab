import pytest

torch = pytest.importorskip("torch")

from CrowdSim.ppo.ppo_policy import RobotActorCritic, flatten_neighbor_observation


def test_flatten_neighbor_observation_matches_legacy_layout():
    obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    neighbors = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    mask = torch.tensor([[True, True, False, True]])

    result = flatten_neighbor_observation(obs, neighbors, mask)
    masked = neighbors * mask.unsqueeze(-1)
    expected = torch.cat((obs, masked[:, 0], masked.flatten(1)), dim=-1)

    assert result.shape == (1, 30)
    torch.testing.assert_close(result, expected)


def test_ppo_uses_one_mlp_for_ego_and_neighbors():
    model = RobotActorCritic(
        obs_dim=5,
        action_dim=2,
        vector_obs_dim=5,
        num_neighbors=4,
        depth_enabled=False,
        map_enabled=False,
    )

    assert model.vector_encoder[0].in_features == 30
    assert not hasattr(model, "neighbor_encoder")


def test_neighbors_are_required():
    model = RobotActorCritic(
        obs_dim=5,
        action_dim=2,
        vector_obs_dim=5,
        num_neighbors=4,
        depth_enabled=False,
        map_enabled=False,
    )

    with pytest.raises(ValueError, match="neighbors and neighbor_mask"):
        model(torch.zeros(1, 5))
