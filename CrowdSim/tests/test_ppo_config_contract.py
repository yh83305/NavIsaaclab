from CrowdSim.utils.config_loader import load_config


def test_ppo_training_and_collection_share_expert_contract():
    training = load_config("CrowdSim/config/env.yaml")
    collection = load_config("CrowdSim/config/collect/ppo.yaml")
    for config in (training, collection):
        navigation = config["navigation"]
        assert navigation["update_hz"] == 25.0
        assert navigation["num_humanoids"] == 10
        assert navigation["num_robots"] == 10
        assert navigation["rl"]["max_linear_velocity"] == 1.0
        assert navigation["rl"]["max_angular_velocity"] == 1.0
        assert config["sensors"]["camera"]["pos"] == [0.25, 0.0, 0.6]
        assert config["sensors"]["camera"]["horizontal_fov"] == 90.0
    assert training["navigation"]["rl"]["reward"]["max_episode_steps"] == 1000
    assert collection["navigation"]["rl"]["max_episode_steps"] == 1000
    assert training["training"]["gamma"] == 0.99
    assert training["training"]["gae_lambda"] == 0.95
