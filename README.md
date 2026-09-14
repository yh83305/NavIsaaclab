# CrowdSim PPO

A compact PyTorch PPO implementation extracted from the CrowdSim robot-navigation
stack. This public package contains the reusable algorithmic components; simulator,
scene, dataset, asset, checkpoint, and deployment code remain outside this repository.

## Included

- Multimodal actor-critic with vector, neighbor, depth, and local-map encoders
- Bounded differential-drive actions
- PPO rollout buffer and trainer
- Success-rate goal curriculum with checkpointable state

## Installation

```bash
python -m pip install -e ".[test]"
```

## Minimal example

```python
import torch

from crowdsim_ppo import RobotActorCritic

model = RobotActorCritic(
    obs_dim=4,
    action_dim=2,
    hidden_dims=(128,),
    map_enabled=False,
    depth_enabled=False,
    num_neighbors=4,
)

batch_size = 8
obs = torch.zeros(batch_size, 4)
neighbors = torch.zeros(batch_size, 4, 5)
neighbor_mask = torch.zeros(batch_size, 4, dtype=torch.bool)

action, raw_action, log_prob, value = model.act(
    obs,
    neighbors=neighbors,
    neighbor_mask=neighbor_mask,
)
```

The normalized action is `[linear_velocity, angular_velocity]`, with ranges
`[0, 1]` and `[-1, 1]`. Environment integration is intentionally application-specific.

## Tests

```bash
pytest
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
