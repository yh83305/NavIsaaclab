# NavIsaaclab PPO

An end-to-end PyTorch PPO example for robot PointGoal navigation. The repository
contains the policy, rollout storage, PPO optimizer, curriculum, a lightweight
vectorized environment, checkpoint/resume logic, and deterministic evaluation.

The included 2-D environment is a simulator-free reference pipeline. It keeps the
same high-level observation and bounded differential-drive action conventions as
the larger CrowdSim stack without requiring Isaac Sim, scene assets, or checkpoints.

## Included

- Multimodal actor-critic with vector, neighbor, depth, and local-map encoders
- Bounded differential-drive actions
- PPO rollout buffer and trainer
- Success-rate goal curriculum with checkpointable state
- Batched PointGoal navigation with moving neighbors
- Training, checkpoint/resume, and evaluation CLIs
- JSON configuration and smoke tests

## Installation

```bash
python -m pip install -e ".[test]"
```

## Train

```bash
crowdsim-ppo-train --config configs/point_goal.json
```

For a quick smoke run:

```bash
crowdsim-ppo-train \
  --config configs/point_goal.json \
  --total-steps 32768 \
  --output output/smoke
```

Training prints one JSON metrics record per PPO update and writes:

```text
output/point_goal_ppo/
├── config.json
├── checkpoint_<step>.pt
└── latest.pt
```

Resume from a checkpoint:

```bash
crowdsim-ppo-train \
  --config configs/point_goal.json \
  --resume output/point_goal_ppo/latest.pt
```

## Evaluate

```bash
crowdsim-ppo-eval \
  --config configs/point_goal.json \
  --checkpoint output/point_goal_ppo/latest.pt \
  --episodes 200
```

Evaluation uses the bounded policy mean and reports success rate, collision rate,
mean episode return, and mean episode length.

## Pipeline

```text
batched environment
  -> vector + nearest-neighbor observations
  -> actor-critic action/value inference
  -> rollout buffer
  -> GAE returns and normalized advantages
  -> clipped PPO actor loss + value loss + entropy bonus
  -> checkpoint and deterministic evaluation
```

The normalized action is `[linear_velocity, angular_velocity]`, with ranges
`[0, 1]` and `[-1, 1]`. The reference environment maps these values to unicycle
linear and angular velocity limits.

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

## Tests

```bash
pytest
```

## Repository layout

```text
configs/point_goal.json          Default environment and PPO settings
src/crowdsim_ppo/point_goal_env.py  Vectorized reference environment
src/crowdsim_ppo/ppo_policy.py      Network, rollout buffer, PPO update
src/crowdsim_ppo/goal_curriculum.py Goal curriculum and state restore
src/crowdsim_ppo/train.py           End-to-end training loop
src/crowdsim_ppo/evaluate.py        Deterministic evaluation
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
