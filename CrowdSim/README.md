# CrowdSim PPO pipeline

This directory contains the original Isaac Lab training path used by the
CrowdSim robot-navigation PPO expert.

## Main entry points

- `ppo/train_ppo.py`: PPO training, logging, checkpointing and resume
- `ppo/train_ppo_finetune.py`: preference-reward PPO fine-tuning
- `crowd_sim.py`: interactive policy rollout and GIF recording
- `tools/eval_policy.py`: grid and episode evaluation
- `flow/collect_ppo_buffer.py`: expert rollout collection
- `flow/recover_ppo_buffer.py`: interrupted collection recovery

## Environment stack

- `world/builder.py`: builds Isaac Lab, ProtoMotions and CrowdSim together
- `nav_manager.py`: observations, rewards, resets, paths and termination
- `protomotions_runtime.py`: restores MaskedMimic and constructs its runtime
- `scene_setup.py`: scene, robot, sensor and USD setup
- `control/drive.py`: normalized PPO actions to differential drive commands
- `world/map.py`, `planning.py`, `sfm.py`: occupancy map, paths and crowd motion

## Configuration

- `config/env.yaml`: default PPO environment, network and optimizer settings
- `config/scenes/`: scene definitions
- `config/collect/ppo.yaml`: expert collection settings
- `config/reward_model_ppo.yaml`: preference-reward fine-tuning settings

Commands are run from the repository root:

```bash
python CrowdSim/ppo/train_ppo.py --headless

python CrowdSim/tools/eval_policy.py \
  --ckpt output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --env-config CrowdSim/config/env.yaml \
  --num-episodes 200 \
  --headless
```

The environment configuration references external scene USDs, motion data and
MaskedMimic checkpoints. See the root `FULL_PIPELINE.md` for the expected paths.
