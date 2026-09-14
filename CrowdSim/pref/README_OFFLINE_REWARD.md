# Offline view-clearance reward model

This pipeline mines close pedestrian interactions from an existing synchronized
PPO raw buffer, labels pairwise preferences with an auditable script teacher,
trains a temporal reward model, and renders the dataset and learned scores.

## Teacher definition

At each sampled frame, all pedestrians are transformed from world coordinates
to the camera's horizontal field of view. The teacher uses the nearest visible
pedestrian distance. A clip is retained only when a pedestrian is within the
interaction radius for a configured fraction of frames and the distance
changes enough to represent an encounter. Higher clearance is always better;
the lower-distance tail and collisions receive extra weight.

This is a geometric field-of-view test, not an image segmentation or occlusion
test. Camera mount and horizontal FOV come from raw-buffer metadata when
available. Legacy buffers use the PPO collection defaults and can be overridden
on the command line.

## 1. Mine interaction clips and preference pairs

```bash
python CrowdSim/pref/build_offline_preferences.py \
  --raw-buffer output/collect_buffer/ppo/raw_*.pt \
  --output output/pref_data/offline_view.pt \
  --segment-steps 16 \
  --frame-stride 3 \
  --interaction-distance 3.0 \
  --max-clips 10000 \
  --max-pairs 50000
```

The default 16 steps at stride 3 correspond to a 1.6-second sequence at 10 Hz
from a 30 Hz raw buffer. Depth is area-resized to 64x64 and stored as float16.
Each clip is stored once; preference pairs contain only two clip indices, their
teacher label, margin, and train/validation split. Splits are assigned by source
episode before pairing, so overlapping windows cannot leak across splits.

For a legacy buffer without camera metadata, pass for example:

```bash
  --camera-mount-pos 0.25 0.0 0.6 --horizontal-fov-deg 90
```

## 2. Train the reward model

Before training, the same visualization command from step 3 can be run without
`--checkpoint` to inspect script-teacher pairs only.

```bash
python CrowdSim/pref/train_offline_reward.py \
  --dataset output/pref_data/offline_view.pt \
  --epochs 50 \
  --batch-size 64 \
  --device cuda
```

The model consumes the same vector observation, executed action, and depth
sequence as the collected policy data. It is trained with Bradley-Terry
pairwise loss plus a small teacher-score calibration loss. The run directory
contains `reward_best.pt`, `reward_latest.pt`, `history.json`, and
`metrics.json`.

## 3. Visualize teacher data and learned rewards

```bash
python CrowdSim/tools/visualize_offline_preferences.py \
  --dataset output/pref_data/offline_view.pt \
  --checkpoint output/reward_models/offline_view/<run>/reward_best.pt \
  --output-dir output/pref_visualization \
  --num-pairs 12
```

Outputs include teacher/model score histograms, pair-margin histograms,
side-by-side pair GIFs, and `summary.json` with teacher/model agreement and
score correlation.

## 4. Closed-loop PPO finetuning

Start from a normal CrowdSim PPO policy and the offline reward checkpoint:

```bash
python CrowdSim/ppo/train_ppo_finetune.py \
  --env-config CrowdSim/config/env.yaml \
  --train-config CrowdSim/config/reward_model_ppo.yaml \
  --base-ppo output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt \
  --reward-ckpt output/reward_models/offline_view/<run>/reward_best.pt \
  --headless \
  --no-wandb
```

The online reward history is automatically aligned to the dataset metadata.
For the default dataset this means 16 observations at 10 Hz, sampled every
three steps from the 30 Hz navigation loop. The reward model is frozen and
receives the executed bounded action. Its clearance score is combined with
raw navigation progress, goal completion, collision, timeout, stuck, and time
terms. The original environment reward has zero weight by default so its
proximity term is not counted a second time. Environment and sensor settings
remain in `CrowdSim/config/env.yaml`; PPO and reward weights live separately in
`CrowdSim/config/reward_model_ppo.yaml` and can also be overridden by CLI.

Legacy reward checkpoints without raw-buffer `source_hz` metadata require
`--reward-source-hz 30`. To resume interrupted finetuning, pass both the same
`--base-ppo` and `--resume <rm-ppo-checkpoint>`; the base policy remains the KL
reference.

Each run writes `robot_ppo_latest.pt`, periodic checkpoints, `metrics.jsonl`,
`summary.json`, `run_config.json`, copies of both configurations, navigation logs, and
`training_rollout.gif`. The checkpoint follows the standard PPO schema.

## 5. Closed-loop policy evaluation

```bash
python CrowdSim/tools/eval_policy.py \
  --ckpt output/crowdsim_reward_model_ppo/latest/robot_ppo_latest.pt \
  --env-config CrowdSim/config/env.yaml \
  --headless \
  --num-episodes 200
```

Evaluation runs the policy in the simulator and writes success, collision,
timeout, and stuck rates to `metrics.json`, alongside `evaluation.gif`, the raw
pickle, and trajectory, visitation, and velocity plots.
