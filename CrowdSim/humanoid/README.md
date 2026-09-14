# Humanoid avoidance diagnostic

This test removes navigation robots and runs MaskedMimic humanoids in a fixed
bidirectional encounter. It measures whether the physical pelvis follows the
SFM command and its first future target.

```bash
python CrowdSim/humanoid/run_avoidance_test.py
```

For a short smoke test:

```bash
python CrowdSim/humanoid/run_avoidance_test.py --steps 300
```

Outputs are written to `output/humanoid_avoidance_eval/`:

- `metrics.json`: distance, collision, velocity, heading and target tracking metrics;
- `humanoid_avoidance.gif`: offline bird's-eye rendering;
- `navigation/trajectory_latest.jsonl`: per-step vectors for deeper inspection.

The GIF uses blue for actual pelvis velocity, green for the filtered SFM
velocity, orange for TTC force, and yellow for the first MaskedMimic future
target.
