# CrowdSim configuration

Configurations are grouped by lifecycle rather than kept in one flat folder:

- `env.yaml`: shared CrowdSim environment and PPO training configuration.
- `scenes/`: scene layouts referenced by the `scene` field.
- `collect/`: expert data collection configurations.
- `train/`: model pretraining and fine-tuning configurations.
- `eval/`: policy evaluation, visualization, and benchmark configurations.

Each executable has a default configuration in the matching directory. Pass a
different file with its `--config` or `--env-config` argument when running an
experiment variant.
