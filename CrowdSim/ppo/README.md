# CrowdSim PPO

本目录是 CrowdSim 移动机器人 PPO 专属流水线的权威说明。PPO 在动态人群中学习
PointGoal 局部导航，并作为 NavDP、FLUX、CFM 等生成式策略的数据采集 expert。

本文中的命令均从仓库根目录运行。当前有效入口是：

| 功能 | 入口 |
|---|---|
| PPO 训练与续训 | `CrowdSim/ppo/train_ppo.py` |
| PPO 在线运行与 GIF | `CrowdSim/crowd_sim.py` |
| PPO 网格评测 | `CrowdSim/tools/eval_policy.py` |
| PPO expert 数据采集 | `CrowdSim/flow/collect_ppo_buffer.py` |
| 采集分片恢复 | `CrowdSim/flow/recover_ppo_buffer.py` |
| Preference reward 微调 | `CrowdSim/ppo/train_ppo_finetune.py` |

旧文档中的 `CrowdSim/train_ppo.py`、独立 `ppo.yaml` 等入口已经失效，不应再使用。

## 1. 环境与配置

激活能够运行本项目 Isaac Sim、IsaacLab 和 ProtoMotions 的环境：

```bash
conda activate env_isaaclab
cd /path/to/NavIsaacLab2.0
```

PPO 训练只使用一个配置文件：`CrowdSim/config/env.yaml`。其中包含：

- `scene`：引用 `CrowdSim/config/scenes/` 下的场景配置；
- `humanoid`：MaskedMimic checkpoint、motion 和人体显示设置；
- `car`：Nova Carter 资产和部署时使用的 PPO checkpoint；
- `sensors`：相机位置、分辨率和坐标约定；
- `navigation`：人车数量、路径规划、SFM、人群安全参数；
- `navigation.rl`：PPO 观测、动作范围、奖励、终止条件和目标课程；
- `network`：PPO actor-critic 网络结构；
- `training`：PPO 优化参数、训练步数和输出目录。

`navigation.num_robots` 是并行 PPO rollout 数。`--num-envs` 是 IsaacLab 环境槽位数，
默认自动取 `max(num_humanoids, num_robots, 1)`；通常不需要手动指定。

网络结构和观测配置必须与 checkpoint 一致。修改以下字段后不能直接加载旧模型：

- `network.*`；
- `navigation.rl.num_neighbors`；
- `navigation.rl.map_size`；
- `navigation.rl.depth_enabled` 和 `depth_size`。

## 2. PPO 的输入、输出与网络

策略每个 25 Hz 仿真步接收四类输入：

1. 机器人向量状态：归一化目标距离、目标方向、线速度和角速度；
2. 最近邻居：最多 4 个邻居，每个为距离、方向和机器人坐标系相对速度；
3. 当前深度：默认 `224×224`，按 `depth_max_range=5 m` 归一化；
4. 局部占据图：默认 `24×24`，覆盖机器人周围 `8 m`。

向量、深度和局部地图分别编码后拼接，进入独立 actor 和 critic head。深度 CNN 保留
`4×4` 空间网格，避免全局池化丢失障碍物左右方向。策略输出两个有界动作：

```text
[normalized_linear_velocity, normalized_angular_velocity]
```

它们由差速驱动层映射到 `navigation.rl.max_linear_velocity` 和
`max_angular_velocity`，并在 PhysX 中只积分一次。

PPO 的观测包含局部地图和显式邻居状态，因此它是 privileged expert；NavDP/FLUX
训练使用的是它产生的视觉和轨迹数据，不代表 PPO 与纯视觉方法具有相同输入。

## 3. 从头训练

默认训练：

```bash
python CrowdSim/ppo/train_ppo.py --headless
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python CrowdSim/ppo/train_ppo.py --headless
```

常用命令行覆盖：

```bash
python CrowdSim/ppo/train_ppo.py \
  --env-config CrowdSim/config/env.yaml \
  --total-steps 5000000 \
  --rollout-steps 256 \
  --ppo-epochs 4 \
  --minibatch-size 256 \
  --lr 3e-4 \
  --save-interval 20000 \
  --headless
```

命令行参数优先于 YAML。W&B 默认启用，project 为 `crowdsim-ppo`；离线调试可加：

```bash
python CrowdSim/ppo/train_ppo.py --headless --no-wandb
```

### 目标课程学习

`navigation.rl.goal_curriculum` 根据最近成功率逐级扩大 reset 难度。当前阶段为：

| 阶段 | 起终点距离 | 初始目标偏角 |
|---|---:|---:|
| `short_straight` | 2–4 m | 0–15° |
| `short_turn` | 3–5 m | 10–45° |
| `medium_turn` | 4–7 m | 20–75° |
| `long_turn` | 5.5–8.5 m | 30–105° |
| `full` | 7–10 m | 30–120° |

默认在最近 200 个 episode 中至少积累 100 个样本后开始判断，每 50 个完成 episode
更新一次；成功率不低于 75% 晋级，不高于 45% 降级。目标距离观测始终使用固定
`10 m` 归一化尺度，课程只改变 reset 分布。

课程阶段、历史窗口和完成 episode 数会随 PPO checkpoint 保存并在续训时恢复。

## 4. Checkpoint 与续训

每次训练创建独立目录：

```text
output/crowdsim_robot_ppo/YYYYMMDD_HHMMSS/
├── robot_ppo_latest.pt
├── robot_ppo_<robot_steps>.pt
├── config/env.yaml
├── navigation/
└── wandb/
```

`output/crowdsim_robot_ppo/latest` 是指向最近一次启动训练的符号链接。续训时建议使用
明确的 run 路径，防止多个任务并发时误读：

```bash
python CrowdSim/ppo/train_ppo.py \
  --resume output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --headless
```

续训会恢复网络、optimizer、robot step 和目标课程状态，但会创建新的输出目录保存后续
checkpoint，不会覆盖原 run。

## 5. 训练指标

终端和 W&B 重点查看：

- `outcome/reached_rate`、`collision_rate`、`timeout_rate`、`stuck_rate`；
- `episode/return` 和 `episode/length`；
- `step/goal_distance`、`progress`、`goal_heading_abs_degrees`；
- 各项 reward 分量，确认某个惩罚没有支配总 reward；
- `loss/policy_loss`、`value_loss`、`entropy`、`approx_kl`、`clip_fraction`；
- `value/explained_variance`；
- `action/*` 的低速、饱和转向比例；
- `curriculum/stage` 和 `curriculum/success_rate`。

保存 checkpoint 时会把近期导航轨迹 GIF 上传到 `viz/trajectory`。

## 6. 在线运行和 GIF

先在 `CrowdSim/config/env.yaml` 中设置：

```yaml
car:
  rl_policy: true
  policy_checkpoint: output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt
```

GUI 运行：

```bash
python CrowdSim/crowd_sim.py \
  --env-config CrowdSim/config/env.yaml
```

Headless 并保存鸟瞰 GIF：

```bash
python CrowdSim/crowd_sim.py \
  --env-config CrowdSim/config/env.yaml \
  --headless \
  --gif-out output/ppo_eval/ppo.gif \
  --gif-every 4 \
  --gif-fps 7.5
```

`crowd_sim.py` 默认持续运行，按 `Ctrl+C` 后结束并写出 GIF；它适合行为检查，不提供
固定 episode benchmark。

## 7. 网格评测

对地图中的起点进行网格采样并统计策略表现：

```bash
python CrowdSim/tools/eval_policy.py \
  --ckpt output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --env-config CrowdSim/config/env.yaml \
  --num-episodes 200 \
  --num-envs 1 \
  --headless
```

可通过 `--search-step`、`--min-dist`、`--grid-cell` 和 `--seed` 控制采样。评测保存的
`.pkl` 可以不启动仿真重新生成图：

```bash
python CrowdSim/tools/eval_policy.py \
  --env-config CrowdSim/config/env.yaml \
  --load-data output/PATH/RESULT.pkl
```

## 8. 使用 PPO 采集 expert 数据

采集使用独立配置 `CrowdSim/config/collect/ppo.yaml`。它必须与训练时保持相同 PPO
网络结构，但可以单独设置人车数量、采集步数、成功样本过滤、GIF 和 W&B。

```bash
python CrowdSim/flow/collect_ppo_buffer.py \
  --env-config CrowdSim/config/collect/ppo.yaml \
  --ppo-ckpt output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt
```

除 checkpoint 路径外，运行参数全部来自 YAML 的 `collection`、`gif` 和 `wandb`
字段。默认 deterministic rollout，并保存同步的：

- PPO 向量观测与动作；
- 独立 `neighbors` 和 `neighbor_mask`；
- 深度、RGB 和局部地图；
- 机器人 pose、目标、episode 结果和时序 metadata。

输出位于：

```text
output/collect_buffer/ppo/
├── raw_YYYYMMDD_HHMMSS.pt
├── checkpoints/ckpt_XXXXXX.pt
├── preview/
└── wandb_collect/
```

分片是追加式且原子保存的。最终 raw buffer 成功落盘后采集器才会删除 recovery shards。
如果磁盘不足导致最终保存失败，不要重新采集，也不要删除 `checkpoints/`。

### 从分片恢复 raw buffer

```bash
python CrowdSim/flow/recover_ppo_buffer.py \
  --checkpoint-dir output/collect_buffer/ppo/checkpoints \
  --output /data/output/collect_buffer/ppo/raw_recovered.pt \
  --depth-max-range 5.0
```

`--ppo-ckpt` 仅用于把 expert checkpoint 路径写入恢复文件的 metadata，不参与拼接，
因此可以省略。恢复脚本不会修改原始 shards。

恢复或采集得到的 raw buffer 再交给具体方法的数据构建器：

- NavDP/FLUX：`CrowdSim/navdp/build_navdp_dataset.py`；
- BEV-CFM：`CrowdSim/bev_cfm/build_bev_cfm_dataset.py`；
- 旧 Flow pipeline：`CrowdSim/flow/build_flow_dataset.py`。

## 9. Preference reward 微调

这是一条可选实验分支，不属于生成 expert 数据的必需步骤：

```bash
python CrowdSim/ppo/train_ppo_finetune.py \
  --resume output/crowdsim_robot_ppo/RUN/robot_ppo_latest.pt \
  --reward-ckpt output/reward_models/best.pt \
  --total-steps 200000 \
  --pref-weight 5.0 \
  --kl-beta 0.05 \
  --headless
```

微调 reward 为环境原始 reward 加 preference reward；`kl-beta` 用于限制策略偏离基准
PPO。没有经过 reward model 数据和尺度验证时，不应把该结果当作默认 expert。

## 10. 验证与常见问题

运行不依赖完整训练的检查：

```bash
python CrowdSim/tools/test_robot_obs.py
python -m pytest -q \
  CrowdSim/tests/test_ppo_goal_curriculum.py \
  CrowdSim/tests/test_drive_kinematics.py \
  CrowdSim/tests/test_robot_spawn_heading.py
```

在 Isaac/PhysX 环境中验证控制命令没有重复积分：

```bash
python CrowdSim/tools/validate_drive_integration.py \
  --config CrowdSim/config/collect/ppo.yaml
```

常见问题：

- checkpoint shape mismatch：训练与运行配置的 network、邻居数、depth/map 不一致；
- 刚启动长时间无输出：Isaac Sim、场景和相机初始化本身较慢，不要过早终止；
- RGB/depth 偶发旧帧：确认 reset 后相机 pose 同步代码存在，并避免绕过标准 env loop；
- 机器人只直走：检查目标偏角分布、`action_abs_angular` 和课程是否停在直行阶段；
- 最终 raw 保存失败：保留 recovery shards，换到空间充足的文件系统再恢复。
