#!/bin/bash
# ============================================================
# CrowdSim 偏好学习全流程自动化脚本
#
# 用法:
#   bash CrowdSim/scripts/run_pref_pipeline.sh [STAGE]
#
# 阶段:
#   train      — 训练原始 PPO 策略
#   collect    — 采集偏好数据
#   reward     — 训练 reward model
#   finetune   — 微调 PPO 策略
#   all        — 从头执行全部 (默认)
#   resume     — 从最后一个未完成阶段继续
#
# 环境变量覆盖:
#   NUM_ENVS=20 HEADLESS=1 TOTAL_STEPS=1000000 bash ...
# ============================================================
set -euo pipefail

# ── 配置 (可通过环境变量覆盖) ──────────────────────────────
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
NUM_ENVS="${NUM_ENVS:-20}"
HEADLESS="${HEADLESS:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"              # 放 GPU1 避免抢显存
GPU_PREFIX="CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"

# --- PPO 训练参数 ---
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"

# --- 偏好采集参数 ---
NUM_EPISODES="${NUM_EPISODES:-500}"
SEGMENT_LEN="${SEGMENT_LEN:-25}"   # must be explicitly set; 50 steps ≈ 1.7 s @ 30 Hz
MAX_PAIRS="${MAX_PAIRS:-2000}"
PREF_BUFFER="${PREF_BUFFER:-output/pref_data/buffer.pkl}"

# --- Reward 训练参数 ---
BATCH_SIZE="${BATCH_SIZE:-256}"
REWARD_EPOCHS="${REWARD_EPOCHS:-200}"
REWARD_LR="${REWARD_LR:-0.0001}"
REWARD_MODEL="${REWARD_MODEL:-output/reward_models/best.pt}"

# --- 微调参数 ---
FINETUNE_STEPS="${FINETUNE_STEPS:-100000}"
PREF_WEIGHT="${PREF_WEIGHT:-5.0}"
KL_BETA="${KL_BETA:-0.05}"
FINETUNE_LR="${FINETUNE_LR:-0.0001}"

# --- 自动解析路径 ---
HEADLESS_FLAG=""
[ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

# checkpoint 路径（从 latest 软链接自动推断）
BASE_POLICY="output/crowdsim_robot_ppo/latest/robot_ppo_latest.pt"
FINETUNE_POLICY="output/crowdsim_robot_ppo_finetune/latest/robot_ppo_finetune_latest.pt"

# ── 日志 ────────────────────────────────────────────────────
log()  { echo -e "\033[1;36m[$(date +%H:%M:%S)]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*"; exit 1; }

# ── 环境检查 ────────────────────────────────────────────────
check_env() {
    log "检查 conda 环境 ..."
    # conda activate 仅在交互式 shell 中可用；非交互脚本需先 source 初始化脚本
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh" \
        || fail "找不到 conda 初始化脚本（conda info --base 是否正常？）"
    conda activate "$CONDA_ENV" || fail "conda 环境 $CONDA_ENV 不存在"
    python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
}

# ── 阶段 1: 训练原始 PPO 策略 ──────────────────────────────
stage_train() {
    log "========================================"
    log "阶段 1/4: 训练原始 PPO 策略"
    log "========================================"
    log "参数: envs=$NUM_ENVS  steps=$TOTAL_STEPS  rollout=$ROLLOUT_STEPS"

    $GPU_PREFIX python CrowdSim/ppo/train_ppo.py \
        --num-envs "$NUM_ENVS" $HEADLESS_FLAG \
        --total-steps "$TOTAL_STEPS" \
        --rollout-steps "$ROLLOUT_STEPS" \
        --save-interval "$SAVE_INTERVAL" \
        || fail "PPO 训练失败"

    ok "PPO 训练完成 → $BASE_POLICY"
}

# ── 阶段 2: 采集偏好数据 ────────────────────────────────────
stage_collect() {
    log "========================================"
    log "阶段 2/4: 采集偏好数据"
    log "========================================"
    log "参数: envs=$NUM_ENVS  episodes=$NUM_EPISODES  seg_len=$SEGMENT_LEN"

    [ -f "$BASE_POLICY" ] || fail "找不到 base policy: $BASE_POLICY"

    # 写临时 env config，覆盖 policy_checkpoint，避免污染版本控制文件
    TMP_ENV_CFG=$(mktemp --suffix=.yaml)
    trap 'rm -f "$TMP_ENV_CFG"' EXIT TERM INT
    sed "s|policy_checkpoint:.*|policy_checkpoint: $BASE_POLICY|" \
        CrowdSim/config/env.yaml > "$TMP_ENV_CFG"

    $GPU_PREFIX python CrowdSim/pref/pref_collect.py \
        --env-config "$TMP_ENV_CFG" \
        --num-envs "$NUM_ENVS" $HEADLESS_FLAG \
        --num-episodes "$NUM_EPISODES" \
        --segment-len "$SEGMENT_LEN" \
        --max-pairs "$MAX_PAIRS" \
        --save-path "$PREF_BUFFER" \
        || fail "偏好采集失败"

    ok "偏好数据已保存 → $PREF_BUFFER"
}

# ── 阶段 3: 训练 reward model ───────────────────────────────
stage_reward() {
    log "========================================"
    log "阶段 3/4: 训练 reward model"
    log "========================================"
    log "参数: batch=$BATCH_SIZE  epochs=$REWARD_EPOCHS  lr=$REWARD_LR"

    BUFFER_FILE=$(ls -t output/pref_data/buffer_*.pkl 2>/dev/null | head -1)
    [ -z "$BUFFER_FILE" ] && fail "找不到偏好 buffer，先执行阶段 2"

    python CrowdSim/pref/pref_train_reward.py \
        --buffer "$BUFFER_FILE" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$REWARD_EPOCHS" \
        --lr "$REWARD_LR" \
        || fail "reward model 训练失败"

    ok "reward model 训练完成"
}

# ── 阶段 4: 微调 PPO 策略 ───────────────────────────────────
stage_finetune() {
    log "========================================"
    log "阶段 4/4: 微调 PPO 策略"
    log "========================================"
    log "参数: pref_weight=$PREF_WEIGHT  kl=$KL_BETA  steps=$FINETUNE_STEPS"

    [ -f "$BASE_POLICY" ] || fail "找不到 base policy: $BASE_POLICY"

    REWARD_CKPT=$(ls -t output/reward_models/best_*.pt 2>/dev/null | head -1)
    [ -z "$REWARD_CKPT" ] && fail "找不到 reward model，先执行阶段 3"

    $GPU_PREFIX python CrowdSim/ppo/train_ppo_finetune.py \
        --resume "$BASE_POLICY" \
        --reward-ckpt "$REWARD_CKPT" \
        --num-envs "$NUM_ENVS" $HEADLESS_FLAG \
        --total-steps "$FINETUNE_STEPS" \
        --pref-weight "$PREF_WEIGHT" \
        --kl-beta "$KL_BETA" \
        --lr "$FINETUNE_LR" \
        --save-interval "$SAVE_INTERVAL" \
        || fail "微调失败"

    ok "微调完成 → $FINETUNE_POLICY"
}

# ── 所有阶段 ────────────────────────────────────────────────
stage_all() {
    stage_train
    stage_collect
    stage_reward
    stage_finetune
    log "========================================"
    log "全流程完成!"
    log "  微调策略: $FINETUNE_POLICY"
    log "  reward model: $(ls -t output/reward_models/best_*.pt | head -1)"
    log "========================================"
}

# ── 主入口 ──────────────────────────────────────────────────
case "${1:-all}" in
    train)    check_env; stage_train ;;
    collect)  check_env; stage_collect ;;
    reward)   check_env; stage_reward ;;
    finetune) check_env; stage_finetune ;;
    all)      check_env; stage_all ;;
    resume)
        check_env
        [ -f "$BASE_POLICY" ] || { stage_train; stage_collect; stage_reward; stage_finetune; exit 0; }
        REWARD_CKPT=$(ls -t output/reward_models/best_*.pt 2>/dev/null | head -1)
        if [ -z "$REWARD_CKPT" ]; then
            BUFFER_FILE=$(ls -t output/pref_data/buffer_*.pkl 2>/dev/null | head -1)
            [ -z "$BUFFER_FILE" ] && { stage_collect; stage_reward; stage_finetune; exit 0; }
            stage_reward; stage_finetune
        else
            stage_finetune
        fi
        ;;
    *)
        echo "用法: $0 {train|collect|reward|finetune|all|resume}"
        exit 1
        ;;
esac
