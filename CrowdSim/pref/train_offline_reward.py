# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a reward model on script-teacher offline preference pairs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CrowdSim.pref.build_offline_preferences import SCHEMA  # noqa: E402
from CrowdSim.pref.offline_reward_model import OfflineRewardModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="output/reward_models/offline_view")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _batch_rewards(model, clips, clip_ids: torch.Tensor, device: torch.device):
    ids = clip_ids.cpu()
    obs = clips["obs"][ids].to(device, non_blocking=device.type == "cuda")
    action = clips["action"][ids].to(device, non_blocking=device.type == "cuda")
    depth_data = clips.get("depth")
    depth = (
        None if depth_data is None else
        depth_data[ids].to(device, non_blocking=device.type == "cuda")
    )
    return model(obs, action, depth)


@torch.no_grad()
def evaluate(model, dataset, indices, device, batch_size: int) -> dict[str, float]:
    model.eval()
    pairs, labels = dataset["pairs"], dataset["labels"]
    rewards_a, rewards_b, targets = [], [], []
    for start in range(0, len(indices), batch_size):
        selection = indices[start:start + batch_size]
        batch_pairs = pairs[selection]
        with _autocast(device):
            reward_a = _batch_rewards(model, dataset["clips"], batch_pairs[:, 0], device)
            reward_b = _batch_rewards(model, dataset["clips"], batch_pairs[:, 1], device)
        rewards_a.append(reward_a.float().cpu())
        rewards_b.append(reward_b.float().cpu())
        targets.append(labels[selection].float().cpu())
    reward_a = torch.cat(rewards_a)
    reward_b = torch.cat(rewards_b)
    target = torch.cat(targets)
    scale = float(model.logit_scale_log.detach().float().cpu().exp().clamp(max=100.0))
    logits = scale * (reward_a - reward_b)
    accuracy = ((logits > 0) == target.bool()).float().mean().item()
    loss = F.binary_cross_entropy_with_logits(logits, target).item()
    margin = torch.where(target.bool(), reward_a - reward_b, reward_b - reward_a)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "preferred_reward_margin": float(margin.mean()),
    }


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset_path = Path(args.dataset).expanduser().resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    metadata = dataset.get("metadata", {})
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}, got {metadata.get('schema')!r}")
    clips = dataset["clips"]
    obs_dim = int(clips["obs"].shape[-1])
    action_dim = int(clips["action"].shape[-1])
    segment_steps = int(clips["obs"].shape[1])
    model = OfflineRewardModel(
        obs_dim=obs_dim, action_dim=action_dim, max_len=max(200, segment_steps)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported()
    )

    pair_split = dataset["pair_split"]
    train_indices = torch.nonzero(pair_split == 0, as_tuple=False).flatten()
    val_indices = torch.nonzero(pair_split == 1, as_tuple=False).flatten()
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Dataset must contain both train and validation pairs")
    loader = DataLoader(
        TensorDataset(train_indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_clip_ids = torch.unique(dataset["pairs"][train_indices].flatten())
    teacher_scores = clips["teacher_score"][train_clip_ids].float()
    teacher_mean = float(teacher_scores.mean())
    teacher_std = max(float(teacher_scores.std()), 1e-6)

    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    best_path = run_dir / "reward_best.pt"
    print(
        f"[OfflineReward] clips={len(clips['obs']):,} "
        f"train_pairs={len(train_indices):,} val_pairs={len(val_indices):,} "
        f"obs={obs_dim} steps={segment_steps} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "rank": 0.0, "cal": 0.0, "correct": 0.0, "n": 0}
        for (selection,) in loader:
            batch_pairs = dataset["pairs"][selection]
            labels = dataset["labels"][selection].float().to(device)
            teacher_a = clips["teacher_score"][batch_pairs[:, 0]].float().to(device)
            teacher_b = clips["teacher_score"][batch_pairs[:, 1]].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                reward_a = _batch_rewards(model, clips, batch_pairs[:, 0], device)
                reward_b = _batch_rewards(model, clips, batch_pairs[:, 1], device)
                logits = model.preference_logits(reward_a, reward_b)
                rank_loss = F.binary_cross_entropy_with_logits(logits, labels)
                target_a = torch.tanh((teacher_a - teacher_mean) / teacher_std)
                target_b = torch.tanh((teacher_b - teacher_mean) / teacher_std)
                calibration = F.mse_loss(reward_a, target_a) + F.mse_loss(reward_b, target_b)
                loss = rank_loss + args.calibration_weight * calibration
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(selection)
            totals["loss"] += float(loss.detach()) * count
            totals["rank"] += float(rank_loss.detach()) * count
            totals["cal"] += float(calibration.detach()) * count
            totals["correct"] += float(((logits > 0) == labels.bool()).sum())
            totals["n"] += count

        validation = evaluate(
            model, dataset, val_indices, device, max(args.batch_size, 64)
        )
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["n"],
            "train_rank_loss": totals["rank"] / totals["n"],
            "train_calibration_loss": totals["cal"] / totals["n"],
            "train_accuracy": totals["correct"] / totals["n"],
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_preferred_reward_margin": validation["preferred_reward_margin"],
        }
        history.append(record)
        print(
            f"[OfflineReward] epoch={epoch:03d}/{args.epochs} "
            f"loss={record['train_loss']:.4f} acc={record['train_accuracy']:.3f} "
            f"val_loss={record['val_loss']:.4f} val_acc={record['val_accuracy']:.3f}"
        )
        payload = {
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "epoch": epoch,
            "metrics": record,
            "teacher_normalization": {"mean": teacher_mean, "std": teacher_std},
            "dataset": str(dataset_path),
            "dataset_metadata": metadata,
            "training_config": vars(args),
        }
        torch.save(payload, run_dir / "reward_latest.pt")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save(payload, best_path)

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"best_val_accuracy": best_accuracy, "epochs": args.epochs}, indent=2),
        encoding="utf-8",
    )
    print(f"[OfflineReward] best_val_accuracy={best_accuracy:.4f} checkpoint={best_path}")
    return best_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
