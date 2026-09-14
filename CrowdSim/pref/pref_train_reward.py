"""Train a preference reward model from CrowdSim preference data.

Loads a buffer saved by ``pref_collect.py`` (or ``crowd_sim.py`` with
pref_collection enabled), samples preference pairs, and trains a transformer
reward network with Bradley-Terry loss.

Usage::

    python CrowdSim/pref_train_reward.py \\
        --buffer output/pref_data/buffer.pkl \\
        --batch-size 128 --epochs 50 \\
        --output output/reward_models/best.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CrowdSim.pref.pref_buffer import CrowdSimPrefBuffer
from CrowdSim.pref.reward_model_utils import RewardNet_Traj_Sys_SelfAttn


# ═══════════════════════════════════════════════════════════════════
# RewardNet wrapper
# ═══════════════════════════════════════════════════════════════════

class CrowdSimRewardNet(torch.nn.Module):
    """Encode CrowdSim episodes → per-step or summed reward.

    Wraps ``RewardNet_Traj_Sys_SelfAttn`` which now handles depth encoding
    internally.  ``_build_batches`` returns raw images; the net encodes them
    with its own depth CNN, so depth features truly flow into the reward.
    """

    def __init__(
        self,
        system_dim: int = 608,   # kept for _build_batches zero-tensor fallback
        depth_size: int = 224,
        depth_embed_dim: int = 64,
        vec_dim: int | None = None,  # if None, derived from system_dim
        **kwargs,
    ):
        super().__init__()
        self._system_dim = int(system_dim)
        self._depth_size = int(depth_size)
        # Derive vec_dim from system_dim if not explicitly provided.
        # system_dim = vec_dim + map_flat_dim + action_dim(2)
        # With map_size=0: vec_dim = system_dim - 2
        if vec_dim is None:
            vec_dim = system_dim - 2
        self.net = RewardNet_Traj_Sys_SelfAttn(
            vec_dim=vec_dim,
            depth_embed_dim=depth_embed_dim,
            **kwargs,
        )

    def _build_batches(
        self,
        episodes: list[dict],
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded ``(B, T, system_dim)`` and ``(B, T, H, W)`` tensors.

        ``system_dim = len(sys_state) + len(action)`` = 608 by default.
        depth is kept as raw images so the net's CNN can process them.
        """
        sys_list, dep_list = [], []
        for ep in episodes:
            s_feats, d_feats = [], []
            for s in ep["steps"]:
                s_feats.append(np.concatenate([
                    np.asarray(s["sys_state"], dtype=np.float32),
                    np.asarray(s["action"],    dtype=np.float32),
                ]))
                d = s.get("depth")
                d_feats.append(
                    np.asarray(d, dtype=np.float32) if d is not None
                    else np.zeros((self._depth_size, self._depth_size), dtype=np.float32)
                )
            if s_feats:
                sys_list.append(np.stack(s_feats))
                dep_list.append(np.stack(d_feats))
        if not sys_list:
            return (
                torch.zeros(len(episodes), 1, self._system_dim, device=device),
                torch.zeros(len(episodes), 1, self._depth_size, self._depth_size, device=device),
            )
        max_t = max(s.shape[0] for s in sys_list)
        sys_b = torch.zeros(len(sys_list), max_t, sys_list[0].shape[-1], device=device)
        dep_b = torch.zeros(
            len(sys_list), max_t, self._depth_size, self._depth_size, device=device,
        )
        for i, (s, d) in enumerate(zip(sys_list, dep_list)):
            sys_b[i, :s.shape[0]] = torch.as_tensor(s, device=device)
            dep_b[i, :d.shape[0]] = torch.as_tensor(d, device=device)
        return sys_b, dep_b

    def forward(
        self,
        episodes: list[dict],
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Return ``(B,)`` summed episode rewards."""
        sys_b, dep_b = self._build_batches(episodes, device)
        return self.net(sys_b, depth_feats=dep_b).sum(dim=1)

    def forward_per_step(
        self,
        episodes: list[dict],
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Return ``(B, T)`` per-step rewards."""
        sys_b, dep_b = self._build_batches(episodes, device)
        return self.net(sys_b, depth_feats=dep_b)


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def train(
    buffer: CrowdSimPrefBuffer,
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda",
    output: str = "output/reward_models/best.pt",
    val_ratio: float = 0.1,
) -> CrowdSimRewardNet:
    # Auto-detect input dimensions from the first buffer step so the model
    # always matches whatever depth_size was used during collection.
    s0 = buffer.pairs[0][0]["steps"][0]
    sys_state = np.asarray(s0["sys_state"], dtype=np.float32)
    action = np.asarray(s0["action"], dtype=np.float32)
    system_dim = len(sys_state) + len(action)
    # The reward net splits system_feats as [vec | (map) | action].
    # With the map removed from the RL obs, the full sys_state is the vector
    # part — vec_dim = len(sys_state), map_size = 0 (no map branch).
    vec_dim = len(sys_state)
    _d0 = s0.get("depth")
    depth_size = int(_d0.shape[0]) if _d0 is not None and hasattr(_d0, "shape") else 224
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_ts = Path(output).parent / f"{Path(output).stem}_{ts}{Path(output).suffix}"
    model = CrowdSimRewardNet(
        system_dim=system_dim, depth_size=depth_size, vec_dim=vec_dim, map_size=0,
    ).to(device)
    print(f"[RewardTrain] system_dim={system_dim}, vec_dim={vec_dim}, "
          f"map_size=0, depth_size={depth_size}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train/val split
    n_pairs = len(buffer.pairs)
    n_val = max(2, int(n_pairs * val_ratio))
    n_train = n_pairs - n_val
    all_pairs = list(buffer.pairs)
    random.shuffle(all_pairs)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    print(f"[RewardTrain] {n_pairs} pairs (train={n_train} val={n_val}), "
          f"batch_size={batch_size}, epochs={epochs}, lr={lr}")

    best_acc = 0.0
    for epoch in range(epochs):
        # Train step
        indices = [random.randrange(n_train) for _ in range(batch_size)]  # noqa: S311
        batch = [train_pairs[i] for i in indices]
        ep_a, ep_b, labels = zip(*batch)
        labels_t = torch.tensor(labels, device=device, dtype=torch.float32)

        r1 = model(list(ep_a), device=device)
        r2 = model(list(ep_b), device=device)

        loss = F.binary_cross_entropy_with_logits(r1 - r2, labels_t)
        l2_reg = 0.01 * (r1.pow(2).mean() + r2.pow(2).mean())
        total_loss = loss + l2_reg

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        train_acc = ((r1 > r2).float() == labels_t).float().mean().item()

        # Validation (batched to avoid OOM on large datasets)
        val_batch = 64
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for vs in range(0, len(val_pairs), val_batch):
                vb = val_pairs[vs:vs + val_batch]
                ep_va_b, ep_vb_b, vlb = zip(*vb)
                vlt = torch.tensor(vlb, device=device, dtype=torch.float32)
                vr1 = model(list(ep_va_b), device=device)
                vr2 = model(list(ep_vb_b), device=device)
                val_correct += int(((vr1 > vr2).float() == vlt).float().sum().item())
                val_total += len(vb)
                val_loss_sum += F.binary_cross_entropy_with_logits(
                    vr1 - vr2, vlt,
                ).item() * len(vb)
        val_acc = val_correct / max(val_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            star = " *" if val_acc > best_acc else ""
            print(f"  epoch {epoch + 1:3d}/{epochs}  "
                  f"loss={total_loss.item():.4f}  acc={train_acc:.3f}  "
                  f"val_acc={val_acc:.3f}  val_loss={val_loss:.4f}{star}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(output_ts) or ".", exist_ok=True)
            torch.save(model.state_dict(), output_ts)

    print(f"[RewardTrain] Best val_acc={best_acc:.3f}  → {output_ts}")
    return model


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train reward model from preference data")
    p.add_argument("--buffer", default="output/pref_data/buffer.pkl")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="output/reward_models/best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    buf = CrowdSimPrefBuffer.load(args.buffer)
    if len(buf) == 0:
        raise RuntimeError(f"No pairs in buffer: {args.buffer}")
    print(f"[RewardTrain] Loaded: {buf}")
    train(buf, args.batch_size, args.epochs, args.lr, args.device, args.output)


if __name__ == "__main__":
    main()
