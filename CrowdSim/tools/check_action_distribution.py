"""Inspect action distributions in built Flow datasets or raw buffers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print linear/angular action statistics without loading the simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--zero-eps", type=float, default=1e-6)
    parser.add_argument("--moving-threshold", type=float, default=0.05)
    return parser.parse_args()


def load_actions(path: Path) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict at the file root, got {type(payload).__name__}")

    traj = payload.get("traj")
    if isinstance(traj, torch.Tensor):
        if traj.ndim != 3 or traj.shape[-1] != 2:
            raise ValueError(f"Expected traj [N,H,2], got {tuple(traj.shape)}")
        traj = traj.detach().float().cpu()
        return traj.reshape(-1, 2), traj[:, 0], f"built dataset traj={tuple(traj.shape)}"

    episodes = payload.get("episodes")
    if isinstance(episodes, (list, tuple)):
        chunks: list[torch.Tensor] = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict) or "action" not in episode:
                raise ValueError(f"Episode {index} has no action tensor")
            action = torch.as_tensor(episode["action"]).detach().float().cpu()
            if action.ndim != 2 or action.shape[-1] != 2:
                raise ValueError(
                    f"Expected episodes[{index}].action [T,2], got {tuple(action.shape)}"
                )
            if action.numel() > 0:
                chunks.append(action)
        if not chunks:
            raise ValueError("Raw buffer contains no actions")
        actions = torch.cat(chunks, dim=0)
        return actions, None, f"raw buffer episodes={len(episodes)} actions={len(actions)}"

    action = payload.get("action")
    if isinstance(action, torch.Tensor):
        action = action.detach().float().cpu()
        if action.ndim != 2 or action.shape[-1] != 2:
            raise ValueError(f"Expected action [N,2], got {tuple(action.shape)}")
        return action, None, f"action tensor={tuple(action.shape)}"

    raise ValueError("No supported 'traj', 'episodes[*].action', or 'action' field found")


def percentage(mask: torch.Tensor) -> float:
    return 100.0 * mask.float().mean().item()


def print_stats(name: str, actions: torch.Tensor, zero_eps: float, moving: float) -> None:
    if actions.numel() == 0:
        print(f"  {name}: empty")
        return

    finite = torch.isfinite(actions).all(dim=-1)
    invalid = int((~finite).sum().item())
    actions = actions[finite]
    if actions.numel() == 0:
        print(f"  {name}: no finite actions (invalid={invalid})")
        return

    linear, angular = actions[:, 0], actions[:, 1]
    quantiles = torch.tensor([0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
    print(f"  [{name}] count={len(actions):,} invalid={invalid:,}")
    print(
        "    linear: "
        f"mean={linear.mean().item():.6f} std={linear.std(unbiased=False).item():.6f} "
        f"negative={percentage(linear < -zero_eps):.2f}% "
        f"zero={percentage(linear.abs() <= zero_eps):.2f}% "
        f"low(0<v<={moving:g})={percentage((linear > zero_eps) & (linear <= moving)):.2f}% "
        f"moving(v>{moving:g})={percentage(linear > moving):.2f}%"
    )
    print("    linear quantiles [0,1,10,25,50,75,90,99,100]%:")
    print("      " + "  ".join(f"{v.item():.6f}" for v in torch.quantile(linear, quantiles)))
    print(
        "    angular: "
        f"mean={angular.mean().item():.6f} std={angular.std(unbiased=False).item():.6f} "
        f"|w|<={zero_eps:g}={percentage(angular.abs() <= zero_eps):.2f}%"
    )
    print("    angular quantiles [0,1,10,25,50,75,90,99,100]%:")
    print("      " + "  ".join(f"{v.item():.6f}" for v in torch.quantile(angular, quantiles)))


def main() -> None:
    args = parse_args()
    for path in args.input:
        print(f"\n=== {path} ===")
        try:
            all_actions, first_actions, schema = load_actions(path)
            print(f"  format: {schema}")
            if first_actions is not None:
                print_stats("first step", first_actions, args.zero_eps, args.moving_threshold)
                print_stats("all horizon steps", all_actions, args.zero_eps, args.moving_threshold)
            else:
                print_stats("all collected steps", all_actions, args.zero_eps, args.moving_threshold)
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
