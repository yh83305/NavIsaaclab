"""Recover a PPO raw buffer from append-only collection checkpoints.

The collector writes each ``ckpt_XXXXXX.pt`` as a list of complete episodes.
This utility validates those shards, joins them in numeric order, and writes the
same top-level structure consumed by the dataset builders::

    {"episodes": [...], "meta": {...}}

Example:
    python CrowdSim/flow/recover_ppo_buffer.py \
        --checkpoint-dir output/collect_buffer/ppo/checkpoints \
        --output /data/output/collect_buffer/ppo/raw_recovered.pt \
        --depth-max-range 5.0

The output should normally be placed on a filesystem with at least as much free
space as the checkpoint shards occupy.  Source checkpoints are never modified.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import threading
from typing import Any

import torch
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join PPO collection checkpoint shards into one raw buffer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing ckpt_XXXXXX.pt shards.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination raw_*.pt file. It may be on another filesystem.",
    )
    parser.add_argument(
        "--ppo-ckpt",
        default="",
        help="Optional PPO policy checkpoint recorded in recovered metadata.",
    )
    parser.add_argument(
        "--depth-max-range",
        required=True,
        type=float,
        help="Metric range used to normalize depth during collection.",
    )
    return parser.parse_args()


def checkpoint_shards(checkpoint_dir: Path) -> list[Path]:
    """Return numerically ordered shards and reject gaps or ambiguous names."""
    candidates = list(checkpoint_dir.glob("ckpt_*.pt"))
    indexed: list[tuple[int, Path]] = []
    invalid: list[str] = []
    for path in candidates:
        suffix = path.stem.removeprefix("ckpt_")
        if suffix.isdigit():
            indexed.append((int(suffix), path))
        else:
            invalid.append(path.name)
    if invalid:
        raise ValueError(f"Unexpected checkpoint filenames: {sorted(invalid)}")
    if not indexed:
        raise FileNotFoundError(f"No ckpt_XXXXXX.pt shards found in {checkpoint_dir}")

    indexed.sort(key=lambda item: item[0])
    ids = [item[0] for item in indexed]
    expected = list(range(ids[0], ids[-1] + 1))
    if ids != expected:
        missing = sorted(set(expected) - set(ids))
        raise ValueError(f"Checkpoint sequence has gaps; missing shard ids: {missing}")
    return [path for _, path in indexed]


def load_shard(path: Path) -> list[dict[str, Any]]:
    """Load tensors through mmap so recovery does not require dataset-sized RAM."""
    episodes = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(episodes, list):
        raise TypeError(f"{path} must contain a list, got {type(episodes).__name__}")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise TypeError(f"{path}: episode {index} is not a dict")
        for field in ("obs", "action", "neighbors", "neighbor_mask"):
            if not isinstance(episode.get(field), torch.Tensor):
                raise ValueError(
                    f"{path}: episode {index} has no tensor {field!r}"
                )
        neighbors = episode["neighbors"]
        neighbor_mask = episode["neighbor_mask"]
        if neighbors.ndim != 3 or neighbors.shape[-1] != 5:
            raise ValueError(
                f"{path}: episode {index} expected neighbors [T,K,5], "
                f"got {tuple(neighbors.shape)}"
            )
        if tuple(neighbor_mask.shape) != tuple(neighbors.shape[:2]):
            raise ValueError(
                f"{path}: episode {index} neighbor_mask "
                f"{tuple(neighbor_mask.shape)} does not match neighbors "
                f"{tuple(neighbors.shape)}"
            )
    return episodes


def infer_meta(
    episodes: list[dict[str, Any]],
    shards: list[Path],
    ppo_ckpt: str,
    depth_max_range: float,
) -> dict:
    first = episodes[0]
    obs = first["obs"]
    depth = first.get("depth")
    neighbors = first.get("neighbors")
    return {
        "schema_version": 2,
        "algorithm": "ppo",
        "num_episodes": len(episodes),
        "total_steps": sum(int(episode["obs"].shape[0]) for episode in episodes),
        "obs_dim": int(obs.shape[1]),
        "depth_size": int(depth.shape[-1]) if isinstance(depth, torch.Tensor) else 0,
        "depth_max_range": depth_max_range,
        "num_neighbors": int(neighbors.shape[1]) if isinstance(neighbors, torch.Tensor) else 0,
        "neighbor_dim": int(neighbors.shape[2]) if isinstance(neighbors, torch.Tensor) else 0,
        "ppo_ckpt": ppo_ckpt,
        "recovered_from": [str(path.resolve()) for path in shards],
    }


def _atomic_save_with_progress(
    payload: dict,
    temporary: Path,
    output: Path,
    estimated_bytes: int,
) -> None:
    """Save atomically while reporting approximate bytes written.

    ``torch.save`` has no progress callback. A lightweight monitor observes
    the temporary file size, which tracks actual filesystem write progress.
    """
    stop = threading.Event()
    progress = tqdm(
        total=max(int(estimated_bytes), 1),
        desc="[Recover] Writing raw buffer (estimated)",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    )

    def monitor() -> None:
        previous = 0
        while not stop.wait(0.5):
            try:
                current = temporary.stat().st_size
            except FileNotFoundError:
                current = 0
            displayed = min(current, progress.total)
            if displayed > previous:
                progress.update(displayed - previous)
                previous = displayed

    watcher = threading.Thread(
        target=monitor,
        name="ppo-raw-save-progress",
        daemon=True,
    )
    watcher.start()
    succeeded = False
    try:
        torch.save(payload, temporary)
        final_bytes = temporary.stat().st_size
        os.replace(temporary, output)
        succeeded = True
    finally:
        stop.set()
        watcher.join(timeout=2.0)
        if succeeded:
            if progress.n < progress.total:
                progress.update(progress.total - progress.n)
            progress.set_postfix_str(
                f"actual={final_bytes / 1024**3:.2f} GiB",
                refresh=False,
            )
        progress.close()


def recover(
    checkpoint_dir: Path,
    output: Path,
    depth_max_range: float,
    ppo_ckpt: str = "",
) -> dict:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if depth_max_range <= 0.0:
        raise ValueError("depth_max_range must be positive")
    shards = checkpoint_shards(checkpoint_dir)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(path.stat().st_size for path in shards)
    free_bytes = shutil.disk_usage(output.parent).free
    # torch.save output is normally close to the sum of shard sizes. Keep a
    # small margin for pickle metadata and filesystem accounting.
    required_bytes = int(source_bytes * 1.02) + 64 * 1024**2
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient space in {output.parent}: checkpoint shards occupy "
            f"{source_bytes / 1024**3:.2f} GiB, recovery requires approximately "
            f"{required_bytes / 1024**3:.2f} GiB, but only "
            f"{free_bytes / 1024**3:.2f} GiB is available. Choose --output on "
            "another filesystem."
        )

    episodes: list[dict[str, Any]] = []
    shard_progress = tqdm(
        shards,
        desc="[Recover] Loading checkpoint shards",
        unit="shard",
        dynamic_ncols=True,
    )
    for path in shard_progress:
        shard_episodes = load_shard(path)
        episodes.extend(shard_episodes)
        shard_progress.set_postfix(
            file=path.name,
            episodes=f"{len(episodes):,}",
        )
    if not episodes:
        raise ValueError("Checkpoint shards contain no episodes")

    payload = {
        "episodes": episodes,
        "meta": infer_meta(episodes, shards, ppo_ckpt, depth_max_range),
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        _atomic_save_with_progress(
            payload,
            temporary,
            output,
            source_bytes,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload["meta"]


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    meta = recover(
        Path(args.checkpoint_dir),
        output,
        args.depth_max_range,
        args.ppo_ckpt,
    )
    print(
        f"[Recover] Saved {output.expanduser().resolve()} "
        f"({output.expanduser().resolve().stat().st_size / 1024**3:.2f} GiB)\n"
        f"  episodes={meta['num_episodes']:,} steps={meta['total_steps']:,}"
    )


if __name__ == "__main__":
    main()
