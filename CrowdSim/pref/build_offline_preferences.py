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

"""Mine interaction clips and script-teacher preference pairs from PPO data.

The output stores each clip once and represents preferences with integer clip
indices.  This is substantially smaller than serialising both clips into every
pair, especially when depth is present.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import glob
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CrowdSim.pref.script_teacher import (  # noqa: E402
    ViewClearanceTeacher,
    ViewClearanceTeacherConfig,
)


SCHEMA = "crowdsim_offline_view_preferences_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-buffer", nargs="+", required=True,
        help="One or more raw PPO .pt buffers (directories are also accepted).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .pt path; defaults to output/pref_data/offline_view_<time>.pt.",
    )
    parser.add_argument("--segment-steps", type=int, default=16)
    parser.add_argument(
        "--frame-stride", type=int, default=3,
        help="Subsample raw 30 Hz frames; 3 produces a 10 Hz reward sequence.",
    )
    parser.add_argument("--window-stride", type=int, default=15)
    parser.add_argument("--max-clips", type=int, default=10_000)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--depth-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pair-score-margin", type=float, default=0.15)
    parser.add_argument(
        "--horizontal-fov-deg", type=float, default=None,
        help="Overrides camera FOV metadata; legacy buffers default to 90 degrees.",
    )
    parser.add_argument("--max-view-distance", type=float, default=5.0)
    parser.add_argument("--interaction-distance", type=float, default=3.0)
    parser.add_argument("--safe-distance", type=float, default=1.2)
    parser.add_argument("--collision-distance", type=float, default=0.7)
    parser.add_argument("--min-interaction-fraction", type=float, default=0.25)
    parser.add_argument("--min-distance-change", type=float, default=0.15)
    parser.add_argument(
        "--camera-mount-pos", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Overrides raw-buffer camera_mount_pos metadata.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _numpy(value: Any, dtype=None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _resolve_sources(values: list[str]) -> list[Path]:
    sources: list[Path] = []
    for value in values:
        wildcard_matches = [Path(match).resolve() for match in glob.glob(value)]
        if wildcard_matches:
            sources.extend(path for path in wildcard_matches if path.is_file())
            continue
        path = Path(value).expanduser().resolve()
        if path.is_file():
            sources.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        final_buffers = sorted(path.glob("raw_*.pt"))
        if final_buffers:
            sources.extend(final_buffers)
        else:
            sources.extend(sorted(path.glob("episodes_*.pt")))
    unique = list(dict.fromkeys(sources))
    if not unique:
        raise FileNotFoundError("No raw PPO .pt files were found")
    return unique


def _load_source(path: Path) -> tuple[list[dict], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        return payload["episodes"], dict(payload.get("meta", {}))
    if isinstance(payload, list):
        return payload, {}
    raise ValueError(f"Unsupported raw buffer format: {path}")


def _resize_depth(depth: torch.Tensor | None, indices: np.ndarray, size: int):
    if depth is None:
        return None
    selected = depth[torch.as_tensor(indices, dtype=torch.long)].float()
    if selected.ndim != 3:
        raise ValueError(f"Depth must be [T,H,W], got {tuple(selected.shape)}")
    if tuple(selected.shape[-2:]) != (size, size):
        selected = F.interpolate(
            selected[:, None], size=(size, size), mode="area"
        )[:, 0]
    return selected.to(torch.float16).contiguous()


def _extract_clip(
    episode: dict,
    indices: np.ndarray,
    *,
    source_id: int,
    source_episode_id: int,
    episode_uid: int,
    start: int,
    depth_size: int,
    teacher_result,
) -> dict:
    world = episode["world"]
    robot_xy = _numpy(world["robot_xy"], np.float32)[indices]
    yaw = _numpy(world["yaw"], np.float32)[indices]
    goal_xy = _numpy(world["goal_xy"], np.float32)[indices]
    human_xy = _numpy(world["hum_xys"], np.float32)[indices]
    obs = episode["obs"][torch.as_tensor(indices, dtype=torch.long)].float().cpu()
    action = episode["action"][torch.as_tensor(indices, dtype=torch.long)].float().cpu()
    return {
        "obs": obs.contiguous(),
        "action": action.contiguous(),
        "depth": _resize_depth(episode.get("depth"), indices, depth_size),
        "robot_xy": torch.from_numpy(robot_xy.copy()),
        "yaw": torch.from_numpy(yaw.copy()),
        "goal_xy": torch.from_numpy(goal_xy.copy()),
        "human_xy": torch.from_numpy(human_xy.copy()),
        "teacher_distance": torch.from_numpy(
            teacher_result.nearest_visible_distance.copy()
        ),
        "visible_count": torch.from_numpy(teacher_result.visible_count.copy()),
        "teacher_score": float(teacher_result.score),
        "interaction_fraction": float(teacher_result.interaction_fraction),
        "collision_fraction": float(teacher_result.collision_fraction),
        "distance_change_m": float(teacher_result.distance_change_m),
        "mean_visible_distance_m": float(
            teacher_result.mean_visible_distance_m
        ),
        "lower_visible_distance_m": float(
            teacher_result.lower_visible_distance_m
        ),
        "goal_distance_m": float(np.linalg.norm(goal_xy[0] - robot_xy[0])),
        "source_id": int(source_id),
        "source_episode_id": int(source_episode_id),
        "episode_uid": int(episode_uid),
        "start": int(start),
    }


def _episode_splits(clips: list[dict], val_ratio: float, rng) -> dict[int, int]:
    episodes = np.asarray(sorted({int(clip["episode_uid"]) for clip in clips}))
    if len(episodes) < 4:
        raise ValueError(
            "At least four source episodes are required so train and validation "
            "can each form cross-episode pairs"
        )
    rng.shuffle(episodes)
    count = min(
        len(episodes) - 2,
        max(2, int(round(len(episodes) * val_ratio))),
    )
    validation = set(episodes[:count].tolist())
    return {int(uid): int(uid in validation) for uid in episodes}


def _compatible(a: dict, b: dict, relaxed: bool = False) -> bool:
    if a["episode_uid"] == b["episode_uid"]:
        return False
    if relaxed:
        return True
    mean_count_a = float(a["visible_count"].float().mean())
    mean_count_b = float(b["visible_count"].float().mean())
    return (
        abs(mean_count_a - mean_count_b) <= 1.0
        and abs(a["goal_distance_m"] - b["goal_distance_m"]) <= 3.0
    )


def _make_pairs(
    clips: list[dict],
    clip_split: np.ndarray,
    *,
    max_pairs: int,
    score_margin: float,
    val_ratio: float,
    rng,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_ids: list[tuple[int, int]] = []
    labels: list[int] = []
    margins: list[float] = []
    splits: list[int] = []
    targets = {
        0: max(1, int(round(max_pairs * (1.0 - val_ratio)))),
        1: max(1, int(round(max_pairs * val_ratio))),
    }
    seen: set[tuple[int, int]] = set()

    for split in (0, 1):
        candidates = np.flatnonzero(clip_split == split)
        if len(candidates) < 2:
            continue
        target = targets[split]
        made = 0
        attempts = 0
        max_attempts = max(2_000, target * 80)
        relaxed = False
        while made < target and attempts < max_attempts:
            attempts += 1
            if attempts == max_attempts // 2:
                relaxed = True
            first, second = rng.choice(candidates, size=2, replace=False).tolist()
            key = tuple(sorted((first, second)))
            if key in seen or not _compatible(clips[first], clips[second], relaxed):
                continue
            margin = float(clips[first]["teacher_score"] - clips[second]["teacher_score"])
            if abs(margin) < score_margin:
                continue
            seen.add(key)
            # Randomise A/B ordering so the network cannot exploit pair position.
            if bool(rng.integers(0, 2)):
                first, second, margin = second, first, -margin
            pair_ids.append((first, second))
            labels.append(int(margin > 0.0))
            margins.append(abs(margin))
            splits.append(split)
            made += 1
    if not pair_ids:
        raise ValueError(
            "No preference pairs met the score margin; lower "
            "--pair-score-margin or collect more diverse interactions"
        )
    return (
        np.asarray(pair_ids, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(margins, dtype=np.float32),
        np.asarray(splits, dtype=np.int8),
    )


def _stack_clips(clips: list[dict]) -> dict[str, Any]:
    human_count = max(clip["human_xy"].shape[1] for clip in clips)
    n, steps = len(clips), clips[0]["human_xy"].shape[0]
    human_xy = torch.zeros(n, steps, human_count, 2, dtype=torch.float32)
    human_mask = torch.zeros(n, steps, human_count, dtype=torch.bool)
    for index, clip in enumerate(clips):
        count = clip["human_xy"].shape[1]
        human_xy[index, :, :count] = clip["human_xy"]
        human_mask[index, :, :count] = torch.isfinite(clip["human_xy"]).all(dim=-1)

    depths = [clip["depth"] for clip in clips]
    if any(value is None for value in depths) and not all(value is None for value in depths):
        raise ValueError("All raw sources must consistently include or omit depth")
    return {
        "obs": torch.stack([clip["obs"] for clip in clips]),
        "action": torch.stack([clip["action"] for clip in clips]),
        "depth": None if depths[0] is None else torch.stack(depths),
        "robot_xy": torch.stack([clip["robot_xy"] for clip in clips]),
        "yaw": torch.stack([clip["yaw"] for clip in clips]),
        "goal_xy": torch.stack([clip["goal_xy"] for clip in clips]),
        "human_xy": human_xy,
        "human_mask": human_mask,
        "teacher_distance": torch.stack(
            [clip["teacher_distance"] for clip in clips]
        ),
        "visible_count": torch.stack([clip["visible_count"] for clip in clips]),
        "teacher_score": torch.tensor(
            [clip["teacher_score"] for clip in clips], dtype=torch.float32
        ),
        "interaction_fraction": torch.tensor(
            [clip["interaction_fraction"] for clip in clips], dtype=torch.float32
        ),
        "collision_fraction": torch.tensor(
            [clip["collision_fraction"] for clip in clips], dtype=torch.float32
        ),
        "distance_change_m": torch.tensor(
            [clip["distance_change_m"] for clip in clips], dtype=torch.float32
        ),
        "mean_visible_distance_m": torch.tensor(
            [clip["mean_visible_distance_m"] for clip in clips], dtype=torch.float32
        ),
        "lower_visible_distance_m": torch.tensor(
            [clip["lower_visible_distance_m"] for clip in clips], dtype=torch.float32
        ),
        "goal_distance_m": torch.tensor(
            [clip["goal_distance_m"] for clip in clips], dtype=torch.float32
        ),
        "source_id": torch.tensor(
            [clip["source_id"] for clip in clips], dtype=torch.int32
        ),
        "source_episode_id": torch.tensor(
            [clip["source_episode_id"] for clip in clips], dtype=torch.int32
        ),
        "episode_uid": torch.tensor(
            [clip["episode_uid"] for clip in clips], dtype=torch.int64
        ),
        "start": torch.tensor([clip["start"] for clip in clips], dtype=torch.int32),
    }


def build_preferences(
    sources: list[Path],
    *,
    teacher_config: ViewClearanceTeacherConfig,
    segment_steps: int,
    frame_stride: int,
    window_stride: int,
    max_clips: int,
    max_pairs: int,
    depth_size: int,
    val_ratio: float,
    score_margin: float,
    seed: int,
    camera_mount_override: tuple[float, float, float] | None = None,
    horizontal_fov_override: float | None = None,
) -> dict[str, Any]:
    if min(segment_steps, frame_stride, window_stride, max_clips, max_pairs, depth_size) <= 0:
        raise ValueError("segment, stride, count and depth-size values must be positive")
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    clips: list[dict] = []
    eligible_seen = 0
    episode_uid = 0
    source_metadata = []

    for source_id, source in enumerate(sources):
        episodes, raw_meta = _load_source(source)
        raw_mount = raw_meta.get("camera_mount_pos")
        mount_value = (
            camera_mount_override
            if camera_mount_override is not None else
            tuple(float(value) for value in raw_mount)
            if raw_mount is not None else teacher_config.camera_mount_pos
        )
        source_teacher_config = ViewClearanceTeacherConfig(
            **{
                **teacher_config.to_dict(),
                "camera_mount_pos": mount_value,
                "horizontal_fov_deg": (
                    horizontal_fov_override
                    if horizontal_fov_override is not None else
                    float(raw_meta.get(
                        "camera_horizontal_fov_deg",
                        teacher_config.horizontal_fov_deg,
                    ))
                ),
            }
        )
        teacher = ViewClearanceTeacher(source_teacher_config)
        source_metadata.append({
            "path": str(source), "meta": raw_meta,
            "teacher_camera_mount_pos": list(mount_value),
            "teacher_config": source_teacher_config.to_dict(),
        })
        print(f"[OfflinePref] scanning {source} episodes={len(episodes):,}")

        for source_episode_id, episode in enumerate(episodes):
            world = episode.get("world")
            missing = {
                "obs", "action", "world"
            } - episode.keys()
            if missing or world is None:
                raise ValueError(f"Raw episode lacks fields: {sorted(missing)}")
            world_missing = {"robot_xy", "yaw", "goal_xy", "hum_xys"} - world.keys()
            if world_missing:
                raise ValueError(
                    f"Raw episode world data lacks: {sorted(world_missing)}"
                )
            robot_xy = _numpy(world["robot_xy"], np.float32)
            yaw = _numpy(world["yaw"], np.float32)
            human_xy = _numpy(world["hum_xys"], np.float32)
            length = len(robot_xy)
            last_start = length - 1 - (segment_steps - 1) * frame_stride
            for start in range(0, max(0, last_start + 1), window_stride):
                indices = start + np.arange(segment_steps) * frame_stride
                result = teacher.evaluate(
                    robot_xy[indices], yaw[indices], human_xy[indices]
                )
                if not result.is_interaction:
                    continue
                eligible_seen += 1
                if len(clips) < max_clips:
                    slot = len(clips)
                else:
                    slot = int(rng.integers(0, eligible_seen))
                    if slot >= max_clips:
                        continue
                clip = _extract_clip(
                    episode, indices, source_id=source_id,
                    source_episode_id=source_episode_id,
                    episode_uid=episode_uid, start=start,
                    depth_size=depth_size, teacher_result=result,
                )
                if slot == len(clips):
                    clips.append(clip)
                else:
                    clips[slot] = clip
            episode_uid += 1

    if len(clips) < 4:
        raise ValueError(
            f"Only {len(clips)} interaction clips found; relax teacher thresholds"
        )
    split_map = _episode_splits(clips, val_ratio, rng)
    clip_split = np.asarray(
        [split_map[int(clip["episode_uid"])] for clip in clips], dtype=np.int8
    )
    pair, label, margin, pair_split = _make_pairs(
        clips, clip_split, max_pairs=max_pairs, score_margin=score_margin,
        val_ratio=val_ratio, rng=rng,
    )
    packed = _stack_clips(clips)
    packed["split"] = torch.from_numpy(clip_split)
    output = {
        "metadata": {
            "schema": SCHEMA,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sources": source_metadata,
            "teacher": teacher_config.to_dict(),
            "segment_steps": segment_steps,
            "frame_stride": frame_stride,
            "window_stride": window_stride,
            "depth_size": depth_size,
            "score_margin": score_margin,
            "seed": seed,
            "eligible_clips_seen": eligible_seen,
            "stored_clips": len(clips),
            "pairs": len(pair),
            "pair_storage": "clip_indices",
            "split_unit": "source_episode",
        },
        "clips": packed,
        "pairs": torch.from_numpy(pair),
        "labels": torch.from_numpy(label),
        "pair_margin": torch.from_numpy(margin),
        "pair_split": torch.from_numpy(pair_split),
    }
    train_pairs = int((pair_split == 0).sum())
    val_pairs = int((pair_split == 1).sum())
    print(
        f"[OfflinePref] eligible={eligible_seen:,} stored={len(clips):,} "
        f"pairs={len(pair):,} train={train_pairs:,} val={val_pairs:,}"
    )
    return output


def main() -> None:
    args = parse_args()
    sources = _resolve_sources(args.raw_buffer)
    mount_override = (
        tuple(float(value) for value in args.camera_mount_pos)
        if args.camera_mount_pos is not None else None
    )
    mount = mount_override or (0.25, 0.0, 0.6)
    config = ViewClearanceTeacherConfig(
        horizontal_fov_deg=args.horizontal_fov_deg or 90.0,
        max_view_distance_m=args.max_view_distance,
        interaction_distance_m=args.interaction_distance,
        safe_distance_m=args.safe_distance,
        collision_distance_m=args.collision_distance,
        min_interaction_fraction=args.min_interaction_fraction,
        min_distance_change_m=args.min_distance_change,
        camera_mount_pos=mount,
    )
    dataset = build_preferences(
        sources, teacher_config=config,
        segment_steps=args.segment_steps, frame_stride=args.frame_stride,
        window_stride=args.window_stride, max_clips=args.max_clips,
        max_pairs=args.max_pairs, depth_size=args.depth_size,
        val_ratio=args.val_ratio, score_margin=args.pair_score_margin,
        seed=args.seed, camera_mount_override=mount_override,
        horizontal_fov_override=args.horizontal_fov_deg,
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else
        PROJECT_ROOT / "output" / "pref_data" /
        f"offline_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output)
    print(f"[OfflinePref] saved {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
