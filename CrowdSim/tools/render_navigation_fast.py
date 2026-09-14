"""Fast CrowdSim navigation renderer — merged video/static visualization.

Video mode (default): renders trajectory JSONL as MP4/GIF with PIL/OpenCV.
Static mode (--static): generates episode-grid PNG for a single car.
Both modes auto-detect episode boundaries so reset jumps are never drawn.

Usage:
  # Video — last 500 frames of car navigation
  python tools/render_navigation_fast.py --output nav.mp4

  # Video — all agents (humanoids + cars) with SFM arrows (default)
  python tools/render_navigation_fast.py --sfm-arrows

  # Static — single-car episode grid (replaces plot_car_trajectories.py)
  python tools/render_navigation_fast.py --static --cars 2 --output car2.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BLACK = (0, 0, 0, 230)
WHITE = (255, 255, 255, 235)
LOCAL_TARGET_OUTLINE = (255, 255, 255, 220)
WAYPOINT_OUTLINE = (20, 20, 20, 220)
VEL_ARROW_COLOR = (0, 135, 255, 230)
DESIRED_ARROW_COLOR = (40, 230, 80, 235)
INTERACT_ARROW_COLOR = (255, 50, 210, 225)
REPULSIVE_ARROW_COLOR = (255, 70, 45, 225)
D_VEL_ARROW_COLOR = (0, 0, 0, 230)
TTC_ARROW_COLOR = (255, 145, 0, 235)
FUTURE_TARGET_COLOR = (255, 215, 0, 235)
YAW_SOURCE_LABELS = {
    "sfm_target": "SFM", "waypoint_fallback": "WP", "previous": "PREV",
    "reached": "DONE", "reset": "RESET", "initial": "INIT",
}
YAW_SOURCE_COLORS = {
    "sfm_target": (40, 230, 80, 230), "waypoint_fallback": (255, 215, 0, 240),
    "previous": (255, 120, 0, 240), "reached": (120, 120, 120, 230),
    "reset": (180, 180, 255, 230), "initial": (180, 180, 255, 230),
}


# ---------------------------------------------------------------------------
# Map canvas
# ---------------------------------------------------------------------------
@dataclass
class MapCanvas:
    static_image: Image.Image
    full_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    resolution: float
    origin_xy: tuple[float, float]

    @property
    def size(self) -> tuple[int, int]:
        return self.static_image.size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast PIL/OpenCV renderer + static episode-grid for CrowdSim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "trajectory_log", nargs="?", default="output/crowdsim_navigation/trajectory_latest.jsonl",
        help="Trajectory JSONL from CrowdNavigationManager.",
    )
    p.add_argument("--path-log", default=None, help="Path JSON. Default: derived from trajectory log.")
    p.add_argument(
        "--output", default="output/crowdsim_navigation/trajectory_latest.mp4",
        help="Output path (.mp4 / .gif / .png). PNG with --static gives grid mode.",
    )
    p.add_argument("--fps", type=float, default=10.0)

    # Frame selection
    p.add_argument("--stride", type=int, default=3, help="Use every Nth frame.")
    p.add_argument("--max-frames", type=int, default=0, help="Render first N frames (0=all).")
    p.add_argument("--tail-frames", type=int, default=500,
                   help="Render last N frames. 0 to disable. Overrides --max-frames.")

    # Rendering
    p.add_argument("--trail-length", type=int, default=200, help="Trail in frames (0=no trail).")
    p.add_argument("--crop-center-pixels", type=int, default=800, help="Center crop. 0=full map.")
    p.add_argument("--scale", type=float, default=1.5, help="Output scale factor.")
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--agent-size", type=int, default=6)
    p.add_argument("--target-size", type=int, default=6)

    # Agent visibility — all agents are always rendered (humanoids + cars).
    # Use --cars-only to restrict to cars (e.g. for single-agent papers).
    p.add_argument("--cars-only", action="store_true", help="Render cars only (default: all agents).")
    p.add_argument("--no-sfm-arrows", action="store_true", help="Hide SFM vector arrows.")
    p.add_argument("--no-initial-paths", action="store_true", help="Hide planned A* paths.")
    p.add_argument("--show-yaw-source-labels", action="store_true")
    p.add_argument("--no-text", action="store_true")

    # Static mode (replaces plot_car_trajectories.py)
    p.add_argument("--static", action="store_true",
                   help="Static PNG grid mode. Use --cars to pick a car, --cols for layout.")
    p.add_argument("--cars", type=str, default=None,
                   help="Car ID (e.g. '2') for static grid mode.")
    p.add_argument("--cols", type=int, default=4, help="Grid columns in static mode.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_trajectory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped[0] != "{":
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "metadata":
                metadata = obj
            elif obj.get("type") == "frame":
                frames.append(obj)
    return metadata, frames


def load_trajectory_tail(
    path: Path, n: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Like load_trajectory() but reads only the tail of the JSONL file.

    Seeks to the last ``n * BYTES_PER_FRAME_EST`` bytes so that long training
    runs don't pay a linear IO cost on every GIF render.  The metadata record
    is recovered from the file header in a separate small read.
    """
    BYTES_PER_FRAME_EST = 2000          # conservative estimate per frame line
    read_bytes = max((n + 10) * BYTES_PER_FRAME_EST, 65536)

    metadata: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []

    file_size = path.stat().st_size
    with path.open("rb") as f:
        # Read header (up to 4 KB) to recover the metadata record.
        header_raw = f.read(min(4096, file_size))
        for line in header_raw.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped[0] != "{":
                continue
            try:
                obj = json.loads(stripped)
                if obj.get("type") == "metadata":
                    metadata = obj
                    break
            except json.JSONDecodeError:
                continue

        # Read tail to recover the most recent frames.
        seek_pos = max(0, file_size - read_bytes)
        f.seek(seek_pos)
        tail_raw = f.read()

    tail_lines = tail_raw.decode("utf-8", errors="replace").splitlines()
    # When we seeked into the middle of the file the first line is likely
    # truncated — skip it to avoid JSON parse errors.
    if seek_pos > 0 and tail_lines:
        tail_lines = tail_lines[1:]

    for line in tail_lines:
        stripped = line.strip()
        if not stripped or stripped[0] != "{":
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "frame":
            frames.append(obj)

    return metadata, frames[-n:]


def load_path_data(metadata: dict, trajectory_path: Path, path_log: str | None) -> dict:
    if path_log:
        p = Path(path_log).expanduser()
        if not p.is_absolute():
            # CLI paths are normally relative to the current project root.
            # Only fall back to trajectory-relative resolution for a basename
            # or another path that does not exist from the current directory.
            cwd_candidate = (Path.cwd() / p).resolve()
            p = cwd_candidate if cwd_candidate.is_file() else trajectory_path.parent / p
    else:
        p = trajectory_path.with_name("paths_latest.json")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Episode boundary detection
# ---------------------------------------------------------------------------
def _find_reset_boundaries_frame(
    reached_arr: np.ndarray,      # (T,) bool per agent
    positions: np.ndarray,        # (T, 2) per agent
    jump_threshold: float = 3.0,
) -> list[int]:
    """Frame indices where a new episode starts (reset happened).

    Vectorised for speed: uses numpy diff instead of Python loops.
    """
    T = len(positions)
    if T == 0:
        return [0]
    breaks: set[int] = {0, T}
    # Reached → not-reached transitions
    if len(reached_arr) == T and T > 1:
        ra = np.asarray(reached_arr, dtype=bool)
        idxs = np.where(ra[:-1] & ~ra[1:])[0] + 1
        breaks.update(idxs.tolist())
    # Large position jumps
    if T > 1:
        diffs = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        idxs = np.where(diffs > jump_threshold)[0] + 1
        breaks.update(idxs.tolist())
    return sorted(breaks)


def _trail_segments_fast(
    positions: np.ndarray,        # (T, 2) for one agent — FULL array
    precomputed_breaks: list[int],
    trail_start: int,
    frame_idx: int,
) -> list[np.ndarray]:
    """Return the current-episode trail using pre-computed episode boundaries.

    Unlike the original _trail_segments(), this does not re-compute breaks on
    every call — callers must precompute them once via _find_reset_boundaries_frame.
    """
    breaks = precomputed_breaks
    for bi in range(len(breaks) - 1):
        b0, b1 = breaks[bi], breaks[bi + 1]
        if b0 <= frame_idx < b1:
            seg_start = max(trail_start, b0)
            seg_end   = frame_idx + 1
            if seg_end > seg_start:
                return [positions[seg_start:seg_end]]
            break
    return []


# Keep the original for backward-compat (used by _trail_segments callers if any).
def _trail_segments(
    positions: np.ndarray,
    reached: np.ndarray | None,
    trail_start: int,
    frame_idx: int,
) -> list[np.ndarray]:
    if reached is not None and len(reached) == len(positions):
        breaks = _find_reset_boundaries_frame(reached, positions)
    else:
        breaks = [0, len(positions)]
    return _trail_segments_fast(positions, breaks, trail_start, frame_idx)


# ---------------------------------------------------------------------------
# Map utilities
# ---------------------------------------------------------------------------
def make_canvas(path_data: dict, metadata: dict, crop_center_pixels: int) -> MapCanvas:
    map_path = Path(str(path_data.get("map_path") or metadata["map_path"]))
    if not map_path.is_absolute():
        map_path = Path.cwd() / map_path
    image = Image.open(map_path).convert("L")
    image = ImageOps.autocontrast(image).convert("RGB")
    w, h = image.size
    crop_box = _centered_crop_box(w, h, crop_center_pixels)
    cropped = image.crop(crop_box)
    resolution = float(path_data.get("map_resolution", metadata.get("map_resolution", 0.05)))
    origin_val = path_data.get("map_origin_xy", metadata.get("map_origin_xy"))
    if origin_val is None:
        origin_val = (-0.5 * (w - 1) * resolution, -0.5 * (h - 1) * resolution)
    return MapCanvas(
        static_image=cropped.convert("RGB"), full_size=(w, h),
        crop_box=crop_box, resolution=resolution,
        origin_xy=(float(origin_val[0]), float(origin_val[1])),
    )


def world_to_crop_pixel(xy, full_w, full_h, resolution, origin_xy, crop_box):
    ox, oy = origin_xy
    cl, ct, _, _ = crop_box
    px = int(round((float(xy[0]) - ox) / resolution))
    py = int(round((full_h - 1) - (float(xy[1]) - oy) / resolution))
    return px - cl, py - ct


def _centered_crop_box(w, h, crop_size):
    if crop_size <= 0:
        return 0, 0, w, h
    cw, ch = min(w, crop_size), min(h, crop_size)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return left, top, left + cw, top + ch


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _agent_color(agent_id: int) -> tuple[int, int, int]:
    # 20 maximally distinct colors
    palette = [
        (230, 40, 40), (40, 160, 230), (40, 210, 80), (230, 180, 30),
        (170, 50, 210), (230, 80, 170), (30, 200, 190), (210, 130, 40),
        (70, 130, 230), (190, 210, 30), (210, 60, 130), (50, 210, 150),
        (230, 100, 70), (80, 180, 230), (160, 210, 50), (230, 60, 100),
        (60, 160, 210), (220, 140, 220), (100, 210, 120), (220, 200, 50),
    ]
    return palette[agent_id % len(palette)]


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def _draw_circle(draw, center, radius, fill, outline):
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)

def _draw_square(draw, center, radius, fill, outline):
    x, y = center
    draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)

def _draw_diamond(draw, center, radius, fill, outline):
    x, y = center
    pts = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
    draw.polygon(pts, fill=fill, outline=outline)

def _draw_goal_marker(draw, center, radius, fill):
    x, y = center
    pts = [(x + int(math.cos(math.pi * 0.5 * i) * radius),
            y + int(math.sin(math.pi * 0.5 * i) * radius)) for i in range(4)]
    draw.polygon(pts, fill=fill, outline=BLACK)

def _draw_x(draw, center, radius, fill, outline):
    x, y = center
    for dx, dy in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
        draw.line((x - radius * dx, y - radius * dy, x + radius * dx, y + radius * dy),
                  fill=outline if dx < 0 else fill, width=4 if dx < 0 else 2)

def _draw_cross(draw, center, radius, fill, outline):
    x, y = center
    draw.line((x - radius, y, x + radius, y), fill=outline, width=4)
    draw.line((x, y - radius, x, y + radius), fill=outline, width=4)
    draw.line((x - radius, y, x + radius, y), fill=fill, width=2)
    draw.line((x, y - radius, x, y + radius), fill=fill, width=2)


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------
def render_video(
    canvas: MapCanvas, metadata: dict, path_data: dict,
    frames: list[dict], output_path: Path,
    fps: float, trail_length: int, scale: float,
    agent_size: int, target_size: int, line_width: int,
    draw_sfm_arrows: bool, draw_planned_paths: bool,
    show_yaw_source_labels: bool, show_text: bool,
    initial_path_records: list[dict] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_h = int(path_data.get("num_humanoids", metadata.get("num_humanoids", 0)))
    num_c = int(path_data.get("num_cars", metadata.get("num_cars", 0)))
    num_agents = num_h + num_c
    full_w, full_h = canvas.full_size
    font = _load_font()

    # Pre-load all positions and reached states (full arrays — no per-frame slice)
    all_positions = np.array([f["positions_xy"] for f in frames], dtype=np.float32)
    all_reached = np.array([f.get("reached", []) for f in frames])
    if all_reached.ndim < 2 or all_reached.shape[1] != num_agents:
        all_reached = np.zeros((len(frames), num_agents), dtype=bool)

    agent_range = range(num_agents)

    if initial_path_records is not None:
        path_records = [dict(r) for r in initial_path_records]
    else:
        path_records = _current_path_records(path_data, num_agents)

    # ── P0-fix: precompute episode breaks once for all agents ──────
    all_breaks: list[list[int]] = [
        _find_reset_boundaries_frame(all_reached[:, a], all_positions[:, a])
        for a in range(num_agents)
    ]

    # ── P1-fix: cache planned-path pixel coordinates ───────────────
    def _path_to_px(record: dict) -> list[tuple[int, int]]:
        pts = record.get("path_xy", [])
        if len(pts) < 2:
            return []
        return [world_to_crop_pixel(xy, full_w, full_h,
                                    canvas.resolution, canvas.origin_xy, canvas.crop_box)
                for xy in pts]

    def _goal_to_px(record: dict) -> tuple[int, int]:
        return world_to_crop_pixel(record.get("goal_xy", [0.0, 0.0]),
                                   full_w, full_h, canvas.resolution,
                                   canvas.origin_xy, canvas.crop_box)

    path_px_cache: list[list]         = [_path_to_px(r) for r in path_records]
    goal_px_cache: list[tuple[int,int]] = [_goal_to_px(r) for r in path_records]

    def _refresh_path_cache(agent_id: int) -> None:
        path_px_cache[agent_id] = _path_to_px(path_records[agent_id])
        goal_px_cache[agent_id] = _goal_to_px(path_records[agent_id])

    writer = _make_video_writer(output_path, canvas.size, fps, scale)
    base_rgba = canvas.static_image.convert("RGBA")   # reuse base

    try:
        for frame_idx, frame in enumerate(frames):
            updates = frame.get("path_updates", [])
            if updates:
                _apply_path_updates(path_records, updates)
                for upd in updates:
                    aid = int(upd.get("agent_id", -1))
                    if 0 <= aid < len(path_px_cache):
                        _refresh_path_cache(aid)

            image   = base_rgba.copy()
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw    = ImageDraw.Draw(overlay, "RGBA")

            frame_positions = np.asarray(frame["positions_xy"], dtype=np.float32)

            trail_start = max(0, frame_idx - trail_length)

            for agent_id in agent_range:
                color = _agent_color(agent_id)
                is_humanoid = agent_id < num_h
                pos_px = world_to_crop_pixel(
                    frame_positions[agent_id], full_w, full_h,
                    canvas.resolution, canvas.origin_xy, canvas.crop_box,
                )

                # Humanoids: only current position as circle
                if is_humanoid:
                    _draw_circle(draw, pos_px, agent_size, (*color, 245), BLACK)
                    continue

                # --- Cars: full rendering ---
                # Planned path — from cache (updated only on path_updates)
                if draw_planned_paths:
                    cached_path_px = path_px_cache[agent_id]
                    if len(cached_path_px) >= 2:
                        draw.line(cached_path_px, fill=(*color, 135),
                                  width=line_width, joint="curve")

                # Goal marker — from cache
                _draw_goal_marker(draw, goal_px_cache[agent_id],
                                  target_size + 2, (*color, 235))

                # Trail — P0-fix: use precomputed breaks, pass FULL position array
                segments = _trail_segments_fast(
                    all_positions[:, agent_id],   # full array, no slice
                    all_breaks[agent_id],
                    trail_start, frame_idx,
                )
                for seg in segments:
                    if len(seg) < 2:
                        continue
                    seg_px = [world_to_crop_pixel(
                        xy, full_w, full_h, canvas.resolution, canvas.origin_xy, canvas.crop_box,
                    ) for xy in seg]
                    draw.line(seg_px, fill=(*color, 215), width=line_width, joint="curve")

                # Agent shape (square for cars)
                _draw_square(draw, pos_px, agent_size, (*color, 245), BLACK)

            # SFM arrows (humanoids only)
            if draw_sfm_arrows:
                _draw_sfm_arrows_for_frame(draw, frame, canvas, num_h)

            # Status text
            if show_text:
                yaw_sources = (
                    _humanoid_yaw_sources(frame, num_h)
                )
                _draw_status_text(draw, font, frame_idx, len(frames), frame,
                                  draw_sfm_arrows, yaw_sources)

            rendered = Image.alpha_composite(image, overlay).convert("RGB")
            writer.write(rendered)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Static PNG grid mode (replaces plot_car_trajectories.py)
# ---------------------------------------------------------------------------
def render_static_grid(
    frames: list[dict], metadata: dict, car_id: int,
    map_path_str: str, origin: tuple, resolution: float,
    stride: int, cols: int, output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection  # P2-fix: vectorised segments

    num_h = int(metadata.get("num_humanoids", 0))
    num_frames = len(frames)
    positions = np.array([f["positions_xy"] for f in frames])
    goals = np.array([f.get("goals_xy", np.zeros((num_h + 1, 2))) for f in frames])
    reached_arr = np.array([f.get("reached", []) for f in frames])

    agent_id = num_h + car_id
    pos = positions[:, agent_id, :]
    goal = goals[:, agent_id, :]
    reached = reached_arr[:, agent_id] if reached_arr.shape[1] > agent_id else None

    breaks = _find_reset_boundaries_frame(
        reached if reached is not None else np.zeros(len(pos), dtype=bool), pos,
    )
    episodes = []
    for ep in range(len(breaks) - 1):
        t0, t1 = breaks[ep], breaks[ep + 1]
        if t1 - t0 < 2:
            continue
        ep_pos = pos[t0:t1]
        ep_goal = goal[t0]
        ep_reached = float(np.linalg.norm(ep_pos[-1] - ep_goal)) <= 0.75
        episodes.append({"pos": ep_pos, "goal": ep_goal, "reached": ep_reached, "idx": ep})

    n_eps = len(episodes)
    n_reached = sum(1 for e in episodes if e["reached"])
    n_cols = min(cols, n_eps)
    n_rows = max(1, int(np.ceil(n_eps / n_cols)))
    print(f"Car {car_id}: {n_eps} episodes ({n_reached} reached), {num_frames} frames.")

    # Load map background
    try:
        from PIL import Image as PILImage
        map_img = np.array(PILImage.open(map_path_str))
    except Exception:
        map_img = None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
    if n_eps == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, ep in enumerate(episodes):
        ax = axes_flat[i]
        color = "#27ae60" if ep["reached"] else "#e74c3c"

        # Map background
        if map_img is not None:
            h_px, w_px = map_img.shape[:2]
            ax.imshow(map_img, extent=[
                origin[0], origin[0] + w_px * resolution,
                origin[1], origin[1] + h_px * resolution,
            ], origin="upper", cmap="gray", alpha=0.5)

        # Auto-zoom
        all_pts = np.concatenate([
            ep["pos"][::max(1, len(ep["pos"]) // 200)],
            ep["goal"].reshape(1, 2),
        ])
        pad = 1.5
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

        # P2-fix: draw trajectory with LineCollection (single draw call)
        sampled = ep["pos"][::stride]
        if len(sampled) >= 2:
            n = len(sampled) - 1
            alphas = [min(0.4 + 0.5 * (j / max(n, 1)), 0.95) for j in range(n)]
            segs   = [sampled[j:j + 2] for j in range(n)]
            # Parse hex color to (r,g,b) in [0,1]
            r_f = int(color[1:3], 16) / 255.0
            g_f = int(color[3:5], 16) / 255.0
            b_f = int(color[5:7], 16) / 255.0
            rgba_colors = [(r_f, g_f, b_f, a) for a in alphas]
            lc = LineCollection(segs, colors=rgba_colors, linewidths=1.6)
            ax.add_collection(lc)

        # Markers
        ax.scatter(*ep["pos"][0], marker="o", s=50, color=color, edgecolors="black", linewidth=0.6, zorder=5)
        ax.scatter(*ep["pos"][-1], marker="D", s=40, color=color, edgecolors="black", linewidth=0.5, zorder=5)
        ax.scatter(*ep["goal"], marker="*", s=120, color="gold", edgecolors="black", linewidth=0.5, zorder=6)
        # Reference line
        ax.plot([ep["pos"][0, 0], ep["goal"][0]], [ep["pos"][0, 1], ep["goal"][1]],
                color="white", alpha=0.3, lw=0.6, ls="--")

        outcome = "✓" if ep["reached"] else "✗"
        ax.set_title(f"Ep{ep['idx']} {outcome}  ({len(ep['pos'])} steps)", fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        ax.set_xlabel(""); ax.set_ylabel("")

    for j in range(n_eps, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Car {car_id}  —  {n_eps} episodes  "
        f"(✓ reached={n_reached}, ✗ incomplete={n_eps - n_reached})  |  "
        "○ start  ◆ end  ★ goal",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    traj_path = Path(args.trajectory_log).expanduser()
    if not traj_path.is_absolute():
        traj_path = Path.cwd() / traj_path

    metadata, frames = load_trajectory(traj_path)
    if not frames:
        raise RuntimeError(f"No trajectory frames found: {traj_path}")

    stride = max(1, int(args.stride))
    path_data = load_path_data(metadata, traj_path, args.path_log)
    num_agents = int(path_data.get("num_agents", len(frames[0]["positions_xy"])))

    # ── Merge path_updates from stride-skipped frames ──────────────────
    # Without this, stride drops frames whose path_updates would be lost,
    # causing the rendered path/goal to lag behind the actual reset.
    if stride > 1:
        for group_start in range(0, len(frames), stride):
            group_end = min(group_start + stride, len(frames))
            collected: list[dict] = []
            for i in range(group_start, group_end):
                collected.extend(frames[i].get("path_updates", []))
            if collected:
                frames[group_start]["path_updates"] = collected

    frames = frames[::stride]

    # ── Pre-process path_updates for tail/max-frames ───────────────────
    # Apply updates ONLY from frames BEFORE the render window so the
    # initial state is correct; updates WITHIN the window are handled
    # incrementally by render_video.
    pre_built_path_records: list[dict] | None = None
    if args.tail_frames > 0:
        tail_start = max(0, len(frames) - args.tail_frames)
        if tail_start > 0:
            pre_built_path_records = _current_path_records(path_data, num_agents)
            for f in frames[:tail_start]:
                _apply_path_updates(pre_built_path_records, f.get("path_updates", []))
        frames = frames[tail_start:]
    elif args.max_frames > 0:
        frames = frames[:args.max_frames]

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    # Static PNG grid mode
    if args.static:
        if args.cars is None:
            raise SystemExit("--static requires --cars <ID> (e.g. --cars 2)")
        car_id = int(args.cars)
        origin = tuple(path_data.get("map_origin_xy", metadata.get("map_origin_xy", [-12.5, -12.5])))
        resolution = float(path_data.get("map_resolution", metadata.get("map_resolution", 0.05)))
        map_path = str(path_data.get("map_path", metadata.get("map_path", "")))
        if not Path(map_path).is_absolute():
            map_path = str(traj_path.parent.parent.parent / map_path)
        render_static_grid(
            frames, metadata, car_id, map_path,
            (float(origin[0]), float(origin[1])), resolution,
            stride, int(args.cols), output_path,
        )
        return

    # Video mode
    canvas = make_canvas(path_data, metadata, max(0, int(args.crop_center_pixels)))
    render_video(
        canvas=canvas, metadata=metadata, path_data=path_data,
        frames=frames, output_path=output_path,
        fps=float(args.fps), trail_length=max(0, int(args.trail_length)),
        scale=max(0.1, float(args.scale)),
        agent_size=max(2, int(args.agent_size)),
        target_size=max(2, int(args.target_size)),
        line_width=max(1, int(args.line_width)),
        draw_sfm_arrows=not args.no_sfm_arrows,
        draw_planned_paths=not args.no_initial_paths,
        show_yaw_source_labels=bool(args.show_yaw_source_labels),
        show_text=not bool(args.no_text),
        initial_path_records=pre_built_path_records,
    )
    print(f"[CrowdSim] Saved navigation render: {output_path}")


# ===================================================================
# Everything below is the original render_navigation_fast helpers,
# kept intact for the video rendering path.
# ===================================================================

def _current_path_records(path_data: dict, num_agents: int) -> list[dict]:
    records = [dict(agent) for agent in path_data.get("agents", [])]
    while len(records) < num_agents:
        records.append({"agent_id": len(records), "goal_xy": [0.0, 0.0], "path_xy": []})
    return records[:num_agents]


def _apply_path_updates(path_records: list[dict], updates: Any) -> None:
    if not updates:
        return
    for update in updates:
        if not isinstance(update, dict):
            continue
        agent_id = int(update.get("agent_id", -1))
        if 0 <= agent_id < len(path_records):
            path_records[agent_id] = dict(update)


class _OpenCvVideoWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: float, scale: float):
        import cv2
        self.cv2 = cv2
        self.scale = scale
        w, h = size
        self.output_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, float(fps), self.output_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")

    def write(self, image: Image.Image) -> None:
        if image.size != self.output_size:
            image = image.resize(self.output_size, Image.Resampling.BILINEAR)
        frame_rgb = np.asarray(image, dtype=np.uint8)
        self.writer.write(self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


class _GifWriter:
    """PIL-based GIF writer with an optional frame-count cap to bound memory use.

    P1-fix: *max_frames* discards old frames when exceeded (keep the tail),
    preventing multi-GB RAM usage on very long renders.
    """

    def __init__(self, path: Path, size: tuple[int, int], fps: float, scale: float,
                 max_frames: int = 600) -> None:
        self.path         = path
        self.scale        = scale
        self.max_frames   = max(1, max_frames)
        w, h              = size
        self.output_size  = (max(1, int(round(w * scale))),
                             max(1, int(round(h * scale))))
        self.duration_ms  = int(round(1000.0 / max(float(fps), 1e-5)))
        self.frames: list[Image.Image] = []

    def write(self, image: Image.Image) -> None:
        if image.size != self.output_size:
            image = image.resize(self.output_size, Image.Resampling.BILINEAR)
        self.frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
        # Drop oldest frame when over the cap so memory stays bounded.
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def close(self) -> None:
        if not self.frames:
            return
        self.frames[0].save(
            self.path, save_all=True, append_images=self.frames[1:],
            duration=self.duration_ms, loop=0, optimize=False,
        )


def _make_video_writer(path: Path, size, fps, scale):
    if path.suffix.lower() == ".gif":
        return _GifWriter(path, size, fps, scale)
    return _OpenCvVideoWriter(path, size, fps, scale)


def _draw_status_text(draw, font, frame_idx, num_frames, frame, show_legend, yaw_sources):
    lines = [
        f"frame {frame_idx + 1}/{num_frames}  "
        f"nav_step {int(frame.get('step', 0))}  "
        f"t {float(frame.get('time', 0.0)):.2f}s"
    ]
    if yaw_sources:
        lines.append(f"yaw: {_format_yaw_source_counts(yaw_sources)}")
    if show_legend:
        lines.append("arrows: blue=vel green=sfm red=wall magenta=agents black=goal orange=TTC")
        lines.append("yellow=first MaskedMimic future target")
    x, y = 12, 10
    boxes = [draw.textbbox((x, y + i * 20), line, font=font) for i, line in enumerate(lines)]
    box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
           max(b[2] for b in boxes), max(b[3] for b in boxes))
    pad = 5
    draw.rounded_rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
                           radius=4, fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text((x, y + i * 20), line, fill=WHITE, font=font)


def _humanoid_yaw_sources(frame: dict, num_humanoids: int) -> list[str]:
    if "humanoid_yaw_source" not in frame:
        return []
    sources = [str(v) for v in frame["humanoid_yaw_source"]]
    return sources[:num_humanoids]


def _format_yaw_source_counts(sources: list[str]) -> str:
    ordered = ["sfm_target", "waypoint_fallback", "previous", "reached", "reset", "initial"]
    counts = {s: sources.count(s) for s in set(sources)}
    parts = []
    for s in ordered:
        c = counts.pop(s, 0)
        if c:
            parts.append(f"{YAW_SOURCE_LABELS.get(s, s)}={c}")
    for s, c in sorted(counts.items()):
        parts.append(f"{s}={c}")
    return " ".join(parts) if parts else "none"


def _draw_sfm_arrows_for_frame(draw, frame, canvas, num_h):
    """Draw SFM vector arrows for humanoids."""
    try:
        desired = np.asarray(frame.get("sfm_desired_velocities_xy", []), dtype=np.float32)
        interact = np.asarray(frame.get("sfm_interact_forces_xy", []), dtype=np.float32)
        repulse = np.asarray(frame.get("sfm_repulsive_forces_xy", []), dtype=np.float32)
        dvel = np.asarray(frame.get("sfm_d_vel_xy", []), dtype=np.float32)
        ttc = np.asarray(frame.get("sfm_ttc_forces_xy", []), dtype=np.float32)
        future_targets = np.asarray(
            frame.get("humanoid_future_first_targets_xy", []), dtype=np.float32
        )
        velocities = np.asarray(frame.get("velocities_xy", []), dtype=np.float32)
        positions = np.asarray(frame["positions_xy"], dtype=np.float32)
    except KeyError:
        return

    for agent_id in range(min(num_h, len(desired))):
        pos = positions[agent_id]
        start_px = world_to_crop_pixel(
            pos, canvas.full_size[0], canvas.full_size[1],
            canvas.resolution, canvas.origin_xy, canvas.crop_box,
        )
        for vec, color, scl in [
            (velocities[agent_id] if agent_id < len(velocities) else np.zeros(2), VEL_ARROW_COLOR, 1.0),
            (desired[agent_id], DESIRED_ARROW_COLOR, 1.0),
            (interact[agent_id] if agent_id < len(interact) else np.zeros(2), INTERACT_ARROW_COLOR, 0.2),
            (repulse[agent_id] if agent_id < len(repulse) else np.zeros(2), REPULSIVE_ARROW_COLOR, 0.2),
            (dvel[agent_id] if agent_id < len(dvel) else np.zeros(2), D_VEL_ARROW_COLOR, 0.2),
            (ttc[agent_id] if agent_id < len(ttc) else np.zeros(2), TTC_ARROW_COLOR, 0.2),
        ]:
            if not np.all(np.isfinite(vec)) or float(np.linalg.norm(vec)) < 1e-5:
                continue
            end = pos + vec * scl
            end_px = world_to_crop_pixel(end, canvas.full_size[0], canvas.full_size[1],
                                         canvas.resolution, canvas.origin_xy, canvas.crop_box)
            draw.line((start_px[0], start_px[1], end_px[0], end_px[1]), fill=color, width=2)
            # Arrowhead
            dx, dy = float(end_px[0] - start_px[0]), float(end_px[1] - start_px[1])
            length = math.hypot(dx, dy)
            if length > 1e-5:
                ux, uy = dx / length, dy / length
                px, py = -uy, ux
                size = 6
                p1 = (end_px[0], end_px[1])
                p2 = (int(end_px[0] - ux * size + px * size * 0.45), int(end_px[1] - uy * size + py * size * 0.45))
                p3 = (int(end_px[0] - ux * size - px * size * 0.45), int(end_px[1] - uy * size - py * size * 0.45))
                draw.polygon([p1, p2, p3], fill=color)
        if agent_id < len(future_targets):
            target_px = world_to_crop_pixel(
                future_targets[agent_id], canvas.full_size[0], canvas.full_size[1],
                canvas.resolution, canvas.origin_xy, canvas.crop_box,
            )
            draw.line((start_px[0], start_px[1], target_px[0], target_px[1]),
                      fill=FUTURE_TARGET_COLOR, width=1)
            _draw_diamond(draw, target_px, 3, FUTURE_TARGET_COLOR, BLACK)


# ---------------------------------------------------------------------------
# Public API: fast GIF bytes (for wandb upload from training loop)
# ---------------------------------------------------------------------------

def render_nav_gif_bytes(
    obstacle_map: np.ndarray,           # (H, W) uint8 — from nav_manager
    resolution: float,
    origin_xy: tuple[float, float],
    num_humanoids: int,
    num_robots: int,
    paths_xy: list[np.ndarray],         # planned A* paths per agent
    goals_xy: np.ndarray,               # (num_agents, 2)
    trajectory_log_path: "Path | str | None",
    n_frames: int = 80,
    fps: float = 8.0,
    agent_size: int = 5,
    trail_length: int = 60,
    scale: float = 1.0,
    crop_center_pixels: int = 0,
) -> bytes | None:
    """Render the last *n_frames* trajectory steps as GIF bytes using the fast
    PIL renderer.  Returns the raw GIF bytes, or *None* on failure.

    Designed to be called from the training loop so wandb gets a high-quality
    animated GIF that is ~400× faster to produce than the matplotlib version.
    """
    import io as _io

    # ── 1. Build a MapCanvas from the obstacle_map array ──────────
    H, W = obstacle_map.shape
    # Convert obstacle map to a PIL Image (white=free, black=obstacle)
    map_arr = ((1 - obstacle_map) * 255).astype(np.uint8)
    map_pil  = Image.fromarray(map_arr, mode="L").convert("RGB")
    crop_box = _centered_crop_box(W, H, crop_center_pixels)
    cropped  = map_pil.crop(crop_box)
    canvas   = MapCanvas(
        static_image=cropped,
        full_size=(W, H),
        crop_box=crop_box,
        resolution=resolution,
        origin_xy=origin_xy,
    )
    full_w, full_h = canvas.full_size

    # ── 2. Load trajectory frames (tail-only to avoid full-file IO) ───────────
    frames: list[dict] = []
    if trajectory_log_path is not None:
        try:
            _, frames = load_trajectory_tail(Path(str(trajectory_log_path)), n_frames)
        except Exception:
            pass
    frames = frames[-n_frames:] if len(frames) >= 1 else []

    num_agents = num_humanoids + num_robots

    # ── 3. Fallback state (current nav_manager values, used when frames==[]) ──
    _fallback_goals: list[list] = [
        goals_xy[i].tolist() if i < len(goals_xy) else [0.0, 0.0]
        for i in range(num_agents)
    ]
    _fallback_paths: list[list] = [
        paths_xy[i].tolist() if i < len(paths_xy) and len(paths_xy[i]) > 0 else []
        for i in range(num_agents)
    ]

    # Replay path_updates to recover the A* path that was active at each frame.
    # This ensures the overlaid path matches the historical frame, not the current one.
    _current_paths: list[list] = [list(p) for p in _fallback_paths]
    frame_paths_snapshot: list[list[list]] = []
    for frame in frames:
        for upd in frame.get("path_updates", []):
            aid = upd.get("agent_id")
            if aid is not None and 0 <= aid < num_agents:
                _current_paths[aid] = upd.get("path_xy", [])
        frame_paths_snapshot.append([list(p) for p in _current_paths])

    # ── 4. Pre-extract positions and precompute episode breaks ─────────────────
    if frames:
        all_positions = np.array([f["positions_xy"] for f in frames], dtype=np.float32)
        all_reached_raw = np.array([f.get("reached", []) for f in frames])
        if all_reached_raw.ndim < 2 or all_reached_raw.shape[1] != num_agents:
            all_reached = np.zeros((len(frames), num_agents), dtype=bool)
        else:
            all_reached = all_reached_raw
        all_breaks = [
            _find_reset_boundaries_frame(all_reached[:, a], all_positions[:, a])
            for a in range(num_agents)
        ]
    else:
        all_positions = None
        all_breaks    = [[0, 1]] * num_agents

    def _path_px(path_xy: list) -> list:
        if len(path_xy) < 2:
            return []
        return [world_to_crop_pixel(xy, full_w, full_h, canvas.resolution,
                                    canvas.origin_xy, canvas.crop_box)
                for xy in path_xy]

    def _goal_px(goal_xy: list) -> tuple:
        return world_to_crop_pixel(goal_xy, full_w, full_h,
                                   canvas.resolution, canvas.origin_xy,
                                   canvas.crop_box)

    # ── 5. Render frames into RGB images (quantise palette once at the end) ───
    duration_ms = int(round(1000.0 / max(fps, 1e-5)))
    rgb_frames: list[Image.Image] = []
    output_size_xy = (
        max(1, int(round(canvas.size[0] * scale))),
        max(1, int(round(canvas.size[1] * scale))),
    )
    base_rgba = canvas.static_image.convert("RGBA")

    render_list = frames if frames else [{"positions_xy": _fallback_goals,
                                          "step": 0, "time": 0.0}]
    for frame_idx, frame in enumerate(render_list):
        frame_positions = np.asarray(frame["positions_xy"], dtype=np.float32)
        trail_start     = max(0, frame_idx - trail_length)

        # Use historical goals/paths when available; fall back to current state.
        if frame_idx < len(frame_paths_snapshot):
            frame_goals = frame.get("goals_xy", _fallback_goals)
            frame_paths = frame_paths_snapshot[frame_idx]
        else:
            frame_goals = _fallback_goals
            frame_paths = _fallback_paths

        image   = base_rgba.copy()
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay, "RGBA")

        for agent_id in range(num_agents):
            color = _agent_color(agent_id)
            is_hum = agent_id < num_humanoids
            pos_px = world_to_crop_pixel(frame_positions[agent_id],
                                         full_w, full_h, canvas.resolution,
                                         canvas.origin_xy, canvas.crop_box)
            if is_hum:
                _draw_circle(draw, pos_px, agent_size, (*color, 230), BLACK)
                continue

            # Planned path + goal (per-frame historical values)
            path_pixels = _path_px(
                frame_paths[agent_id] if agent_id < len(frame_paths) else []
            )
            if path_pixels:
                draw.line(path_pixels, fill=(*color, 100), width=1, joint="curve")
            goal_pixels = _goal_px(
                frame_goals[agent_id] if agent_id < len(frame_goals) else [0.0, 0.0]
            )
            _draw_goal_marker(draw, goal_pixels, agent_size + 2, (*color, 220))

            # Trail with precomputed breaks
            if all_positions is not None:
                segs = _trail_segments_fast(
                    all_positions[:, agent_id],
                    all_breaks[agent_id],
                    trail_start, frame_idx,
                )
                for seg in segs:
                    if len(seg) < 2:
                        continue
                    seg_px = [world_to_crop_pixel(
                        xy, full_w, full_h, canvas.resolution,
                        canvas.origin_xy, canvas.crop_box) for xy in seg]
                    draw.line(seg_px, fill=(*color, 200), width=2, joint="curve")

            _draw_square(draw, pos_px, agent_size, (*color, 240), BLACK)

        rendered = Image.alpha_composite(image, overlay).convert("RGB")
        if rendered.size != output_size_xy:
            rendered = rendered.resize(output_size_xy, Image.Resampling.BILINEAR)
        rgb_frames.append(rendered)

    if not rgb_frames:
        return None

    # Quantise all frames to a shared palette derived from the first frame.
    # Per-frame ADAPTIVE palettes cause colour flicker in GIF playback.
    ref_p = rgb_frames[0].convert("P", palette=Image.Palette.ADAPTIVE)
    quantized: list[Image.Image] = [ref_p] + [
        f.quantize(palette=ref_p) for f in rgb_frames[1:]
    ]

    buf = _io.BytesIO()
    quantized[0].save(
        buf, format="GIF", save_all=True,
        append_images=quantized[1:],
        duration=duration_ms, loop=0, optimize=False,
    )
    return buf.getvalue()


if __name__ == "__main__":
    main()
