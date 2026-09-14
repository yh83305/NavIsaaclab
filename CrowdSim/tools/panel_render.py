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
"""Shared per-episode panel renderer for CrowdSim data collection + offline viz.

Produces one GIF per episode with a fixed layout, every frame::

    ┌──────────────────────────────────────────────────────────────┐
    │ [bird's-eye] │ [RGB / depth]  │  [obs bar chart]  │  [action] │
    │  (world)     │                │                   │           │
    ├──────────────┴────────────────┴───────────────────┴───────────┤
    │  Step info: episode / step / goal_dist / ...                  │
    └───────────────────────────────────────────────────────────────┘

Used by:
  * ``CrowdSim/flow/collect_cbf_buffer.py``  (real-time, per-episode trigger)
  * ``CrowdSim/flow/collect_ppo_buffer.py``  (real-time, per-episode trigger)
  * ``CrowdSim/tools/visualize_flow_dataset.py`` (offline, any episode)

The bird's-eye panel draws all humanoids (grey dots) + all robots (coloured
squares with heading); the episode's focal robot is highlighted with a trail,
CBF barrier ring, and goal diamond + tolerance circle.  CBF (single robot)
degenerates to drawing just itself.

Episode data contract (both the on-disk buffer and the in-memory live steps)::

    episode = {
      "obs":    (T, obs_dim),
      "depth":  (T, H, W) | None,
      "rgb":    (T, 3, h, w) | None,
      "action": (T, 2),
      "camera_local_trajectory": (T, 24, 3) | (T, K, 24, 3),  # optional,
                                      # cumulative camera-local trajectories
      "world": {                       # optional; absent → bird's-eye skipped
        "robot_xy":       (T, 2),      # focal robot world xy
        "world_camera_position_xy": (T, 2),  # required with camera trajectory
        "world_camera_yaw": (T,),            # required with camera trajectory
        "yaw":            (T,),        # focal robot yaw
        "goal_xy":        (T, 2),
        "hum_xys":        (T, num_h, 2),
        "all_robot_xys":  (T, n_r, 2), # all active robots (n_r=1 for CBF)
        "all_robot_yaws": (T, n_r),
        "focal_idx":      int,
      }
    }
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# torch is optional at import time — renderers call ``.cpu().numpy()`` only
# when given tensors, so a torch-less environment (pure offline viz) works.
try:  # pragma: no cover
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None


# ── obs layout (must match nav_manager._build_robot_rl_observations) ────────
# Vector channel: goal(3) + ego motion(2). Neighbors are stored separately as
# [T, K, 5] and are appended only for display below.
OBS_LABELS_FIXED = [
    "goal_dist", "goal_sin", "goal_cos",
    "fwd_speed", "ang_vel",
]


def _neighbor_labels(n_neighbors: int) -> list[str]:
    labels: list[str] = []
    for i in range(n_neighbors):
        labels.extend([
            f"nb{i}_dist", f"nb{i}_sin", f"nb{i}_cos",
            f"nb{i}_vfwd", f"nb{i}_vlat",
        ])
    return labels


def obs_labels_for_dim(obs_dim: int) -> list[str]:
    labels = list(OBS_LABELS_FIXED)
    if len(labels) < obs_dim:
        labels += [f"unk_{i}" for i in range(obs_dim - len(labels))]
    return labels[:obs_dim]


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "DejaVuSansMono.ttf", "LiberationMono.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _to_np(x: Any) -> np.ndarray:
    """Best-effort tensor→numpy (no-op for numpy/scalars)."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def camera_local_trajectory_to_world(
    camera_local_trajectory: Any,
    world_camera_position_xy: Any,
    world_camera_yaw: float,
) -> np.ndarray:
    """Convert cumulative camera forward/left points to world XY coordinates.

    Accepts ``[..., H, 2 or 3]`` and preserves all leading dimensions.  The
    helper is shared by offline dataset inspection and online NavDP demos so
    that coordinate-sign mistakes are visible in the collection panel.
    """
    camera_local_trajectory = _to_np(camera_local_trajectory).astype(
        np.float32, copy=False
    )
    if (
        camera_local_trajectory.ndim < 2
        or camera_local_trajectory.shape[-1] not in (2, 3)
    ):
        raise ValueError(
            "Camera-local trajectory must be [...,H,2 or 3], got "
            f"{tuple(camera_local_trajectory.shape)}"
        )
    c, s = math.cos(float(world_camera_yaw)), math.sin(float(world_camera_yaw))
    rotation_camera_to_world = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return (
        camera_local_trajectory[..., :2] @ rotation_camera_to_world.T
        + _to_np(world_camera_position_xy)
    )


# ── panel renderers ─────────────────────────────────────────────────────────

def render_rgb(rgb_tensor: Any) -> Image.Image:
    """Render RGB tensor (H, W, 3) or (3, H, W) uint8 → PIL Image."""
    arr = _to_np(rgb_tensor)
    if arr.ndim == 3 and arr.shape[0] == 3:   # CHW → HWC
        arr = arr.transpose(1, 2, 0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _depth_colormap(norm: np.ndarray) -> np.ndarray:
    """Map normalized depth values to RGB using the panel's colour map."""
    norm = np.clip(norm, 0.0, 1.0)
    try:
        from matplotlib import cm
        return (cm.turbo(norm)[..., :3] * 255).astype(np.uint8)
    except ImportError:
        value = (norm * 255).astype(np.uint8)
        return np.stack([value, value, value], axis=-1)


def render_depth(depth_tensor: Any, max_range: float = 5.0) -> Image.Image:
    """Render normalized depth ``[0,1]`` as a colour-mapped image.

    ``max_range`` defines the physical value represented by 1.0 and is used
    by :func:`add_depth_colorbar` for metre labels; it must not be applied a
    second time to the already-normalized tensor.
    """
    arr = _to_np(depth_tensor).astype(np.float32)
    rgb = _depth_colormap(arr)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.transpose(1, 2, 0)
    return Image.fromarray(rgb).convert("RGB")


def add_depth_colorbar(
    image: Image.Image,
    max_range: float,
    font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    """Overlay a vertical colour bar with metric depth labels."""
    if font is None:
        font = _load_font(12)
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    margin = max(8, image.width // 40)
    bar_w = max(12, image.width // 24)
    bar_h = max(80, int(image.height * 0.66))
    bar_x1 = image.width - margin
    bar_x0 = bar_x1 - bar_w
    bar_y0 = max(margin + 18, (image.height - bar_h) // 2)
    bar_y1 = min(image.height - margin, bar_y0 + bar_h)

    labels = (f"{max_range:g}m", f"{max_range / 2:g}m", "0m")
    label_width = max(
        draw.textbbox((0, 0), label, font=font)[2] for label in labels
    )
    box_x0 = max(0, bar_x0 - label_width - 12)
    draw.rounded_rectangle(
        (box_x0 - 4, bar_y0 - 22, bar_x1 + 4, bar_y1 + 4),
        radius=5,
        fill=(0, 0, 0, 155),
    )
    draw.text((box_x0, bar_y0 - 20), "depth", fill=(255, 255, 255, 235), font=font)

    # Top is far/max depth; bottom is near/zero depth, matching Turbo mapping.
    values = np.linspace(1.0, 0.0, max(bar_y1 - bar_y0, 1), dtype=np.float32)
    colors = _depth_colormap(values)
    for offset, color in enumerate(colors):
        y = bar_y0 + offset
        draw.line((bar_x0, y, bar_x1, y), fill=tuple(int(v) for v in color) + (255,))
    draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), outline=(255, 255, 255, 220), width=1)

    tick_values = (
        (bar_y0, labels[0]),
        ((bar_y0 + bar_y1) // 2, labels[1]),
        (bar_y1, labels[2]),
    )
    for y, label in tick_values:
        draw.line((bar_x0 - 4, y, bar_x0, y), fill=(255, 255, 255, 230), width=1)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_h = bbox[3] - bbox[1]
        draw.text(
            (bar_x0 - label_width - 7, y - text_h // 2),
            label,
            fill=(255, 255, 255, 235),
            font=font,
        )

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def render_obs_bars(obs_vec: np.ndarray, labels: list[str], width: int = 400,
                    height: int = 480, font: ImageFont.ImageFont = None) -> Image.Image:
    if font is None:
        font = _load_font(12)
    img = Image.new("RGB", (width, height), (30, 30, 35))
    draw = ImageDraw.Draw(img)
    n = len(obs_vec)
    bar_h = max(8, (height - 20) // max(n, 1))
    mid_x = width // 2
    for i, (val, label) in enumerate(zip(obs_vec, labels)):
        y = 10 + i * bar_h
        val_clipped = max(-2.0, min(2.0, float(val)))
        bar_len = int((val_clipped / 2.0) * (width // 2 - 120))
        colour = (80, 200, 120) if val_clipped >= 0 else (220, 80, 80)
        draw.text((5, y), f"{label}", fill=(200, 200, 210), font=font)
        y0, y1 = y + 2, max(y + 3, y + bar_h - 4)
        draw.rectangle((120, y0, width - 10, y1), fill=(50, 50, 60))
        draw.line((mid_x, y0, mid_x, y1), fill=(90, 90, 100))
        by0, by1 = y + 3, max(y + 4, y + bar_h - 5)
        if bar_len >= 0:
            draw.rectangle((mid_x, by0, mid_x + bar_len, by1), fill=colour)
        else:
            draw.rectangle((mid_x + bar_len, by0, mid_x, by1), fill=colour)
        draw.text((width - 60, y), f"{float(val):+.2f}", fill=(180, 180, 200), font=font)
    return img


def render_action(action_vec: np.ndarray, width: int = 200, height: int = 480,
                  font: ImageFont.ImageFont = None) -> Image.Image:
    if font is None:
        font = _load_font(14)
    img = Image.new("RGB", (width, height), (30, 30, 35))
    draw = ImageDraw.Draw(img)
    v_lin = float(action_vec[0])
    v_ang = float(action_vec[1])
    draw.text((10, 10), "Action", fill=(255, 255, 255), font=_load_font(16))
    cx, cy, r = width // 2, 130, 60
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(100, 100, 120), width=2)
    draw.line((cx, cy - r, cx, cy + r), fill=(60, 60, 70))
    draw.line((cx - r, cy, cx + r, cy), fill=(60, 60, 70))
    angle = v_ang * math.pi / 2
    arrow_len = max(5, abs(v_lin) * r)
    ex = cx + math.cos(angle - math.pi / 2) * arrow_len
    ey = cy + math.sin(angle - math.pi / 2) * arrow_len
    draw.line((cx, cy, ex, ey), fill=(80, 200, 120), width=3)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(80, 200, 120))
    bar_y = 230
    for i, (val, label) in enumerate([(v_lin, "v_lin"), (v_ang, "v_ang")]):
        y = bar_y + i * 60
        draw.text((10, y), label, fill=(200, 200, 210), font=font)
        draw.rectangle((80, y + 18, width - 10, y + 34), fill=(50, 50, 60))
        mid = 80 + (width - 90) // 2
        draw.line((mid, y + 18, mid, y + 34), fill=(90, 90, 100))
        bar_len = int(float(val) * (width - 90) / 2)
        colour = (80, 200, 120) if val >= 0 else (220, 80, 80)
        if bar_len >= 0:
            draw.rectangle((mid, y + 19, mid + bar_len, y + 33), fill=colour)
        else:
            draw.rectangle((mid + bar_len, y + 19, mid, y + 33), fill=colour)
        draw.text((width - 55, y), f"{float(val):+.2f}", fill=(180, 180, 200), font=font)
    return img


def render_info_bar(ep_idx: int, step_idx: int, ep_len: int, obs_vec: np.ndarray,
                    width: int, font: ImageFont.ImageFont) -> Image.Image:
    h = 30
    img = Image.new("RGB", (width, h), (20, 20, 25))
    draw = ImageDraw.Draw(img)
    goal_dist = float(obs_vec[0]) if len(obs_vec) > 0 else 0
    fwd_speed = float(obs_vec[3]) if len(obs_vec) > 3 else 0
    ang_vel = float(obs_vec[4]) if len(obs_vec) > 4 else 0
    txt = (f"Ep {ep_idx}  Step {step_idx}/{ep_len}  |  "
           f"goal_dist={goal_dist:.2f}  fwd={fwd_speed:.2f}  ang={ang_vel:.2f}")
    draw.text((10, 6), txt, fill=(255, 255, 255), font=font)
    return img


# ── bird's-eye renderer (multi-robot) ───────────────────────────────────────

_AGENT_COLORS = [
    (40, 160, 230), (230, 40, 40), (40, 210, 80), (230, 180, 30),
    (170, 50, 210), (230, 80, 170), (30, 200, 190), (210, 130, 40),
]


def _agent_color(idx: int) -> tuple[int, int, int]:
    return _AGENT_COLORS[idx % len(_AGENT_COLORS)]


class BirdseyeRenderer:
    """Renders one bird's-eye tile per step, side-by-side with the obs panels.

    Background: the occupancy-map image (loaded once, autocontrast + centred
    crop).  Falls back to a placeholder panel if the map is absent.

    Per frame draws: humanoids (grey dots) → all robots (coloured squares with
    heading arrow) → robot trails, barrier rings and goals → action text.
    """

    ROBOT_COLOR = (40, 160, 230)
    GOAL_COLOR = (249, 199, 79)
    HUM_COLOR = (110, 112, 128)
    BLACK = (0, 0, 0, 220)
    WHITE = (255, 255, 255, 220)

    def __init__(self, map_path, origin, resolution, tile_px: int = 320,
                 robot_radius: float = 0.3, goal_tolerance: float = 0.75,
                 trail_len: int = 80, teleport_thresh_m: float = 0.5,
                 trajectory_selected_width: int = 3,
                 trajectory_candidate_width: int = 1,
                 trajectory_selected_alpha: int = 230,
                 trajectory_candidate_alpha: int = 65,
                 trajectory_point_radius: int = 0,
                 local_view_size_m: float = 0.0,
                 local_view_robot: int = 0):
        self.origin = (float(origin[0]), float(origin[1]))
        self.res = float(resolution)
        self.tile_px = int(tile_px)
        self.robot_radius = float(robot_radius)
        self.goal_tolerance = float(goal_tolerance)
        self.trail_len = int(trail_len)
        self.teleport_thresh_m = float(teleport_thresh_m)
        self.trajectory_selected_width = int(trajectory_selected_width)
        self.trajectory_candidate_width = int(trajectory_candidate_width)
        self.trajectory_selected_alpha = int(trajectory_selected_alpha)
        self.trajectory_candidate_alpha = int(trajectory_candidate_alpha)
        self.trajectory_point_radius = int(trajectory_point_radius)
        self.local_view_size_m = float(local_view_size_m)
        self.local_view_robot = int(local_view_robot)
        self.bg = None
        self._cl = self._ct = 0
        self._H_px = 0
        self._map_scale = 1.0
        if map_path and Path(map_path).exists():
            from PIL import ImageOps
            bg = ImageOps.autocontrast(Image.open(map_path).convert("L")).convert("RGB")
            W, H = bg.size
            # Keep the complete map. Large 1 cm maps are resized once here;
            # world-to-pixel coordinates use the same scale below.
            max_side = 900
            self._H_px = H
            self._map_scale = min(1.0, max_side / max(W, H))
            if self._map_scale < 1.0:
                bg = bg.resize(
                    (
                        max(1, int(round(W * self._map_scale))),
                        max(1, int(round(H * self._map_scale))),
                    ),
                    Image.Resampling.NEAREST,
                )
            self.bg = bg
        # Expected rendered tile height: tile_px wide, height keeps the
        # (cropped) map aspect ratio.  Used by the panel layout to size row 1.
        if self.bg is not None:
            self.out_h = max(1, int(self.tile_px * self.bg.size[1] / self.bg.size[0]))
        else:
            self.out_h = self.tile_px
        self.font = _load_font(12)

    @property
    def has_map(self) -> bool:
        return self.bg is not None

    def _w2p(self, xy) -> tuple[int, int]:
        ox, oy = self.origin
        px = int(round(
            ((float(xy[0]) - ox) / self.res - self._cl) * self._map_scale
        ))
        py = int(round(
            ((self._H_px - 1) - (float(xy[1]) - oy) / self.res - self._ct)
            * self._map_scale
        ))
        return px, py

    def render(self, world_step: dict, action: np.ndarray,
               trail: list, focal_idx: int | None) -> Image.Image:
        """Render one bird's-eye frame.

        ``world_step`` carries per-step ``all_robot_xys (n_r,2)``,
        ``all_robot_yaws (n_r,)``, ``hum_xys (num_h,2)``, ``goal_xy (2,)``.
        NavDP evaluation may provide world-camera trajectories as
        ``[R,K,H,2]``, world goals as
        ``[R,2]``, actions as ``[R,2]`` and one trail per robot.  Other callers
        may still provide the single-focal forms.
        """
        if not self.has_map:
            img = Image.new("RGB", (self.tile_px, self.tile_px), (25, 25, 35))
            draw = ImageDraw.Draw(img)
            draw.text((10, self.tile_px // 2 - 10),
                      "bird's-eye: map not found\n(pass --map-path)",
                      fill=(180, 180, 200), font=self.font)
            return img

        base = self.bg.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Optional route-guidance reference. The complete sparse A* route is
        # drawn as connected points and the currently tracked waypoint is
        # highlighted, making controller/reference disagreement visible.
        route_sparse_path = world_step.get("route_sparse_path")
        if route_sparse_path is not None:
            route_sparse_path = np.asarray(route_sparse_path, dtype=np.float32)
            route_sparse_path = route_sparse_path[
                np.isfinite(route_sparse_path).all(axis=-1)
            ]
            if len(route_sparse_path):
                pixels = [self._w2p(point) for point in route_sparse_path]
                if len(pixels) >= 2:
                    draw.line(pixels, fill=(255, 215, 64, 190), width=2)
                for px, py in pixels:
                    draw.ellipse((px - 3, py - 3, px + 3, py + 3),
                                 fill=(255, 215, 64, 225), outline=self.BLACK)
        route_sparse_target = world_step.get("route_sparse_target")
        if route_sparse_target is not None:
            px, py = self._w2p(route_sparse_target)
            draw.ellipse((px - 6, py - 6, px + 6, py + 6),
                         outline=(255, 80, 80, 255), width=3)

        # Optional NavDP candidates. Draw all candidates lightly and the
        # first/selected trajectory brightly.  Coordinates are already world
        # XY here; conversion from camera-local is performed by the caller.
        world_camera_trajectories = world_step.get("world_camera_trajectories")
        if world_camera_trajectories is not None:
            world_camera_trajectories = np.asarray(
                world_camera_trajectories, dtype=np.float32
            )
            if world_camera_trajectories.ndim == 2:
                world_camera_trajectories = world_camera_trajectories[None, None]
            elif world_camera_trajectories.ndim == 3:
                world_camera_trajectories = world_camera_trajectories[None]
            for robot_id, world_camera_candidate_trajectories in enumerate(
                world_camera_trajectories
            ):
                color = _agent_color(robot_id if focal_idx is None else focal_idx)
                for candidate_id, world_camera_candidate_trajectory in enumerate(
                    world_camera_candidate_trajectories
                ):
                    world_camera_candidate_trajectory = (
                        world_camera_candidate_trajectory[
                            np.isfinite(world_camera_candidate_trajectory).all(axis=-1)
                        ]
                    )
                    if len(world_camera_candidate_trajectory) < 2:
                        continue
                    pixels = [
                        self._w2p(world_camera_point)
                        for world_camera_point in world_camera_candidate_trajectory
                    ]
                    selected = candidate_id == 0
                    draw.line(
                        pixels,
                        fill=(*color, self.trajectory_selected_alpha if selected else self.trajectory_candidate_alpha),
                        width=self.trajectory_selected_width if selected else self.trajectory_candidate_width,
                        joint="curve",
                    )
                    if self.trajectory_point_radius > 0:
                        radius = self.trajectory_point_radius if selected else max(1, self.trajectory_point_radius - 1)
                        alpha = self.trajectory_selected_alpha if selected else self.trajectory_candidate_alpha
                        for px, py in pixels:
                            draw.ellipse(
                                (px - radius, py - radius, px + radius, py + radius),
                                fill=(*color, alpha),
                            )

        # Humanoids
        for hxy in np.asarray(world_step["hum_xys"]):
            hp = self._w2p(hxy)
            r = 3
            draw.ellipse((hp[0] - r, hp[1] - r, hp[0] + r, hp[1] + r),
                         fill=(*self.HUM_COLOR, 200))

        # All robots (coloured squares + heading).  Focal gets a brighter
        # outline; non-focal drawn slightly translucent so the focal stands out.
        all_xy = np.asarray(world_step["all_robot_xys"])
        all_yaw = np.asarray(world_step["all_robot_yaws"]).reshape(-1)
        n_r = all_xy.shape[0]
        for i in range(n_r):
            rp = self._w2p(all_xy[i])
            col = _agent_color(i)
            rs = 3
            alpha = 245 if focal_idx is None or i == focal_idx else 150
            draw.rectangle((rp[0] - rs, rp[1] - rs, rp[0] + rs, rp[1] + rs),
                           fill=(*col, alpha), outline=self.BLACK)
            yaw = float(all_yaw[i])
            arr_len = rs * (3 if focal_idx is None or i == focal_idx else 2)
            ax = rp[0] + int(arr_len * math.cos(yaw))
            ay = rp[1] - int(arr_len * math.sin(yaw))
            draw.line([rp, (ax, ay)], fill=(*col, 230), width=2)

        # Robot overlays: trail, footprint ring, goal diamond + tolerance circle.
        robot_ids = range(n_r) if focal_idx is None else (focal_idx,)
        trails = trail if focal_idx is None else (trail,)
        world_goal_xy = world_step.get("all_goal_xys")
        if world_goal_xy is None:
            world_goal_xy = np.asarray(world_step["goal_xy"])[None]
        bar_px = max(1, int(round(
            self.robot_radius / self.res * self._map_scale
        )))
        tol_px = max(1, int(round(
            self.goal_tolerance / self.res * self._map_scale
        )))
        for trail_index, robot_id in enumerate(robot_ids):
            color = _agent_color(robot_id)
            robot_trail = trails[trail_index]
            if len(robot_trail) >= 2:
                draw.line(robot_trail, fill=(*color, 180), width=2, joint="curve")
            rp = self._w2p(all_xy[robot_id])
            draw.ellipse(
                (rp[0] - bar_px, rp[1] - bar_px, rp[0] + bar_px, rp[1] + bar_px),
                outline=(*color, 90), width=1,
            )
            gp = self._w2p(world_goal_xy[robot_id if focal_idx is None else 0])
            gs = 4
            draw.polygon(
                [(gp[0], gp[1] - gs), (gp[0] + gs, gp[1]),
                 (gp[0], gp[1] + gs), (gp[0] - gs, gp[1])],
                fill=(*color, 220), outline=self.BLACK,
            )
            draw.ellipse(
                (gp[0] - tol_px, gp[1] - tol_px, gp[0] + tol_px, gp[1] + tol_px),
                outline=(*color, 140), width=1,
            )

        # Action text
        actions = np.asarray(action)
        actual_actions = world_step.get("actual_actions")
        if actions.ndim == 1:
            v_str = f"v={float(actions[0]):.2f}  ω={float(actions[1]):.2f}"
        else:
            if actual_actions is None:
                v_str = "  ".join(
                    f"R{i}: {float(value[0]):.2f}/{float(value[1]):.2f}"
                    for i, value in enumerate(actions)
                )
            else:
                actual_actions = np.asarray(actual_actions)
                v_str = "  ".join(
                    f"R{i} C:{float(command[0]):.2f}/{float(command[1]):.2f} "
                    f"A:{float(actual[0]):.2f}/{float(actual[1]):.2f}"
                    for i, (command, actual) in enumerate(
                        zip(actions, actual_actions)
                    )
                )
        bbox = (draw.textbbox((8, 6), v_str, font=self.font)
                if hasattr(draw, "textbbox") else (8, 6, 8 + len(v_str) * 8, 22))
        draw.rectangle((bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3),
                       fill=(0, 0, 0, 150))
        draw.text((8, 6), v_str, fill=self.WHITE, font=self.font)

        composited = Image.alpha_composite(base, overlay).convert("RGB")
        out = composited
        # Scale width to tile_px, height proportional — keep aspect ratio.
        if out.size[0] != self.tile_px:
            aspect = out.size[1] / out.size[0]
            out = out.resize((self.tile_px, max(1, int(self.tile_px * aspect))),
                             Image.Resampling.BILINEAR)
        if self.local_view_size_m > 0.0 and n_r > 0:
            robot_id = min(max(self.local_view_robot, 0), n_r - 1)
            cx, cy = self._w2p(all_xy[robot_id])
            crop_px = max(2, int(round(
                self.local_view_size_m / self.res * self._map_scale
            )))
            half = crop_px // 2
            local = Image.new("RGB", (crop_px, crop_px), (25, 25, 35))
            left, top = cx - half, cy - half
            source_box = (
                max(0, left), max(0, top),
                min(composited.width, left + crop_px),
                min(composited.height, top + crop_px),
            )
            if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
                patch = composited.crop(source_box)
                local.paste(patch, (source_box[0] - left, source_box[1] - top))
            local = local.resize((out.height, out.height), Image.Resampling.NEAREST)
            label_draw = ImageDraw.Draw(local, "RGBA")
            label = f"R{robot_id} local {self.local_view_size_m:g}m x {self.local_view_size_m:g}m"
            bbox = label_draw.textbbox((10, 8), label, font=self.font)
            label_draw.rectangle((bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3), fill=(0, 0, 0, 170))
            label_draw.text((10, 8), label, fill=self.WHITE, font=self.font)
            combined = Image.new("RGB", (out.width + out.height, out.height), (25, 25, 35))
            combined.paste(out, (0, 0))
            combined.paste(local, (out.width, 0))
            out = combined
        return out


# ── per-step world stacking helper (used by collect scripts) ─────────────────

def stack_world(steps: list[dict]) -> dict:
    """Stack per-step ``world`` dicts (numpy) into episode-level arrays.

    Each step's ``world`` has scalar/array numpy fields; this produces the
    ``(T, ...)`` arrays the renderer + on-disk buffer expect.
    """
    result = {
        "robot_xy":       np.stack([s["world"]["robot_xy"]       for s in steps]),
        "yaw":            np.stack([s["world"]["yaw"]            for s in steps]),
        "goal_xy":        np.stack([s["world"]["goal_xy"]        for s in steps]),
        "hum_xys":        np.stack([s["world"]["hum_xys"]        for s in steps]),
        "all_robot_xys":  np.stack([s["world"]["all_robot_xys"]  for s in steps]),
        "all_robot_yaws": np.stack([s["world"]["all_robot_yaws"] for s in steps]),
        "focal_idx":      int(steps[0]["world"]["focal_idx"]),
    }
    for key in ("world_camera_position_xy", "world_camera_yaw"):
        if all(key in step["world"] for step in steps):
            result[key] = np.stack([step["world"][key] for step in steps])
    if all("route_sparse_path" in step["world"] for step in steps):
        result["route_sparse_path"] = steps[0]["world"]["route_sparse_path"]
    if all("route_sparse_target" in step["world"] for step in steps):
        result["route_sparse_target"] = np.stack(
            [step["world"]["route_sparse_target"] for step in steps]
        )
    return result


# ── unified per-episode renderer ────────────────────────────────────────────

class RenderOpts:
    """Layout/render knobs shared by real-time + offline paths."""

    def __init__(self, *, show_rgb: bool = True, show_depth: bool = True,
                 depth_max_range: float = 5.0, tile_px: int = 320,
                 row2_h: int = 200, gap: int = 10, max_frames: int = 0,
                 show_diagnostics: bool = True):
        self.show_rgb = show_rgb
        self.show_depth = show_depth
        self.depth_max_range = depth_max_range
        self.tile_px = tile_px
        self.row2_h = row2_h
        self.gap = gap
        self.max_frames = max_frames  # 0 = all
        self.show_diagnostics = bool(show_diagnostics)


def render_episode_frames(episode: dict, birdseye: "BirdseyeRenderer | None",
                          opts: RenderOpts, ep_idx: int = 0,
                          frame_indices: list[int] | None = None) -> list[Image.Image]:
    """Render every frame of one episode into the unified panel canvas.

    Returns a list of PIL images (one per kept frame).  Bird's-eye is included
    iff ``birdseye`` is supplied AND ``episode["world"]`` exists.
    """
    font_small = _load_font(12)
    font_med = _load_font(14)

    obs_arr = _to_np(episode["obs"])
    neighbor_arr = (
        _to_np(episode["neighbors"]) if episode.get("neighbors") is not None else None
    )
    neighbor_mask_arr = (
        _to_np(episode["neighbor_mask"]).astype(bool)
        if episode.get("neighbor_mask") is not None else None
    )
    act_arr = _to_np(episode["action"])
    T = obs_arr.shape[0]
    n_frames = min(T, opts.max_frames) if opts.max_frames > 0 else T
    if frame_indices is None:
        indices = list(range(n_frames))
    else:
        indices = [int(index) for index in frame_indices]
        invalid = [index for index in indices if index < 0 or index >= T]
        if invalid:
            raise IndexError(
                f"Frame indices outside episode length {T}: {invalid}"
            )
    obs_dim = obs_arr.shape[-1]
    labels = obs_labels_for_dim(obs_dim)
    if neighbor_arr is not None:
        labels += _neighbor_labels(neighbor_arr.shape[1])

    has_rgb = opts.show_rgb and episode.get("rgb") is not None
    has_depth = opts.show_depth and episode.get("depth") is not None
    has_bird = birdseye is not None and episode.get("world") is not None

    row1_img = opts.tile_px
    # Portrait maps use a stacked sensor column so row 1 contains no large
    # unused block below square RGB/depth inputs.
    row1_h = birdseye.out_h if has_bird else row1_img
    row2_h = opts.row2_h if opts.show_diagnostics else 0
    gap = opts.gap
    bird_w = row1_img if has_bird else 0
    rgb_w = row1_img if has_rgb else 0
    depth_w = row1_img if has_depth else 0
    stack_sensor_tiles = (
        has_bird and has_rgb and has_depth and row1_h > row1_img + gap
    )
    if stack_sensor_tiles:
        n_tiles = 2
        row1_w = bird_w + row1_img + gap * (n_tiles + 1)
        sensor_slot_h = max(1, (row1_h - gap) // 2)
    else:
        n_tiles = int(has_bird) + int(has_rgb) + int(has_depth)
        row1_w = bird_w + rgb_w + depth_w + gap * (n_tiles + 1)
        sensor_slot_h = row1_img
    row2_w = row1_w
    action_panel_w = max(140, int(row2_w * 0.35))
    obs_panel_w = row2_w - action_panel_w - gap * 3
    total_w = row1_w
    total_h = (
        row1_h + row2_h + gap * 3 + 16
        if opts.show_diagnostics else row1_h + gap * 2
    )

    world = episode.get("world")
    camera_local_trajectory = episode.get("camera_local_trajectory")
    if camera_local_trajectory is not None and has_bird:
        missing_camera_pose = [
            key
            for key in ("world_camera_position_xy", "world_camera_yaw")
            if key not in world
        ]
        if missing_camera_pose:
            raise ValueError(
                "camera_local_trajectory requires camera world pose fields: "
                f"{missing_camera_pose}"
            )
    focal_idx = int(world["focal_idx"]) if has_bird else 0

    frames: list[Image.Image] = []
    trail: list[tuple[int, int]] = []
    prev_xy: np.ndarray | None = None

    random_access = frame_indices is not None
    for t in indices:
        canvas = Image.new("RGB", (total_w, total_h), (15, 15, 20))
        obs_vec = obs_arr[t]
        display_obs = obs_vec
        if neighbor_arr is not None:
            neighbor_t = neighbor_arr[t].copy()
            if neighbor_mask_arr is not None:
                neighbor_t[~neighbor_mask_arr[t]] = 0.0
            display_obs = np.concatenate([obs_vec, neighbor_t.reshape(-1)])
        action_vec = act_arr[t]

        # Row 1: bird's-eye + RGB + depth
        y1 = gap
        x = gap
        if has_bird:
            world_robot_xy = np.asarray(world["robot_xy"][t])
            if random_access:
                start = max(0, t - birdseye.trail_len + 1)
                history = np.asarray(world["robot_xy"][start:t + 1])
                if len(history) > 1:
                    jumps = np.linalg.norm(np.diff(history, axis=0), axis=1)
                    teleports = np.flatnonzero(
                        jumps > birdseye.teleport_thresh_m
                    )
                    if len(teleports):
                        history = history[int(teleports[-1]) + 1:]
                trail = [birdseye._w2p(xy) for xy in history]
            else:
                if (
                    prev_xy is not None
                    and float(np.linalg.norm(world_robot_xy - prev_xy))
                    > birdseye.teleport_thresh_m
                ):
                    trail = []
                prev_xy = world_robot_xy.copy()
                trail.append(birdseye._w2p(world_robot_xy))
                if len(trail) > birdseye.trail_len:
                    trail.pop(0)
            world_t = {
                "all_robot_xys": np.asarray(world["all_robot_xys"][t]),
                "all_robot_yaws": np.asarray(world["all_robot_yaws"][t]),
                "hum_xys": np.asarray(world["hum_xys"][t]),
                "goal_xy": np.asarray(world["goal_xy"][t]),
            }
            if "route_sparse_path" in world:
                world_t["route_sparse_path"] = np.asarray(world["route_sparse_path"])
            if "route_sparse_target" in world:
                world_t["route_sparse_target"] = np.asarray(
                    world["route_sparse_target"][t]
                )
            if camera_local_trajectory is not None:
                world_camera_position_xy = np.asarray(
                    world["world_camera_position_xy"][t]
                )
                world_camera_yaw = float(world["world_camera_yaw"][t])
                world_t["world_camera_trajectories"] = (
                    camera_local_trajectory_to_world(
                        camera_local_trajectory[t],
                        world_camera_position_xy,
                        world_camera_yaw,
                    )
                )
            canvas.paste(birdseye.render(world_t, action_vec, trail, focal_idx), (x, y1))
            x += row1_img + gap
        if has_rgb and stack_sensor_tiles:
            rgb_img = ImageOps.fit(
                render_rgb(episode["rgb"][t]),
                (row1_img, sensor_slot_h),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
            canvas.paste(rgb_img, (x, y1))
        elif has_rgb:
            rgb_img = render_rgb(episode["rgb"][t])
            rgb_img = rgb_img.resize((row1_img, row1_img), Image.Resampling.BILINEAR)
            canvas.paste(rgb_img, (x, y1))
            x += row1_img + gap
        if has_depth and stack_sensor_tiles:
            dep_img = render_depth(
                episode["depth"][t], max_range=opts.depth_max_range
            )
            dep_img = ImageOps.fit(
                dep_img,
                (row1_img, sensor_slot_h),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
            dep_img = add_depth_colorbar(
                dep_img, max_range=opts.depth_max_range, font=font_small
            )
            canvas.paste(dep_img, (x, y1 + sensor_slot_h + gap))
        elif has_depth:
            dep_img = render_depth(episode["depth"][t], max_range=opts.depth_max_range)
            dep_img = dep_img.resize((row1_img, row1_img), Image.Resampling.BILINEAR)
            dep_img = add_depth_colorbar(
                dep_img, max_range=opts.depth_max_range, font=font_small
            )
            canvas.paste(dep_img, (x, y1))

        if opts.show_diagnostics:
            # Row 2: obs bars + action
            y2 = gap * 2 + row1_h
            x = gap
            obs_img = render_obs_bars(display_obs, labels, width=obs_panel_w,
                                      height=row2_h, font=font_small)
            canvas.paste(obs_img, (x, y2))
            x += obs_panel_w + gap
            act_img = render_action(action_vec, width=action_panel_w,
                                    height=row2_h, font=font_med)
            canvas.paste(act_img, (x, y2))

            # Info bar — pinned to the bottom edge of the canvas.
            y3 = total_h - 30
            canvas.paste(
                render_info_bar(ep_idx, t, T, obs_vec, total_w, font_small),
                (0, y3),
            )
        frames.append(canvas)

    return frames


# ── GIF save + wandb upload ─────────────────────────────────────────────────

def save_gif(frames: list[Image.Image], path, fps: float) -> bool:
    """Save frames as a GIF.  Returns True if written."""
    if not frames:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(fps, 1)))
    pal = [f.convert("P", palette=Image.Palette.ADAPTIVE) for f in frames]
    pal[0].save(str(path), save_all=True, append_images=pal[1:],
                duration=duration_ms, loop=0, optimize=False)
    kb = path.stat().st_size // 1024
    print(f"[GIF] Saved → {path}  ({len(frames)} frames @ {fps}fps, {kb} KB)")
    return True


def render_episode_gif(episode: dict, birdseye: "BirdseyeRenderer | None",
                       gif_base, fps: float, *, ep_idx: int,
                       wandb_run=None, wandb_step: int = 0,
                       opts: "RenderOpts | None" = None) -> None:
    """Render one episode's full-panel GIF + save + upload to wandb.

    ``gif_base`` is the base path (e.g. ``.../run.gif``); the actual file is
    ``.../run_ep{N:04d}.gif`` so successive episodes don't overwrite.  Used by
    both real-time collect scripts (cbf + ppo) on their per-episode trigger.
    """
    if birdseye is None:
        return
    if opts is None:
        opts = RenderOpts(
            show_rgb=episode.get("rgb") is not None,
            show_depth=episode.get("depth") is not None,
        )
    gif_base = Path(gif_base)
    out_path = gif_base.with_name(f"{gif_base.stem}_ep{ep_idx:04d}.gif")
    frames = render_episode_frames(episode, birdseye, opts, ep_idx=ep_idx)
    if save_gif(frames, out_path, fps):
        upload_gif_wandb(out_path, fps, wandb_run, wandb_step=wandb_step)


def upload_gif_wandb(gif_path, fps: float, wandb_run, wandb_step: int = 0) -> None:
    """Upload a GIF file to a wandb run (best-effort, no-op if wandb absent)."""
    if wandb_run is None:
        return
    try:
        import wandb
        with open(gif_path, "rb") as f:
            data = f.read()
        buf = io.BytesIO(data)
        wandb_run.log(
            {"eval/video": wandb.Video(buf, format="gif", fps=int(fps))},
            step=wandb_step,
        )
        print(f"[W&B] Uploaded GIF → eval/video (step {wandb_step})")
    except Exception as exc:  # noqa: BLE001
        print(f"[W&B] WARNING: GIF upload failed: {exc}")
