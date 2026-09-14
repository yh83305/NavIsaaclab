"""Render CrowdSim navigation path logs on top of the occupancy map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


HUMANOID_COLORS = [
    (230, 74, 25),
    (244, 143, 177),
    (255, 183, 77),
    (171, 71, 188),
]
CAR_COLORS = [
    (0, 188, 212),
    (3, 169, 244),
    (77, 208, 225),
    (38, 166, 154),
]
START_COLOR = (40, 220, 80)
GOAL_COLOR = (255, 235, 59)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CrowdSim output/crowdsim_navigation path logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path_log",
        nargs="?",
        default="output/crowdsim_navigation/paths_latest.json",
        help="Path JSON produced by CrowdSim navigation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to the path log basename with .png.",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Override occupancy map path. By default uses map_path from JSON.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1800,
        help="Maximum output image width/height. Use 0 to keep full resolution.",
    )
    parser.add_argument("--line-width", type=int, default=6)
    parser.add_argument("--point-radius", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = resolve_path(args.path_log)
    data = json.loads(log_path.read_text(encoding="utf-8"))

    map_path = resolve_path(args.map) if args.map else Path(data["map_path"]).expanduser()
    image = Image.open(map_path).convert("L")
    base = ImageOps.autocontrast(image).convert("RGB")
    draw_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_overlay)

    resolution = float(data["map_resolution"])
    origin_xy = tuple(data.get("map_origin_xy", default_center_origin(base.size, resolution)))
    width, height = base.size
    font = load_font()

    for agent in data["agents"]:
        draw_agent(
            draw=draw,
            font=font,
            agent=agent,
            width=width,
            height=height,
            resolution=resolution,
            origin_xy=origin_xy,
            line_width=args.line_width,
            point_radius=args.point_radius,
        )

    rendered = Image.alpha_composite(base.convert("RGBA"), draw_overlay).convert("RGB")
    rendered = draw_legend(rendered, data)

    if args.max_size and max(rendered.size) > args.max_size:
        rendered.thumbnail((args.max_size, args.max_size), Image.Resampling.LANCZOS)

    output_path = resolve_output_path(args.output, log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    print(f"[CrowdSim] Saved path visualization: {output_path}")


def draw_agent(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    agent: dict[str, Any],
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
    line_width: int,
    point_radius: int,
) -> None:
    agent_type = str(agent["agent_type"])
    local_id = int(agent["local_id"])
    color = agent_color(agent_type, local_id)
    label = f"{agent_type[0].upper()}{local_id}"
    path = agent.get("path_xy", [])
    if len(path) >= 2:
        pixels = [world_to_pixel(xy, width, height, resolution, origin_xy) for xy in path]
        draw.line(pixels, fill=(*color, 215), width=line_width, joint="curve")
        for idx, pixel in enumerate(pixels):
            if idx == 0 or idx == len(pixels) - 1:
                continue
            draw_circle(draw, pixel, max(2, point_radius // 3), (*color, 180), outline=None)

    start_px = world_to_pixel(agent["start_xy"], width, height, resolution, origin_xy)
    goal_px = world_to_pixel(agent["goal_xy"], width, height, resolution, origin_xy)
    draw_circle(draw, start_px, point_radius, (*START_COLOR, 240), outline=(0, 0, 0, 220))
    draw_circle(draw, goal_px, point_radius, (*GOAL_COLOR, 240), outline=(0, 0, 0, 220))
    draw_label(draw, font, start_px, f"{label} S", color=(255, 255, 255, 235))
    draw_label(draw, font, goal_px, f"{label} G", color=(255, 255, 255, 235))


def world_to_pixel(
    xy: list[float] | tuple[float, float],
    width: int,
    height: int,
    resolution: float,
    origin_xy: tuple[float, float],
) -> tuple[int, int]:
    origin_x, origin_y = origin_xy
    pixel_x = int(round((float(xy[0]) - origin_x) / resolution))
    pixel_y = int(round((height - 1) - (float(xy[1]) - origin_y) / resolution))
    pixel_x = max(0, min(width - 1, pixel_x))
    pixel_y = max(0, min(height - 1, pixel_y))
    return pixel_x, pixel_y


def default_center_origin(size: tuple[int, int], resolution: float) -> tuple[float, float]:
    width, height = size
    return (-0.5 * (width - 1) * resolution, -0.5 * (height - 1) * resolution)


def draw_circle(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None,
) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(box, fill=fill, outline=outline, width=2 if outline else 1)


def draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    center: tuple[int, int],
    text: str,
    color: tuple[int, int, int, int],
) -> None:
    x, y = center
    pos = (x + 12, y - 16)
    bbox = draw.textbbox(pos, text, font=font)
    pad = 3
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=3,
        fill=(0, 0, 0, 155),
    )
    draw.text(pos, text, fill=color, font=font)


def draw_legend(image: Image.Image, data: dict[str, Any]) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font()
    lines = [
        f"created_at: {data.get('created_at', '')}",
        f"humanoids: {data.get('num_humanoids', 0)}",
        f"cars: {data.get('num_cars', 0)}",
        "green=start, yellow=goal",
        "warm paths=humanoid, cyan paths=car",
    ]
    text_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    line_height = 18
    x0, y0 = 16, 16
    draw.rounded_rectangle(
        (x0 - 8, y0 - 8, x0 + text_width + 12, y0 + line_height * len(lines) + 8),
        radius=5,
        fill=(0, 0, 0, 155),
    )
    for idx, line in enumerate(lines):
        draw.text((x0, y0 + idx * line_height), line, fill=(255, 255, 255, 235), font=font)
    return image


def agent_color(agent_type: str, local_id: int) -> tuple[int, int, int]:
    if agent_type == "humanoid":
        return HUMANOID_COLORS[local_id % len(HUMANOID_COLORS)]
    return CAR_COLORS[local_id % len(CAR_COLORS)]


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def resolve_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def resolve_output_path(output: str | None, log_path: Path) -> Path:
    if output:
        return resolve_path(output)
    return log_path.with_suffix(".png")


if __name__ == "__main__":
    main()
