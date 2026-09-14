"""Occupancy map metadata helpers for CrowdSim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAP_RESOLUTION = 100.0 / 1999.0
DEFAULT_FREE_THRESHOLD = 200


@dataclass(frozen=True)
class OccupancyMapMetadata:
    image_path: Path
    metadata_path: Path | None
    resolution: float
    origin_xy: tuple[float, float]
    free_threshold: int
    negate: int = 0
    occupied_thresh: float | None = None
    free_thresh: float | None = None


def load_occupancy_map_metadata(path_like: str | Path) -> OccupancyMapMetadata:
    """Load map image path, resolution, origin, and free threshold from a map YAML.

    If ``path_like`` points to an image, a sidecar YAML with the same stem is used
    when present. Otherwise the historical CrowdSim defaults are used.
    """
    path = Path(path_like).expanduser().resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_from_yaml(path)

    sidecar = path.with_suffix(".yaml")
    if sidecar.exists():
        return _load_from_yaml(sidecar, image_override=path)

    return OccupancyMapMetadata(
        image_path=path,
        metadata_path=None,
        resolution=DEFAULT_MAP_RESOLUTION,
        origin_xy=_center_origin(path, DEFAULT_MAP_RESOLUTION),
        free_threshold=DEFAULT_FREE_THRESHOLD,
    )


def _load_from_yaml(
    yaml_path: Path,
    image_override: Path | None = None,
) -> OccupancyMapMetadata:
    data = _load_simple_yaml(yaml_path)
    image_path = image_override or (yaml_path.parent / str(data["image"])).resolve()
    resolution = float(data["resolution"])
    origin = data.get("origin", [0.0, 0.0, 0.0])
    negate = int(data.get("negate", 0))
    free_thresh = data.get("free_thresh")
    occupied_thresh = data.get("occupied_thresh")
    free_threshold = _free_thresh_to_gray_threshold(
        free_thresh=float(free_thresh) if free_thresh is not None else None,
        negate=negate,
    )
    return OccupancyMapMetadata(
        image_path=Path(image_path).expanduser().resolve(),
        metadata_path=yaml_path,
        resolution=resolution,
        origin_xy=(float(origin[0]), float(origin[1])),
        free_threshold=free_threshold,
        negate=negate,
        occupied_thresh=float(occupied_thresh) if occupied_thresh is not None else None,
        free_thresh=float(free_thresh) if free_thresh is not None else None,
    )


def _free_thresh_to_gray_threshold(free_thresh: float | None, negate: int) -> int:
    if free_thresh is None:
        return DEFAULT_FREE_THRESHOLD
    if negate:
        threshold = 255.0 * free_thresh
    else:
        threshold = 255.0 * (1.0 - free_thresh)
    return int(round(max(0.0, min(255.0, threshold))))


def _center_origin(image_path: Path, resolution: float) -> tuple[float, float]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return (0.0, 0.0)
    return (
        -0.5 * (width - 1) * resolution,
        -0.5 * (height - 1) * resolution,
    )


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", maxsplit=1)
        data[key.strip()] = _parse_scalar(raw_value.strip())
    return data


def _parse_scalar(value: str):
    value = _strip_inline_comment(value).strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(value):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            return value[:idx]
    return value
